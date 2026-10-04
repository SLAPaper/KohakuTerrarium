import { createPinia, setActivePinia } from "pinia"
import { beforeEach, describe, expect, it, vi } from "vitest"
import { terrariumAPI } from "@/utils/api"
import { useChatStore } from "./chat.js"

describe("chat branch operation state", () => {
  beforeEach(() => {
    setActivePinia(createPinia())
  })

  it("reconciles a predicted branch collision from processing_start atomically", () => {
    const chat = useChatStore()
    chat.activeTab = "main"
    chat.tabs = ["main"]
    chat.messagesByTab = { main: [] }
    chat.eventsByTab = {
      main: [
        {
          type: "processing_start",
          turn_index: 1,
          branch_id: 2,
          _optimistic: true,
        },
      ],
    }
    chat.branchViewByTab = { main: { 1: 2 } }
    chat._streamingBranchByTab = { main: { turnIndex: 1, branchId: 2 } }
    chat._branchResyncPendingByTab = {
      main: { active: true, expectedBranchByTurn: { 1: 2 } },
    }
    chat.branchOperationByTab.main = {
      type: "edit",
      phase: "starting",
      turnIndex: 1,
      predictedBranch: 2,
      requestId: "request-1",
    }

    chat._onMessage({
      type: "processing_start",
      source: "main",
      turn_index: 1,
      branch_id: 3,
    })

    expect(chat.branchViewByTab.main).toEqual({ 1: 3 })
    expect(chat._streamingBranchByTab.main).toEqual({ turnIndex: 1, branchId: 3 })
    expect(chat._branchResyncPendingByTab.main.expectedBranchByTurn).toEqual({ 1: 3 })
    expect(chat.eventsByTab.main[0]).toMatchObject({ turn_index: 1, branch_id: 3 })
    expect(chat.branchOperationByTab.main).toMatchObject({ phase: "accepted", branchId: 3 })
  })

  it("ignores processing_start from another request id", () => {
    const chat = useChatStore()
    chat._instanceGeneration = 2
    chat.activeTab = "main"
    chat.tabs = ["main"]
    chat.messagesByTab = { main: [] }
    chat.branchOperationByTab.main = {
      type: "edit",
      phase: "starting",
      turnIndex: 1,
      predictedBranch: 2,
      requestId: "request-1",
      instanceGeneration: 2,
    }

    chat._onMessage({
      type: "processing_start",
      source: "main",
      turn_index: 1,
      branch_id: 3,
      request_id: "request-2",
    })

    expect(chat.branchOperationByTab.main).toMatchObject({
      phase: "starting",
      requestId: "request-1",
    })
  })

  it("clears branch operations on idle, reset, and connection errors", () => {
    const chat = useChatStore()
    chat.activeTab = "main"
    chat.tabs = ["main"]
    chat.messagesByTab = { main: [] }
    chat.branchOperationByTab.main = { type: "edit", phase: "accepted" }

    chat._onMessage({ type: "idle", source: "main" })
    expect(chat.branchOperationByTab.main).toBeNull()

    chat.branchOperationByTab.main = { type: "edit", phase: "starting" }
    chat._onMessage({ type: "error", source: "main", content: "failed" })
    expect(chat.branchOperationByTab.main).toBeNull()
    expect(chat.branchOperationErrorByTab.main).toBe("failed")

    chat.branchOperationByTab.main = { type: "edit", phase: "starting" }
    chat.resetForRouteSwitch()
    expect(chat.branchOperationByTab).toEqual({})
  })

  it("keeps operations independent across creature tabs", () => {
    const chat = useChatStore()
    chat._setBranchOperation("alpha", { type: "edit", phase: "starting" })
    chat._setBranchOperation("beta", { type: "regenerate", phase: "starting" })

    expect(chat.branchOperationByTab.alpha.type).toBe("edit")
    expect(chat.branchOperationByTab.beta.type).toBe("regenerate")

    chat._failBranchOperation("alpha", new Error("collision"))
    expect(chat.branchOperationByTab.alpha).toBeNull()
    expect(chat.branchOperationErrorByTab.alpha).toBe("collision")
    expect(chat.branchOperationByTab.beta.type).toBe("regenerate")
  })

  describe("branch view capture on send", () => {
    it("rejects an invalid explicit selection without sending on another branch", async () => {
      const chat = useChatStore()
      chat.activeTab = "main"
      chat.tabs = ["main"]
      chat.messagesByTab = { main: [] }
      const wsSend = vi.fn()
      chat._ws = { readyState: WebSocket.OPEN, send: wsSend }
      for (const view of [{ 1: "wrong" }, { 1: true }, { 0: 1 }, { 1: 0 }]) {
        chat.branchViewByTab.main = view
        await expect(chat.send("must not run")).rejects.toThrow(/branch/i)
        expect(wsSend).not.toHaveBeenCalled()
        expect(chat.messagesByTab.main).toEqual([])
      }
    })
    it("attaches detached branch_view copy when sending under an older branch view and resists subsequent mutations", async () => {
      const chat = useChatStore()
      chat.activeTab = "main"
      chat.tabs = ["main"]
      chat.messagesByTab = { main: [] }
      const wsSend = vi.fn()
      chat._ws = { readyState: WebSocket.OPEN, send: wsSend }

      chat.selectBranch(1, 1)
      expect(chat.branchViewByTab.main).toEqual({ 1: 1 })

      await chat.send("continue from branch 1")

      expect(wsSend).toHaveBeenCalledTimes(1)
      const payload = JSON.parse(wsSend.mock.calls[0][0])
      expect(payload).toMatchObject({
        type: "input",
        target: "main",
        content: [{ type: "text", text: "continue from branch 1" }],
        branch_view: { 1: 1 },
      })

      // Mutating original branchViewByTab after send must not change the payload that was already sent
      chat.branchViewByTab.main[1] = 99
      chat.branchViewByTab.main[2] = 42
      expect(payload.branch_view).toEqual({ 1: 1 })
    })

    it("keeps branch views isolated across active tabs and omits branch_view when empty", async () => {
      const chat = useChatStore()
      chat.tabs = ["alpha", "beta", "gamma"]
      chat.messagesByTab = { alpha: [], beta: [], gamma: [] }
      const wsSend = vi.fn()
      chat._ws = { readyState: WebSocket.OPEN, send: wsSend }

      chat.branchViewByTab = {
        alpha: { 1: 2 },
        beta: { 2: 4 },
        gamma: {},
      }

      // Send on alpha
      chat.activeTab = "alpha"
      await chat.send("msg from alpha")
      const alphaPayload = JSON.parse(wsSend.mock.calls[0][0])
      expect(alphaPayload.target).toBe("alpha")
      expect(alphaPayload.branch_view).toEqual({ 1: 2 })

      // Send on beta
      chat.activeTab = "beta"
      await chat.send("msg from beta")
      const betaPayload = JSON.parse(wsSend.mock.calls[1][0])
      expect(betaPayload.target).toBe("beta")
      expect(betaPayload.branch_view).toEqual({ 2: 4 })

      // Send on gamma (empty map) -> legacy payload without branch_view
      chat.activeTab = "gamma"
      await chat.send("msg from gamma")
      const gammaPayload = JSON.parse(wsSend.mock.calls[2][0])
      expect(gammaPayload.target).toBe("gamma")
      expect(gammaPayload.branch_view).toBeUndefined()
      expect("branch_view" in gammaPayload).toBe(false)
    })
  })

  describe("ancestor branch switching and descendant overrides (F5)", () => {
    it("prunes descendant overrides when changing an ancestor branch", () => {
      const chat = useChatStore()
      chat.activeTab = "main"
      chat.tabs = ["main"]
      chat.messagesByTab = { main: [] }
      chat.eventsByTab = {
        main: [
          { event_id: 1, turn_index: 1, branch_id: 1 },
          { event_id: 2, turn_index: 1, branch_id: 2 },
          { event_id: 3, turn_index: 2, branch_id: 1 },
          { event_id: 4, turn_index: 3, branch_id: 1 },
        ],
      }
      chat.branchViewByTab = {
        main: { 1: 1, 2: 3, 3: 5 },
      }

      // Switching ancestor turn 1 from 1 to 2 prunes later descendant overrides (turns 2 and 3)
      chat.selectBranch(1, 2)
      expect(chat.branchViewByTab.main).toEqual({ 1: 2 })

      // Switching intermediate ancestor turn 2 prunes turn 3 but preserves turn 1
      chat.branchViewByTab.main = { 1: 1, 2: 2, 3: 4 }
      chat.selectBranch(2, 5)
      expect(chat.branchViewByTab.main).toEqual({ 1: 1, 2: 5 })
    })

    it("preserves descendant overrides when re-selecting the same effective ancestor (explicit and implicit)", () => {
      const chat = useChatStore()
      chat.activeTab = "main"
      chat.tabs = ["main"]
      chat.messagesByTab = { main: [] }
      chat.eventsByTab = {
        main: [
          { event_id: 1, turn_index: 1, branch_id: 1 },
          { event_id: 2, turn_index: 1, branch_id: 2 },
          { event_id: 3, turn_index: 2, branch_id: 1 },
        ],
      }

      // Explicit same ancestor: turn 1 explicitly 1 in view
      chat.branchViewByTab = {
        main: { 1: 1, 2: 3 },
      }
      chat.selectBranch(1, 1)
      expect(chat.branchViewByTab.main).toEqual({ 1: 1, 2: 3 })

      // Implicit same ancestor: view has no turn 1 key, but effective branch is 2 (highest candidate)
      chat.branchViewByTab = {
        main: { 2: 3 },
      }
      chat.selectBranch(1, 2)
      expect(chat.branchViewByTab.main).toEqual({ 1: 2, 2: 3 })

      // Implicit ancestor changed: selecting branch 1 (differs from effective branch 2) prunes turn 2
      chat.branchViewByTab = {
        main: { 2: 3 },
      }
      chat.selectBranch(1, 1)
      expect(chat.branchViewByTab.main).toEqual({ 1: 1 })
    })

    it("sends correct branch_view payload across ancestor roundtrip", async () => {
      const chat = useChatStore()
      chat.activeTab = "main"
      chat.tabs = ["main"]
      chat.messagesByTab = { main: [] }
      chat.eventsByTab = {
        main: [
          { event_id: 1, turn_index: 1, branch_id: 1 },
          { event_id: 2, turn_index: 1, branch_id: 2 },
          { event_id: 3, turn_index: 2, branch_id: 1 },
        ],
      }
      const wsSend = vi.fn()
      chat._ws = { readyState: WebSocket.OPEN, send: wsSend }

      chat.branchViewByTab = { main: { 1: 1, 2: 2 } }

      // Switch ancestor to branch 2 -> prunes turn 2 descendant
      chat.selectBranch(1, 2)
      expect(chat.branchViewByTab.main).toEqual({ 1: 2 })

      await chat.send("message on branch 2")
      expect(wsSend).toHaveBeenCalledTimes(1)
      const payload1 = JSON.parse(wsSend.mock.calls[0][0])
      expect(payload1.branch_view).toEqual({ 1: 2 })

      // Switch ancestor back to branch 1
      chat.selectBranch(1, 1)
      expect(chat.branchViewByTab.main).toEqual({ 1: 1 })

      await chat.send("message back on branch 1")
      expect(wsSend).toHaveBeenCalledTimes(2)
      const payload2 = JSON.parse(wsSend.mock.calls[1][0])
      expect(payload2.branch_view).toEqual({ 1: 1 })
    })
  })

  describe("pending branch operation send guard", () => {
    it("rejects send before any optimistic or WS side effects when branch operation is pending, while other tabs and channel tabs remain unaffected", async () => {
      const chat = useChatStore()
      chat.tabs = ["main", "secondary", "ch:general"]
      chat.messagesByTab = { main: [], secondary: [], "ch:general": [] }
      const wsSend = vi.fn()
      chat._ws = { readyState: WebSocket.OPEN, send: wsSend }

      // Set pending branch operation on 'main'
      chat.branchOperationByTab.main = {
        type: "edit",
        phase: "starting",
        turnIndex: 1,
        predictedBranch: 3,
        requestId: "req-pending-1",
      }
      chat.activeTab = "main"

      // Attempt send on 'main' with pending operation: must reject with identifiable pending error
      await expect(chat.send("blocked msg")).rejects.toThrow(/pending|branch operation/i)

      // Guarded: no WS send, no optimistic messages, no queued messages, no recent user inputs, not processing
      expect(wsSend).not.toHaveBeenCalled()
      expect(chat.messagesByTab.main).toEqual([])
      expect(chat.queuedMessagesByTab.main).toBeUndefined()
      expect(Object.keys(chat._recentUserInputs)).toHaveLength(0)
      expect(chat.processingByTab.main).toBeFalsy()

      // Slash command goal RPC is also rejected before any command execution
      const commandSpy = vi.spyOn(terrariumAPI, "executeCreatureCommand")
      await expect(chat.send("/goal test")).rejects.toThrow(/pending|branch operation/i)
      expect(commandSpy).not.toHaveBeenCalled()
      commandSpy.mockRestore()

      // Other tab 'secondary' has no pending operation and sends normally
      chat.activeTab = "secondary"
      await chat.send("msg from secondary")
      expect(wsSend).toHaveBeenCalledTimes(1)
      const secPayload = JSON.parse(wsSend.mock.calls[0][0])
      expect(secPayload.target).toBe("secondary")
      expect(secPayload.content).toEqual([{ type: "text", text: "msg from secondary" }])
      expect(chat.messagesByTab.secondary).toHaveLength(1)

      // Channel tab allows original path and does not get blocked by creature branch operation guard
      chat.activeTab = "ch:general"
      chat.branchOperationByTab["ch:general"] = { type: "edit", phase: "starting" }
      const channelSpy = vi.spyOn(terrariumAPI, "sendToChannel").mockResolvedValue({})
      await chat.send("channel msg")
      expect(channelSpy).toHaveBeenCalledTimes(1)
      channelSpy.mockRestore()
    })
  })
})
