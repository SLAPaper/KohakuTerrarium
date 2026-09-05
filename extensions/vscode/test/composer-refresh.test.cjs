const assert = require('node:assert/strict')
const path = require('node:path')
const { createRequire } = require('node:module')
const { pathToFileURL } = require('node:url')
const test = require('node:test')

const root = path.resolve(__dirname, '..')
const frontendRequire = createRequire(path.resolve(root, '../../src/kohakuterrarium-frontend/package.json'))
const { JSDOM, VirtualConsole } = frontendRequire('jsdom')
const importLocal = (name) => import(pathToFileURL(require.resolve(name)))

async function buildWebview() {
  const { build } = await importLocal('vite')
  const { default: config } = await import(pathToFileURL(path.join(root, 'vite.config.mjs')))
  const { default: vue } = await importLocal('@vitejs/plugin-vue')
  const { default: autoImport } = await importLocal('unplugin-auto-import/vite')
  const result = await build({
    ...config,
    configFile: false,
    logLevel: 'silent',
    plugins: [vue(), autoImport({ imports: ['vue', 'pinia'], dts: false })],
    build: { ...config.build, write: false, sourcemap: false, minify: false },
  })
  const outputs = (Array.isArray(result) ? result : [result]).flatMap((item) => item.output)
  return outputs.find((item) => item.type === 'chunk' && item.isEntry).code
}

async function settle() {
  // Host replies, store initialization and Vue's rendering each queue work.
  for (let i = 0; i < 8; i++) await new Promise((resolve) => setImmediate(resolve))
}

test('real App preserves text and files across pending Refresh, isolates creature IDs, and resets on configuration change', async () => {
  const code = await buildWebview()
  const errors = []
  const virtualConsole = new VirtualConsole()
  virtualConsole.on('jsdomError', (error) => errors.push(error))
  virtualConsole.on('error', (...args) => errors.push(args))
  const dom = new JSDOM('<!doctype html><div id="app"></div>', {
    url: 'https://webview.test/',
    runScripts: 'outside-only',
    pretendToBeVisual: true,
    virtualConsole,
  })
  const { window } = dom
  const { document } = window
  const requests = []
  const session = {
    runtimeId: 'runtime-a',
    title: 'Test Session',
    kind: 'biome',
    isLive: true,
    creatures: [
      { id: 'creature-a', name: 'same-name' },
      { id: 'creature-b', name: 'same-name' },
    ],
  }
  let readyId
  let holdList = false
  let selectionVersion = 1
  const selection = (id) => ({ session: session.runtimeId, targetCreatureId: id })
  const receive = (data) => window.dispatchEvent(new window.MessageEvent('message', { data }))
  const reply = (request, data) => receive({ type: `${request.type}.result`, requestId: request.requestId, data })
  window.acquireVsCodeApi = () => ({
    postMessage(message) {
      requests.push(message)
      queueMicrotask(() => {
        if (message.type === 'session.list' && !holdList) reply(message, [session])
        if (message.type === 'http.history') reply(message, { events: [] })
        if (message.type === 'ws.open') receive({ type: 'ws.opened', socketId: message.socketId })
        if (message.type === 'ws.close') receive({ type: 'ws.closed', socketId: message.socketId })
        if (message.type === 'ws.send') receive({ type: 'ws.send.result', socketId: message.socketId, sendId: message.sendId })
      })
    },
  })
  const textarea = () => {
    const element = document.querySelector('textarea')
    assert.ok(element, 'selected conversation renders the real Composer')
    return element
  }
  const assertBuffer = (text, file) => {
    assert.equal(textarea().value, text)
    const chips = [...document.querySelectorAll('button[aria-label^="Remove "]')]
    assert.deepEqual(
      chips.map((chip) => chip.getAttribute('aria-label')),
      file ? [`Remove ${file}`] : [],
    )
  }
  async function typeAndAttach(text, name) {
    assert.equal(textarea().disabled, false, 'Composer is connected and editable')
    textarea().value = text
    textarea().dispatchEvent(new window.Event('input', { bubbles: true }))
    const input = [...document.querySelectorAll('input[type="file"]')].find((item) => item.accept !== 'image/*')
    assert.ok(input, 'real file input exists')
    Object.defineProperty(input, 'files', { configurable: true, value: [new window.File(['notes'], name, { type: 'text/plain' })] })
    input.dispatchEvent(new window.Event('change', { bubbles: true }))
    await settle()
    assertBuffer(text, name)
  }
  async function answerReady(request, creatureId, connectionId = 'opaque-service-a') {
    readyId = request.requestId
    reply(request, { available: true, automatic: true, readyId, connectionId, selectionVersion, selection: selection(creatureId) })
    await settle()
    assert.equal(textarea().disabled, false)
  }
  async function select(creatureId) {
    receive({
      type: 'selection.changed',
      readyId,
      data: { selection: selection(creatureId), changed: true, selectionVersion: ++selectionVersion },
    })
    await settle()
  }
  try {
    window.eval(code)
    await settle()
    const initial = requests.find((message) => message.type === 'ready')
    assert.ok(initial)
    await answerReady(initial, 'creature-a')
    await typeAndAttach('unsent A', 'a.txt')

    const refresh = document.querySelector('button[aria-label="Refresh Sessions"]')
    assert.ok(refresh)
    refresh.click()
    await settle()
    const readyRequests = requests.filter((message) => message.type === 'ready')
    assert.equal(readyRequests.length, 2, 'Refresh actually issued a new ready request')
    const pending = readyRequests[1]
    assert.notEqual(pending.requestId, initial.requestId)
    assertBuffer('unsent A', 'a.txt') // No ready response has been delivered yet.
    await select('creature-b')
    assertBuffer('unsent A', 'a.txt')
    await answerReady(pending, 'creature-a')
    assertBuffer('unsent A', 'a.txt')

    await select('creature-b')
    assertBuffer('', null)
    await typeAndAttach('unsent B', 'b.txt')
    await select('creature-a')
    assertBuffer('unsent A', 'a.txt')
    await select('creature-b')
    assertBuffer('unsent B', 'b.txt')

    receive({ type: 'configuration.changed' })
    await settle()
    assert.equal(document.querySelector('textarea'), null, 'configuration reset detaches the conversation')
    const reset = requests.filter((message) => message.type === 'ready').at(-1)
    assert.notEqual(reset.requestId, pending.requestId)
    await answerReady(reset, 'creature-b')
    assertBuffer('', null)
    const activeSocket = requests.filter((message) => message.type === 'ws.open').at(-1).socketId
    receive({
      type: 'ws.frame',
      socketId: activeSocket,
      data: JSON.stringify({
        type: 'notification',
        source: 'same-name',
        event_id: 'toast-error',
        surface: 'toast',
        payload: { level: 'error', text: '<img src=x onerror=alert(1)> task failed', duration_ms: 60_000 },
      }),
    })
    await settle()
    const toast = document.querySelector('.kt-notification[role="alert"]')
    assert.ok(toast, 'toast ui_event must be visible instead of silently discarded')
    assert.match(toast.textContent, /task failed/)
    assert.equal(toast.querySelector('img'), null, 'untrusted notification text is not HTML')
    assert.equal(toast.getAttribute('aria-live'), 'assertive')
    const dismiss = toast.querySelector('button[aria-label="Dismiss notification"]')
    assert.ok(dismiss)
    dismiss.click()
    await settle()
    assert.equal(document.querySelector('.kt-notification'), null)
    async function submitGoal(text) {
      textarea().value = text
      textarea().dispatchEvent(new window.Event('input', { bubbles: true }))
      await settle()
      textarea().dispatchEvent(new window.KeyboardEvent('keydown', { key: 'Enter', bubbles: true }))
      await settle()
      return requests.filter((message) => message.type === 'goal.execute').at(-1)
    }
    const firstGoal = await submitGoal('/goal list')
    assert.ok(firstGoal, 'pure goal reaches the Host rather than the no-op shim')
    assert.equal(firstGoal.args, 'list')
    assert.equal(firstGoal.readyId, readyId)
    assert.equal(firstGoal.selectionVersion, selectionVersion)
    assert.equal('creature' in firstGoal || 'session' in firstGoal || 'command' in firstGoal, false)
    assertBuffer('/goal list', null)
    reply(firstGoal, { command: 'goal', success: true, output: 'Goals: one active commitment', error: '' })
    await settle()
    assertBuffer('', null)
    assert.match(document.body.textContent, /Goals: one active commitment/)
    const rejectedGoal = await submitGoal('/goal invalid')
    reply(rejectedGoal, { command: 'goal', success: false, output: '', error: 'Unknown goal operation' })
    await settle()
    assertBuffer('/goal invalid', null)
    assert.match(document.body.textContent, /Unknown goal operation/)
    const failedGoal = await submitGoal('/goal list')
    receive({ type: 'error', requestId: failedGoal.requestId, error: 'Goal service unavailable' })
    await settle()
    assertBuffer('/goal list', null)
    assert.match(document.body.textContent, /Goal service unavailable/)
    const lateGoal = await submitGoal('/goal list')
    await select('creature-a')
    reply(lateGoal, { command: 'goal', success: true, output: 'Old private goal result', error: '' })
    await settle()
    assert.doesNotMatch(document.body.textContent, /Old private goal result/)
    await typeAndAttach('/goal list', 'goal-not-command.txt')
    const goalsBeforeAttachment = requests.filter((message) => message.type === 'goal.execute').length
    textarea().dispatchEvent(new window.KeyboardEvent('keydown', { key: 'Enter', bubbles: true }))
    for (let attempt = 0; attempt < 50 && !requests.some((message) => message.type === 'ws.send'); attempt++) {
      await new Promise((resolve) => setTimeout(resolve, 10))
    }
    await settle()
    assert.equal(requests.filter((message) => message.type === 'goal.execute').length, goalsBeforeAttachment)
    const sentAttachment = requests.filter((message) => message.type === 'ws.send').at(-1)
    assert.ok(sentAttachment, 'goal text with attachments stays on the chat transport')
    assert.equal(JSON.parse(sentAttachment.data).content.length > 1, true)
    assertBuffer('', null)
    await select('creature-a')
    assertBuffer('', null) // Inactive buckets must be cleared as well.
    await typeAndAttach('service A only', 'private.txt')
    refresh.click()
    await settle()
    const replaced = requests.filter((message) => message.type === 'ready').at(-1)
    receive({
      type: 'selection.changed',
      readyId: replaced.requestId,
      connectionId: 'opaque-service-b',
      data: { selection: selection('creature-a'), changed: true, selectionVersion: ++selectionVersion },
    })
    await settle()
    assertBuffer('', null)
    await answerReady(replaced, 'creature-a', 'opaque-service-b')
    assertBuffer('', null)
    holdList = true
    await select('creature-b')
    const delayedList = requests.filter((message) => message.type === 'session.list').at(-1)
    const beforeBlockedGoal = requests.filter((message) => message.type === 'goal.execute').length
    await submitGoal('/goal list')
    assert.match(document.body.textContent, /reconcil|reconnect/i)
    assertBuffer('/goal list', null)
    assert.equal(requests.filter((message) => message.type === 'goal.execute').length, beforeBlockedGoal)
    refresh.click()
    await settle()
    const failed = requests.filter((message) => message.type === 'ready').at(-1)
    assert.notEqual(failed.requestId, replaced.requestId, 'Refresh starts without waiting for old topology I/O')
    receive({ type: 'error', requestId: failed.requestId, error: 'connection failed' })
    await settle()
    holdList = false
    reply(delayedList, [session])
    await settle()
    assert.equal(document.querySelector('textarea'), null, 'late selection cannot resurrect a failed ready')
    receive({
      type: 'selection.changed',
      readyId: failed.requestId,
      connectionId: 'opaque-service-b',
      data: { selection: selection('creature-b'), changed: true, selectionVersion: ++selectionVersion },
    })
    await settle()
    assert.equal(document.querySelector('textarea'), null, 'notifications arriving after failure are also rejected')
    assert.deepEqual(errors, [], 'built webview emitted no browser/runtime errors')
  } finally {
    window.close() // Disposes jsdom's animation frames and request/store timers.
  }
})
