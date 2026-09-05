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
