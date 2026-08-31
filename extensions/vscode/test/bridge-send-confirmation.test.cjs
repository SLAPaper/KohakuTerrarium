const assert = require('node:assert/strict')
const fs = require('node:fs')
const path = require('node:path')
const test = require('node:test')

let BridgeWebSocket

test.before(async () => {
  const source = fs.readFileSync(path.resolve(__dirname, '../src/webview/bridge.js'), 'utf8')
  const moduleUrl = `data:text/javascript;base64,${Buffer.from(
    `const markRaw = value => value;${source}`,
  ).toString('base64')}`
  ;({ BridgeWebSocket } = await import(moduleUrl))
})

test.beforeEach(() => {
  BridgeWebSocket.disposeAll(Error('test reset'))
  BridgeWebSocket.sockets.clear()
  BridgeWebSocket.nextId = 1
  BridgeWebSocket.nextSendId = 1
  BridgeWebSocket.confirmationTimeout = 1000
})

test('captured send retains WebSocket signature and resolves only its correlated Host result', async () => {
  const posts = []
  BridgeWebSocket.post = (message) => posts.push(message)
  const first = new BridgeWebSocket('ws://bridge')
  const second = new BridgeWebSocket('ws://bridge')
  BridgeWebSocket.receive({ type: 'ws.opened', id: first.id })
  BridgeWebSocket.receive({ type: 'ws.opened', id: second.id })

  const captured = BridgeWebSocket.captureSend(() => first.send('one'))
  const other = BridgeWebSocket.captureSend(() => second.send('two'))
  assert.equal(captured.value, undefined)
  assert.equal(posts.at(-2).sendId !== posts.at(-1).sendId, true)

  BridgeWebSocket.receive({ type: 'ws.send.result', id: second.id, sendId: posts.at(-1).sendId })
  await other.confirmation
  let settled = false
  captured.confirmation.finally(() => (settled = true))
  await Promise.resolve()
  assert.equal(settled, false)

  BridgeWebSocket.receive({ type: 'ws.send.result', id: first.id, sendId: posts.at(-2).sendId })
  await captured.confirmation
})

test('capturing a send fails when the chat store emits no socket frame', () => {
  assert.throws(
    () => BridgeWebSocket.captureSend(() => undefined, { requireConfirmation: true }),
    /did not reach the Host/,
  )
})

test('correlated failure, close, timeout, and disposal reject pending confirmation', async () => {
  BridgeWebSocket.post = () => {}
  const socket = new BridgeWebSocket('ws://bridge')
  BridgeWebSocket.receive({ type: 'ws.opened', id: socket.id })
  const failed = BridgeWebSocket.captureSend(() => socket.send('failed'))
  const failedSendId = BridgeWebSocket.nextSendId - 1
  BridgeWebSocket.receive({ type: 'ws.send.error', id: socket.id, sendId: failedSendId, error: 'not open' })
  await assert.rejects(failed.confirmation, /not open/)

  const closed = BridgeWebSocket.captureSend(() => socket.send('closed'))
  BridgeWebSocket.receive({ type: 'ws.closed', id: socket.id })
  await assert.rejects(closed.confirmation, /closed/i)

  const timeoutSocket = new BridgeWebSocket('ws://bridge')
  BridgeWebSocket.receive({ type: 'ws.opened', id: timeoutSocket.id })
  BridgeWebSocket.confirmationTimeout = 1
  const timed = BridgeWebSocket.captureSend(() => timeoutSocket.send('timed'))
  await assert.rejects(timed.confirmation, /timed out/i)

  BridgeWebSocket.confirmationTimeout = 1000
  const disposed = BridgeWebSocket.captureSend(() => timeoutSocket.send('disposed'))
  BridgeWebSocket.disposeAll(Error('disposed'))
  await assert.rejects(disposed.confirmation, /disposed/)
  BridgeWebSocket.receive({ type: 'ws.send.result', id: timeoutSocket.id, sendId: BridgeWebSocket.nextSendId - 1 })
})
