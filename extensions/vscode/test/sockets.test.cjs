const assert = require('node:assert/strict')
const test = require('node:test')

const { SocketOwners } = require('../src/host/sockets.cjs')

function fakeSocket() {
  return {
    sent: [],
    closeCount: 0,
    send(data) {
      this.sent.push(data)
    },
    close() {
      this.closeCount++
    },
  }
}

test('socket owners ignore stale callbacks and preserve a reused id owner', () => {
  const messages = []
  const owners = new SocketOwners()
  const generation = owners.begin()
  const first = fakeSocket()
  const second = fakeSocket()

  owners.open(generation, 1, () => first, { postMessage: (message) => messages.push(message) })
  owners.open(generation, 1, () => second, { postMessage: (message) => messages.push(message) })
  first.onopen()
  first.onmessage({ data: 'stale' })
  first.onclose({ code: 1000 })
  second.onopen()
  second.onmessage({ data: Buffer.from('current') })

  assert.equal(first.closeCount, 1)
  assert.deepEqual(messages, [
    { type: 'ws.opened', id: 1 },
    { type: 'ws.frame', id: 1, data: 'current' },
  ])
  assert.equal(owners.send(generation, 1, 'input'), true)
  assert.deepEqual(second.sent, ['input'])
})

test('closing a generation notifies the Bridge, closes sockets, and rejects future sends', () => {
  const messages = []
  const owners = new SocketOwners()
  const generation = owners.begin()
  const socket = fakeSocket()
  owners.open(generation, 3, () => socket, { postMessage: (message) => messages.push(message) })

  owners.closeGeneration(generation)

  assert.deepEqual(messages, [{ type: 'ws.closed', id: 3, code: 1000 }])
  assert.equal(socket.closeCount, 1)
  assert.equal(owners.send(generation, 3, 'input'), false)
})

test('asynchronous socket errors emit error and close exactly once', () => {
  const messages = []
  const owners = new SocketOwners()
  const generation = owners.begin()
  const socket = fakeSocket()
  owners.open(generation, 6, () => socket, { postMessage: (message) => messages.push(message) })

  socket.onerror()
  socket.onclose?.({ code: 1006 })

  assert.deepEqual(messages, [
    { type: 'ws.error', id: 6 },
    { type: 'ws.closed', id: 6, code: 1011 },
  ])
  assert.equal(owners.send(generation, 6, 'input'), false)
})

test('failed socket creation emits a deterministic error and close frame', () => {
  const messages = []
  const owners = new SocketOwners()
  const generation = owners.begin()

  assert.throws(
    () => owners.open(generation, 4, () => { throw Error('connect failed') }, { postMessage: (message) => messages.push(message) }),
    /connect failed/,
  )

  assert.deepEqual(messages, [
    { type: 'ws.error', id: 4 },
    { type: 'ws.closed', id: 4, code: 1011 },
  ])
})

test('local close emits a deterministic close frame before releasing ownership', () => {
  const messages = []
  const owners = new SocketOwners()
  const generation = owners.begin()
  const socket = fakeSocket()
  owners.open(generation, 5, () => socket, { postMessage: (message) => messages.push(message) })

  assert.equal(owners.closeSocket(generation, 5), true)

  assert.deepEqual(messages, [{ type: 'ws.closed', id: 5, code: 1000 }])
  assert.equal(owners.send(generation, 5, 'input'), false)
})
