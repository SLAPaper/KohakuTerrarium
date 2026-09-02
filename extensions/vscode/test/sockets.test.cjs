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

test('stale generation opens and closes emit a terminal close envelope', () => {
  const messages = []
  const owners = new SocketOwners()
  const generation = owners.begin()
  const view = { postMessage: (message) => messages.push(message) }

  assert.equal(owners.open(generation - 1, 9, fakeSocket, view), null)
  assert.equal(owners.closeSocket(generation - 1, 10, view), false)
  assert.deepEqual(messages, [
    { type: 'ws.closed', socketId: 9, code: 1008 },
    { type: 'ws.closed', socketId: 10, code: 1008 },
  ])
})

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
    { type: 'ws.opened', socketId: 1 },
    { type: 'ws.frame', socketId: 1, data: 'current' },
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

  assert.deepEqual(messages, [{ type: 'ws.closed', socketId: 3, code: 1000 }])
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
    { type: 'ws.error', socketId: 6 },
    { type: 'ws.closed', socketId: 6, code: 1011 },
  ])
  assert.equal(owners.send(generation, 6, 'input'), false)
})

test('failed socket creation emits a deterministic error and close frame', () => {
  const messages = []
  const owners = new SocketOwners()
  const generation = owners.begin()

  assert.throws(
    () =>
      owners.open(
        generation,
        4,
        () => {
          throw Error('connect failed')
        },
        { postMessage: (message) => messages.push(message) },
      ),
    /connect failed/,
  )

  assert.deepEqual(messages, [{ type: 'ws.closed', socketId: 4, code: 1011 }])
})

test('local close without an owner still emits a terminal close frame', () => {
  const messages = []
  const owners = new SocketOwners()
  const generation = owners.begin()

  assert.equal(owners.closeSocket(generation, 7, { postMessage: (message) => messages.push(message) }), false)
  assert.deepEqual(messages, [{ type: 'ws.closed', socketId: 7, code: 1000 }])
})

test('local close emits a deterministic close frame before releasing ownership', () => {
  const messages = []
  const owners = new SocketOwners()
  const generation = owners.begin()
  const socket = fakeSocket()
  owners.open(generation, 5, () => socket, { postMessage: (message) => messages.push(message) })

  assert.equal(owners.closeSocket(generation, 5), true)

  assert.deepEqual(messages, [{ type: 'ws.closed', socketId: 5, code: 1000 }])
  assert.equal(owners.send(generation, 5, 'input'), false)
})
