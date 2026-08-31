const assert = require('node:assert/strict')
const test = require('node:test')

const { TopologyWatcher } = require('../src/host/topology.cjs')

function fakeSocket() {
  return {
    closeCount: 0,
    close() {
      this.closeCount++
    },
  }
}

test('topology watcher treats only relevant frames as invalidations', async () => {
  const calls = []
  const socket = fakeSocket()
  const watcher = new TopologyWatcher({
    socketFactory: () => socket,
    endpoint: 'http://127.0.0.1:8000',
    token: 'secret',
    onInvalidate: async (frame) => calls.push(frame.type),
  })

  watcher.start()
  socket.onmessage({ data: Buffer.from(JSON.stringify({ type: 'turn_completed' })) })
  socket.onmessage({ data: Buffer.from(JSON.stringify({ type: 'topology_changed' })) })
  socket.onmessage({ data: Buffer.from(JSON.stringify({ type: 'creature_stopped' })) })
  socket.onmessage({ data: Buffer.from('not-json') })
  await new Promise((resolve) => setImmediate(resolve))

  assert.deepEqual(calls, ['topology_changed', 'creature_stopped'])
})

test('topology watcher ignores late frames from a replaced socket', async () => {
  const calls = []
  const sockets = []
  const watcher = new TopologyWatcher({
    socketFactory: () => {
      const socket = fakeSocket()
      sockets.push(socket)
      return socket
    },
    endpoint: 'http://127.0.0.1:8000',
    token: 'secret',
    onInvalidate: async (frame) => calls.push(frame.marker),
  })

  watcher.start()
  watcher.start()
  sockets[0].onmessage({
    data: JSON.stringify({ type: 'topology_changed', marker: 'stale' }),
  })
  sockets[1].onmessage({
    data: JSON.stringify({ type: 'topology_changed', marker: 'current' }),
  })
  await new Promise((resolve) => setImmediate(resolve))

  assert.deepEqual(calls, ['current'])
})

test('topology watcher closes explicitly and never reconnects itself', () => {
  const sockets = []
  const watcher = new TopologyWatcher({
    socketFactory: () => {
      const socket = fakeSocket()
      sockets.push(socket)
      return socket
    },
    endpoint: 'http://127.0.0.1:8000',
    token: 'secret',
    onInvalidate: async () => {},
  })

  watcher.start()
  assert.equal(typeof sockets[0].onerror, 'function')
  sockets[0].onerror({ message: 'connection refused' })
  sockets[0].onclose?.({ code: 1006 })
  watcher.close()

  assert.equal(sockets.length, 1)
  assert.equal(sockets[0].closeCount, 1)
})
