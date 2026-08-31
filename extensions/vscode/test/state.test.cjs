const assert = require('node:assert/strict')
const test = require('node:test')

const { ConnectionStateWriter } = require('../src/host/state.cjs')

test('connection state updates are serialized and the newest configuration wins', async () => {
  let value = { endpoint: 'old', selection: null }
  let releaseOld
  const workspaceState = {
    get: () => value,
    async update(_key, next) {
      if (next.endpoint === 'old' && next.selection) {
        await new Promise((resolve) => { releaseOld = resolve })
      }
      value = next
    },
  }
  const writer = new ConnectionStateWriter(workspaceState, 'connection')
  const old = writer.update(() => ({ endpoint: 'old', selection: { session: 'stale' } }))
  await new Promise((resolve) => setImmediate(resolve))
  const configured = writer.update(() => ({ endpoint: 'new', selection: null }))

  releaseOld()
  await Promise.all([old, configured])

  assert.deepEqual(value, { endpoint: 'new', selection: null })
})

test('a guarded stale selection write can no-op at execution time', async () => {
  let value = { endpoint: 'new', selection: null }
  const workspaceState = {
    get: () => value,
    async update(_key, next) {
      value = next
    },
  }
  const writer = new ConnectionStateWriter(workspaceState, 'connection')

  await writer.update((current) =>
    current.endpoint === 'old'
      ? { endpoint: 'old', selection: { session: 'stale' } }
      : undefined,
  )

  assert.deepEqual(value, { endpoint: 'new', selection: null })
})
