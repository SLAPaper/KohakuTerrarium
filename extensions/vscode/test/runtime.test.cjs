const assert = require('node:assert/strict')
const test = require('node:test')

const { RuntimeHost } = require('../src/host/runtime.cjs')

function harness() {
  const updates = []
  const posts = []
  const socketCalls = []
  const client = {
    createdPayload: null,
    listOpen: async () => [],
    resume: async () => ({
      instance_id: 'graph-resumed',
      type: 'agent',
      session_name: 'saved-one',
      session: {
        session_id: 'graph-resumed',
        name: 'saved-one',
        creatures: [{ creature_id: 'creature-resumed', name: 'resumed' }],
      },
    }),
    stop: async () => ({ status: 'stopped' }),
    async createCreature(payload) {
      this.createdPayload = payload
      return {
        session_id: 'graph-created',
        type: 'creature',
        config_name: 'swe',
        creatures: [{ creature_id: 'creature-created', name: 'swe' }],
      }
    },
    active: async () => ({
      session_id: 'graph-live',
      type: 'terrarium',
      config_name: 'team',
      creatures: [{ creature_id: 'creature-beta', name: 'beta' }],
    }),
    history: async () => ({ events: [] }),
    interrupt: async () => ({ ok: true }),
  }
  const state = {
    selection: null,
    async updateSelection(selection) {
      this.selection = selection
      updates.push(selection)
    },
  }
  const sockets = {
    beginCount: 0,
    begin() {
      this.beginCount++
      return this.beginCount
    },
    open(generation, id, factory) {
      socketCalls.push({ generation, id, socket: factory() })
    },
    send() {
      return true
    },
    closeSocket() {
      return true
    },
  }
  const host = new RuntimeHost({
    client,
    state,
    sockets,
    post: (message) => posts.push(message),
    getDefaultCreature: () => '@kt-biome/creatures/swe',
    getWorkspacePath: () => 'C:/workspace',
    socketFactory: (url, protocols) => ({ url, protocols }),
    webSocketBase: 'ws://127.0.0.1:8000',
    token: 'host-secret',
  })
  return { client, host, posts, socketCalls, sockets, state, updates }
}

test('session.create uses Host-owned config and workspace then returns a normalized live session', async () => {
  const { client, host, posts } = harness()

  await host.handle({ type: 'session.create', id: 1 })

  assert.equal(posts[0].type, 'session.create.result')
  assert.deepEqual(client.createdPayload, {
    configPath: '@kt-biome/creatures/swe',
    pwd: 'C:/workspace',
    name: 'VS Code session',
  })
  assert.equal(posts[0].data.runtimeId, 'graph-created')
  assert.deepEqual(posts[0].data.creatures, [{ id: 'creature-created', name: 'swe' }])
  assert.equal(posts[0].data.createdPayload, undefined)
})

test('session.select validates the active runtime, persists stable identity, and resets socket ownership', async () => {
  const { host, posts, sockets, state } = harness()

  await host.handle({ type: 'session.select', id: 2, session: 'graph-live', creatureId: 'creature-beta' })

  assert.equal(sockets.beginCount, 2)
  assert.deepEqual(state.selection, {
    session: 'graph-live',
    graph: 'graph-live',
    creature: 'beta',
    targetCreatureId: 'creature-beta',
  })
  assert.deepEqual(posts[0], {
    type: 'session.select.result',
    id: 2,
    data: state.selection,
  })
})

test('session.select preserves the old socket generation when persistence fails', async () => {
  const { host, sockets, state } = harness()
  state.updateSelection = async () => {
    throw Error('persistence failed')
  }
  const before = sockets.beginCount

  await assert.rejects(
    host.handle({ type: 'session.select', id: 8, session: 'graph-live', creatureId: 'creature-beta' }),
    /persistence failed/,
  )

  assert.equal(sockets.beginCount, before)
})

test('ws.open omits the token subprotocol for loopback-bypass mode', async () => {
  const { host, socketCalls, state } = harness()
  state.selection = {
    session: 'graph-live',
    graph: 'graph-live',
    creature: 'beta',
    targetCreatureId: 'creature-beta',
  }
  host.token = ''

  await host.handle({ type: 'ws.open', id: 7 })

  assert.deepEqual(socketCalls[0].socket.protocols, [])
})

test('ws.open constructs the authenticated selected-creature route on the Host', async () => {
  const { host, socketCalls, state } = harness()
  state.selection = {
    session: 'graph live',
    graph: 'graph live',
    creature: 'beta/name',
    targetCreatureId: 'creature-beta',
  }
  host.generation = 4

  await host.handle({ type: 'ws.open', id: 9 })

  assert.equal(socketCalls[0].generation, 4)
  assert.equal(socketCalls[0].id, 9)
  assert.equal(
    socketCalls[0].socket.url,
    'ws://127.0.0.1:8000/ws/sessions/graph%20live/creatures/beta%2Fname/chat',
  )
  assert.deepEqual(socketCalls[0].socket.protocols, ['kt-token.host-secret'])
})

test('clearing selection persists before rotating socket ownership', async () => {
  const { host, sockets, state } = harness()
  state.selection = {
    session: 'graph-old',
    graph: 'graph-old',
    creature: 'old',
    targetCreatureId: 'creature-old',
  }
  const before = sockets.beginCount

  await host.clearSelection()

  assert.equal(state.selection, null)
  assert.equal(sockets.beginCount, before + 1)
})

test('session.resume validates a dormant session and returns a normalized live session', async () => {
  const { client, host, posts } = harness()
  client.listOpen = async () => [{ savedName: 'saved-one', isLive: false }]

  await host.handle({ type: 'session.resume', id: 10, savedName: 'saved-one' })

  assert.deepEqual(posts[0].data, {
    conversationId: null,
    runtimeId: 'graph-resumed',
    savedName: 'saved-one',
    title: 'saved-one',
    isLive: true,
    kind: 'creature',
    creatures: [{ id: 'creature-resumed', name: 'resumed' }],
  })
})

test('session.stop clears ownership only after the Host stop succeeds', async () => {
  const { client, host, posts, sockets, state } = harness()
  state.selection = {
    session: 'graph-selected',
    graph: 'graph-selected',
    creature: 'beta',
    targetCreatureId: 'creature-beta',
  }
  host.generation = sockets.begin()
  const before = sockets.beginCount

  await host.handle({
    type: 'session.stop',
    id: 11,
    session: 'graph-selected',
    creatureId: 'creature-beta',
  })

  assert.equal(state.selection, null)
  assert.equal(sockets.beginCount, before + 1)
  assert.deepEqual(posts.at(-1), { type: 'session.stop.result', id: 11, data: { ok: true } })
  client.stop = async () => { throw Error('stop failed') }
  state.selection = {
    session: 'graph-next',
    graph: 'graph-next',
    creature: 'beta',
    targetCreatureId: 'creature-beta',
  }
  const failedBefore = sockets.beginCount
  await assert.rejects(
    host.handle({ type: 'session.stop', id: 12, session: 'graph-next', creatureId: 'creature-beta' }),
    /stop failed/,
  )
  assert.notEqual(state.selection, null)
  assert.equal(sockets.beginCount, failedBefore)
})

test('selection reconciliation relocates by stable Creature id and rotates transport ownership', async () => {
  const { client, host, sockets, state } = harness()
  state.selection = {
    session: 'graph-old',
    graph: 'graph-old',
    creature: 'old-name',
    targetCreatureId: 'creature-beta',
  }
  client.listOpen = async () => [
    {
      runtimeId: 'graph-new',
      isLive: true,
      creatures: [{ id: 'creature-beta', name: 'new-name' }],
    },
  ]
  const before = sockets.beginCount

  const result = await host.reconcileSelection()

  assert.deepEqual(result.selection, {
    session: 'graph-new',
    graph: 'graph-new',
    creature: 'new-name',
    targetCreatureId: 'creature-beta',
  })
  assert.deepEqual(state.selection, result.selection)
  assert.equal(sockets.beginCount, before + 1)
})

test('selection reconciliation fails closed when the stable target disappears', async () => {
  const { client, host, sockets, state } = harness()
  state.selection = {
    session: 'graph-old',
    graph: 'graph-old',
    creature: 'beta',
    targetCreatureId: 'creature-beta',
  }
  client.listOpen = async () => [
    { runtimeId: 'graph-other', isLive: true, creatures: [{ id: 'creature-alpha', name: 'alpha' }] },
  ]
  const before = sockets.beginCount

  const result = await host.reconcileSelection()

  assert.equal(result.selection, null)
  assert.equal(state.selection, null)
  assert.equal(sockets.beginCount, before + 1)
})

test('selection reconciliation preserves stale ownership when the service is unavailable', async () => {
  const { client, host, sockets, state } = harness()
  state.selection = {
    session: 'graph-stale',
    graph: 'graph-stale',
    creature: 'beta',
    targetCreatureId: 'creature-beta',
  }
  client.listOpen = async () => { throw Error('service unavailable') }
  const before = sockets.beginCount

  await assert.rejects(host.reconcileSelection(), /service unavailable/)

  assert.deepEqual(state.selection, {
    session: 'graph-stale',
    graph: 'graph-stale',
    creature: 'beta',
    targetCreatureId: 'creature-beta',
  })
  assert.equal(sockets.beginCount, before)
})

test('selection reconciliation keeps ownership stable when the attachment is unchanged', async () => {
  const { client, host, sockets, state } = harness()
  state.selection = {
    session: 'graph-live',
    graph: 'graph-live',
    creature: 'beta',
    targetCreatureId: 'creature-beta',
  }
  client.listOpen = async () => [
    { runtimeId: 'graph-live', isLive: true, creatures: [{ id: 'creature-beta', name: 'beta' }] },
  ]
  const before = sockets.beginCount

  const result = await host.reconcileSelection()

  assert.deepEqual(result.selection, state.selection)
  assert.equal(sockets.beginCount, before)
})

test('session.reconcile returns the authoritative selection for explicit recovery', async () => {
  const { client, host, posts, state } = harness()
  state.selection = {
    session: 'graph-old',
    graph: 'graph-old',
    creature: 'old-name',
    targetCreatureId: 'creature-beta',
  }
  client.listOpen = async () => [
    { runtimeId: 'graph-new', isLive: true, creatures: [{ id: 'creature-beta', name: 'beta' }] },
  ]

  await host.handle({ type: 'session.reconcile', id: 13 })

  assert.deepEqual(posts.at(-1), {
    type: 'session.reconcile.result',
    id: 13,
    data: {
      selection: {
        session: 'graph-new',
        graph: 'graph-new',
        creature: 'beta',
        targetCreatureId: 'creature-beta',
      },
      changed: true,
    },
  })
})

test('history and interrupt always use the persisted selected target', async () => {
  const { client, host, posts, state } = harness()
  const calls = []
  state.selection = {
    session: 'graph-selected',
    graph: 'graph-selected',
    creature: 'beta',
    targetCreatureId: 'creature-beta',
  }
  client.history = async (...args) => {
    calls.push(['history', ...args])
    return { events: [] }
  }
  client.interrupt = async (...args) => {
    calls.push(['interrupt', ...args])
    return { ok: true }
  }

  await host.handle({ type: 'http.history', id: 3, session: 'graph-selected', creature: 'beta' })
  await host.handle({ type: 'http.interrupt', id: 4, session: 'graph-selected', creature: 'beta' })

  assert.deepEqual(calls, [
    ['history', 'graph-selected', 'beta'],
    ['interrupt', 'graph-selected', 'beta'],
  ])
  assert.equal(posts[0].type, 'http.history.result')
  assert.equal(posts[1].type, 'http.interrupt.result')
})
