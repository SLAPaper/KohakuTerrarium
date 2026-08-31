const assert = require('node:assert/strict')
const test = require('node:test')

const { allowedMessage, validateEndpoint } = require('../src/host/protocol.cjs')
const { createClient, normalizeSession, validateCapabilities } = require('../src/host/client.cjs')

test('protocol accepts the formal vertical-slice messages with strict fields', () => {
  for (const message of [
    { type: 'ready', id: 1 },
    { type: 'session.list', id: 2 },
    { type: 'session.reconcile', id: 18 },
    { type: 'session.clearSelection', id: 19 },
    { type: 'session.create', id: 3 },
    { type: 'session.resume', id: 14, savedName: 'saved-one' },
    { type: 'session.stop', id: 15, session: 'graph-1', creatureId: 'creature-alpha' },
    { type: 'session.select', id: 4, session: 'graph-1', creatureId: 'creature-alpha' },
    { type: 'http.history', id: 5, session: 'graph-1', creature: 'alpha' },
    { type: 'http.interrupt', id: 6, session: 'graph-1', creature: 'alpha' },
    { type: 'context.compact', id: 20 },
    { type: 'context.clear', id: 21 },
    { type: 'ws.open', id: 7 },
    { type: 'ws.send', id: 7, data: '{"type":"input"}' },
    { type: 'ws.close', id: 7 },
  ]) assert.equal(allowedMessage(message), true, message.type)

  for (const message of [
    { type: 'session.select', id: 8, session: 'graph-1' },
    { type: 'session.select', id: 13, session: 'graph-1', creature: 'alpha' },
    { type: 'http.history', id: 9, session: 'graph-1', creature: '' },
    { type: 'ws.send', id: 10 },
    { type: 'session.create', id: 11, endpoint: 'http://127.0.0.1:8000' },
    { type: 'session.create', id: 12, configRef: '@secret/config' },
    { type: 'session.resume', id: 16, savedName: '' },
    { type: 'session.stop', id: 17, session: 'graph-1', creatureId: '' },
    { type: 'context.compact', id: 20, command: 'stop' },
    { type: 'context.clear', id: 21, args: ['anything'] },
    { type: 'command.execute', id: 22, command: 'clear', args: [] },
  ]) assert.equal(allowedMessage(message), false, message.type)
})

test('endpoint validation allows explicit loopback HTTP only', () => {
  assert.equal(validateEndpoint('http://127.0.0.1:8000'), 'http://127.0.0.1:8000')
  assert.equal(validateEndpoint('http://[::1]:9000'), 'http://[::1]:9000')
  for (const value of [
    'http://127.0.0.1',
    'https://127.0.0.1:8000',
    'http://localhost:8000',
    'http://127.0.0.1:8000/path',
    'http://user:pass@127.0.0.1:8000',
  ]) assert.throws(() => validateEndpoint(value))
})

test('Host client omits credentials for loopback-bypass mode', async () => {
  const calls = []
  const client = createClient({
    endpoint: 'http://127.0.0.1:8000',
    token: '',
    fetchImpl: async (url, options) => {
      calls.push({ url, options })
      return { ok: true, status: 200, json: async () => ({ sessions: [] }) }
    },
  })

  await client.listOpen()

  assert.equal(calls[0].options.headers['X-KT-Host-Token'], undefined)
})

test('Host client authenticates with X-KT-Host-Token and constructs creature routes', async () => {
  const calls = []
  const fetchImpl = async (url, options) => {
    calls.push({ url, options })
    return { ok: true, status: 200, json: async () => ({ ok: true }) }
  }
  const client = createClient({ endpoint: 'http://127.0.0.1:8000', token: 'host-secret', fetchImpl })

  await client.history('graph one', 'alpha/beta')
  await client.interrupt('graph one', 'alpha/beta')
  await client.creatureCommand('graph one', 'alpha/beta', 'clear', '--force')

  assert.equal(calls[0].url, 'http://127.0.0.1:8000/api/sessions/graph%20one/creatures/alpha%2Fbeta/history')
  assert.equal(calls[0].options.headers['X-KT-Host-Token'], 'host-secret')
  assert.equal(calls[0].options.headers.authorization, undefined)
  assert.equal(calls[0].options.redirect, 'error')
  assert.equal(calls[1].url, 'http://127.0.0.1:8000/api/sessions/graph%20one/creatures/alpha%2Fbeta/interrupt')
  assert.equal(calls[1].options.method, 'POST')
  assert.equal(calls[2].url, 'http://127.0.0.1:8000/api/sessions/graph%20one/creatures/alpha%2Fbeta/command')
  assert.equal(calls[2].options.method, 'POST')
  assert.deepEqual(JSON.parse(calls[2].options.body), { command: 'clear', args: '--force' })
})

test('capability validation accepts local bypass and rejects multi-user mode', () => {
  assert.doesNotThrow(() =>
    validateCapabilities({
      schema: 1,
      auth: {
        host_token: { enabled: true, loopback_bypass: true },
        admin_token: { enabled: false },
        multi_user: { enabled: false },
      },
    }),
  )
  assert.throws(() =>
    validateCapabilities({
      schema: 1,
      auth: {
        host_token: { enabled: true, loopback_bypass: true },
        multi_user: { enabled: true },
      },
    }),
  )
})

test('session normalization preserves stable ids and hides generated graph titles', () => {
  const session = normalizeSession({
    conversation_id: 'conversation-1',
    runtime_id: 'graph_deadbeef',
    display_name: 'graph_deadbeef',
    is_live: true,
    type: 'terrarium',
    creatures: [
      { creature_id: 'creature-beta', name: 'Beta' },
      { creature_id: 'creature-alpha', name: 'Alpha' },
    ],
  })

  assert.deepEqual(session, {
    conversationId: 'conversation-1',
    runtimeId: 'graph_deadbeef',
    savedName: null,
    title: 'Alpha, Beta',
    isLive: true,
    kind: 'terrarium',
    creatures: [
      { id: 'creature-beta', name: 'Beta' },
      { id: 'creature-alpha', name: 'Alpha' },
    ],
  })
  assert.throws(() => normalizeSession({ is_live: true, creatures: [{ name: 'Missing id' }] }))
})
