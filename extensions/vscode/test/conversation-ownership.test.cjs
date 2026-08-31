const assert = require('node:assert/strict')
const test = require('node:test')

let createConversationAttachments
let createConversationDrafts
let createConversationOwnership
let isConversationSuperseded

test.before(async () => {
  ;({
    createConversationAttachments,
    createConversationDrafts,
    createConversationOwnership,
    isConversationSuperseded,
  } = await import('../src/webview/conversationOwnership.mjs'))
})

function deferred() {
  let resolve
  const promise = new Promise((done) => (resolve = done))
  return { promise, resolve }
}

function fixture() {
  let state = { readyId: 1, runtimeId: 'session-a', creatureId: 'root', name: 'Root' }
  const ownership = createConversationOwnership(() => state)
  return { ownership, setState: (value) => (state = value) }
}

test('accepted attachments are scoped by Session and Creature and restored on return', () => {
  let state = { readyId: 1, runtimeId: 'session-a', creatureId: 'root' }
  const buckets = createConversationAttachments(() => state)
  const fromA = [{ name: 'a.txt' }]
  buckets.set(fromA)

  state = { readyId: 1, runtimeId: 'session-b', creatureId: 'root' }
  assert.deepEqual(buckets.get(), [])
  buckets.set([{ name: 'b.txt' }])

  state = { readyId: 1, runtimeId: 'session-a', creatureId: 'other' }
  assert.deepEqual(buckets.get(), [])
  state = { readyId: 1, runtimeId: 'session-a', creatureId: 'root' }
  assert.equal(buckets.get(), fromA)
})

test('runtime epoch replacement fails closed while a stable epoch preserves attachments', () => {
  let state = { readyId: 1, runtimeId: 'session-a', creatureId: 'root' }
  const buckets = createConversationAttachments(() => state)
  const accepted = [{ name: 'a.txt' }]
  buckets.set(accepted)
  assert.equal(buckets.get(), accepted)

  state = { ...state }
  assert.equal(buckets.get(), accepted)
  state = { ...state, readyId: 2 }
  assert.deepEqual(buckets.get(), [])
})

test('clear only removes the captured conversation bucket', () => {
  let state = { readyId: 1, runtimeId: 'session-a', creatureId: 'root' }
  const buckets = createConversationAttachments(() => state)
  const owned = buckets.capture()
  buckets.set([{ name: 'a.txt' }], owned)
  state = { readyId: 1, runtimeId: 'session-b', creatureId: 'root' }
  buckets.set([{ name: 'b.txt' }])
  buckets.clear(owned)
  assert.deepEqual(buckets.get(), [{ name: 'b.txt' }])
})

test('confirmed submissions remove only the submitted attachment objects', () => {
  const state = { readyId: 1, runtimeId: 'session-a', creatureId: 'root' }
  const buckets = createConversationAttachments(() => state)
  const owned = buckets.capture()
  const submitted = { name: 'same.txt' }
  const laterSameName = { name: 'same.txt' }
  buckets.set([submitted, laterSameName], owned)

  buckets.removeSubmitted([submitted], owned)

  assert.deepEqual(buckets.get(owned), [laterSameName])
})

test('confirmed submissions preserve attachments added while confirmation is pending', () => {
  const state = { readyId: 1, runtimeId: 'session-a', creatureId: 'root' }
  const buckets = createConversationAttachments(() => state)
  const owned = buckets.capture()
  const submitted = { name: 'a.txt' }
  const later = { name: 'b.txt' }
  buckets.set([submitted, later], owned)

  buckets.removeSubmitted([submitted], owned)

  assert.deepEqual(buckets.get(owned), [later])
})

test('confirmed submissions leave a re-added identical object in the bucket', () => {
  const state = { readyId: 1, runtimeId: 'session-a', creatureId: 'root' }
  const buckets = createConversationAttachments(() => state)
  const owned = buckets.capture()
  const submitted = { name: 'a.txt' }
  buckets.set([submitted, submitted], owned)

  buckets.removeSubmitted([submitted], owned)

  assert.deepEqual(buckets.get(owned), [submitted])
})

test('drafts are bucketed and late cleanup only clears the submitted owner', () => {
  let state = { readyId: 1, runtimeId: 'session-a', creatureId: 'root' }
  const drafts = createConversationDrafts(() => state)
  const owned = drafts.capture()
  drafts.set('submitted A')

  state = { readyId: 1, runtimeId: 'session-b', creatureId: 'root' }
  drafts.set('current B')
  drafts.clear(owned)
  assert.equal(drafts.get(), 'current B')

  state = owned
  assert.equal(drafts.get(), '')
})

test('dispatch permits authoritative completion after ownership switches', async () => {
  const { ownership, setState } = fixture()
  const confirmation = deferred()
  let dispatched = false
  const operation = ownership.dispatch(async (assertCurrent) => {
    assertCurrent()
    dispatched = true
    await confirmation.promise
    return 'accepted'
  })

  setState({ readyId: 1, runtimeId: 'session-b', creatureId: 'root', name: 'Root' })
  confirmation.resolve()
  assert.equal(await operation, 'accepted')
  assert.equal(dispatched, true)
})

test('attachment transform from another Session with the same creature name is superseded', async () => {
  const { ownership, setState } = fixture()
  const conversion = deferred()
  const transformed = ownership.transform(() => conversion.promise)('file')

  setState({ readyId: 1, runtimeId: 'session-b', creatureId: 'root', name: 'Root' })
  conversion.resolve('a-bytes')

  await assert.rejects(transformed, isConversationSuperseded)
})

test('attachment transform applies while exact ownership remains current', async () => {
  const { ownership } = fixture()
  const conversion = deferred()
  const transformed = ownership.transform(() => conversion.promise)('file')
  conversion.resolve('a-bytes')
  assert.equal(await transformed, 'a-bytes')
})

test('supplemental name changes do not supersede the same stable ownership', async () => {
  const { ownership, setState } = fixture()
  const conversion = deferred()
  const transformed = ownership.transform(() => conversion.promise)('file')

  setState({ readyId: 1, runtimeId: 'session-a', creatureId: 'root', name: 'Renamed' })
  conversion.resolve('a-bytes')

  assert.equal(await transformed, 'a-bytes')
})

test('runtime refresh supersedes attachment transform for the same Session and Creature', async () => {
  const { ownership, setState } = fixture()
  const conversion = deferred()
  const transformed = ownership.transform(() => conversion.promise)('file')

  setState({ readyId: 2, runtimeId: 'session-a', creatureId: 'root', name: 'Root' })
  conversion.resolve('old-bytes')

  await assert.rejects(transformed, isConversationSuperseded)
})

test('send rechecks ownership after content conversion and before dispatch', async () => {
  const { ownership, setState } = fixture()
  const conversion = deferred()
  let sent = false
  const operation = ownership.run(async (assertCurrent) => {
    const content = await conversion.promise
    assertCurrent()
    sent = true
    return content
  })

  setState({ readyId: 1, runtimeId: 'session-b', creatureId: 'root', name: 'Root' })
  conversion.resolve('a-bytes')

  await assert.rejects(operation, isConversationSuperseded)
  assert.equal(sent, false)
})
