const assert = require('node:assert/strict')
const test = require('node:test')

let createConversationAttachments
let createConversationOwnership
let isConversationSuperseded

test.before(async () => {
  ;({ createConversationAttachments, createConversationOwnership, isConversationSuperseded } = await import(
    '../src/webview/conversationOwnership.mjs'
  ))
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
