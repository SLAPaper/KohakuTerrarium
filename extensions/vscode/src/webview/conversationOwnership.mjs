const SUPERSEDED = Symbol('conversation-superseded')

export function isConversationSuperseded(error) {
  return error?.code === SUPERSEDED
}

function supersededError() {
  const error = new Error('Conversation operation was superseded')
  error.code = SUPERSEDED
  error.silent = true
  return error
}

function sameOwnership(left, right) {
  return (
    left?.readyId === right?.readyId &&
    left?.runtimeId === right?.runtimeId &&
    left?.creatureId === right?.creatureId
  )
}

function ownershipKey(owned) {
  if (owned?.readyId == null || owned?.runtimeId == null || owned?.creatureId == null) return ''
  return JSON.stringify([owned.readyId, owned.runtimeId, owned.creatureId])
}

export function createConversationAttachments(current) {
  const buckets = new Map()
  const capture = () => ({ ...current() })
  const keyFor = (owned = capture()) => ownershipKey(owned)

  return {
    capture,
    get(owned) {
      return buckets.get(keyFor(owned)) || []
    },
    set(value, owned) {
      const key = keyFor(owned)
      if (key) buckets.set(key, value)
    },
    clear(owned) {
      buckets.delete(keyFor(owned))
    },
  }
}

export function createConversationOwnership(current) {
  const capture = () => ({ ...current() })
  const assertion = (owned) => () => {
    if (!sameOwnership(owned, current())) throw supersededError()
  }

  return {
    transform(transformer) {
      return async (...args) => {
        const owned = capture()
        const assertCurrent = assertion(owned)
        assertCurrent()
        const result = await transformer(...args)
        assertCurrent()
        return result
      }
    },
    async run(operation) {
      const owned = capture()
      const assertCurrent = assertion(owned)
      assertCurrent()
      const result = await operation(assertCurrent)
      assertCurrent()
      return result
    },
  }
}
