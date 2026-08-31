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

function createConversationBuckets(current, empty) {
  const buckets = new Map()
  const capture = () => ({ ...current() })
  const keyFor = (owned = capture()) => ownershipKey(owned)

  return {
    capture,
    get(owned) {
      return buckets.get(keyFor(owned)) ?? empty
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

export function createConversationAttachments(current) {
  const buckets = createConversationBuckets(current, [])
  return {
    ...buckets,
    removeSubmitted(submitted, owned) {
      const remaining = new Map()
      for (const attachment of submitted) remaining.set(attachment, (remaining.get(attachment) || 0) + 1)
      buckets.set(
        buckets.get(owned).filter((attachment) => {
          const count = remaining.get(attachment) || 0
          if (!count) return true
          if (count === 1) remaining.delete(attachment)
          else remaining.set(attachment, count - 1)
          return false
        }),
        owned,
      )
    },
  }
}

export function createConversationDrafts(current) {
  return createConversationBuckets(current, '')
}

export function createConversationOwnership(current) {
  const capture = () => ({ ...current() })
  const assertion = (owned) => () => {
    if (!sameOwnership(owned, current())) throw supersededError()
  }

  return {
    isCurrent(owned) {
      return sameOwnership(owned, current())
    },
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
    async dispatch(operation) {
      const owned = capture()
      const assertCurrent = assertion(owned)
      assertCurrent()
      return operation(assertCurrent)
    },
  }
}
