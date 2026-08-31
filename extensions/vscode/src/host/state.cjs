class ConnectionStateWriter {
  constructor(workspaceState, key) {
    this.workspaceState = workspaceState
    this.key = key
    this.queue = Promise.resolve()
  }

  read() {
    return this.workspaceState.get(this.key) || {}
  }

  update(mutator) {
    const operation = this.queue.then(async () => {
      const current = this.read()
      const next = typeof mutator === 'function' ? mutator(current) : mutator
      if (next === undefined) return current
      await this.workspaceState.update(this.key, next)
      return next
    })
    this.queue = operation.catch(() => {})
    return operation
  }
}

module.exports = { ConnectionStateWriter }
