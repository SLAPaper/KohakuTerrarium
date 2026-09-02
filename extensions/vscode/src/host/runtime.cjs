const { normalizeSession } = require('./client.cjs')

const contextCapabilities = new WeakMap()

function encode(value) {
  return encodeURIComponent(value)
}

function normalizeActive(active) {
  return normalizeSession({
    conversation_id: active.conversation_id ?? null,
    runtime_id: active.session_id ?? active.runtime_id,
    display_name: active.display_name ?? active.config_name ?? active.name,
    is_live: true,
    type: active.type,
    creatures: active.creatures,
  })
}

class RuntimeHost {
  constructor({
    client,
    state,
    sockets,
    post,
    getDefaultCreature,
    getWorkspacePath,
    socketFactory,
    webSocketBase,
    token,
    runtimeEpoch = null,
  }) {
    this.client = client
    this.state = state
    this.sockets = sockets
    this.post = post
    this.getDefaultCreature = getDefaultCreature
    this.getWorkspacePath = getWorkspacePath
    this.socketFactory = socketFactory
    this.webSocketBase = webSocketBase
    this.token = token
    this.runtimeEpoch = runtimeEpoch
    this.selectionOperationTail = Promise.resolve()
    this.selectionVersion = 0
    this.generation = this.sockets.begin()
  }

  requireSelection(message) {
    const selection = this.state.selection
    if (!selection || selection.session !== message.session || selection.creature !== message.creature) {
      throw Error('Selected Creature ownership changed')
    }
    return selection
  }

  enqueueSelectionOperation(operation) {
    const result = this.selectionOperationTail.then(operation)
    this.selectionOperationTail = result.catch(() => {})
    return result
  }

  clearSelection() {
    return this.enqueueSelectionOperation(() => this.clearSelectionOwned())
  }

  async clearSelectionOwned() {
    if (!this.state.selection) {
      return { selection: null, changed: false, selectionVersion: this.selectionVersion }
    }
    await this.state.updateSelection(null)
    this.generation = this.sockets.begin()
    this.selectionVersion++
    return { selection: null, changed: true, selectionVersion: this.selectionVersion }
  }

  reconcileSelection() {
    return this.enqueueSelectionOperation(() => this.reconcileSelectionOwned())
  }

  async reconcileSelectionOwned() {
    const current = this.state.selection
    if (!current?.targetCreatureId) {
      return { selection: null, changed: false, selectionVersion: this.selectionVersion }
    }
    const sessions = await this.client.listOpen()
    const session = sessions.find(
      (candidate) => candidate.isLive && candidate.creatures.some((creature) => creature.id === current.targetCreatureId),
    )
    const creature = session?.creatures.find((candidate) => candidate.id === current.targetCreatureId)
    const selection =
      session && creature
        ? {
            session: session.runtimeId,
            graph: session.runtimeId,
            creature: creature.name,
            targetCreatureId: current.targetCreatureId,
          }
        : null
    const changed = !selection || selection.session !== current.session || selection.creature !== current.creature
    if (!changed) {
      return { selection: current, changed: false, selectionVersion: this.selectionVersion }
    }
    await this.state.updateSelection(selection)
    this.generation = this.sockets.begin()
    this.selectionVersion++
    return { selection, changed: true, selectionVersion: this.selectionVersion }
  }

  async selectOwned(message) {
    const active = await this.client.active(message.session)
    const selected = active.creatures?.find((creature) => String(creature.creature_id ?? creature.id) === message.creatureId)
    if (!selected?.name) throw Error('Selected Creature is not in the active Session')
    const selection = {
      session: active.session_id ?? message.session,
      graph: active.session_id ?? message.session,
      creature: selected.name,
      targetCreatureId: message.creatureId,
    }
    const changed =
      !this.state.selection ||
      this.state.selection.session !== selection.session ||
      this.state.selection.creature !== selection.creature ||
      this.state.selection.targetCreatureId !== selection.targetCreatureId
    if (changed) {
      await this.state.updateSelection(selection)
      this.generation = this.sockets.begin()
      this.selectionVersion++
    }
    return { selection, changed, selectionVersion: this.selectionVersion }
  }

  acquireContextCommand() {
    const selected = this.state.selection
    if (!selected) return null
    const capability = Object.freeze({})
    contextCapabilities.set(capability, {
      runtime: this,
      runtimeEpoch: this.runtimeEpoch,
      selected,
      selectionVersion: this.selectionVersion,
    })
    return capability
  }

  ownsContextCommand(capability) {
    const owned = contextCapabilities.get(capability)
    return (
      owned?.runtime === this &&
      owned.runtimeEpoch === this.runtimeEpoch &&
      owned.selected === this.state.selection &&
      owned.selectionVersion === this.selectionVersion
    )
  }

  async contextCommandOwned(message, capability) {
    if (!this.ownsContextCommand(capability)) throw Error('Selected Creature ownership changed')
    const { selected } = contextCapabilities.get(capability)
    const command = message.type === 'context.compact' ? 'compact' : 'clear'
    const args = command === 'clear' ? '--force' : ''
    const data = await this.client.creatureCommand(selected.session, selected.creature, command, args)
    if (!this.ownsContextCommand(capability)) throw Error('Selected Creature ownership changed')
    return data
  }

  async stopOwned(message) {
    const selected = this.state.selection
    if (!selected || selected.session !== message.session || selected.targetCreatureId !== message.creatureId) {
      throw Error('Session ownership changed')
    }
    await this.client.stop(selected.session)
    return this.clearSelectionOwned()
  }

  async handle(message) {
    switch (message.type) {
      case 'session.clearSelection': {
        const result = await this.clearSelection()
        this.post({
          type: 'session.clearSelection.result',
          requestId: message.requestId,
          data: { ok: true, selectionVersion: result.selectionVersion, readyId: this.runtimeEpoch },
        })
        return
      }
      case 'session.reconcile': {
        const data = await this.reconcileSelection()
        this.post({ type: 'session.reconcile.result', requestId: message.requestId, data: { ...data, readyId: this.runtimeEpoch } })
        return
      }
      case 'session.list': {
        this.post({ type: 'session.list.result', requestId: message.requestId, data: await this.client.listOpen() })
        return
      }
      case 'session.create': {
        const configPath = this.getDefaultCreature()
        const pwd = this.getWorkspacePath()
        if (!configPath) throw Error('Configure kohakuterrarium.defaultCreature first')
        if (!pwd) throw Error('Open a workspace folder before creating a Session')
        const created = await this.client.createCreature({
          configPath,
          pwd,
          name: 'VS Code Session',
        })
        const data = normalizeActive(created)
        this.post({ type: 'session.create.result', requestId: message.requestId, data })
        return
      }
      case 'session.resume': {
        const open = await this.client.listOpen()
        if (!open.some((session) => !session.isLive && session.savedName === message.savedName)) {
          throw Error('Saved session is not an open dormant Session')
        }
        const resumed = await this.client.resume(message.savedName)
        const data = normalizeActive({
          ...resumed.session,
          session_id: resumed.instance_id ?? resumed.session?.session_id,
          type: resumed.type,
          config_name: resumed.session_name,
        })
        data.savedName = resumed.session_name ?? message.savedName
        this.post({ type: 'session.resume.result', requestId: message.requestId, data })
        return
      }
      case 'session.select': {
        const result = await this.enqueueSelectionOperation(() => this.selectOwned(message))
        this.post({
          type: 'session.select.result',
          requestId: message.requestId,
          data: { ...result.selection, selectionVersion: result.selectionVersion, readyId: this.runtimeEpoch },
        })
        return
      }
      case 'session.stop': {
        const result = await this.enqueueSelectionOperation(() => this.stopOwned(message))
        this.post({
          type: 'session.stop.result',
          requestId: message.requestId,
          data: { ok: true, selectionVersion: result.selectionVersion, readyId: this.runtimeEpoch },
        })
        return
      }
      case 'http.history': {
        const selected = this.requireSelection(message)
        this.post({
          type: 'http.history.result',
          requestId: message.requestId,
          data: await this.client.history(selected.session, selected.creature),
        })
        return
      }
      case 'http.interrupt': {
        const selected = this.requireSelection(message)
        this.post({
          type: 'http.interrupt.result',
          requestId: message.requestId,
          data: await this.client.interrupt(selected.session, selected.creature),
        })
        return
      }
      case 'context.compact':
      case 'context.clear': {
        const capability = message.contextCapability || this.acquireContextCommand()
        if (!capability) throw Error('Select a Creature before managing context')
        const data = await this.enqueueSelectionOperation(() => this.contextCommandOwned(message, capability))
        this.post({ type: `${message.type}.result`, requestId: message.requestId, data })
        return
      }
      case 'ws.open': {
        const selected = this.state.selection
        if (!selected) throw Error('Select a Creature before opening chat')
        const route = `/ws/sessions/${encode(selected.session)}/creatures/${encode(selected.creature)}/chat`
        this.sockets.open(
          this.generation,
          message.socketId,
          () => this.socketFactory(`${this.webSocketBase}${route}`, this.token ? [`kt-token.${this.token}`] : []),
          { postMessage: this.post },
        )
        return
      }
      case 'ws.send':
        if (!(await this.sockets.send(this.generation, message.socketId, message.data))) {
          throw Error('Chat socket is not open')
        }
        this.post({
          type: 'ws.send.result',
          socketId: message.socketId,
          sendId: message.sendId,
          readyId: this.runtimeEpoch,
        })
        return
      case 'ws.close':
        this.sockets.closeSocket(this.generation, message.socketId, { postMessage: this.post })
        return
      default:
        throw Error(`Unsupported message: ${message.type}`)
    }
  }

  dispose() {
    this.sockets.closeGeneration(this.generation)
  }
}

module.exports = { RuntimeHost, normalizeActive }
