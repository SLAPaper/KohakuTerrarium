function chatInstance(task) {
  if (!task.runtimeId) throw Error('Live task has no runtime')
  return {
    id: task.runtimeId,
    graph_id: task.runtimeId,
    session_id: task.runtimeId,
    type: task.kind,
    config_name: task.title,
    creatures: task.creatures.map((creature) => ({
      id: creature.id,
      creature_id: creature.id,
      name: creature.name,
      is_root: false,
    })),
    channels: [],
  }
}

export function createTaskShell({ api, chat }) {
  async function attach(task, creatureId) {
    const selected = task.creatures.find((creature) => creature.id === creatureId)
    if (!selected?.id) throw Error('Unknown Creature target')
    const instance = chatInstance(task)
    const ownership = await api.select({
      session: task.runtimeId,
      creatureId: selected.id,
    })
    chat.unbindFromInstance()
    chat.initForInstance(instance, { initialTab: selected.name })
    return {
      task,
      instance,
      target: selected.name,
      targetCreatureId: ownership.targetCreatureId,
    }
  }

  async function attachWhenSingle(task) {
    if (task.creatures.length === 1) return attach(task, task.creatures[0].id)
    await api.clearSelection()
    chat.unbindFromInstance()
    return { task, target: null, targetCreatureId: null }
  }

  return {
    list: () => api.list(),
    restore: (task, selection) => {
      const creature = task.creatures.find(
        (candidate) => candidate.id === selection.targetCreatureId,
      )
      if (!creature || task.runtimeId !== selection.session) {
        throw Error('Reconciled Creature is not in Open Tasks')
      }
      const instance = chatInstance(task)
      chat.unbindFromInstance()
      chat.initForInstance(instance, { initialTab: creature.name })
      return {
        task,
        instance,
        target: creature.name,
        targetCreatureId: creature.id,
      }
    },
    create: async () => attachWhenSingle(await api.create()),
    resume: async (savedName) => attachWhenSingle(await api.resume(savedName)),
    detach: async (current) => {
      await api.detach({
        session: current.task.runtimeId,
        creatureId: current.targetCreatureId,
      })
      chat.unbindFromInstance()
    },
    open: (task, creatureId) => attach(task, creatureId),
  }
}
