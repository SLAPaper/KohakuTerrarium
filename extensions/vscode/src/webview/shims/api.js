export const terrariumAPI = {
  getHistory: (graph, target) => globalThis.__ktVsCodeHistory(graph, target),
  interruptCreature: (graph, target) => globalThis.__ktVsCodeInterrupt(graph, target),
  executeCreatureCommand: async () => {},
  sendToChannel: async () => {},
  promoteCreatureTask: async () => {},
}

export const agentAPI = {
  regenerate: async () => {},
  editMessage: async () => {},
  rewindTo: async () => {},
}
