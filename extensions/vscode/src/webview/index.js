import { BridgeWebSocket } from './bridge.js'
import { createVisibilityInterval } from './shims/visibility.js'
import './style.css'

const vscode = acquireVsCodeApi()
const app = document.querySelector('#app')
const pending = new Map()
let nextId = 2
let sessions = []
let selection = null
let status = 'Connecting to local KohakuTerrarium…'

function request(type, data = {}) {
  return new Promise((resolve, reject) => {
    const id = nextId++
    const timer = setTimeout(() => {
      pending.delete(id)
      reject(Error('KohakuTerrarium request timed out'))
    }, 30000)
    pending.set(id, { resolve, reject, timer })
    vscode.postMessage({ type, id, ...data })
  })
}

function button(label, action, disabled = false) {
  const node = document.createElement('button')
  node.textContent = label
  node.disabled = disabled
  node.addEventListener('click', () => action().catch(showError))
  return node
}

function showError(error) {
  status = error?.message || 'KohakuTerrarium request failed'
  render()
}

async function refresh() {
  sessions = await request('session.list')
  render()
}

async function select(session, creature) {
  selection = await request('session.select', {
    session: session.runtimeId,
    creatureId: creature.id,
  })
  status = `Selected ${creature.name}`
  render()
}

async function createSession() {
  const created = await request('session.create')
  await refresh()
  if (created.creatures.length === 1) await select(created, created.creatures[0])
}

async function resumeSession(session) {
  const resumed = await request('session.resume', { savedName: session.savedName })
  await refresh()
  if (resumed.creatures.length === 1) await select(resumed, resumed.creatures[0])
}

async function stopSession() {
  if (!selection) return
  await request('session.stop', {
    session: selection.session,
    creatureId: selection.targetCreatureId,
  })
  selection = null
  await refresh()
}

function render() {
  app.replaceChildren()
  const title = document.createElement('h1')
  title.textContent = 'KohakuTerrarium'
  app.append(title, Object.assign(document.createElement('p'), { textContent: status }))
  const controls = document.createElement('section')
  controls.append(button('New Session', createSession), button('Refresh', refresh))
  app.append(controls)
  for (const session of sessions) {
    const row = document.createElement('article')
    row.className = 'session'
    row.append(Object.assign(document.createElement('strong'), { textContent: session.title }))
    if (session.isLive) {
      for (const creature of session.creatures) row.append(button(creature.name, () => select(session, creature)))
    } else if (session.savedName) row.append(button('Resume', () => resumeSession(session)))
    app.append(row)
  }
  if (selection) app.append(button('Stop Session', stopSession))
}

BridgeWebSocket.post = (message) => vscode.postMessage(message)
globalThis.WebSocket = BridgeWebSocket
window.addEventListener('message', ({ data: message }) => {
  BridgeWebSocket.receive(message)
  if (message?.type === 'configuration.changed') {
    selection = null
    vscode.postMessage({ type: 'ready', id: nextId++ })
    return
  }
  if (message?.type === 'ready.result') {
    selection = message.data.selection
    status = message.data.available
      ? message.data.automatic ? 'Connected automatically to local KT' : 'Connected using a local override'
      : 'No local KohakuTerrarium service found. Run “kt serve start”, then refresh.'
    if (message.data.available) refresh().catch(showError)
    else render()
    return
  }
  if (message?.type === 'selection.changed') {
    selection = message.data.selection
    render()
    return
  }
  const operation = pending.get(message?.id)
  if (!operation) return
  pending.delete(message.id)
  clearTimeout(operation.timer)
  if (message.type === 'error') operation.reject(Error(message.error))
  else operation.resolve(message.data)
})

const reconciliation = createVisibilityInterval(() => {
  request('session.reconcile').then((result) => {
    selection = result.selection
    render()
  }).catch(showError)
}, 15000, { immediate: false })
reconciliation.start()
window.addEventListener('pagehide', () => reconciliation.stop(), { once: true })
render()
vscode.postMessage({ type: 'ready', id: 1 })
