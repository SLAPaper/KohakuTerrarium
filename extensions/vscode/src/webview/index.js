import {
  ChatTranscriptSection,
  ConversationMessage,
  MarkdownRenderer,
} from '@kohakuterrarium/chat-ui'
import { createPinia } from 'pinia'
import { computed, createApp, h, ref } from 'vue'

import { useChatStore } from '@/stores/chat'

import { BridgeWebSocket } from './bridge.js'
import { createTaskShell } from './taskShell.js'
import './style.css'

const vscode = acquireVsCodeApi()
const pending = new Map()
let nextRequestId = 1000

function rejectPending(error) {
  for (const request of pending.values()) {
    clearTimeout(request.timer)
    request.reject(error)
  }
  pending.clear()
}

function request(type, data = {}) {
  return new Promise((resolve, reject) => {
    const id = nextRequestId++
    const timer = setTimeout(() => {
      pending.delete(id)
      reject(Error('KohakuTerrarium request timed out'))
    }, 30000)
    pending.set(id, { resolve, reject, timer })
    vscode.postMessage({ type, id, ...data })
  })
}

BridgeWebSocket.post = (message) => vscode.postMessage(message)
globalThis.WebSocket = BridgeWebSocket

const App = {
  setup() {
    const chat = useChatStore()
    const available = ref(false)
    const automatic = ref(true)
    const tasks = ref([])
    const current = ref(null)
    const draft = ref('')
    const error = ref('')
    const busy = ref(false)
    let reconciliation = Promise.resolve()

    const api = {
      list: () => request('task.list'),
      create: () => request('task.create'),
      resume: (savedName) => request('task.resume', { savedName }),
      detach: ({ session, creatureId }) => request('task.detach', { session, creatureId }),
      clearSelection: () => request('task.clearSelection'),
      reconcile: () => request('task.reconcile'),
      select: ({ session, creatureId }) => request('task.select', { session, creatureId }),
    }
    const shell = createTaskShell({ api, chat })
    const tab = computed(() => current.value?.target || '')
    const messages = computed(() => chat.messagesByTab[tab.value] || [])

    globalThis.__ktVsCodeHistory = (session, creature) => request('http.history', { session, creature })
    globalThis.__ktVsCodeInterrupt = (session, creature) => request('http.interrupt', { session, creature })

    async function reloadTasks() {
      tasks.value = await shell.list()
    }

    async function reconcileTasks() {
      vscode.postMessage({ type: 'ready', id: nextRequestId++ })
    }

    async function applySelection(selection, changed = true) {
      tasks.value = await shell.list()
      if (!selection) {
        chat.unbindFromInstance()
        current.value = null
        return
      }
      if (!changed && current.value?.targetCreatureId === selection.targetCreatureId) {
        const task = tasks.value.find(
          (candidate) =>
            candidate.isLive &&
            candidate.runtimeId === selection.session &&
            candidate.creatures.some((creature) => creature.id === selection.targetCreatureId),
        )
        if (task) current.value = { ...current.value, task }
        return
      }
      const task = tasks.value.find(
        (candidate) =>
          candidate.isLive &&
          candidate.runtimeId === selection.session &&
          candidate.creatures.some((creature) => creature.id === selection.targetCreatureId),
      )
      if (!task) {
        chat.unbindFromInstance()
        current.value = null
        throw Error('Selected Creature is no longer open')
      }
      current.value = shell.restore(task, selection)
    }

    async function createTask() {
      busy.value = true
      error.value = ''
      try {
        current.value = await shell.create()
        await reloadTasks()
      } catch (cause) {
        error.value = cause.message
      } finally {
        busy.value = false
      }
    }

    async function resumeTask(task) {
      busy.value = true
      error.value = ''
      try {
        current.value = await shell.resume(task.savedName)
        await reloadTasks()
      } catch (cause) {
        error.value = cause.message
      } finally {
        busy.value = false
      }
    }

    async function detachTask() {
      if (!current.value?.targetCreatureId) return
      busy.value = true
      error.value = ''
      try {
        await shell.detach(current.value)
        current.value = null
        await reloadTasks()
      } catch (cause) {
        error.value = cause.message
      } finally {
        busy.value = false
      }
    }

    async function openTask(task, creatureId) {
      busy.value = true
      error.value = ''
      try {
        current.value = await shell.open(task, creatureId)
      } catch (cause) {
        error.value = cause.message
      } finally {
        busy.value = false
      }
    }

    function send() {
      const content = draft.value.trim()
      if (!content || !current.value) return
      chat.send(content)
      draft.value = ''
    }

    function submitReply(message, actionId, values) {
      chat.submitUIReply(tab.value, message.eventId, actionId, values)
    }

    function renderSharedText(content, breaks = false) {
      return h(MarkdownRenderer, { content, breaks })
    }

    function renderTranscriptMessage(message, { reply }) {
      return h(ConversationMessage, {
        message,
        renderText: renderSharedText,
        onReply: ({ actionId, values }) => reply(actionId, values),
      })
    }

    function renderTask(task) {
      return h('article', { class: 'task', key: task.conversationId || task.runtimeId }, [
        h('div', `${task.isLive ? 'open' : 'dormant'} — ${task.title}`),
        ...(task.isLive
          ? task.creatures.map((creature) =>
              h(
                'button',
                {
                  disabled: busy.value || !available.value,
                  onClick: () => openTask(task, creature.id),
                },
                creature.name,
              ),
            )
          : [
              h(
                'button',
                {
                  disabled: busy.value || !available.value || !task.savedName,
                  onClick: () => resumeTask(task),
                },
                'Resume',
              ),
            ]),
      ])
    }

    window.addEventListener('message', ({ data: message }) => {
      BridgeWebSocket.receive(message)
      if (message?.type === 'configuration.changed') {
        rejectPending(Error('KohakuTerrarium configuration changed'))
        chat.unbindFromInstance()
        current.value = null
        available.value = false
        automatic.value = false
        vscode.postMessage({ type: 'ready', id: nextRequestId++ })
        return
      }
      if (message?.type === 'ready.result') {
        available.value = message.data.available === true
        automatic.value = message.data.automatic !== false
        if (available.value) {
          error.value = ''
          reconciliation = reconciliation
            .then(() => applySelection(message.data.selection))
            .catch((cause) => (error.value = cause.message))
        } else {
          chat.unbindFromInstance()
          current.value = null
          tasks.value = []
          error.value = 'No local KohakuTerrarium service found. Run “kt serve start”, then press Refresh.'
        }
        return
      }
      if (message?.type === 'selection.changed') {
        reconciliation = reconciliation
          .then(() => applySelection(message.data.selection, message.data.changed))
          .catch((cause) => (error.value = cause.message))
        return
      }
      if (message?.id && pending.has(message.id)) {
        const promise = pending.get(message.id)
        pending.delete(message.id)
        clearTimeout(promise.timer)
        if (message.type === 'error') promise.reject(Error(message.error))
        else promise.resolve(message.data)
      }
    })

    vscode.postMessage({ type: 'ready', id: 1 })

    return () =>
      h('main', { class: 'kt-conversation-host' }, [
        h('header', [
          h('h1', 'KohakuTerrarium'),
          h(
            'p',
            available.value
              ? automatic.value
                ? 'Connected automatically to local KT'
                : 'Connected using a local override'
              : 'Waiting for a local KT service',
          ),
        ]),
        h('section', { class: 'taskrail' }, [
          h('h2', 'Open Tasks'),
          h('button', { disabled: busy.value || !available.value, onClick: createTask }, 'New Task'),
          h(
            'button',
            {
              disabled: busy.value,
              onClick: () => {
                error.value = ''
                reconciliation = reconciliation.then(reconcileTasks).catch((cause) => (error.value = cause.message))
              },
            },
            'Refresh',
          ),
          ...tasks.value.map(renderTask),
        ]),
        current.value
          ? h('section', { class: 'selection' }, [
              h('div', `Current task: ${current.value.task.title}`),
              h('div', `Creature: ${current.value.target || 'Choose a creature'}`),
              current.value.targetCreatureId
                ? h('button', { disabled: busy.value, onClick: detachTask }, 'Detach')
                : null,
            ])
          : null,
        h('p', { class: 'status' }, error.value),
        h(ChatTranscriptSection, {
          messages: messages.value,
          totalCount: messages.value.length,
          emptyTitle: 'No messages yet',
          emptySubtitle: 'Choose a task and send a message',
          processing: chat.processingByTab[tab.value],
          processingLabel: 'Kohaku is working',
          reconnecting: chat.wsStatus === 'reconnecting',
          reconnectLabel: 'Reconnecting',
          renderMessage: renderTranscriptMessage,
          onReply: ({ message, actionId, values }) => submitReply(message, actionId, values),
        }),
        current.value && chat.processingByTab[tab.value]
          ? h('button', { onClick: () => chat.interrupt(tab.value) }, 'Stop Turn')
          : null,
        current.value?.target
          ? h(
              'form',
              {
                class: 'composer',
                onSubmit: (event) => {
                  event.preventDefault()
                  send()
                },
              },
              [
                h('input', {
                  value: draft.value,
                  placeholder: 'Send to selected Creature',
                  onInput: (event) => (draft.value = event.target.value),
                }),
                h('button', { type: 'submit' }, 'Send'),
              ],
            )
          : null,
      ])
  },
}

createApp(App).use(createPinia()).mount('#app')
