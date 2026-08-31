import {
  buildMessageParts,
  ChatComposer,
  ChatTranscriptSection,
  ConversationMessage,
  MarkdownRenderer,
} from '@kohakuterrarium/chat-ui'
import { createPinia } from 'pinia'
import { computed, createApp, h, nextTick, onBeforeUnmount, ref, watch } from 'vue'

import { useChatStore } from '@/stores/chat'

import { BridgeWebSocket } from './bridge.js'
import { renderCarbonIcon } from './carbonIcons.mjs'
import { applyContextCommandOutcome } from './contextCommandResult.mjs'
import {
  createConversationMessageOrchestrator,
  createConversationScrollController,
} from './conversationScroll.mjs'
import {
  createConversationAttachments,
  createConversationDrafts,
  createConversationOwnership,
  isConversationSuperseded,
} from './conversationOwnership.mjs'
import { createReadyCoordinator } from './readyCoordinator.mjs'
import { createSelectionVersionOwner } from './selectionVersion.mjs'
import { createSessionShell } from './sessionShell.js'
import { createSubmitGate, isComposerSubmitDisabled } from './submitGate.mjs'
import { applyTopologySelection } from './topologySelection.mjs'
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

function request(type, data = {}, onSend = () => {}) {
  return new Promise((resolve, reject) => {
    const id = nextRequestId++
    onSend(id)
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
    const sessions = ref([])
    const currentSession = ref(null)
    const composerOwner = () => ({
      readyId: latestReadyRequestId,
      runtimeId: currentSession.value?.session?.runtimeId,
      creatureId: currentSession.value?.targetCreatureId,
    })
    const draftBuckets = createConversationDrafts(composerOwner)
    const draftRevision = ref(0)
    const draft = computed({
      get: () => {
        draftRevision.value
        return draftBuckets.get()
      },
      set: (value) => {
        draftBuckets.set(value)
        draftRevision.value += 1
      },
    })
    const attachmentBuckets = createConversationAttachments(composerOwner)
    const attachmentRevision = ref(0)
    const attachments = computed({
      get: () => {
        attachmentRevision.value
        return attachmentBuckets.get()
      },
      set: (value) => {
        attachmentBuckets.set(value)
        attachmentRevision.value += 1
      },
    })
    const error = ref('')
    const status = ref('')
    const busy = ref(false)
    const contextBusy = ref(false)
    const sessionsExpanded = ref(false)
    const brandUri = document.querySelector('#app')?.dataset.brandUri || ''
    let reconciliation = Promise.resolve()
    let contextOperation = 0
    const selectionVersions = createSelectionVersionOwner()
    let activeSelectionReadyId = null
    const currentConversationOwnership = () => ({
      readyId: latestReadyRequestId,
      runtimeId: currentSession.value?.session?.runtimeId,
      creatureId: currentSession.value?.targetCreatureId,
      name: currentSession.value?.target,
    })
    const conversationOwnership = createConversationOwnership(currentConversationOwnership)
    const submitGate = createSubmitGate()
    const submitRevision = ref(0)
    const submitBusy = computed(() => {
      submitRevision.value
      attachmentRevision.value
      return submitGate.busy(currentConversationOwnership())
    })
    const attachmentTransform = conversationOwnership.transform((file) => file)
    const selectionRequest = async (type, data) => {
      const result = await request(type, data)
      if (result.readyId !== latestReadyRequestId && result.readyId !== activeSelectionReadyId) return result
      selectionVersions.acceptResult(result.readyId, result.selectionVersion, result.readyId === latestReadyRequestId)
      activeSelectionReadyId = result.readyId
      return result
    }

    const api = {
      list: () => request('session.list'),
      create: () => request('session.create'),
      resume: (savedName) => request('session.resume', { savedName }),
      stop: ({ session, creatureId }) =>
        selectionRequest('session.stop', { session, creatureId }),
      clearSelection: () => selectionRequest('session.clearSelection'),
      reconcile: () => request('session.reconcile'),
      select: ({ session, creatureId }) =>
        selectionRequest('session.select', { session, creatureId }),
    }
    const shell = createSessionShell({ api, chat })
    const tab = computed(() => currentSession.value?.target || '')
    const messages = computed(() => chat.messagesByTab[tab.value] || [])
    const scrollIdentity = computed(() => {
      const session = currentSession.value?.session?.runtimeId
      const creature = currentSession.value?.targetCreatureId
      return session && creature ? `${session}:${creature}` : ''
    })
    watch(scrollIdentity, () => {
      attachmentRevision.value += 1
      draftRevision.value += 1
    })
    const scroll = createConversationScrollController({
      schedule: (callback) => nextTick(callback),
    })
    const messageChanges = createConversationMessageOrchestrator(scroll)

    function messageSequenceSnapshot(items) {
      return items.map((message) => ({ id: message.id, eventId: message.eventId }))
    }

    function messageTailSignature(items) {
      const last = items[items.length - 1]
      if (!last) return '0'
      const contentLength =
        typeof last.content === 'string'
          ? last.content.length
          : Array.isArray(last.content)
            ? last.content.length
            : 0
      const parts = Array.isArray(last.parts)
        ? last.parts
            .map((part) =>
              part.type === 'text'
                ? `t:${part.content?.length || 0}`
                : `o:${part.status || ''}:${part.result?.length || 0}:${part.children?.length || 0}`,
            )
            .join('|')
        : ''
      return `${items.length}:${last.id}:${last.role}:${contentLength}:${parts}`
    }

    watch(
      [scrollIdentity, () => messages.value.length],
      ([identity, count]) => scroll.setIdentity(identity, { hasMessages: count > 0 }),
      { immediate: true },
    )
    watch(
      () => ({
        identity: scrollIdentity.value,
        sequence: messageSequenceSnapshot(messages.value),
        tail: messageTailSignature(messages.value),
      }),
      (current, previous) => {
        if (!previous || (current.sequence.length === 0 && previous.sequence.length === 0)) return
        messageChanges.beforeMessagesChange(
          previous.identity,
          previous.sequence,
          current.identity,
          current.sequence,
        )
      },
      { flush: 'sync' },
    )
    watch(
      () => ({
        identity: scrollIdentity.value,
        sequence: messageSequenceSnapshot(messages.value),
        tail: messageTailSignature(messages.value),
      }),
      (current, previous) => {
        if (previous && (current.tail !== previous.tail || current.identity !== previous.identity))
          messageChanges.afterMessagesChange(current.identity, current.sequence)
      },
      { flush: 'post' },
    )
    watch(
      () => chat.processingByTab[tab.value],
      (processing) => {
        if (processing) scroll.onMessagesUpdated({ hasMessages: messages.value.length > 0 })
      },
    )
    onBeforeUnmount(() => scroll.dispose())

    globalThis.__ktVsCodeHistory = (session, creature) => request('http.history', { session, creature })
    globalThis.__ktVsCodeInterrupt = (session, creature) => request('http.interrupt', { session, creature })

    async function reloadSessions() {
      sessions.value = await shell.list()
    }

    const applySelection = (selection, changed = true, isCurrent = () => true) =>
      applyTopologySelection({
        selection,
        changed,
        shell,
        chat,
        getCurrentSession: () => currentSession.value,
        setCurrentSession: (value) => (currentSession.value = value),
        setSessions: (value) => (sessions.value = value),
        isCurrent,
      })

    let latestReadyRequestId = null
    const readyCoordinator = createReadyCoordinator({
      requestReady: () =>
        request('ready', {}, (id) => {
          latestReadyRequestId = id
          attachmentRevision.value += 1
          draftRevision.value += 1
        }),
      async applyReady(result, isCurrent) {
        if (result.available === true) {
          activeSelectionReadyId = result.readyId
          const versioned = selectionVersions.acceptBaseline(activeSelectionReadyId, result.selectionVersion)
          const readyVersion = result.selectionVersion
          await applySelection(
            result.selection,
            true,
            () =>
              isCurrent() &&
              (!versioned || selectionVersions.highest() === readyVersion),
          )
        }
        if (!isCurrent()) return
        available.value = result.available === true
        automatic.value = result.automatic !== false
        if (available.value) {
          error.value = ''
        } else {
          chat.unbindFromInstance()
          currentSession.value = null
          sessions.value = []
          error.value = 'No local KohakuTerrarium service found. Run “kt serve start”, then press Refresh.'
        }
      },
      async applyFailure(cause, isCurrent) {
        if (isCurrent()) error.value = cause.message
      },
    })

    const reconcileSessions = () => readyCoordinator.reconcile()

    async function createSession() {
      busy.value = true
      error.value = ''
      try {
        currentSession.value = await shell.create()
        await reloadSessions()
      } catch (cause) {
        error.value = cause.message
      } finally {
        busy.value = false
      }
    }

    async function resumeSession(session) {
      busy.value = true
      error.value = ''
      try {
        currentSession.value = await shell.resume(session.savedName)
        await reloadSessions()
      } catch (cause) {
        error.value = cause.message
      } finally {
        busy.value = false
      }
    }

    async function stopSession() {
      if (!currentSession.value?.targetCreatureId) return
      busy.value = true
      error.value = ''
      try {
        await shell.stop(currentSession.value)
        currentSession.value = null
        await reloadSessions()
      } catch (cause) {
        error.value = cause.message
      } finally {
        busy.value = false
      }
    }

    async function openSession(session, creatureId) {
      busy.value = true
      error.value = ''
      try {
        currentSession.value = await shell.open(session, creatureId)
      } catch (cause) {
        error.value = cause.message
      } finally {
        busy.value = false
      }
    }

    async function send({ text = draft.value, attachments: submittedAttachments = attachments.value } = {}) {
      if ((!text.trim() && submittedAttachments.length === 0) || !currentSession.value?.target) return
      const submitToken = submitGate.acquire(currentConversationOwnership())
      if (!submitToken) return
      submitRevision.value += 1
      const submittedText = text
      const submitted = [...submittedAttachments]
      const submittedOwner = attachmentBuckets.capture()
      try {
        await conversationOwnership.dispatch(async (assertCurrent) => {
          const content = submitted.length
            ? await buildMessageParts(submittedText, submitted)
            : submittedText
          assertCurrent()
          const sent = BridgeWebSocket.captureSend(() => chat.send(content))
          const outcome = await sent.value
          if (sent.confirmation != null) await sent.confirmation
          return outcome
        })
        if (draftBuckets.get(submittedOwner) === submittedText) {
          draftBuckets.clear(submittedOwner)
          draftRevision.value += 1
        }
        attachmentBuckets.removeSubmitted(submitted, submittedOwner)
        attachmentRevision.value += 1
        if (conversationOwnership.isCurrent(submittedOwner)) {
          scroll.forceFollow()
          error.value = ''
        }
      } catch (cause) {
        if (!isConversationSuperseded(cause) && conversationOwnership.isCurrent(submittedOwner))
          error.value = cause?.message || String(cause)
      } finally {
        submitGate.release(submitToken)
        submitRevision.value += 1
      }
    }

    async function manageContext(type) {
      if (!available.value || !currentSession.value?.target || busy.value || contextBusy.value) return
      const ownedReadyId = activeSelectionReadyId
      const ownedTarget = currentSession.value.targetCreatureId
      const operation = ++contextOperation
      const isCurrent = () =>
        operation === contextOperation &&
        ownedReadyId === activeSelectionReadyId &&
        ownedTarget === currentSession.value?.targetCreatureId
      contextBusy.value = true
      error.value = ''
      status.value = ''
      try {
        const response = await request(type)
        applyContextCommandOutcome(response, isCurrent(), (kind, text) => {
          if (kind === 'error') error.value = text
          else status.value = text
        })
      } catch (cause) {
        if (isCurrent()) error.value = cause?.message || String(cause)
      } finally {
        if (operation === contextOperation) contextBusy.value = false
      }
    }

    function onComposerError(problem) {
      error.value =
        problem?.error?.message ||
        (problem?.code === 'too-large'
          ? `${problem.name} is too large to attach`
          : `Could not attach ${problem?.name || 'file'}`)
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

    function icon(name) {
      const paths = {
        add: 'M17 15V8h-2v7H8v2h7v7h2v-7h7v-2z',
        chevron: 'M16 22 6 12l1.4-1.4 8.6 8.6 8.6-8.6L26 12z',
        refresh: 'M12 10H6.78A11 11 0 0 1 27 16h2A13 13 0 0 0 6 7.68V4H4v8h8Zm8 12h5.22A11 11 0 0 1 5 16H3a13 13 0 0 0 23 8.32V28h2v-8h-8Z',
        send: 'm27.45 15.11-22-11a1 1 0 0 0-1.08.12 1 1 0 0 0-.33 1L7 16 4 26.74A1 1 0 0 0 5 28a1 1 0 0 0 .45-.11l22-11a1 1 0 0 0 0-1.78m-20.9 10L8.76 17H18v-2H8.76L6.55 6.89 24.76 16Z',
        stop: 'M16 2a14 14 0 1 0 14 14A14 14 0 0 0 16 2m6 18a2 2 0 0 1-2 2h-8a2 2 0 0 1-2-2v-8a2 2 0 0 1 2-2h8a2 2 0 0 1 2 2Z',
      }
      return h('svg', { class: `action-icon action-icon--${name}`, viewBox: '0 0 32 32', 'aria-hidden': 'true' }, [
        h('path', { d: paths[name] }),
      ])
    }

    function actionButton(label, iconName, { text, ...options } = {}) {
      return h(
        'button',
        { type: 'button', title: label, 'aria-label': label, ...options },
        [icon(iconName), text ? h('span', text) : null],
      )
    }

    function renderSession(session) {
      const selected = currentSession.value?.session?.runtimeId === session.runtimeId
      const common = {
        disabled: busy.value || !available.value,
        class: ['session-row', selected ? 'is-active' : ''],
      }
      if (!session.isLive) {
        return h('button', {
          ...common,
          key: session.conversationId || session.savedName,
          disabled: common.disabled || !session.savedName,
          'aria-label': `Resume Session ${session.title}`,
          onClick: () => resumeSession(session),
        }, [h('span', { class: 'status-dot status-dot--dormant', 'aria-hidden': 'true' }), h('span', { class: 'row-label' }, session.title)])
      }
      if (session.creatures.length === 1) {
        const creature = session.creatures[0]
        return h('button', {
          ...common,
          key: session.conversationId || session.runtimeId,
          'aria-label': `Open Session ${session.title}`,
          onClick: () => openSession(session, creature.id),
        }, [h('span', { class: 'status-dot', 'aria-hidden': 'true' }), h('span', { class: 'row-label' }, session.title)])
      }
      return h('div', { class: 'session-group', key: session.conversationId || session.runtimeId }, [
        h('div', { class: ['session-row', 'session-row--label', selected ? 'is-active' : ''] }, [
          h('span', { class: 'status-dot', 'aria-hidden': 'true' }),
          h('span', { class: 'row-label' }, session.title),
        ]),
        ...session.creatures.map((creature) => h('button', {
          class: ['session-row', 'creature-row', currentSession.value?.targetCreatureId === creature.id ? 'is-active' : ''],
          disabled: common.disabled,
          'aria-label': `Open Creature ${creature.name} in Session ${session.title}`,
          onClick: () => openSession(session, creature.id),
        }, [h('span', { class: 'row-label' }, creature.name)])),
      ])
    }

    const receiveHostMessage = ({ data: message }) => {
      BridgeWebSocket.receive(message)
      if (message?.type === 'configuration.changed') {
        BridgeWebSocket.disposeAll(Error('KohakuTerrarium configuration changed'))
        rejectPending(Error('KohakuTerrarium configuration changed'))
        chat.unbindFromInstance()
        currentSession.value = null
        available.value = false
        automatic.value = false
        readyCoordinator.reconcile()
        return
      }
      if (message?.type === 'selection.changed') {
        const eventReadyId = message.readyId ?? activeSelectionReadyId
        const pendingRuntime = eventReadyId === latestReadyRequestId
        if (!pendingRuntime && eventReadyId !== activeSelectionReadyId) return
        const notification = selectionVersions.beginNotification(
          eventReadyId,
          message.data.selectionVersion,
          pendingRuntime,
        )
        if (!notification) return
        const ownedReadyId = message.readyId
        const isCurrent = () =>
          notification.isCurrent() &&
          (ownedReadyId === undefined || ownedReadyId === activeSelectionReadyId || ownedReadyId === latestReadyRequestId)
        reconciliation = reconciliation
          .then(() => applySelection(message.data.selection, message.data.changed, isCurrent))
          .catch((cause) => {
            if (isCurrent()) error.value = cause.message
          })
        return
      }
      if (message?.id && pending.has(message.id)) {
        const promise = pending.get(message.id)
        pending.delete(message.id)
        clearTimeout(promise.timer)
        if (message.type === 'error') promise.reject(Error(message.error))
        else promise.resolve(message.data)
      }
    }
    window.addEventListener('message', receiveHostMessage)
    onBeforeUnmount(() => {
      window.removeEventListener('message', receiveHostMessage)
      readyCoordinator.invalidate()
      BridgeWebSocket.disposeAll(Error('KohakuTerrarium webview disposed'))
      rejectPending(Error('KohakuTerrarium webview disposed'))
    })

    readyCoordinator.reconcile()

    return () => {
      const currentSummary = currentSession.value
        ? `${currentSession.value.session.title} · ${currentSession.value.target || 'Choose a Creature'}`
        : sessions.value.length
          ? `${sessions.value.length} Session${sessions.value.length === 1 ? '' : 's'} · No Session selected`
          : 'No Sessions'
      return h('main', { class: 'kt-conversation-host' }, [
        h('header', { class: 'app-header' }, [
          brandUri ? h('img', { class: 'brand-mark', src: brandUri, alt: '' }) : null,
          h('div', { class: 'header-copy' }, [
            h('h1', 'KohakuTerrarium'),
            h('p', available.value ? (automatic.value ? 'Connected locally' : 'Connected by override') : 'Waiting for local KT'),
          ]),
        ]),
        h('section', { class: 'session-region', 'aria-label': 'Sessions' }, [
          h('div', { class: 'session-toolbar' }, [
            h('button', {
              type: 'button',
              class: 'session-disclosure',
              'aria-expanded': sessionsExpanded.value,
              'aria-controls': 'session-list',
              onClick: () => (sessionsExpanded.value = !sessionsExpanded.value),
            }, [icon('chevron'), h('span', { class: 'session-summary' }, currentSummary)]),
            h('div', { class: 'session-actions' }, [
              actionButton('New Session', 'add', { disabled: busy.value || !available.value, onClick: createSession }),
              actionButton('Refresh Sessions', 'refresh', {
                disabled: busy.value,
                onClick: () => {
                  error.value = ''
                  reconciliation = reconciliation.then(reconcileSessions).catch((cause) => (error.value = cause.message))
                },
              }),
            ]),
          ]),
          sessionsExpanded.value
            ? h('div', { id: 'session-list', class: 'session-list' }, [
                ...sessions.value.map(renderSession),
                currentSession.value?.targetCreatureId
                  ? actionButton('Stop Session', 'stop', { class: 'stop-session', disabled: busy.value, onClick: stopSession, text: 'Stop Session' })
                  : null,
              ])
            : null,
        ]),
        error.value ? h('p', { class: 'status is-error', role: 'alert' }, error.value) : null,
        status.value ? h('p', { class: 'status', role: 'status', 'aria-live': 'polite' }, status.value) : null,
        h('section', { class: 'chat-region' }, [
          h(ChatTranscriptSection, {
            messages: messages.value,
            totalCount: messages.value.length,
            emptyTitle: 'No messages yet',
            emptySubtitle: 'Choose a Session and send a message',
            processing: chat.processingByTab[tab.value],
            processingLabel: 'Kohaku is working',
            reconnecting: chat.wsStatus === 'reconnecting',
            reconnectLabel: 'Reconnecting',
            renderMessage: renderTranscriptMessage,
            onViewportReady: ((identity) => (viewport) =>
              scroll.onViewportReady(viewport, identity))(scrollIdentity.value),
            onScroll: ((identity) => (event) => scroll.onScroll(event, identity))(
              scrollIdentity.value,
            ),
            onReply: ({ message, actionId, values }) => submitReply(message, actionId, values),
          }),
        ]),
        h('section', { class: 'composer-region', 'aria-label': 'Message composer' }, [
          currentSession.value?.target
            ? h(ChatComposer, {
                modelValue: draft.value,
                attachments: attachments.value,
                processing: !!chat.processingByTab[tab.value],
                disabled:
                  !currentSession.value?.target ||
                  isComposerSubmitDisabled(submitBusy.value, !!chat.processingByTab[tab.value]),
                contextActionsDisabled:
                  busy.value ||
                  contextBusy.value ||
                  !available.value ||
                  !!chat.processingByTab[tab.value],
                managedSubmit: true,
                attachmentTransform,
                showContextActions: true,
                placeholder: 'Send to selected Creature',
                labels: {
                  attachFile: 'Attach file',
                  attachImage: 'Attach image',
                  compact: 'Compact context',
                  clear: 'Clear context',
                  message: 'Message',
                  removeAttachment: 'Remove {name}',
                  send: 'Send',
                  stop: 'Stop generation',
                },
                'onUpdate:modelValue': (value) => (draft.value = value),
                'onUpdate:attachments': (value) => (attachments.value = value),
                onSubmit: send,
                onInterrupt: () => chat.interrupt(tab.value),
                onCompact: () => manageContext('context.compact'),
                onClear: () => manageContext('context.clear'),
                onError: onComposerError,
              }, {
                'compact-icon': () => renderCarbonIcon('collapse-all'),
                'clear-icon': () => renderCarbonIcon('clean'),
              })
            : h('p', { class: 'composer-placeholder' }, 'Select a Session to start chatting'),
        ]),
      ])
    }
  },
}

createApp(App).use(createPinia()).mount('#app')
