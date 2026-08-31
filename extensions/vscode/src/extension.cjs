const crypto = require('node:crypto')
const vscode = require('vscode')
const WebSocket = require('ws')

const { createClient, validateCapabilities } = require('./host/client.cjs')
const { resolveLocalConnection } = require('./host/connection.cjs')
const { discoverInstalledKt, probeCapabilities } = require('./host/localDiscovery.cjs')
const { publicError } = require('./host/errors.cjs')
const { allowedMessage, validateEndpoint } = require('./host/protocol.cjs')
const { RuntimeHost } = require('./host/runtime.cjs')
const { SocketOwners } = require('./host/sockets.cjs')
const { ConnectionStateWriter } = require('./host/state.cjs')
const { TopologyWatcher } = require('./host/topology.cjs')
const { renderWebviewHtml } = require('./host/webview.cjs')

const CONFIG_KEY = 'kohakuterrarium.connection'
const TOKEN_KEY = 'kohakuterrarium.hostToken'

function tokenRequired(capabilities) {
  const policy = capabilities.auth?.host_token || {}
  return policy.enabled === true && policy.loopback_bypass !== true
}

async function resolveConnection(context, stored) {
  let discovery
  if (stored.manual === true && stored.endpoint) {
    const endpoint = validateEndpoint(stored.endpoint)
    const capabilities = validateCapabilities(await probeCapabilities(endpoint))
    discovery = {
      endpoint,
      capabilities,
      source: 'manual',
      requiresToken: tokenRequired(capabilities),
    }
  } else {
    try {
      discovery = await discoverInstalledKt()
    } catch (error) {
      if (!stored.endpoint) throw error
      const endpoint = validateEndpoint(stored.endpoint)
      const capabilities = validateCapabilities(await probeCapabilities(endpoint))
      discovery = {
        endpoint,
        capabilities,
        source: 'legacy',
        requiresToken: tokenRequired(capabilities),
      }
    }
  }

  return resolveLocalConnection({
    discover: async () => discovery,
    getStoredToken: () => context.secrets.get(TOKEN_KEY),
    promptToken: () =>
      vscode.window.showInputBox({
        prompt: 'The local KT service requires its host token',
        password: true,
        ignoreFocusOut: true,
      }),
    storeToken: (token) => context.secrets.store(TOKEN_KEY, token),
    verify: async ({ endpoint, token }) => {
      await createClient({ endpoint, token }).listOpen()
    },
  })
}

async function configure(context, stateWriter, onConfigured = () => {}) {
  const previous = stateWriter.read()
  const rawEndpoint = await vscode.window.showInputBox({
    prompt: 'Advanced local KT endpoint override',
    value: previous.endpoint || 'http://127.0.0.1:8001',
    ignoreFocusOut: true,
  })
  if (!rawEndpoint) return
  const endpoint = validateEndpoint(rawEndpoint)
  const capabilities = validateCapabilities(await probeCapabilities(endpoint))
  let token = ''
  if (tokenRequired(capabilities)) {
    token =
      (await vscode.window.showInputBox({
        prompt: 'The overridden local KT service requires its host token',
        password: true,
        ignoreFocusOut: true,
      })) || ''
    if (!token) throw Error('Host token is required by the local service')
  }
  await createClient({ endpoint, token }).listOpen()
  if (token) await context.secrets.store(TOKEN_KEY, token)
  await stateWriter.update(() => ({
    endpoint,
    manual: true,
    source: 'manual',
    selection: previous.endpoint === endpoint ? previous.selection || null : null,
  }))
  await onConfigured()
  await vscode.window.showInformationMessage('KohakuTerrarium local override configured')
  await vscode.commands.executeCommand('kohakuterrarium.chat.focus')
}

function webSocketBase(endpoint) {
  return endpoint.replace(/^http:/, 'ws:')
}

function activate(context) {
  const liveViews = new Set()
  const stateWriter = new ConnectionStateWriter(context.workspaceState, CONFIG_KEY)
  async function rediscoverViews() {
    for (const entry of liveViews) {
      entry.disposeRuntime()
      await entry.webview.postMessage({ type: 'configuration.changed' })
    }
  }

  context.subscriptions.push(
    vscode.commands.registerCommand('kohakuterrarium.configure', () =>
      configure(context, stateWriter, rediscoverViews).catch(() =>
        vscode.window.showErrorMessage('Could not configure local KohakuTerrarium'),
      ),
    ),
    vscode.commands.registerCommand('kohakuterrarium.useAutomaticDiscovery', async () => {
      await stateWriter.update((current) => ({
        manual: false,
        selection: current.selection || null,
      }))
      await rediscoverViews()
      await vscode.window.showInformationMessage('KohakuTerrarium will use automatic local discovery')
    }),
  )

  const provider = {
    resolveWebviewView(view) {
      const webview = view.webview
      webview.options = {
        enableScripts: true,
        localResourceRoots: [vscode.Uri.joinPath(context.extensionUri, 'dist')],
      }
      webview.html = renderWebviewHtml({
        cspSource: webview.cspSource,
        scriptUri: String(
          webview.asWebviewUri(vscode.Uri.joinPath(context.extensionUri, 'dist', 'webview.js')),
        ),
        styleUri: String(
          webview.asWebviewUri(vscode.Uri.joinPath(context.extensionUri, 'dist', 'webview.css')),
        ),
        nonce: crypto.randomBytes(16).toString('base64'),
      })

      let runtime = null
      let runtimePromise = null
      let topology = null
      let activeConnection = null
      let epoch = 0
      const entry = {
        webview,
        disposeRuntime() {
          epoch++
          topology?.close()
          topology = null
          runtime?.dispose()
          runtime = null
          runtimePromise = null
          activeConnection = null
        },
      }
      liveViews.add(entry)

      const sendError = (message, error) => {
        console.error(`KohakuTerrarium ${message.type} failed`, error)
        const safe = publicError(message.type)
        return webview.postMessage({
          type: 'error',
          id: message.id,
          error: safe.message,
          code: safe.code,
        })
      }

      async function buildRuntime() {
        const initial = stateWriter.read()
        const connection = await resolveConnection(context, initial)
        const stored = await stateWriter.update((current) => ({
          endpoint: connection.endpoint,
          manual: current.manual === true,
          source: connection.source,
          selection: current.selection || null,
        }))
        const runtimeEpoch = epoch
        activeConnection = connection
        const state = {
          selection: stored.selection || null,
          async updateSelection(selection) {
            if (runtimeEpoch !== epoch) throw Error('Runtime ownership changed')
            await stateWriter.update((current) => {
              if (runtimeEpoch !== epoch || current.endpoint !== connection.endpoint) return undefined
              return { ...current, selection }
            })
            if (runtimeEpoch !== epoch) throw Error('Runtime ownership changed')
            this.selection = selection
          },
        }
        const createdRuntime = new RuntimeHost({
          client: createClient(connection),
          state,
          sockets: new SocketOwners(),
          post: (response) => webview.postMessage(response),
          getDefaultCreature: () =>
            vscode.workspace.getConfiguration('kohakuterrarium').get('defaultCreature', ''),
          getWorkspacePath: () => vscode.workspace.workspaceFolders?.[0]?.uri.fsPath || null,
          socketFactory: (url, protocols) => new WebSocket(url, protocols),
          webSocketBase: webSocketBase(connection.endpoint),
          token: connection.token,
        })
        topology = new TopologyWatcher({
          socketFactory: (url, protocols) => new WebSocket(url, protocols),
          endpoint: connection.endpoint,
          token: connection.token,
          onInvalidate: async () => {
            if (runtimeEpoch !== epoch || runtime !== createdRuntime) return
            const result = await createdRuntime.reconcileSelection()
            await webview.postMessage({ type: 'selection.changed', data: result })
          },
        })
        if (runtimeEpoch !== epoch) {
          createdRuntime.dispose()
          throw Error('Runtime ownership changed')
        }
        runtime = createdRuntime
        topology.start()
        return createdRuntime
      }

      function ensureRuntime() {
        if (runtime) return Promise.resolve(runtime)
        if (!runtimePromise) {
          runtimePromise = buildRuntime().finally(() => {
            runtimePromise = null
          })
        }
        return runtimePromise
      }

      const disposable = webview.onDidReceiveMessage(async (message) => {
        if (!allowedMessage(message)) return
        try {
          if (message.type === 'ready') {
            try {
              if (runtime) entry.disposeRuntime()
              const current = await ensureRuntime()
              const reconciled = await current.reconcileSelection()
              webview.postMessage({
                type: 'ready.result',
                id: message.id,
                data: {
                  available: true,
                  automatic: activeConnection.source !== 'manual',
                  selection: reconciled.selection,
                },
              })
            } catch (error) {
              entry.disposeRuntime()
              webview.postMessage({
                type: 'ready.result',
                id: message.id,
                data: { available: false, automatic: true, selection: null },
              })
              sendError(message, error)
            }
            return
          }
          await (await ensureRuntime()).handle(message)
          if (message.type === 'session.reconcile') topology?.start()
        } catch (error) {
          sendError(message, error)
        }
      })

      view.onDidDispose(() => {
        disposable.dispose()
        entry.disposeRuntime()
        liveViews.delete(entry)
      })
    },
  }

  context.subscriptions.push(
    vscode.window.registerWebviewViewProvider('kohakuterrarium.chat', provider),
  )
}

function deactivate() {}

module.exports = { activate, configure, deactivate, resolveConnection, tokenRequired, webSocketBase }
