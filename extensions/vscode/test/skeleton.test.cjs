const assert = require('node:assert/strict')
const fs = require('node:fs')
const path = require('node:path')
const test = require('node:test')

const ROOT = path.resolve(__dirname, '..')

function read(relative) {
  return fs.readFileSync(path.join(ROOT, relative), 'utf8')
}

test('manifest defines a workspace sidebar extension and deterministic package scripts', () => {
  const manifest = JSON.parse(read('package.json'))

  assert.equal(manifest.name, 'kohakuterrarium-vscode')
  assert.equal(manifest.private, true)
  assert.equal(manifest.main, './dist/extension.cjs')
  assert.deepEqual(manifest.extensionKind, ['workspace'])
  assert.ok(manifest.activationEvents.includes('onView:kohakuterrarium.chat'))
  assert.deepEqual(manifest.contributes.views.kohakuterrarium, [
    { type: 'webview', id: 'kohakuterrarium.chat', name: 'KohakuTerrarium' },
  ])
  assert.deepEqual(manifest.contributes.commands, [
    {
      command: 'kohakuterrarium.configure',
      title: 'KohakuTerrarium: Configure Local Connection Override',
    },
    {
      command: 'kohakuterrarium.useAutomaticDiscovery',
      title: 'KohakuTerrarium: Use Automatic Local Discovery',
    },
  ])
  assert.equal(manifest.scripts.test, 'node --test test/*.test.cjs')
  assert.equal(manifest.scripts.build, 'node scripts/build.cjs')
  assert.equal(manifest.scripts.package, 'npm run build && vsce package --no-dependencies')
  assert.equal(manifest.repository.url, 'https://github.com/Kohaku-Lab/KohakuTerrarium')
  assert.equal(manifest.files, undefined)
  const ignored = read('.vscodeignore')
  for (const entry of ['src/**', 'test/**', 'scripts/**', 'node_modules/**', 'package-lock.json']) {
    assert.match(ignored, new RegExp(entry.replaceAll('*', '\\*')))
  }
})

test('fixed protocol accepts only shell messages and rejects host capabilities from the webview', () => {
  const { allowedMessage } = require('../src/host/protocol.cjs')

  for (const message of [
    { type: 'ready', id: 1 },
    { type: 'session.list', id: 2 },
    { type: 'session.create', id: 3 },
  ]) {
    assert.equal(allowedMessage(message), true)
  }

  for (const message of [
    { type: 'session.create', id: 4, token: 'secret' },
    { type: 'session.create', id: 5, configPath: 'C:/secret/config' },
    { type: 'session.create', id: 6, pwd: 'C:/workspace' },
    { type: 'unknown', id: 7 },
    { type: 'ready' },
  ]) {
    assert.equal(allowedMessage(message), false)
  }
})

test('webview shell uses a strict CSP and only packaged resources', () => {
  const { renderWebviewHtml } = require('../src/host/webview.cjs')
  const html = renderWebviewHtml({
    cspSource: 'vscode-webview://origin',
    scriptUri: 'vscode-webview://origin/dist/webview.js',
    styleUri: 'vscode-webview://origin/dist/webview.css',
    nonce: 'fixed-test-nonce',
  })

  assert.match(html, /default-src 'none'/)
  assert.match(html, /img-src vscode-webview:\/\/origin data:/)
  assert.match(html, /style-src vscode-webview:\/\/origin/)
  assert.match(html, /script-src 'nonce-fixed-test-nonce'/)
  assert.match(html, /src="vscode-webview:\/\/origin\/dist\/webview\.js"/)
  assert.match(html, /href="vscode-webview:\/\/origin\/dist\/webview\.css"/)
  assert.doesNotMatch(html, /unsafe-inline/)
  assert.doesNotMatch(html, /https?:\/\//)
})

test('build script is self-contained inside the formal extension package', () => {
  const build = read('scripts/build.cjs')

  assert.match(build, /require\(['"]esbuild['"]\)/)
  assert.match(build, /path\.join\(root, 'src', 'extension\.cjs'\)/)
  assert.match(build, /path\.join\(root, 'vite\.config\.mjs'\)/)
  assert.doesNotMatch(build, /vscode-chat-sidebar-tracer/)
  assert.doesNotMatch(build, /kohakuterrarium-frontend[\\/]node_modules/)
})
