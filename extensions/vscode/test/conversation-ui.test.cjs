const assert = require('node:assert/strict')
const fs = require('node:fs')
const path = require('node:path')
const test = require('node:test')

const root = path.resolve(__dirname, '..')
const frontend = path.resolve(root, '..', '..', 'src', 'kohakuterrarium-frontend', 'src')

function read(file) {
  return fs.readFileSync(file, 'utf8')
}

test('Dashboard and VS Code import one production conversation message component', () => {
  const shared = path.join(frontend, 'components', 'chat', 'shared', 'ConversationMessage.js')
  const dashboard = read(path.join(frontend, 'components', 'chat', 'ChatMessage.vue'))
  const webview = read(path.join(root, 'src', 'webview', 'index.js'))

  assert.equal(fs.existsSync(shared), true)
  assert.match(dashboard, /from ["']@kohakuterrarium\/chat-ui["']/)
  assert.match(webview, /from ["']@kohakuterrarium\/chat-ui["']/)
  assert.doesNotMatch(webview, /function renderMessage\(/)
  assert.doesNotMatch(webview, /function renderPart\(/)
  assert.doesNotMatch(webview, /function renderInteractive\(/)
})

test('VS Code conversation text uses the public shared Markdown renderer', () => {
  const webview = read(path.join(root, 'src', 'webview', 'index.js'))

  assert.match(
    webview,
    /import\s*\{[^}]*\bMarkdownRenderer\b[^}]*\}\s*from ['"]@kohakuterrarium\/chat-ui['"]/s,
  )
  assert.match(
    webview,
    /function renderSharedText\(content, breaks = false\)\s*\{\s*return h\(MarkdownRenderer, \{ content, breaks \}\)\s*\}/,
  )
  const messages = [...webview.matchAll(/h\(ConversationMessage,\s*\{([\s\S]*?)\n\s*\}\)/g)]
  assert.ok(messages.length > 0)
  for (const message of messages) assert.match(message[1], /renderText:\s*renderSharedText/)
})

test('Dashboard and VS Code consume one host-neutral transcript section', () => {
  const shared = path.join(frontend, 'components', 'chat', 'shared', 'ChatTranscriptSection.js')
  const dashboard = read(path.join(frontend, 'components', 'chat', 'ChatPanel.vue'))
  const webview = read(path.join(root, 'src', 'webview', 'index.js'))

  assert.equal(fs.existsSync(shared), true)
  assert.match(dashboard, /from ["']@kohakuterrarium\/chat-ui["']/)
  assert.match(webview, /from ["']@kohakuterrarium\/chat-ui["']/)
  assert.doesNotMatch(webview, /messages\.value\.map/)
  assert.match(webview, /processing:\s*chat\.processingByTab\[tab\.value\]/)
  assert.match(webview, /processingLabel:\s*['"][^'"]+['"]/)
  assert.doesNotMatch(webview, /`\$\{error\.value\} \$\{chat\.wsStatus\}`/)
})

test('VS Code binds transcript viewport callbacks to the rendered conversation identity', () => {
  const webview = read(path.join(root, 'src', 'webview', 'index.js'))

  assert.match(webview, /createConversationScrollController/)
  assert.match(
    webview,
    /onViewportReady:\s*\(\(identity\)\s*=>\s*\(viewport\)\s*=>\s*scroll\.onViewportReady\(viewport, identity\)\)\(scrollIdentity\.value\)/,
  )
  assert.match(
    webview,
    /onScroll:\s*\(\(identity\)\s*=>\s*\(event\)\s*=>\s*scroll\.onScroll\(event, identity\)\)\(\s*scrollIdentity\.value,\s*\)/,
  )
  assert.match(webview, /scroll\.forceFollow\(\)\s*\n\s*chat\.send\(content\)/)
  assert.match(webview, /messageTailSignature\(messages\.value\)/)
})

test('shared conversation CSS is the only message visual source used by both hosts', () => {
  const sharedCss = path.join(frontend, 'components', 'chat', 'shared', 'conversation-message.css')
  const dashboard = read(path.join(frontend, 'components', 'chat', 'ChatMessage.vue'))
  const webview = read(path.join(root, 'src', 'webview', 'index.js'))
  const extensionCss = read(path.join(root, 'src', 'webview', 'style.css'))

  assert.equal(fs.existsSync(sharedCss), true)
  const component = read(path.join(frontend, 'components', 'chat', 'shared', 'ConversationMessage.js'))
  assert.match(component, /import ["']\.\/conversation-message\.css["']/)
  assert.doesNotMatch(dashboard, /conversation-message\.css/)
  assert.doesNotMatch(webview, /conversation-message\.css/)
  assert.doesNotMatch(extensionCss, /\.message\s*\{/)
  assert.doesNotMatch(extensionCss, /\.tool\s*\{/)
  assert.match(extensionCss, /--kt-conversation-accent:/)
})
