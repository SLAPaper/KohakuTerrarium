const assert = require('node:assert/strict')
const fs = require('node:fs')
const path = require('node:path')
const test = require('node:test')

const root = path.resolve(__dirname, '..')
const read = (relative) => fs.readFileSync(path.join(root, relative), 'utf8')

test('webview build uses the shared transcript boundary without later UI dependencies', () => {
  const config = read('vite.config.mjs')
  const source = read('src/webview/index.js')

  assert.match(config, /kohakuterrarium-frontend/)
  assert.match(source, /ChatTranscriptSection|ConversationMessage|useChatStore/)
  assert.match(source, /from ['"]@kohakuterrarium\/chat-ui['"]/)
  assert.match(source, /MarkdownRenderer/)
  assert.doesNotMatch(source, /Composer/)
})

test('BridgeWebSocket forwards opaque frames without parsing them', () => {
  const source = read('src/webview/bridge.js')

  assert.match(source, /onmessage\?\.\(\{ data: message\.data/)
  assert.doesNotMatch(source, /JSON\.parse/)
  assert.doesNotMatch(source, /activity_type|text_delta|ui_event/)
})

test('webview renderer preserves task lifecycle controls around the shared transcript', () => {
  const source = read('src/webview/index.js')

  for (const label of ['New Task', 'Refresh', 'Resume', 'Detach']) {
    assert.match(source, new RegExp(label))
  }
  assert.match(source, /ChatTranscriptSection/)
})
