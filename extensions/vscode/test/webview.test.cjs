const assert = require('node:assert/strict')
const fs = require('node:fs')
const path = require('node:path')
const test = require('node:test')

const root = path.resolve(__dirname, '..')
const read = (relative) => fs.readFileSync(path.join(root, relative), 'utf8')

test('foundation webview build is isolated from shared frontend components', () => {
  const config = read('vite.config.mjs')
  const source = read('src/webview/index.js')

  assert.doesNotMatch(config, /kohakuterrarium-frontend|vue|pinia|element-plus/)
  for (const type of ['session.list', 'session.create', 'session.resume', 'session.select', 'session.stop']) {
    assert.match(source, new RegExp(type.replace('.', '\\.')))
  }
  assert.doesNotMatch(source, /ConversationMessage|Markdown|Composer|useChatStore/)
})

test('BridgeWebSocket forwards opaque frames without parsing them', () => {
  const source = read('src/webview/bridge.js')

  assert.match(source, /onmessage\?\.\(\{ data: message\.data/)
  assert.doesNotMatch(source, /JSON\.parse/)
  assert.doesNotMatch(source, /activity_type|text_delta|ui_event/)
})

test('foundation renderer is plain and has only session lifecycle controls', () => {
  const source = read('src/webview/index.js')

  for (const label of ['New Session', 'Refresh', 'Resume', 'Stop Session']) {
    assert.match(source, new RegExp(label))
  }
  assert.match(source, /createVisibilityInterval/)
})
