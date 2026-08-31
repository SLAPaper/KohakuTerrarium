function publicError(type) {
  switch (type) {
    case 'session.create':
      return {
        code: 'session_create_failed',
        message: 'Could not create the session. Check the Creature setting and workspace.',
      }
    case 'session.resume':
      return {
        code: 'session_resume_failed',
        message: 'Could not resume that session. Check its saved workspace mappings.',
      }
    case 'session.stop':
      return {
        code: 'session_stop_failed',
        message: 'Could not stop the selected session.',
      }
    case 'session.select':
      return {
        code: 'session_select_failed',
        message: 'Could not select that Creature. Refresh Open Sessions and try again.',
      }
    case 'session.reconcile':
      return {
        code: 'session_reconcile_failed',
        message: 'Could not refresh the selected Creature. Check the KohakuTerrarium service.',
      }
    case 'session.list':
      return {
        code: 'session_list_failed',
        message: 'Could not load Open Sessions. Check the KohakuTerrarium connection.',
      }
    case 'http.history':
      return {
        code: 'history_failed',
        message: 'Could not load chat history for the selected Creature.',
      }
    case 'http.interrupt':
      return {
        code: 'interrupt_failed',
        message: 'Could not stop the current turn.',
      }
    case 'ws.open':
    case 'ws.send':
    case 'ws.close':
      return {
        code: 'chat_transport_failed',
        message: 'The chat connection failed. Reconnect to the selected Creature.',
      }
    default:
      return {
        code: 'request_failed',
        message: 'The KohakuTerrarium request failed.',
      }
  }
}

module.exports = { publicError }
