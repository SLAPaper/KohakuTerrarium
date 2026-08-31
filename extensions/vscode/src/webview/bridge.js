export class BridgeWebSocket {
  static CONNECTING = 0
  static OPEN = 1
  static CLOSING = 2
  static CLOSED = 3
  static sockets = new Map()
  static nextId = 1
  static nextSendId = 1
  static confirmationTimeout = 30000
  static capture = null
  static post = () => {}

  constructor(url) {
    this.url = url
    this.readyState = BridgeWebSocket.CONNECTING
    this.id = BridgeWebSocket.nextId++
    this.pendingSends = new Map()
    BridgeWebSocket.sockets.set(this.id, this)
    BridgeWebSocket.post({ type: 'ws.open', id: this.id })
  }

  send(data) {
    if (this.readyState !== BridgeWebSocket.OPEN) throw Error('WebSocket not open')
    const sendId = BridgeWebSocket.nextSendId++
    const confirmation = new Promise((resolve, reject) => {
      const timer = setTimeout(() => {
        this.pendingSends.delete(sendId)
        reject(Error('Chat send confirmation timed out'))
      }, BridgeWebSocket.confirmationTimeout)
      this.pendingSends.set(sendId, { resolve, reject, timer })
    })
    // Standard WebSocket callers do not observe Host confirmations.
    confirmation.catch(() => {})
    BridgeWebSocket.capture?.push(confirmation)
    BridgeWebSocket.post({ type: 'ws.send', id: this.id, sendId, data })
  }

  close() {
    if (this.readyState >= BridgeWebSocket.CLOSING) return
    this.readyState = BridgeWebSocket.CLOSING
    this.rejectPending(Error('Chat socket closed'))
    BridgeWebSocket.post({ type: 'ws.close', id: this.id })
  }

  rejectPending(error) {
    for (const pending of this.pendingSends.values()) {
      clearTimeout(pending.timer)
      pending.reject(error)
    }
    this.pendingSends.clear()
  }

  static captureSend(callback) {
    const previous = this.capture
    const confirmations = []
    this.capture = confirmations
    try {
      const value = callback()
      if (confirmations.length > 1) throw Error('Chat submit generated multiple WebSocket frames')
      return { value, confirmation: confirmations[0] || null }
    } finally {
      this.capture = previous
    }
  }

  static settleSend(socket, message, error) {
    const pending = socket.pendingSends.get(message.sendId)
    if (!pending) return
    socket.pendingSends.delete(message.sendId)
    clearTimeout(pending.timer)
    if (error) pending.reject(error)
    else pending.resolve()
  }

  static receive(message) {
    const socket = this.sockets.get(message.id)
    if (!socket) return
    if (message.type === 'ws.opened') {
      socket.readyState = this.OPEN
      socket.onopen?.({ target: socket })
    } else if (message.type === 'ws.frame') {
      socket.onmessage?.({ data: message.data, target: socket })
    } else if (message.type === 'ws.closed') {
      socket.readyState = this.CLOSED
      socket.rejectPending(Error('Chat socket closed'))
      this.sockets.delete(message.id)
      socket.onclose?.({ code: message.code || 1000, target: socket })
    } else if (message.type === 'ws.error') {
      socket.onerror?.({ target: socket })
    } else if (message.type === 'ws.send.result') {
      this.settleSend(socket, message)
    } else if (message.type === 'ws.send.error') {
      this.settleSend(socket, message, Error(message.error || 'Chat send was rejected'))
    }
  }

  static disposeAll(error = Error('WebSocket bridge disposed')) {
    for (const socket of this.sockets.values()) socket.rejectPending(error)
  }
}
