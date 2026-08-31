export class BridgeWebSocket {
  static CONNECTING = 0
  static OPEN = 1
  static CLOSING = 2
  static CLOSED = 3
  static sockets = new Map()
  static nextId = 1
  static post = () => {}

  constructor(url) {
    this.url = url
    this.readyState = BridgeWebSocket.CONNECTING
    this.id = BridgeWebSocket.nextId++
    BridgeWebSocket.sockets.set(this.id, this)
    BridgeWebSocket.post({ type: 'ws.open', id: this.id })
  }

  send(data) {
    if (this.readyState !== BridgeWebSocket.OPEN) throw Error('WebSocket not open')
    BridgeWebSocket.post({ type: 'ws.send', id: this.id, data })
  }

  close() {
    if (this.readyState >= BridgeWebSocket.CLOSING) return
    this.readyState = BridgeWebSocket.CLOSING
    BridgeWebSocket.post({ type: 'ws.close', id: this.id })
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
      this.sockets.delete(message.id)
      socket.onclose?.({ code: message.code || 1000, target: socket })
    } else if (message.type === 'ws.error') {
      socket.onerror?.({ target: socket })
    }
  }
}
