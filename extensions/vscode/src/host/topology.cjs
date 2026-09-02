class TopologyWatcher {
  constructor({ socketFactory, endpoint, token, onInvalidate }) {
    this.socketFactory = socketFactory
    this.endpoint = endpoint
    this.token = token
    this.onInvalidate = onInvalidate
    this.socket = null
    this.queue = Promise.resolve()
  }

  start() {
    this.close()
    const url = this.endpoint.replace(/^http:/, 'ws:') + '/ws/runtime/graph'
    const socket = this.socketFactory(url, this.token ? [`kt-token.${this.token}`] : [])
    this.socket = socket
    socket.onerror = () => {}
    socket.onmessage = (event) => {
      if (this.socket !== socket) return
      let frame
      try {
        frame = JSON.parse(String(event.data))
      } catch {
        return
      }
      if (!['topology_changed', 'creature_stopped'].includes(frame.type)) return
      this.queue = this.queue.then(() => (this.socket === socket ? this.onInvalidate(frame) : undefined)).catch(() => {})
    }
    return socket
  }

  close() {
    const socket = this.socket
    this.socket = null
    socket?.close()
  }
}

module.exports = { TopologyWatcher }
