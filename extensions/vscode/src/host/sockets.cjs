class SocketOwners {
  constructor() {
    this.sockets = new Map()
    this.generation = 0
  }

  begin() {
    this.close()
    return this.generation
  }

  open(generation, id, factory, view) {
    if (generation !== this.generation) return null
    this.sockets.get(id)?.close()
    let socket
    try {
      socket = factory()
    } catch (error) {
      view.postMessage({ type: 'ws.error', id })
      view.postMessage({ type: 'ws.closed', id, code: 1011 })
      throw error
    }
    this.sockets.set(id, socket)
    const owned = () => this.generation === generation && this.sockets.get(id) === socket
    socket.onopen = () => owned() && view.postMessage({ type: 'ws.opened', id })
    socket.onmessage = (event) => owned() && view.postMessage({ type: 'ws.frame', id, data: String(event.data) })
    socket.onerror = () => {
      if (!owned()) return
      view.postMessage({ type: 'ws.error', id })
      this.sockets.delete(id)
      view.postMessage({ type: 'ws.closed', id, code: 1011 })
      socket.close()
    }
    socket.onclose = (event) => {
      if (!owned()) return
      this.sockets.delete(id)
      view.postMessage({ type: 'ws.closed', id, code: event.code })
    }
    return socket
  }

  send(generation, id, data) {
    if (generation !== this.generation) return false
    const socket = this.sockets.get(id)
    if (!socket) return false
    socket.send(data)
    return true
  }

  closeSocket(generation, id) {
    if (generation !== this.generation) return false
    const socket = this.sockets.get(id)
    if (!socket) return false
    socket.onclose?.({ code: 1000 })
    socket.close()
    return true
  }

  closeGeneration(generation) {
    if (generation !== this.generation) return false
    this.close()
    return true
  }

  close() {
    for (const socket of [...this.sockets.values()]) {
      socket.onclose?.({ code: 1000 })
      socket.close()
    }
    this.sockets.clear()
    this.generation++
  }
}

module.exports = { SocketOwners }
