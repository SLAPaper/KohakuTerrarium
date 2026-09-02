class SocketOwners {
  constructor() {
    this.sockets = new Map()
    this.generation = 0
  }

  begin() {
    this.close()
    return this.generation
  }

  open(generation, socketId, factory, view) {
    if (generation !== this.generation) {
      view.postMessage({ type: 'ws.closed', socketId, code: 1008 })
      return null
    }
    this.sockets.get(socketId)?.close()
    let socket
    try {
      socket = factory()
    } catch (error) {
      view.postMessage({ type: 'ws.closed', socketId, code: 1011 })
      throw error
    }
    this.sockets.set(socketId, socket)
    const owned = () => this.generation === generation && this.sockets.get(socketId) === socket
    socket.onopen = () => owned() && view.postMessage({ type: 'ws.opened', socketId })
    socket.onmessage = (event) => owned() && view.postMessage({ type: 'ws.frame', socketId, data: String(event.data) })
    socket.onerror = () => {
      if (!owned()) return
      view.postMessage({ type: 'ws.error', socketId })
      this.sockets.delete(socketId)
      view.postMessage({ type: 'ws.closed', socketId, code: 1011 })
      socket.close()
    }
    socket.onclose = (event) => {
      if (!owned()) return
      this.sockets.delete(socketId)
      view.postMessage({ type: 'ws.closed', socketId, code: event.code })
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

  closeSocket(generation, id, view = null) {
    if (generation !== this.generation) {
      view?.postMessage({ type: 'ws.closed', socketId: id, code: 1008 })
      return false
    }
    const socket = this.sockets.get(id)
    if (!socket) {
      view?.postMessage({ type: 'ws.closed', socketId: id, code: 1000 })
      return false
    }
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
