async function resolveLocalConnection({ discover, getStoredToken, promptToken, storeToken = async () => {}, verify }) {
  const local = await discover()
  if (!local.requiresToken) {
    const connection = { ...local, token: '' }
    await verify(connection)
    return connection
  }

  const storedToken = (await getStoredToken()) || ''
  if (storedToken) {
    const storedConnection = { ...local, token: storedToken }
    try {
      await verify(storedConnection)
      return storedConnection
    } catch (error) {
      if (error?.status !== 401) throw error
    }
  }

  const token = (await promptToken()) || ''
  if (!token) throw Error('Host token is required by the local service')
  const connection = { ...local, token }
  await verify(connection)
  await storeToken(token)
  return connection
}

module.exports = { resolveLocalConnection }
