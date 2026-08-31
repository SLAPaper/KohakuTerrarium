# KohakuTerrarium for VS Code

First-party VS Code workspace extension for creating and operating KohakuTerrarium sessions from a sidebar.

## First-release scope

- Automatically discover a same-host KohakuTerrarium daemon from `~/.kohakuterrarium/run/web.json`.
- Fall back to a bounded probe of local KT ports when using foreground `kt web` or when daemon state is stale.
- Use the default loopback auth bypass without reading, requesting, storing, or sending a host token.
- Reuse a token from VS Code `SecretStorage` only when the local service explicitly disables loopback bypass; prompt once only when that strict service has no stored token.
- List live and dormant sessions.
- Create a session from `kohakuterrarium.defaultCreature` and the current workspace folder.
- Select a Creature by stable Creature ID.
- Render a minimal plain foundation shell; shared transcript, Markdown, composer, and product layout arrive in later migration boundaries.
- Stop and resume sessions.
- Relocate the selected Creature after graph merge/split events and fail closed when it disappears.
- Recover explicitly with Refresh after the KT service restarts; the extension does not run an infinite reconnect loop.

The first release supports Tunnel Browser plus a KohakuTerrarium service on the same host. Remote KT endpoints and multi-user auth are out of scope.

## Use

1. Start the local daemon:

   ```bash
   kt serve start
   ```

2. Set `kohakuterrarium.defaultCreature` to a trusted Creature path or installed `@package/...` reference if you want to create new sessions. Existing sessions can be viewed and resumed without this setting.
3. Open the KohakuTerrarium Activity Bar view.

The extension discovers the daemon URL and connects automatically. With the normal local KT defaults, there is no endpoint or token prompt.

If no daemon is running, start it and press **Refresh**. A foreground `kt web` process is also discovered on the bounded default local port range.

### Strict local auth

If the local service has host-token auth enabled and `loopback_bypass = false`, the extension reuses the token from VS Code `SecretStorage`. It asks only for the token when no valid token has been saved. The endpoint is still discovered automatically.

### Advanced override

Use **KohakuTerrarium: Configure Local Connection Override** only for a nonstandard local port that cannot be discovered. Return to the normal behavior with **KohakuTerrarium: Use Automatic Local Discovery**.

The Webview never receives the token, endpoint, Creature config reference, workspace path, or `pwd`. HTTP, WebSocket, filesystem-sensitive settings, discovery, and selection ownership stay in the Workspace Extension Host.

## Development

```bash
cd extensions/vscode
npm ci
npm test
npm run build
npm run package
```

The VSIX contains only the bundled Extension Host, bundled Webview, stylesheet, manifest, icon, license, and README. Source files, tests, source maps, scripts, dependencies, and lockfiles are excluded.
