"""Serving static frontend launcher plus legacy shims.

The legacy ``KohakuManager`` facade and the ``StreamOutput`` /
``ChannelEvent`` helpers were removed in Phase 3 of the studio cleanup.
The :class:`Terrarium` engine (``kohakuterrarium.terrarium``) and the
studio sessions modules (``kohakuterrarium.studio.sessions``) own those
responsibilities now. A narrow ``AgentSession`` module remains as an
external-package compatibility shim.
"""

__all__: list[str] = []
