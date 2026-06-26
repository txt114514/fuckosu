# Visualization

`src/visualization` provides the Chinese training UI and gallery export layer.
Training code should depend only on `visualization.lib`.

Common options:

```text
--progress-ui auto|rich|plain|off
--progress-language zh-CN
```

TTY runs use Rich when available. Non-TTY runs use plain output. `off` disables
interactive rendering without disabling structured events or training.

Docs:

- [Architecture](VISUALIZATION_ARCHITECTURE.md)
- [Public API](VISUALIZATION_API.md)
- [Terminal UI](TERMINAL_UI.md)
- [Gallery Migration](GALLERY_MIGRATION.md)
