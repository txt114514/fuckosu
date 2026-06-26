# Gallery 迁移

旧实现：

```text
src/traning/lib/visualization/gallery.py
```

新实现：

```text
src/visualization/core/gallery/exporter.py
```

训练业务通过 `visualization.lib.gallery_api.export_best_trial_gallery` 调用新实现。
旧路径保留 re-export 以兼容现有测试和命令。
