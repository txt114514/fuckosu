"""验证图集 API 迁移后的新旧导入路径仍指向同一实现。"""

from __future__ import annotations

import unittest

from traning.lib.visualization.gallery import save_best_trial_gallery as old_gallery
from visualization.lib.gallery_api import export_best_trial_gallery
from visualization.core.gallery.exporter import save_best_trial_gallery as new_gallery


class VisualizationGalleryMigrationTests(unittest.TestCase):
    def test_old_gallery_import_reexports_new_implementation(self) -> None:
        self.assertIs(old_gallery, new_gallery)
        self.assertTrue(callable(export_best_trial_gallery))


if __name__ == "__main__":
    unittest.main()
