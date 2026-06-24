from __future__ import annotations

import unittest

from environment import (
    REQUIRED_PACKAGES,
    collect_environment_report,
)


class EnvironmentCheckTests(unittest.TestCase):
    def test_collect_environment_report_is_non_destructive(self) -> None:
        report = collect_environment_report()

        self.assertTrue(report.python_version)
        self.assertTrue(report.python_executable)
        self.assertIsNotNone(report.torch.cuda_available)

    def test_required_package_specs_are_reported(self) -> None:
        report = collect_environment_report()

        expected = {spec.label for spec in REQUIRED_PACKAGES}
        actual = {check.spec.label for check in report.packages}
        self.assertTrue(expected.issubset(actual))


if __name__ == "__main__":
    unittest.main()
