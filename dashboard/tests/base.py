import tempfile
import unittest
from pathlib import Path

from tests import fake_dataset


class DatasetTestCase(unittest.TestCase):
    """Each test gets a fresh synthetic dataset on disk and clean module-level caches."""

    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        # a private parent dir: tests that write 'outside the dataroot' must not touch a shared /tmp
        self.root = Path(tmp.name) / "data"
        self.ds, stack = fake_dataset.install(self.root)
        self.addCleanup(stack.close)
