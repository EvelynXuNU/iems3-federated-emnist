"""Regression checks for reuse of published experiment records."""

from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch

from fedemnist.core import Client, Config, run, save_json


class PublishedResultsTest(unittest.TestCase):
    def test_missing_checkpoint_retrains_and_complete_run_reuses(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "clients"
            root.mkdir()
            generator = np.random.default_rng(11)
            for client_id in range(100):
                x = generator.random((3, 28, 28), dtype=np.float32)
                y = np.full(3, client_id % 62, dtype=np.int64)
                np.savez(root / f"client_{client_id:03d}.npz", train_x=x[:2], train_y=y[:2],
                         val_x=x[2:], val_y=y[2:])
            save_json(root / "manifest.json", {"source_sha256": "synthetic-regression-data"})
            config = Config(clients=4, epochs=1, rounds=1)
            output = Path(temporary) / "results"
            first = run(root, output, config)
            destination = output / config.name
            original = torch.load(destination / "model.pt", weights_only=True)

            # Public clones retain summaries and histories but omit these ignored files.
            (destination / "model.pt").unlink()
            (destination / "confusion.npy").unlink()
            repeated = run(root, output, config)
            restored = torch.load(destination / "model.pt", weights_only=True)
            self.assertEqual(first["val_acc"], repeated["val_acc"])
            self.assertTrue(all(torch.equal(original[key], restored[key]) for key in original))

            with patch.object(Client, "train", side_effect=AssertionError("A complete run should be reused")):
                self.assertEqual(run(root, output, config), repeated)

            # A failed replacement must leave the completion marker invalidated.
            (destination / "model.pt").unlink()
            with patch.object(Client, "train", side_effect=RuntimeError("Simulated interruption")):
                with self.assertRaisesRegex(RuntimeError, "Simulated interruption"):
                    run(root, output, config)
            self.assertFalse((destination / "summary.json").exists())


if __name__ == "__main__":
    unittest.main()
