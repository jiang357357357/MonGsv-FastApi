"""Epoch values must be measured, mapped correctly, and never interpolated."""
import logging
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock

import torch
from torch.utils.tensorboard import SummaryWriter

from Code.FastApi.Base.Training.telemetry import read_telemetry
from GPT_SoVITS.training_metrics import EpochLoss


class TrainingTelemetryTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)

    def writer(self, target, run=""):
        return SummaryWriter(str(self.root / "train" / target / run))

    def log(self, text):
        path = self.root / "training.log"
        path.write_text(text)
        return str(path)

    def test_old_sovits_samples_only_fill_their_actual_epochs(self):
        with self.writer("sovits") as writer:
            writer.add_scalar("loss/g/total", 38.56024, 0)
            writer.add_scalar("loss/g/total", 32.48245, 100)
        log = self.log("0/12 [00:00<?, ?it/s]INFO:dataset:Train Epoch: 1 [0%]\n"
                       "4/12 [00:03<00:07, 1.13it/s]INFO:dataset:Train Epoch: 9 [33%]\n"
                       "INFO:dataset:====> Epoch: 12\n")
        data = read_telemetry(str(self.root), "sovits", log, True, 12)
        self.assertEqual(len(data["epoch_loss_history"]), 12)
        recorded = [row for row in data["epoch_loss_history"] if row["value"] is not None]
        self.assertEqual([(row["epoch"], row["step"], row["kind"]) for row in recorded], [(1, 0, "sample"), (9, 100, "sample")])
        self.assertEqual(data["current_loss_epoch"], 9)
        self.assertEqual(data["epoch_loss_source"], "step_sample")
        self.assertEqual(data["missing_epochs"], [2, 3, 4, 5, 6, 7, 8, 10, 11, 12])

    def test_gpt_epoch_means_use_logged_epoch_markers_instead_of_point_order(self):
        with self.writer("gpt") as writer:
            for epoch, step, value in [(0, 10, 6000), (2, 32, 5000), (3, 43, 4000)]:
                writer.add_scalar("total_loss_epoch", value, step)
                writer.add_scalar("epoch", epoch, step)
        data = read_telemetry(str(self.root), "gpt", completed=True, total_epochs=4)
        self.assertEqual([row["value"] for row in data["epoch_loss_history"]], [6000, None, 5000, 4000])
        self.assertEqual(data["missing_epochs"], [2])
        self.assertEqual(data["current_loss_epoch"], 4)
        self.assertEqual(data["epoch_loss_source"], "epoch_mean")

    def test_new_sovits_records_each_epoch_mean_without_more_step_logs(self):
        with self.writer("sovits") as writer:
            writer.add_scalar("loss/g/total", 99, 0)
            for epoch in range(1, 4):
                metric = EpochLoss()
                for value in [epoch, epoch + 2]:
                    metric.add(torch.tensor(float(value), requires_grad=True))
                self.assertFalse(metric.total.requires_grad)
                metric.write(writer, logging.getLogger("telemetry-test"), epoch, epoch * 2)
        data = read_telemetry(str(self.root), "sovits", completed=True, total_epochs=3)
        self.assertEqual([row["value"] for row in data["epoch_loss_history"]], [2, 3, 4])
        self.assertEqual(data["current_epoch"], 3)
        self.assertEqual(data["current_loss"], 4)
        self.assertEqual(data["missing_epochs"], [])
        self.assertEqual(len(data["loss_history"]), 1)
        self.assertEqual(data["epoch_loss_source"], "epoch_mean")

    def test_resume_keeps_old_missing_epochs_and_new_means_override_samples(self):
        with self.writer("sovits", "old") as writer:
            writer.add_scalar("loss/g/total", 38, 0)
            writer.add_scalar("loss/g/total", 32, 100)
        with self.writer("sovits", "resumed") as writer:
            writer.add_scalar("loss/g/total", 99, 156)
            for epoch in range(13, 17):
                writer.add_scalar("loss/g/total_epoch", 30 - epoch, epoch)
                writer.add_scalar("epoch", epoch, epoch * 12)
        log = self.log("0/12 [00:00<?, ?it/s]INFO:dataset:Train Epoch: 1 [0%]\n")
        data = read_telemetry(str(self.root), "sovits", log, True, 16)
        self.assertEqual(data["epoch_loss_history"][12]["value"], 17)
        self.assertEqual(data["epoch_loss_history"][12]["kind"], "mean")
        self.assertEqual(data["epoch_loss_history"][11]["kind"], "missing")
        self.assertEqual(data["epoch_loss_source"], "mixed")
        self.assertEqual(data["current_loss_epoch"], 16)

    def test_samples_are_not_assigned_epochs_without_loader_size_evidence(self):
        with self.writer("sovits") as writer:
            writer.add_scalar("loss/g/total", 38, 0)
            writer.add_scalar("loss/g/total", 32, 100)
        data = read_telemetry(str(self.root), "sovits", completed=True, total_epochs=12)
        self.assertTrue(all(row["value"] is None for row in data["epoch_loss_history"]))
        self.assertEqual(len(data["loss_history"]), 2)
        self.assertIsNone(data["current_loss_epoch"])

    def test_epoch_records_are_not_lost_to_tensorboard_reservoir_sampling(self):
        with self.writer("sovits") as writer:
            for epoch in range(1, 1003):
                writer.add_scalar("loss/g/total_epoch", epoch / 10, epoch)
        data = read_telemetry(str(self.root), "sovits", completed=True, total_epochs=1002)
        self.assertEqual(len(data["epoch_loss_history"]), 1002)
        self.assertEqual(data["missing_epochs"], [])
        self.assertAlmostEqual(data["epoch_loss_history"][0]["value"], .1)

    def test_time_windows_and_eval_events_do_not_leak_into_epoch_means(self):
        with self.writer("gpt") as writer:
            for epoch, when in [(0, 100), (1, 200), (2, 300)]:
                writer.add_scalar("total_loss_epoch", epoch + 1, epoch * 10, walltime=when)
                writer.add_scalar("epoch", epoch, epoch * 10, walltime=when)
        with self.writer("gpt", "eval") as writer:
            writer.add_scalar("total_loss_epoch", 999, 10, walltime=200)
            writer.add_scalar("epoch", 1, 10, walltime=200)
        data = read_telemetry(str(self.root), "gpt", completed=True, total_epochs=3, window_start=150, window_end=250)
        self.assertEqual([row["value"] for row in data["epoch_loss_history"]], [None, 2, None])

    def test_empty_epochs_do_not_produce_a_fabricated_zero_loss(self):
        writer = Mock()
        EpochLoss().write(writer, Mock(), 1, 0)
        writer.add_scalar.assert_not_called()


if __name__ == "__main__":
    unittest.main()
