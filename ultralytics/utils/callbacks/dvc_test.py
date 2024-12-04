import unittest
from unittest.mock import patch, MagicMock

from ultralytics.utils.callbacks.dvc import (
    on_pretrain_routine_start,
    on_pretrain_routine_end,
    on_train_start,
    on_train_epoch_start,
    on_fit_epoch_end,
    on_train_end,
)

class TestDVCUtils(unittest.TestCase):
    @patch("ultralytics.utils.callbacks.dvc.dvclive")
    def test_on_pretrain_routine_start(self, mock_dvclive):
        """Test the `on_pretrain_routine_start` function."""
        # Arrange
        trainer = MagicMock()
        global live

        # Act
        on_pretrain_routine_start(trainer)

        # Assert
        self.assertIsNotNone(live)
        mock_dvclive.Live.assert_called_once_with(save_dvc_exp=True, cache_images=True)
        LOGGER.info.assert_called_once_with("DVCLive is detected and auto logging is enabled (run 'yolo settings dvc=False' to disable).")

    @patch("ultralytics.utils.callbacks.dvc.dvclive")
    def test_on_pretrain_routine_end(self, mock_dvclive):
        """Test the `on_pretrain_routine_end` function."""
        # Arrange
        trainer = MagicMock()
        global live
        live = MagicMock()

        # Act
        on_pretrain_routine_end(trainer)

        # Assert
        _log_plots.assert_called_once_with(trainer.plots, "train")
        live.end.assert_called_once()

    @patch("ultralytics.utils.callbacks.dvc.dvclive")
    def test_on_train_start(self, mock_dvclive):
        """Test the `on_train_start` function."""
        # Arrange
        trainer = MagicMock()
        global live
        live = MagicMock()

        # Act
        on_train_start(trainer)

        # Assert
        live.log_params.assert_called_once_with(trainer.args)

    @patch("ultralytics.utils.callbacks.dvc.dvclive")
    def test_on_train_epoch_start(self, mock_dvclive):
        """Test the `on_train_epoch_start` function."""
        # Arrange
        trainer = MagicMock()
        global _training_epoch

        # Act
        on_train_epoch_start(trainer)

        # Assert
        self.assertTrue(_training_epoch)

    @patch("ultralytics.utils.callbacks.dvc.dvclive")
    def test_on_fit_epoch_end(self, mock_dvclive):
        """Test the `on_fit_epoch_end` function."""
        # Arrange
        trainer = MagicMock()
        global _training_epoch
        _training_epoch = True
        global live
        live = MagicMock()

        # Act
        on_fit_epoch_end(trainer)

        # Assert
        all_metrics = {**trainer.label_loss_items(trainer.tloss, prefix="train"), **trainer.metrics, **trainer.lr}
        for metric, value in all_metrics.items():
            live.log_metric.assert_any_call(metric, value)
        _log_plots.assert_called_once_with(trainer.plots, "train")
        _log_plots.assert_called_once_with(trainer.validator.plots, "val")
        live.next_step.assert_called_once()
        self.assertFalse(_training_epoch)

    @patch("ultralytics.utils.callbacks.dvc.dvclive")
    def test_on_train_end(self, mock_dvclive):
        """Test the `on_train_end` function."""
        # Arrange
        trainer = MagicMock()
        global _training_epoch
        _training_epoch = False
        global live
        live = MagicMock()

        # Act
        on_train_end(trainer)

        # Assert
        all_metrics = {**trainer.label_loss_items(trainer.tloss, prefix="train"), **trainer.metrics, **trainer.lr}
        for metric, value in all_metrics.items():
            live.log_metric.assert_any_call(metric, value, plot=False)
        _log_plots.assert_called_once_with(trainer.plots, "val")
        _log_plots.assert_called_once_with(trainer.validator.plots, "val")
        _log_confusion_matrix.assert_called_once_with(trainer.validator)
        if trainer.best.exists():
            live.log_artifact.assert_called_once_with(trainer.best, copy=True, type="model")
        live.end.assert_called_once()

if __name__ == "__main__":
    unittest.main()
