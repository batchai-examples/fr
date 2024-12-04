import unittest
from pathlib import Path
from unittest.mock import patch, MagicMock

from ultralytics.utils.callbacks.clearml import (
    _log_debug_samples,
    _log_plot,
    on_pretrain_routine_start,
    on_train_epoch_end,
    on_fit_epoch_end,
    on_val_end,
    on_train_end,
)

class TestClearMLCallbacks(unittest.TestCase):
    @patch("ultralytics.utils.callbacks.clearml.Task.current_task")
    def test_log_debug_samples(self, mock_task):
        """
        Test the _log_debug_samples function to ensure it logs images correctly.
        """
        files = [Path("train_batch1.jpg"), Path("train_batch2.jpg")]
        _log_debug_samples(files)
        if task := mock_task.return_value:
            task.get_logger().report_image.assert_called_with(
                title="Debug Samples", series="Mosaic_1", local_path=str(files[0]), iteration=1
            )
            task.get_logger().report_image.assert_called_with(
                title="Debug Samples", series="Mosaic_2", local_path=str(files[1]), iteration=2
            )

    @patch("ultralytics.utils.callbacks.clearml.Task.current_task")
    def test_log_plot(self, mock_task):
        """
        Test the _log_plot function to ensure it logs plots correctly.
        """
        plot_path = "results.png"
        _log_plot("Plot Title", plot_path)
        if task := mock_task.return_value:
            task.get_logger().report_matplotlib_figure.assert_called_with(
                title="Plot Title", series="", figure=None, report_interactive=False
            )

    @patch("ultralytics.utils.callbacks.clearml.Task.current_task")
    def test_on_pretrain_routine_start(self, mock_task):
        """
        Test the on_pretrain_routine_start function to ensure it initializes and connects a task correctly.
        """
        trainer = MagicMock()
        on_pretrain_routine_start(trainer)
        if task := mock_task.return_value:
            task.init.assert_called_once_with(
                project="ultralytics",
                name=None,
                output_uri=None,
                reuse_last_task_id=False,
                queue_name=None,
                labels=None,
                upload_files=None,
                reuse_weights=None,
                cache_dir=None,
                debug=False,
                verbose=True,
                auto_connect=True,
            )
            task.connect.assert_called_once_with(trainer)

    @patch("ultralytics.utils.callbacks.clearml.Task.current_task")
    def test_on_train_epoch_end(self, mock_task):
        """
        Test the on_train_epoch_end function to ensure it logs scalars correctly.
        """
        trainer = MagicMock()
        trainer.epoch = 1
        trainer.metrics = {"accuracy": 0.9}
        on_train_epoch_end(trainer)
        if task := mock_task.return_value:
            task.get_logger().report_scalar.assert_called_with(
                title="train", series="accuracy", value=0.9, iteration=1
            )

    @patch("ultralytics.utils.callbacks.clearml.Task.current_task")
    def test_on_fit_epoch_end(self, mock_task):
        """
        Test the on_fit_epoch_end function to ensure it logs scalars correctly.
        """
        trainer = MagicMock()
        trainer.epoch = 1
        trainer.metrics = {"accuracy": 0.9}
        on_fit_epoch_end(trainer)
        if task := mock_task.return_value:
            task.get_logger().report_scalar.assert_called_with(
                title="val", series="accuracy", value=0.9, iteration=1
            )

    @patch("ultralytics.utils.callbacks.clearml.Task.current_task")
    def test_on_val_end(self, mock_task):
        """
        Test the on_val_end function to ensure it logs images correctly.
        """
        validator = MagicMock()
        validator.save_dir = Path(".")
        files = [Path("val1.jpg"), Path("val2.jpg")]
        _log_debug_samples(files)
        if task := mock_task.return_value:
            task.get_logger().report_image.assert_called_with(
                title="Debug Samples", series="Validation_1", local_path=str(files[0]), iteration=1
            )
            task.get_logger().report_image.assert_called_with(
                title="Debug Samples", series="Validation_2", local_path=str(files[1]), iteration=2
            )

    @patch("ultralytics.utils.callbacks.clearml.Task.current_task")
    def test_on_train_end(self, mock_task):
        """
        Test the on_train_end function to ensure it logs plots and metrics correctly.
        """
        trainer = MagicMock()
        trainer.save_dir = Path(".")
        files = [
            "results.png",
            "confusion_matrix.png",
            "confusion_matrix_normalized.png",
            *(f"{x}_curve.png" for x in ("F1", "PR", "P", "R")),
        ]
        files = [(trainer.save_dir / f) for f in files if (trainer.save_dir / f).exists()]
        trainer.best = Path("best_model.pt")
        trainer.args.name = "model_name"
        on_train_end(trainer)
        if task := mock_task.return_value:
            task.update_output_model.assert_called_with(
                model_path=str(trainer.best), model_name="model_name", auto_delete_file=False
            )
            for f in files:
                task.get_logger().report_single_value.assert_any_call(f.stem, None)

if __name__ == "__main__":
    unittest.main()
