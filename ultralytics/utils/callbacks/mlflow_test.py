import os
import unittest
from unittest.mock import patch, MagicMock

from ultralytics.utils.callbacks.mlflow import on_pretrain_routine_end, on_train_epoch_end, on_fit_epoch_end, on_train_end, callbacks

class TestMLflowCallbacks(unittest.TestCase):
    @patch("ultralytics.utils.callbacks.mlflow.mlflow")
    def test_on_pretrain_routine_end(self, mock_mlflow):
        """
        Test the on_pretrain_routine_end function with various scenarios.
        
        Steps:
        1. Mock the mlflow module and its functions.
        2. Set environment variables for MLFLOW_TRACKING_URI, MLFLOW_EXPERIMENT_NAME, and MLFLOW_RUN.
        3. Call the on_pretrain_routine_end function with a mock trainer object.
        4. Verify that the mlflow functions are called with the correct arguments.
        """
        os.environ["MLFLOW_TRACKING_URI"] = "http://127.0.0.1:5000"
        os.environ["MLFLOW_EXPERIMENT_NAME"] = "test_experiment"
        os.environ["MLFLOW_RUN"] = "test_run"

        mock_trainer = MagicMock()
        mock_trainer.args.project = "test_project"
        mock_trainer.args.name = "test_name"
        mock_trainer.lr = {"lr": 0.01}
        mock_trainer.tloss = {"loss": 0.5}

        on_pretrain_routine_end(mock_trainer)

        mock_mlflow.set_tracking_uri.assert_called_once_with("http://127.0.0.1:5000")
        mock_mlflow.set_experiment.assert_called_once_with("test_project")
        mock_mlflow.autolog.assert_called_once()
        mock_mlflow.start_run.assert_called_once_with(run_name="test_run")
        mock_mlflow.log_params.assert_called_once_with({"project": "test_project", "name": "test_name"})

    @patch("ultralytics.utils.callbacks.mlflow.mlflow")
    def test_on_train_epoch_end(self, mock_mlflow):
        """
        Test the on_train_epoch_end function with various scenarios.
        
        Steps:
        1. Mock the mlflow module and its functions.
        2. Call the on_train_epoch_end function with a mock trainer object.
        3. Verify that the mlflow.log_metrics function is called with the correct arguments.
        """
        mock_trainer = MagicMock()
        mock_trainer.lr = {"lr": 0.01}
        mock_trainer.label_loss_items.return_value = {"loss": 0.5}

        on_train_epoch_end(mock_trainer)

        mock_mlflow.log_metrics.assert_called_once_with(metrics={"lr": 0.01, "train_loss": 0.5}, step=mock_trainer.epoch)

    @patch("ultralytics.utils.callbacks.mlflow.mlflow")
    def test_on_fit_epoch_end(self, mock_mlflow):
        """
        Test the on_fit_epoch_end function with various scenarios.
        
        Steps:
        1. Mock the mlflow module and its functions.
        2. Call the on_fit_epoch_end function with a mock trainer object.
        3. Verify that the mlflow.log_metrics function is called with the correct arguments.
        """
        mock_trainer = MagicMock()
        mock_trainer.metrics = {"loss": 0.5}

        on_fit_epoch_end(mock_trainer)

        mock_mlflow.log_metrics.assert_called_once_with(metrics={"loss": 0.5}, step=mock_trainer.epoch)

    @patch("ultralytics.utils.callbacks.mlflow.mlflow")
    def test_on_train_end(self, mock_mlflow):
        """
        Test the on_train_end function with various scenarios.
        
        Steps:
        1. Mock the mlflow module and its functions.
        2. Set environment variables for MLFLOW_KEEP_RUN_ACTIVE.
        3. Call the on_train_end function with a mock trainer object.
        4. Verify that the mlflow.log_artifact and mlflow.end_run functions are called with the correct arguments.
        """
        os.environ["MLFLOW_KEEP_RUN_ACTIVE"] = "False"

        mock_trainer = MagicMock()
        mock_trainer.best.parent = "/path/to/save_dir"
        mock_trainer.save_dir.glob.return_value = [MagicMock(suffix=".pt"), MagicMock(suffix=".yaml")]

        on_train_end(mock_trainer)

        mock_mlflow.log_artifact.assert_called_once_with("/path/to/save_dir/weights/best.pt")
        mock_mlflow.log_artifact.assert_called_with("/path/to/save_dir/last.pt")
        mock_mlflow.log_artifact.assert_called_once_with("/path/to/save_dir/file.png")
        mock_mlflow.log_artifact.assert_called_once_with("/path/to/save_dir/file.jpg")
        mock_mlflow.log_artifact.assert_called_once_with("/path/to/save_dir/file.csv")
        mock_mlflow.log_artifact.assert_called_once_with("/path/to/save_dir/file.pt")
        mock_mlflow.log_artifact.assert_called_once_with("/path/to/save_dir/file.yaml")
        mock_mlflow.end_run.assert_called_once()

    def test_callbacks(self):
        """
        Test the callbacks dictionary.
        
        Steps:
        1. Verify that the callbacks dictionary is not empty when mlflow is available.
        2. Verify that the callbacks dictionary is empty when mlflow is not available.
        """
        os.environ["MLFLOW_TRACKING_URI"] = "http://127.0.0.1:5000"

        self.assertTrue(callbacks)

        del os.environ["MLFLOW_TRACKING_URI"]

        self.assertFalse(callbacks)


if __name__ == "__main__":
    unittest.main()
