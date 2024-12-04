import unittest
from unittest.mock import patch, MagicMock
import numpy as np
import pandas as pd

class TestWandbCallbacks(unittest.TestCase):
    @patch('ultralytics.utils.callbacks.wb.wb')
    def test_on_pretrain_routine_start(self, mock_wb):
        from ultralytics.utils.callbacks.wb import on_pretrain_routine_start
        trainer = MagicMock()
        trainer.args.project = "YOLOv8"
        trainer.args.name = "test_run"

        on_pretrain_routine_start(trainer)

        mock_wb.init.assert_called_once_with(project="YOLOv8", name="test_run", config=vars(trainer.args))

    @patch('ultralytics.utils.callbacks.wb.wb')
    def test_on_fit_epoch_end(self, mock_wb):
        from ultralytics.utils.callbacks.wb import on_fit_epoch_end
        trainer = MagicMock()
        trainer.metrics = {"accuracy": 0.9}
        trainer.epoch = 2
        trainer.plots = {}
        trainer.validator.plots = {}

        on_fit_epoch_end(trainer)

        mock_wb.run.log.assert_called_once_with({"accuracy": 0.9}, step=3)
        self.assertIn("run_1_model", _processed_plots)

    @patch('ultralytics.utils.callbacks.wb.wb')
    def test_on_train_epoch_end(self, mock_wb):
        from ultralytics.utils.callbacks.wb import on_train_epoch_end
        trainer = MagicMock()
        trainer.label_loss_items.return_value = {"loss": 0.5}
        trainer.lr = 0.01
        trainer.epoch = 2
        trainer.plots = {}

        on_train_epoch_end(trainer)

        mock_wb.run.log.assert_called_once_with({"train_loss": 0.5}, step=3)
        mock_wb.run.log.assert_called_once_with({"lr": 0.01}, step=3)
        self.assertIn("run_1_model", _processed_plots)

    @patch('ultralytics.utils.callbacks.wb.wb')
    def test_on_train_end(self, mock_wb):
        from ultralytics.utils.callbacks.wb import on_train_end
        trainer = MagicMock()
        trainer.best.exists.return_value = True
        trainer.validator.metrics.curves = ["accuracy"]
        trainer.validator.metrics.curves_results = [[np.array([0.1]), np.array([0.2]), "x", "y"]]
        trainer.validator.metrics.names = {"class1": "name1"}
        trainer.epoch = 2
        trainer.plots = {}
        trainer.validator.plots = {}

        on_train_end(trainer)

        mock_wb.Artifact.assert_called_once_with(type="model", name=f"run_{mock_wb.run.id}_model")
        mock_wb.run.log_artifact.assert_called_once()
        mock_wb.plot.line.assert_called_once()

    @patch('ultralytics.utils.callbacks.wb.wb')
    def test_plot_curve(self, mock_wb):
        from ultralytics.utils.callbacks.wb import _plot_curve
        x = np.array([0.1, 0.2])
        y = np.array([0.3, 0.4])
        names = ["class1"]
        id = "curves/accuracy"
        title = "Accuracy Curve"
        x_title = "Epochs"
        y_title = "Accuracy"

        _plot_curve(x, y, names, id, title, x_title, y_title)

        mock_wb.Table.assert_called_once()
        mock_wb.plot.line.assert_called_once()

    @patch('ultralytics.utils.callbacks.wb.wb')
    def test_log_plots(self, mock_wb):
        from ultralytics.utils.callbacks.wb import _log_plots
        plots = {"plot1": {"timestamp": 1}}
        step = 2

        with patch.dict(_processed_plots, clear=True):
            _log_plots(plots, step)

        self.assertIn("plot1", _processed_plots)
        mock_wb.run.log.assert_called_once()

    @patch('ultralytics.utils.callbacks.wb.wb')
    def test_on_pretrain_routine_start_no_wb(self, mock_wb):
        from ultralytics.utils.callbacks.wb import on_pretrain_routine_start
        trainer = MagicMock()
        trainer.args.project = "YOLOv8"
        trainer.args.name = "test_run"

        with patch.dict('ultralytics.utils.callbacks.wb.wb', clear=True):
            on_pretrain_routine_start(trainer)

        mock_wb.init.assert_not_called()

    @patch('ultralytics.utils.callbacks.wb.wb')
    def test_on_fit_epoch_end_no_plots(self, mock_wb):
        from ultralytics.utils.callbacks.wb import on_fit_epoch_end
        trainer = MagicMock()
        trainer.metrics = {"accuracy": 0.9}
        trainer.epoch = 2
        trainer.plots = {}
        trainer.validator.plots = {}

        with patch.dict('ultralytics.utils.callbacks.wb._processed_plots', clear=True):
            on_fit_epoch_end(trainer)

        mock_wb.run.log.assert_called_once_with({"accuracy": 0.9}, step=3)
        self.assertNotIn("run_1_model", _processed_plots)

    @patch('ultralytics.utils.callbacks.wb.wb')
    def test_on_train_epoch_end_no_plots(self, mock_wb):
        from ultralytics.utils.callbacks.wb import on_train_epoch_end
        trainer = MagicMock()
        trainer.label_loss_items.return_value = {"loss": 0.5}
        trainer.lr = 0.01
        trainer.epoch = 2
        trainer.plots = {}

        with patch.dict('ultralytics.utils.callbacks.wb._processed_plots', clear=True):
            on_train_epoch_end(trainer)

        mock_wb.run.log.assert_called_once_with({"train_loss": 0.5}, step=3)
        mock_wb.run.log.assert_called_once_with({"lr": 0.01}, step=3)
        self.assertNotIn("run_1_model", _processed_plots)

if __name__ == '__main__':
    unittest.main()
