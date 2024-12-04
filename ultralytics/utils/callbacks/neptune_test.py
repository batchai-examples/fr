import unittest
from unittest.mock import patch, MagicMock
from pathlib import Path

# Mocking neptune and its dependencies
class MockNeptune:
    def __init__(self, *args, **kwargs):
        self.run = None

    def init_run(self, *args, **kwargs):
        self.run = MagicMock()
        return self.run

    def __getattr__(self, item):
        if item == "types":
            return {"File": MagicMock()}
        return MagicMock()

class MockTrainer:
    def __init__(self, args=None, tloss=None, lr=None, metrics=None, save_dir=None, best=None):
        self.args = args or MagicMock()
        self.tloss = tloss or {}
        self.lr = lr or {}
        self.metrics = metrics or {}
        self.save_dir = save_dir or Path("test_save_dir")
        self.best = best or MagicMock()

class TestNeptuneCallbacks(unittest.TestCase):

    @patch('ultralytics.utils.callbacks.neptune.neptune', new_callable=MockNeptune)
    def test_on_pretrain_routine_start(self, mock_neptune):
        """
        Test the on_pretrain_routine_start function.
        
        Steps:
        1. Mock the neptune library and its dependencies.
        2. Create a mock trainer instance.
        3. Call the on_pretrain_routine_start function.
        4. Verify that neptune.init_run is called with correct arguments.
        """
        from ultralytics.utils.callbacks.neptune import on_pretrain_routine_start
        on_pretrain_routine_start(MockTrainer())
        mock_neptune.assert_called_once_with(project="YOLOv8", name=None, tags=["YOLOv8"])

    @patch('ultralytics.utils.callbacks.neptune.neptune', new_callable=MockNeptune)
    def test_on_train_epoch_end(self, mock_neptune):
        """
        Test the on_train_epoch_end function.
        
        Steps:
        1. Mock the neptune library and its dependencies.
        2. Create a mock trainer instance with tloss, lr, and save_dir attributes.
        3. Call the on_train_epoch_end function.
        4. Verify that _log_scalars is called twice (for tloss and lr).
        """
        from ultralytics.utils.callbacks.neptune import on_train_epoch_end
        on_train_epoch_end(MockTrainer(tloss={"loss": 0.5}, lr={"lr": 0.01}))
        mock_neptune.return_value.run.assert_has_calls([
            mock.call["Configuration/Hyperparameters"].set({"loss": "", "lr": ""}),
            mock.call["train/loss"].append(value=0.5, step=2),
            mock.call["train/lr"].append(value=0.01, step=2)
        ])

    @patch('ultralytics.utils.callbacks.neptune.neptune', new_callable=MockNeptune)
    def test_on_fit_epoch_end(self, mock_neptune):
        """
        Test the on_fit_epoch_end function.
        
        Steps:
        1. Mock the neptune library and its dependencies.
        2. Create a mock trainer instance with metrics attribute.
        3. Call the on_fit_epoch_end function.
        4. Verify that _log_scalars is called once.
        """
        from ultralytics.utils.callbacks.neptune import on_fit_epoch_end
        on_fit_epoch_end(MockTrainer(metrics={"accuracy": 0.9}))
        mock_neptune.return_value.run.assert_has_calls([
            mock.call["Configuration/Model"].set({"accuracy": ""}),
            mock.call["fit/accuracy"].append(value=0.9, step=2)
        ])

    @patch('ultralytics.utils.callbacks.neptune.neptune', new_callable=MockNeptune)
    def test_on_val_end(self, mock_neptune):
        """
        Test the on_val_end function.
        
        Steps:
        1. Mock the neptune library and its dependencies.
        2. Create a mock trainer instance with save_dir attribute.
        3. Call the on_val_end function.
        4. Verify that _log_plot is called once for each file in save_dir.
        """
        from ultralytics.utils.callbacks.neptune import on_val_end
        on_val_end(MockTrainer(save_dir=Path("test_save_dir")))
        mock_neptune.return_value.run.assert_has_calls([
            mock.call["weights/test_task/best"].upload(File=str(Path("test_save_dir") / "best"))
        ])

    @patch('ultralytics.utils.callbacks.neptune.neptune', new_callable=MockNeptune)
    def test_on_train_end(self, mock_neptune):
        """
        Test the on_train_end function.
        
        Steps:
        1. Mock the neptune library and its dependencies.
        2. Create a mock trainer instance with save_dir and best attributes.
        3. Call the on_train_end function.
        4. Verify that _log_plot is called once for each file in save_dir.
        """
        from ultralytics.utils.callbacks.neptune import on_train_end
        on_train_end(MockTrainer(save_dir=Path("test_save_dir"), best=MagicMock(name="best_model")))
        mock_neptune.return_value.run.assert_has_calls([
            mock.call["weights/test_task/best_model"].upload(File=str(Path("test_save_dir") / "best_model"))
        ])

if __name__ == '__main__':
    unittest.main()
