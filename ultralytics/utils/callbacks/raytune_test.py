import unittest
from unittest.mock import patch, MagicMock

from ultralytics.utils.callbacks.raytune import on_fit_epoch_end, callbacks

class TestRayTuneCallback(unittest.TestCase):
    @patch("ultralytics.utils.SETTINGS", {"raytune": True})
    def test_on_fit_epoch_end_with_ray_tune_enabled(self):
        """
        Test the `on_fit_epoch_end` function when Ray Tune is enabled.
        
        Steps:
        1. Mock the necessary dependencies and objects.
        2. Call the `on_fit_epoch_end` function with a mock trainer object.
        3. Verify that the metrics are reported to Ray Tune.
        """
        # Arrange
        ray = MagicMock()
        tune = MagicMock()
        session = MagicMock()
        
        with patch("ultralytics.utils.callbacks.raytune(ray, tune, session)"):
            trainer = MagicMock(metrics={"loss": 0.5}, epoch=1)
            
            # Act
            on_fit_epoch_end(trainer)
            
            # Assert
            session.report.assert_called_once_with({"loss": 0.5, "epoch": 1})

    @patch("ultralytics.utils.SETTINGS", {"raytune": False})
    def test_on_fit_epoch_end_with_ray_tune_disabled(self):
        """
        Test the `on_fit_epoch_end` function when Ray Tune is disabled.
        
        Steps:
        1. Mock the necessary dependencies and objects.
        2. Call the `on_fit_epoch_end` function with a mock trainer object.
        3. Verify that no metrics are reported to Ray Tune.
        """
        # Arrange
        ray = MagicMock()
        tune = MagicMock()
        session = MagicMock()
        
        with patch("ultralytics.utils.callbacks.raytune(ray, tune, session)"):
            trainer = MagicMock(metrics={"loss": 0.5}, epoch=1)
            
            # Act
            on_fit_epoch_end(trainer)
            
            # Assert
            session.report.assert_not_called()

    def test_callbacks_with_ray_tune_enabled(self):
        """
        Test the `callbacks` dictionary when Ray Tune is enabled.
        
        Steps:
        1. Mock the necessary dependencies and objects.
        2. Verify that the `on_fit_epoch_end` callback is included in the `callbacks` dictionary.
        """
        # Arrange
        ray = MagicMock()
        tune = MagicMock()
        session = MagicMock()
        
        with patch("ultralytics.utils.callbacks.raytune(ray, tune, session)"):
            expected_callbacks = {
                "on_fit_epoch_end": on_fit_epoch_end,
            }
            
            # Act
            actual_callbacks = callbacks
            
            # Assert
            self.assertEqual(actual_callbacks, expected_callbacks)

    def test_callbacks_with_ray_tune_disabled(self):
        """
        Test the `callbacks` dictionary when Ray Tune is disabled.
        
        Steps:
        1. Mock the necessary dependencies and objects.
        2. Verify that no callbacks are included in the `callbacks` dictionary.
        """
        # Arrange
        ray = MagicMock()
        tune = None
        session = MagicMock()
        
        with patch("ultralytics.utils.callbacks.raytune(ray, tune, session)"):
            expected_callbacks = {}
            
            # Act
            actual_callbacks = callbacks
            
            # Assert
            self.assertEqual(actual_callbacks, expected_callbacks)

if __name__ == "__main__":
    unittest.main()
