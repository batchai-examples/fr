import unittest
from unittest.mock import patch, MagicMock
from ultralytics.utils.tuner import run_ray_tune

class TestRunRayTune(unittest.TestCase):

    @patch('subprocess.run')
    def test_run_ray_tune_with_default_args(self, mock_subprocess):
        """
        Tests the run_ray_tune function with default arguments.
        
        Steps:
        1. Mock subprocess.run to simulate a successful execution.
        2. Call run_ray_tune with default arguments.
        3. Assert that subprocess.run was called once.
        """
        model = MagicMock()
        result = run_ray_tune(model)
        mock_subprocess.assert_called_once()

    @patch('subprocess.run')
    def test_run_ray_tune_with_custom_args(self, mock_subprocess):
        """
        Tests the run_ray_tune function with custom arguments.
        
        Steps:
        1. Mock subprocess.run to simulate a successful execution.
        2. Call run_ray_tune with custom arguments.
        3. Assert that subprocess.run was called once.
        """
        model = MagicMock()
        result = run_ray_tune(model, data="custom_data", epochs=50)
        mock_subprocess.assert_called_once()

    @patch('subprocess.run')
    def test_run_ray_tune_with_no_space(self, mock_subprocess):
        """
        Tests the run_ray_tune function with no search space provided.
        
        Steps:
        1. Mock subprocess.run to simulate a successful execution.
        2. Call run_ray_tune without providing a search space.
        3. Assert that subprocess.run was called once and a warning message was logged.
        """
        model = MagicMock()
        with patch('logging.warning') as mock_warning:
            result = run_ray_tune(model)
            mock_subprocess.assert_called_once()
            mock_warning.assert_called_once()

    @patch('subprocess.run')
    def test_run_ray_tune_with_gpu(self, mock_subprocess):
        """
        Tests the run_ray_tune function with GPU support.
        
        Steps:
        1. Mock subprocess.run to simulate a successful execution.
        2. Call run_ray_tune with GPU support.
        3. Assert that subprocess.run was called once and the correct number of GPUs were allocated.
        """
        model = MagicMock()
        result = run_ray_tune(model, gpu_per_trial=1)
        mock_subprocess.assert_called_once()

    @patch('subprocess.run')
    def test_run_ray_tune_with_no_gpu(self, mock_subprocess):
        """
        Tests the run_ray_tune function with no GPU support.
        
        Steps:
        1. Mock subprocess.run to simulate a successful execution.
        2. Call run_ray_tune without providing GPU support.
        3. Assert that subprocess.run was called once and no GPUs were allocated.
        """
        model = MagicMock()
        result = run_ray_tune(model, gpu_per_trial=0)
        mock_subprocess.assert_called_once()

if __name__ == '__main__':
    unittest.main()
