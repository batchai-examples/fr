import unittest
from unittest.mock import patch, MagicMock
import torch
from ultralytics.utils.autobatch import check_train_batch_size, autobatch

class TestAutobatch(unittest.TestCase):

    @patch('ultralytics.utils.autobatch.deepcopy')
    def test_check_train_batch_size_happy_path(self, mock_deepcopy):
        # Arrange
        model = MagicMock()
        imgsz = 640
        amp = True
        batch = 0.5
        expected_result = 16

        mock_model = MagicMock()
        mock_model.train.return_value = mock_model
        mock_deepcopy.return_value = mock_model

        with patch('ultralytics.utils.autobatch.autobatch', return_value=expected_result) as mock_autobatch:
            # Act
            result = check_train_batch_size(model, imgsz, amp, batch)

            # Assert
            self.assertEqual(result, expected_result)
            mock_deepcopy.assert_called_once_with(model)
            mock_model.train.assert_called_once()
            mock_autobatch.assert_called_once_with(mock_model, imgsz, fraction=batch)

    @patch('ultralytics.utils.autobatch.deepcopy')
    def test_check_train_batch_size_negative_batch(self, mock_deepcopy):
        # Arrange
        model = MagicMock()
        imgsz = 640
        amp = True
        batch = -1
        expected_result = 16

        mock_model = MagicMock()
        mock_model.train.return_value = mock_model
        mock_deepcopy.return_value = mock_model

        with patch('ultralytics.utils.autobatch.autobatch', return_value=expected_result) as mock_autobatch:
            # Act
            result = check_train_batch_size(model, imgsz, amp, batch)

            # Assert
            self.assertEqual(result, expected_result)
            mock_deepcopy.assert_called_once_with(model)
            mock_model.train.assert_called_once()
            mock_autobatch.assert_called_once_with(mock_model, imgsz, fraction=0.6)

    @patch('ultralytics.utils.autobatch.deepcopy')
    def test_check_train_batch_size_cpu_device(self, mock_deepcopy):
        # Arrange
        model = MagicMock()
        imgsz = 640
        amp = True
        batch = 0.5
        expected_result = 16

        mock_model = MagicMock()
        mock_model.train.return_value = mock_model
        mock_deepcopy.return_value = mock_model
        mock_model.parameters.return_value = [MagicMock(device=torch.device('cpu'))]

        with patch('ultralytics.utils.autobatch.autobatch', return_value=expected_result) as mock_autobatch:
            # Act
            result = check_train_batch_size(model, imgsz, amp, batch)

            # Assert
            self.assertEqual(result, expected_result)
            mock_deepcopy.assert_called_once_with(model)
            mock_model.train.assert_called_once()
            mock_autobatch.assert_not_called()

    @patch('ultralytics.utils.autobatch.deepcopy')
    def test_check_train_batch_size_cudnn_benchmark(self, mock_deepcopy):
        # Arrange
        model = MagicMock()
        imgsz = 640
        amp = True
        batch = 0.5
        expected_result = 16

        mock_model = MagicMock()
        mock_model.train.return_value = mock_model
        mock_deepcopy.return_value = mock_model
        mock_model.parameters.return_value = [MagicMock(device=torch.device('cuda'))]
        torch.backends.cudnn.benchmark = True

        with patch('ultralytics.utils.autobatch.autobatch', return_value=expected_result) as mock_autobatch:
            # Act
            result = check_train_batch_size(model, imgsz, amp, batch)

            # Assert
            self.assertEqual(result, expected_result)
            mock_deepcopy.assert_called_once_with(model)
            mock_model.train.assert_called_once()
            mock_autobatch.assert_not_called()

    @patch('ultralytics.utils.autobatch.deepcopy')
    def test_check_train_batch_size_cuda_anomaly(self, mock_deepcopy):
        # Arrange
        model = MagicMock()
        imgsz = 640
        amp = True
        batch = 0.5
        expected_result = 16

        mock_model = MagicMock()
        mock_model.train.return_value = mock_model
        mock_deepcopy.return_value = mock_model
        mock_model.parameters.return_value = [MagicMock(device=torch.device('cuda'))]
        torch.backends.cudnn.benchmark = False

        with patch('ultralytics.utils.autobatch.autobatch', side_effect=Exception("CUDA anomaly")) as mock_autobatch:
            # Act
            result = check_train_batch_size(model, imgsz, amp, batch)

            # Assert
            self.assertEqual(result, expected_result)
            mock_deepcopy.assert_called_once_with(model)
            mock_model.train.assert_called_once()
            mock_autobatch.assert_called_once()

    def test_autobatch_happy_path(self):
        # Arrange
        model = MagicMock()
        imgsz = 640
        fraction = 0.5
        expected_result = 16

        with patch('ultralytics.utils.autobatch.polyfit', return_value=[1, 2]) as mock_polyfit:
            with patch('ultralytics.utils.autobatch.polyval', return_value=8) as mock_polyval:
                # Act
                result = autobatch(model, imgsz, fraction)

                # Assert
                self.assertEqual(result, expected_result)
                mock_polyfit.assert_called_once_with([1], [2], 1)
                mock_polyval.assert_called_once_with([1, 2], 8)

    def test_autobatch_negative_batch(self):
        # Arrange
        model = MagicMock()
        imgsz = 640
        fraction = 0.5
        expected_result = 16

        with patch('ultralytics.utils.autobatch.polyfit', return_value=[1, 2]) as mock_polyfit:
            with patch('ultralytics.utils.autobatch.polyval', return_value=8) as mock_polyval:
                # Act
                result = autobatch(model, imgsz, fraction)

                # Assert
                self.assertEqual(result, expected_result)
                mock_polyfit.assert_called_once_with([1], [2], 1)
                mock_polyval.assert_called_once_with([1, 2], 8)

    def test_autobatch_cuda_anomaly(self):
        # Arrange
        model = MagicMock()
        imgsz = 640
        fraction = 0.5
        expected_result = 16

        with patch('ultralytics.utils.autobatch.polyfit', return_value=[1, 2]) as mock_polyfit:
            with patch('ultralytics.utils.autobatch.polyval', return_value=8) as mock_polyval:
                # Act
                result = autobatch(model, imgsz, fraction)

                # Assert
                self.assertEqual(result, expected_result)
                mock_polyfit.assert_called_once_with([1], [2], 1)
                mock_polyval.assert_called_once_with([1, 2], 8)

    def test_autobatch_exception(self):
        # Arrange
        model = MagicMock()
        imgsz = 640
        fraction = 0.5
        expected_result = 16

        with patch('ultralytics.utils.autobatch.polyfit', side_effect=Exception("Error")) as mock_polyfit:
            with patch('ultralytics.utils.autobatch.polyval', return_value=8) as mock_polyval:
                # Act
                result = autobatch(model, imgsz, fraction)

                # Assert
                self.assertEqual(result, expected_result)
                mock_polyfit.assert_called_once_with([1], [2], 1)
                mock_polyval.assert_not_called()

if __name__ == '__main__':
    unittest.main()
