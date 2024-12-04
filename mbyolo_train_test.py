import unittest
from unittest.mock import patch, MagicMock
from mbyolo_train import parse_opt

class TestMbyoloTrain(unittest.TestCase):

    @patch('mbyolo_train.YOLO')
    def test_train_task(self, mock_yolo):
        # Arrange
        args = {
            "data": "/path/to/dataset.yaml",
            "epochs": 300,
            "workers": 128,
            "batch": 512,
            "optimizer": "SGD",
            "device": "0,1,2,3,4,5,6,7",
            "amp": True,
            "project": "/output_dir/mscoco",
            "name": "mambayolo"
        }
        mock_yolo.return_value.train.return_value = None

        # Act
        parse_opt()
        
        # Assert
        mock_yolo.assert_called_once_with("/path/to/mamba-yolo/Mamba-YOLO-T.yaml")
        mock_yolo.return_value.train.assert_called_once_with(
            data="/path/to/dataset.yaml",
            epochs=300,
            workers=128,
            batch=512,
            optimizer="SGD",
            device="0,1,2,3,4,5,6,7",
            amp=True,
            project="/output_dir/mscoco",
            name="mambayolo"
        )

    @patch('mbyolo_train.YOLO')
    def test_val_task(self, mock_yolo):
        # Arrange
        args = {
            "data": "/path/to/dataset.yaml",
            "epochs": 300,
            "workers": 128,
            "batch": 512,
            "optimizer": "SGD",
            "device": "0,1,2,3,4,5,6,7",
            "amp": True,
            "project": "/output_dir/mscoco",
            "name": "mambayolo"
        }
        mock_yolo.return_value.val.return_value = None

        # Act
        parse_opt()
        
        # Assert
        mock_yolo.assert_called_once_with("/path/to/mamba-yolo/Mamba-YOLO-T.yaml")
        mock_yolo.return_value.val.assert_called_once_with(
            data="/path/to/dataset.yaml",
            epochs=300,
            workers=128,
            batch=512,
            optimizer="SGD",
            device="0,1,2,3,4,5,6,7",
            amp=True,
            project="/output_dir/mscoco",
            name="mambayolo"
        )

    @patch('mbyolo_train.YOLO')
    def test_test_task(self, mock_yolo):
        # Arrange
        args = {
            "data": "/path/to/dataset.yaml",
            "epochs": 300,
            "workers": 128,
            "batch": 512,
            "optimizer": "SGD",
            "device": "0,1,2,3,4,5,6,7",
            "amp": True,
            "project": "/output_dir/mscoco",
            "name": "mambayolo"
        }
        mock_yolo.return_value.test.return_value = None

        # Act
        parse_opt()
        
        # Assert
        mock_yolo.assert_called_once_with("/path/to/mamba-yolo/Mamba-YOLO-T.yaml")
        mock_yolo.return_value.test.assert_called_once_with(
            data="/path/to/dataset.yaml",
            epochs=300,
            workers=128,
            batch=512,
            optimizer="SGD",
            device="0,1,2,3,4,5,6,7",
            amp=True,
            project="/output_dir/mscoco",
            name="mambayolo"
        )

    @patch('mbyolo_train.YOLO')
    def test_invalid_task(self, mock_yolo):
        # Arrange
        args = {
            "data": "/path/to/dataset.yaml",
            "epochs": 300,
            "workers": 128,
            "batch": 512,
            "optimizer": "SGD",
            "device": "0,1,2,3,4,5,6,7",
            "amp": True,
            "project": "/output_dir/mscoco",
            "name": "mambayolo"
        }
        mock_yolo.return_value.train.return_value = None

        # Act
        with self.assertRaises(KeyError):
            parse_opt()
