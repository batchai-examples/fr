import unittest
from ultralytics.models.rtdetr.model import RTDETR

class TestRTDETRModel(unittest.TestCase):
    """
    Test cases for the RTDETR model class.
    """

    def test_init_with_default_model(self):
        """
        Test case to verify that the RTDETR model initializes with the default model file 'rtdetr-l.pt'.
        """
        # Arrange
        expected_model_path = "rtdetr-l.pt"

        # Act
        rt_detr = RTDETR()

        # Assert
        self.assertEqual(rt_detr.model, expected_model_path)

    def test_init_with_custom_model(self):
        """
        Test case to verify that the RTDETR model initializes with a custom model file.
        """
        # Arrange
        custom_model_path = "custom_rtdetr.pt"
        expected_model_path = custom_model_path

        # Act
        rt_detr = RTDETR(model=custom_model_path)

        # Assert
        self.assertEqual(rt_detr.model, expected_model_path)

    def test_init_with_invalid_model_extension(self):
        """
        Test case to verify that the RTDETR model raises a NotImplementedError when initialized with an invalid model file extension.
        """
        # Arrange
        invalid_model_path = "invalid_rtdetr.txt"

        # Act & Assert
        with self.assertRaises(NotImplementedError):
            RTDETR(model=invalid_model_path)

    def test_task_map_detection(self):
        """
        Test case to verify that the task map for detection is correctly defined.
        """
        # Arrange
        rt_detr = RTDETR()
        expected_task_map = {
            "detect": {
                "predictor": RTDETRPredictor,
                "validator": RTDETRValidator,
                "trainer": RTDETRTrainer,
                "model": RTDETRDetectionModel,
            }
        }

        # Act
        task_map = rt_detr.task_map

        # Assert
        self.assertEqual(task_map, expected_task_map)

    def test_task_map_missing_task(self):
        """
        Test case to verify that the task map does not contain a missing task.
        """
        # Arrange
        rt_detr = RTDETR()
        expected_task_map_keys = {"detect"}

        # Act
        task_map_keys = set(rt_detr.task_map.keys())

        # Assert
        self.assertEqual(task_map_keys, expected_task_map_keys)

    def test_task_map_predictor(self):
        """
        Test case to verify that the predictor class is correctly mapped in the task map.
        """
        # Arrange
        rt_detr = RTDETR()
        expected_predictor_class = RTDETRPredictor

        # Act
        predictor_class = rt_detr.task_map["detect"]["predictor"]

        # Assert
        self.assertEqual(predictor_class, expected_predictor_class)

    def test_task_map_validator(self):
        """
        Test case to verify that the validator class is correctly mapped in the task map.
        """
        # Arrange
        rt_detr = RTDETR()
        expected_validator_class = RTDETRValidator

        # Act
        validator_class = rt_detr.task_map["detect"]["validator"]

        # Assert
        self.assertEqual(validator_class, expected_validator_class)

    def test_task_map_trainer(self):
        """
        Test case to verify that the trainer class is correctly mapped in the task map.
        """
        # Arrange
        rt_detr = RTDETR()
        expected_trainer_class = RTDETRTrainer

        # Act
        trainer_class = rt_detr.task_map["detect"]["trainer"]

        # Assert
        self.assertEqual(trainer_class, expected_trainer_class)

    def test_task_map_model(self):
        """
        Test case to verify that the model class is correctly mapped in the task map.
        """
        # Arrange
        rt_detr = RTDETR()
        expected_model_class = RTDETRDetectionModel

        # Act
        model_class = rt_detr.task_map["detect"]["model"]

        # Assert
        self.assertEqual(model_class, expected_model_class)

    def test_task_map_invalid_task(self):
        """
        Test case to verify that the task map raises a KeyError when accessing an invalid task.
        """
        # Arrange
        rt_detr = RTDETR()

        # Act & Assert
        with self.assertRaises(KeyError):
            rt_detr.task_map["invalid_task"]

!!!!test_end!!!!
