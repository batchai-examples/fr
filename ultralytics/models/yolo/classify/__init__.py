import pytest
from ultralytics.models.yolo.classify.predict import ClassificationPredictor
from ultralytics.models.yolo.classify.train import ClassificationTrainer
from ultralytics.models.yolo.classify.val import ClassificationValidator

class TestClassificationPredictor:
    """
    Test cases for the ClassificationPredictor class.
    """

    def test_predict_with_valid_input(self):
        """
        Test case to verify that the predict method works correctly with valid input.
        """
        predictor = ClassificationPredictor()
        # Arrange
        input_data = {"image_path": "path/to/image.jpg"}
        
        # Act
        result = predictor.predict(input_data)
        
        # Assert
        assert isinstance(result, dict), "The result should be a dictionary"
        assert "predictions" in result, "The result should contain predictions"

    def test_predict_with_invalid_input(self):
        """
        Test case to verify that the predict method raises an error with invalid input.
        """
        predictor = ClassificationPredictor()
        # Arrange
        input_data = {"image_path": None}
        
        # Act & Assert
        with pytest.raises(ValueError, match="Invalid input: image_path cannot be None"):
            predictor.predict(input_data)

class TestClassificationTrainer:
    """
    Test cases for the ClassificationTrainer class.
    """

    def test_train_with_valid_input(self):
        """
        Test case to verify that the train method works correctly with valid input.
        """
        trainer = ClassificationTrainer()
        # Arrange
        input_data = {"train_dataset": "path/to/train/dataset", "epochs": 10}
        
        # Act
        result = trainer.train(input_data)
        
        # Assert
        assert isinstance(result, dict), "The result should be a dictionary"
        assert "loss" in result, "The result should contain loss"

    def test_train_with_invalid_input(self):
        """
        Test case to verify that the train method raises an error with invalid input.
        """
        trainer = ClassificationTrainer()
        # Arrange
        input_data = {"train_dataset": None, "epochs": 10}
        
        # Act & Assert
        with pytest.raises(ValueError, match="Invalid input: train_dataset cannot be None"):
            trainer.train(input_data)

class TestClassificationValidator:
    """
    Test cases for the ClassificationValidator class.
    """

    def test_validate_with_valid_input(self):
        """
        Test case to verify that the validate method works correctly with valid input.
        """
        validator = ClassificationValidator()
        # Arrange
        input_data = {"val_dataset": "path/to/val/dataset"}
        
        # Act
        result = validator.validate(input_data)
        
        # Assert
        assert isinstance(result, dict), "The result should be a dictionary"
        assert "accuracy" in result, "The result should contain accuracy"

    def test_validate_with_invalid_input(self):
        """
        Test case to verify that the validate method raises an error with invalid input.
        """
        validator = ClassificationValidator()
        # Arrange
        input_data = {"val_dataset": None}
        
        # Act & Assert
        with pytest.raises(ValueError, match="Invalid input: val_dataset cannot be None"):
            validator.validate(input_data)
