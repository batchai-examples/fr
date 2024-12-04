import pytest

from ultralytics.models.yolo.pose import PosePredictor, PoseTrainer, PoseValidator


def test_pose_predictor_creation():
    """
    Test the creation of a PosePredictor instance.
    
    Steps:
        1. Attempt to create an instance of PosePredictor.
        2. Verify that the instance is not None.
    """
    predictor = PosePredictor()
    assert predictor is not None, "PosePredictor instance should be created successfully"


def test_pose_trainer_creation():
    """
    Test the creation of a PoseTrainer instance.
    
    Steps:
        1. Attempt to create an instance of PoseTrainer.
        2. Verify that the instance is not None.
    """
    trainer = PoseTrainer()
    assert trainer is not None, "PoseTrainer instance should be created successfully"


def test_pose_validator_creation():
    """
    Test the creation of a PoseValidator instance.
    
    Steps:
        1. Attempt to create an instance of PoseValidator.
        2. Verify that the instance is not None.
    """
    validator = PoseValidator()
    assert validator is not None, "PoseValidator instance should be created successfully"


def test_all_models_imported():
    """
    Test if all models are correctly imported.
    
    Steps:
        1. Check if 'PoseTrainer' is in __all__.
        2. Check if 'PoseValidator' is in __all__.
        3. Check if 'PosePredictor' is in __all__.
    """
    assert "PoseTrainer" in __all__, "'PoseTrainer' should be included in __all__"
    assert "PoseValidator" in __all__, "'PoseValidator' should be included in __all__"
    assert "PosePredictor" in __all__, "'PosePredictor' should be included in __all__"

!!!!test_end!!!!
