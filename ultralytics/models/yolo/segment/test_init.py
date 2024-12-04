import unittest
from ultralytics.models.yolo.segment import SegmentationPredictor, SegmentationTrainer, SegmentationValidator

class TestSegmentationModels(unittest.TestCase):

    def test_segmentation_predictor(self):
        """
        Test the SegmentationPredictor class.
        
        Steps:
        1. Create an instance of SegmentationPredictor.
        2. Assert that the instance is not None.
        """
        predictor = SegmentationPredictor()
        self.assertIsNotNone(predictor)

    def test_segmentation_trainer(self):
        """
        Test the SegmentationTrainer class.
        
        Steps:
        1. Create an instance of SegmentationTrainer.
        2. Assert that the instance is not None.
        """
        trainer = SegmentationTrainer()
        self.assertIsNotNone(trainer)

    def test_segmentation_validator(self):
        """
        Test the SegmentationValidator class.
        
        Steps:
        1. Create an instance of SegmentationValidator.
        2. Assert that the instance is not None.
        """
        validator = SegmentationValidator()
        self.assertIsNotNone(validator)

    def test_all_models_imported(self):
        """
        Test if all models are correctly imported.
        
        Steps:
        1. Check if 'SegmentationPredictor', 'SegmentationTrainer', and 'SegmentationValidator' are in __all__.
        2. Assert that each model is not None.
        """
        from ultralytics.models.yolo.segment import __all__
        self.assertIn('SegmentationPredictor', __all__)
        self.assertIn('SegmentationTrainer', __all__)
        self.assertIn('SegmentationValidator', __all__)

        predictor = SegmentationPredictor()
        trainer = SegmentationTrainer()
        validator = SegmentationValidator()

        self.assertIsNotNone(predictor)
        self.assertIsNotNone(trainer)
        self.assertIsNotNone(validator)

if __name__ == '__main__':
    unittest.main()
