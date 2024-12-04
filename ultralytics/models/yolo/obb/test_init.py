import unittest
from ultralytics.models.yolo.obb import OBBPredictor, OBBTrainer, OBBValidator

class TestOBBInit(unittest.TestCase):
    """
    Test cases for the __init__.py file in ultralytics/models/yolo/obb directory.
    
    This test suite includes tests for the following:
    - The presence of expected classes in the __all__ variable
    """

    def test_all_classes_present(self):
        """
        Test that all expected classes are present in the __all__ variable.
        
        Steps:
        1. Check if 'OBBPredictor' is in __all__
        2. Check if 'OBBTrainer' is in __all__
        3. Check if 'OBBValidator' is in __all__
        """
        expected_classes = ['OBBPredictor', 'OBBTrainer', 'OBBValidator']
        for cls in expected_classes:
            self.assertIn(cls, __all__)

if __name__ == '__main__':
    unittest.main()
