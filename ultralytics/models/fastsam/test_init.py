import unittest
from ultralytics.models.fastsam import FastSAM, FastSAMPredictor, FastSAMPrompt, FastSAMValidator

class TestFastSAM(unittest.TestCase):
    def test_fast_sam_import(self):
        """
        Test if FastSAM can be imported successfully.
        """
        # Import the FastSAM class and check if it is not None
        self.assertIsNotNone(FastSAM)

    def test_fast_sampredictor_import(self):
        """
        Test if FastSAMPredictor can be imported successfully.
        """
        # Import the FastSAMPredictor class and check if it is not None
        self.assertIsNotNone(FastSAMPredictor)

    def test_fast_samprompt_import(self):
        """
        Test if FastSAMPrompt can be imported successfully.
        """
        # Import the FastSAMPrompt class and check if it is not None
        self.assertIsNotNone(FastSAMPrompt)

    def test_fastsamvalidator_import(self):
        """
        Test if FastSAMValidator can be imported successfully.
        """
        # Import the FastSAMValidator class and check if it is not None
        self.assertIsNotNone(FastSAMValidator)


if __name__ == '__main__':
    unittest.main()
