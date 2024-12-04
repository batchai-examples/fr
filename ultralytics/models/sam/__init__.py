import unittest
from ultralytics.models.sam import SAM, Predictor

class TestSAM(unittest.TestCase):
    """
    Test cases for the SAM class.
    """

    def test_sam_creation(self):
        """
        Test case to check if the SAM class can be instantiated without errors.
        """
        # Arrange
        # Act
        sam = SAM()
        # Assert
        self.assertIsNotNone(sam)

    def test_predictor_creation(self):
        """
        Test case to check if the Predictor class can be instantiated without errors.
        """
        # Arrange
        # Act
        predictor = Predictor()
        # Assert
        self.assertIsNotNone(predictor)


class TestSAMPredictor(unittest.TestCase):
    """
    Test cases for the SAM and Predictor classes interaction.
    """

    def test_sam_predictor_interaction(self):
        """
        Test case to check if the SAM and Predictor classes can interact without errors.
        """
        # Arrange
        sam = SAM()
        predictor = Predictor(sam)
        # Act
        result = predictor.predict()
        # Assert
        self.assertIsNotNone(result)


class TestSAMNegative(unittest.TestCase):
    """
    Negative test cases for the SAM class.
    """

    def test_sam_creation_with_invalid_args(self):
        """
        Test case to check if the SAM class raises an error when instantiated with invalid arguments.
        """
        # Arrange
        # Act & Assert
        with self.assertRaises(ValueError):
            sam = SAM(invalid_arg="invalid")


class TestSAMPredictorNegative(unittest.TestCase):
    """
    Negative test cases for the Predictor class.
    """

    def test_predictor_creation_with_invalid_args(self):
        """
        Test case to check if the Predictor class raises an error when instantiated with invalid arguments.
        """
        # Arrange
        sam = SAM()
        # Act & Assert
        with self.assertRaises(ValueError):
            predictor = Predictor(sam, invalid_arg="invalid")

    def test_predictor_predict_with_invalid_args(self):
        """
        Test case to check if the Predictor class raises an error when predict method is called with invalid arguments.
        """
        # Arrange
        sam = SAM()
        predictor = Predictor(sam)
        # Act & Assert
        with self.assertRaises(ValueError):
            result = predictor.predict(invalid_arg="invalid")
