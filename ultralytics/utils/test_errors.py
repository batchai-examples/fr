import unittest
from ultralytics.utils.errors import HUBModelError, emojis

class TestHUBModelError(unittest.TestCase):
    """
    Test cases for the HUBModelError class.
    
    This test suite includes tests for both happy path and edge cases of the HUBModelError exception.
    """

    def setUp(self):
        """Setup method to initialize any necessary resources before each test."""
        self.default_message = "Model not found. Please check model URL and try again."
        self.custom_message = "Custom error message"

    def test_default_error_message(self):
        """
        Test the default error message of HUBModelError.
        
        This test checks if the default error message is correctly processed by the emojis function.
        """
        # Arrange
        expected_message = emojis(self.default_message)
        
        # Act
        exception = HUBModelError()
        
        # Assert
        self.assertEqual(str(exception), expected_message)

    def test_custom_error_message(self):
        """
        Test the custom error message of HUBModelError.
        
        This test checks if a custom error message is correctly processed by the emojis function.
        """
        # Arrange
        expected_message = emojis(self.custom_message)
        
        # Act
        exception = HUBModelError(self.custom_message)
        
        # Assert
        self.assertEqual(str(exception), expected_message)

    def test_empty_error_message(self):
        """
        Test an empty error message of HUBModelError.
        
        This test checks if an empty error message is handled gracefully and does not raise an exception.
        """
        # Arrange
        expected_message = emojis("")
        
        # Act
        exception = HUBModelError("")
        
        # Assert
        self.assertEqual(str(exception), expected_message)

    def test_none_error_message(self):
        """
        Test a None error message of HUBModelError.
        
        This test checks if a None error message is handled gracefully and does not raise an exception.
        """
        # Arrange
        expected_message = emojis(None)
        
        # Act
        exception = HUBModelError(None)
        
        # Assert
        self.assertEqual(str(exception), expected_message)

if __name__ == '__main__':
    unittest.main()
