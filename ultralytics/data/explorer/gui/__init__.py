import unittest

class TestInit(unittest.TestCase):
    """
    Test cases for the __init__.py file in ultralytics/data/explorer/gui/
    """

    def test_init_file(self):
        """
        Test that the __init__.py file is not empty and contains the expected content.
        """
        # Step 1: Read the content of the __init__.py file
        with open("ultralytics/data/explorer/gui/__init__.py", "r") as f:
            content = f.read()

        # Step 2: Check that the file is not empty
        self.assertNotEqual(content, "", "The __init__.py file should not be empty")

        # Step 3: Check that the expected content is present in the file
        self.assertIn("# Ultralytics YOLO 🚀, AGPL-3.0 license", content, "Expected content is missing from the __init__.py file")
