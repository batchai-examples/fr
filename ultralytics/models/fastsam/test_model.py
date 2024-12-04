import unittest
from pathlib import Path
from ultralytics.models.fastsam.model import FastSAM

class TestFastSAM(unittest.TestCase):
    def setUp(self):
        self.model = FastSAM("last.pt")

    def test_init_with_default_model(self):
        """
        Test the __init__ method with the default model.
        
        Steps:
            1. Create an instance of FastSAM with the default model "FastSAM-x.pt".
            2. Verify that the model attribute is set to "FastSAM-x.pt".
        """
        self.assertEqual(self.model.model, "FastSAM-x.pt")

    def test_init_with_custom_model(self):
        """
        Test the __init__ method with a custom model.
        
        Steps:
            1. Create an instance of FastSAM with a custom model "custom.pt".
            2. Verify that the model attribute is set to "custom.pt".
        """
        custom_model = FastSAM("custom.pt")
        self.assertEqual(custom_model.model, "custom.pt")

    def test_init_with_yaml_model(self):
        """
        Test the __init__ method with a YAML model.
        
        Steps:
            1. Attempt to create an instance of FastSAM with a YAML model "model.yaml".
            2. Verify that an AssertionError is raised.
        """
        with self.assertRaises(AssertionError):
            FastSAM("model.yaml")

    def test_task_map(self):
        """
        Test the task_map property.
        
        Steps:
            1. Access the task_map property of the model instance.
            2. Verify that the returned dictionary contains the correct keys and values.
        """
        task_map = self.model.task_map
        expected_keys = {"segment"}
        expected_values = {
            "segment": {
                "predictor": FastSAMPredictor,
                "validator": FastSAMValidator
            }
        }
        self.assertEqual(set(task_map.keys()), expected_keys)
        self.assertDictEqual(task_map["segment"], expected_values)

if __name__ == "__main__":
    unittest.main()
