import unittest
from ultralytics.models.nas.model import NAS

class TestNASModel(unittest.TestCase):
    def setUp(self):
        self.model = NAS('yolo_nas_s.pt')

    def test_load_pretrained_model(self):
        """
        Test that the model loads a pre-trained model correctly.
        
        Steps:
            1. Create an instance of NAS with a valid pre-trained model path.
            2. Verify that the model is loaded without raising any exceptions.
        """
        self.assertIsNotNone(self.model.model)

    def test_load_invalid_model_path(self):
        """
        Test that the model raises an assertion error when given an invalid model path.
        
        Steps:
            1. Create an instance of NAS with an invalid model path (e.g., a non-existent file).
            2. Verify that an AssertionError is raised.
        """
        with self.assertRaises(AssertionError):
            NAS('non_existent_model.pt')

    def test_info_method(self):
        """
        Test the info method of the NAS model.
        
        Steps:
            1. Call the info method on the NAS instance.
            2. Verify that the method returns a string containing information about the model.
        """
        info_output = self.model.info()
        self.assertIsInstance(info_output, str)
        self.assertIn('Model', info_output)

    def test_task_map_property(self):
        """
        Test the task_map property of the NAS model.
        
        Steps:
            1. Access the task_map property on the NAS instance.
            2. Verify that the returned dictionary contains the correct keys and values.
        """
        task_map = self.model.task_map
        self.assertIsInstance(task_map, dict)
        self.assertIn('detect', task_map)
        self.assertIn('predictor', task_map['detect'])
        self.assertIn('validator', task_map['detect'])

if __name__ == '__main__':
    unittest.main()
