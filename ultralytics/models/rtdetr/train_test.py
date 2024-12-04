import unittest
from ultralytics.models.rtdetr.train import RTDETRTrainer, RTDETRDetectionModel, RTDETRDataset, RTDETRValidator

class TestRTDETRTrainer(unittest.TestCase):
    def setUp(self):
        self.args = {
            "model": "rtdetr-l.yaml",
            "data": "coco8.yaml",
            "imgsz": 640,
            "epochs": 3
        }
        self.trainer = RTDETRTrainer(overrides=self.args)

    def test_get_model(self):
        """
        Test the get_model method of RTDETRTrainer.
        
        Steps:
        1. Initialize the trainer with default arguments.
        2. Call the get_model method without weights.
        3. Verify that the model is an instance of RTDETRDetectionModel.
        """
        model = self.trainer.get_model()
        self.assertIsInstance(model, RTDETRDetectionModel)

    def test_get_model_with_weights(self):
        """
        Test the get_model method of RTDETRTrainer with pre-trained weights.
        
        Steps:
        1. Initialize the trainer with default arguments.
        2. Call the get_model method with a mock weights path.
        3. Verify that the model is an instance of RTDETRDetectionModel and has loaded weights.
        """
        mock_weights_path = "path/to/mock/weights.pt"
        model = self.trainer.get_model(weights=mock_weights_path)
        self.assertIsInstance(model, RTDETRDetectionModel)
        self.assertTrue(hasattr(model, "state_dict"))

    def test_build_dataset(self):
        """
        Test the build_dataset method of RTDETRTrainer.
        
        Steps:
        1. Initialize the trainer with default arguments.
        2. Call the build_dataset method with a mock image path and mode 'val'.
        3. Verify that the dataset is an instance of RTDETRDataset.
        """
        img_path = "path/to/mock/images"
        dataset = self.trainer.build_dataset(img_path, mode="val")
        self.assertIsInstance(dataset, RTDETRDataset)

    def test_get_validator(self):
        """
        Test the get_validator method of RTDETRTrainer.
        
        Steps:
        1. Initialize the trainer with default arguments.
        2. Call the get_validator method.
        3. Verify that the validator is an instance of RTDETRValidator.
        """
        validator = self.trainer.get_validator()
        self.assertIsInstance(validator, RTDETRValidator)

    def test_preprocess_batch(self):
        """
        Test the preprocess_batch method of RTDETRTrainer.
        
        Steps:
        1. Initialize the trainer with default arguments.
        2. Create a mock batch dictionary.
        3. Call the preprocess_batch method with the mock batch.
        4. Verify that the preprocessed batch contains the expected keys and types.
        """
        mock_batch = {
            "img": torch.randn(2, 3, 640, 640),
            "bboxes": torch.tensor([[100, 100, 200, 200], [300, 300, 400, 400]]),
            "cls": torch.tensor([0, 1])
        }
        preprocessed_batch = self.trainer.preprocess_batch(mock_batch)
        self.assertIn("img", preprocessed_batch)
        self.assertIn("gt_bbox", preprocessed_batch)
        self.assertIn("gt_class", preprocessed_batch)
        self.assertIsInstance(preprocessed_batch["img"], torch.Tensor)
        self.assertIsInstance(preprocessed_batch["gt_bbox"], list)
        self.assertIsInstance(preprocessed_batch["gt_class"], list)

if __name__ == "__main__":
    unittest.main()
