import unittest
from ultralytics.models.yolo.classify.train import ClassificationTrainer

class TestClassificationTrainer(unittest.TestCase):
    def setUp(self):
        self.args = {
            "model": "yolov8n-cls.pt",
            "data": "imagenet10",
            "epochs": 3,
            "imgsz": 224
        }
        self.trainer = ClassificationTrainer(overrides=self.args)

    def test_set_model_attributes(self):
        """
        Test the set_model_attributes method to ensure it correctly sets the model's class names from the loaded dataset.
        """
        # Mock data for testing
        mock_data = {"names": ["class1", "class2"]}
        self.trainer.data = mock_data

        # Call the method under test
        self.trainer.set_model_attributes()

        # Assert that the model's names are set correctly
        self.assertEqual(self.trainer.model.names, mock_data["names"])

    def test_get_model_with_local_weights(self):
        """
        Test the get_model method to ensure it returns a modified PyTorch model configured for training YOLO with local weights.
        """
        # Mock data for testing
        mock_weights = "path/to/local/weights.pt"

        # Call the method under test
        model = self.trainer.get_model(weights=mock_weights)

        # Assert that the model is an instance of ClassificationModel and has been modified
        self.assertIsInstance(model, ClassificationModel)
        self.assertTrue(hasattr(model, "transforms"))

    def test_get_model_with_default_weights(self):
        """
        Test the get_model method to ensure it returns a modified PyTorch model configured for training YOLO with default weights.
        """
        # Call the method under test
        model = self.trainer.get_model()

        # Assert that the model is an instance of ClassificationModel and has been modified
        self.assertIsInstance(model, ClassificationModel)
        self.assertTrue(hasattr(model, "transforms"))

    def test_get_dataloader_with_train_mode(self):
        """
        Test the get_dataloader method to ensure it returns a PyTorch DataLoader with transforms for training.
        """
        # Mock data for testing
        mock_dataset_path = "path/to/dataset"

        # Call the method under test
        dataloader = self.trainer.get_dataloader(mock_dataset_path, mode="train")

        # Assert that the dataloader is an instance of DataLoader and has the correct transforms
        self.assertIsInstance(dataloader, DataLoader)
        self.assertTrue(hasattr(dataloader.dataset, "torch_transforms"))

    def test_get_dataloader_with_eval_mode(self):
        """
        Test the get_dataloader method to ensure it returns a PyTorch DataLoader with transforms for evaluation.
        """
        # Mock data for testing
        mock_dataset_path = "path/to/dataset"

        # Call the method under test
        dataloader = self.trainer.get_dataloader(mock_dataset_path, mode="eval")

        # Assert that the dataloader is an instance of DataLoader and has the correct transforms
        self.assertIsInstance(dataloader, DataLoader)
        self.assertTrue(hasattr(dataloader.dataset, "torch_transforms"))

    def test_preprocess_batch(self):
        """
        Test the preprocess_batch method to ensure it correctly preprocesses a batch of images and classes.
        """
        # Mock data for testing
        mock_batch = {
            "img": torch.randn(16, 3, 224, 224),
            "cls": torch.randint(0, 10, (16,))
        }

        # Call the method under test
        preprocessed_batch = self.trainer.preprocess_batch(mock_batch)

        # Assert that the batch is correctly preprocessed
        self.assertTrue(preprocessed_batch["img"].device.type == "cpu")
        self.assertTrue(preprocessed_batch["cls"].device.type == "cpu")

    def test_progress_string(self):
        """
        Test the progress_string method to ensure it returns a formatted string showing training progress.
        """
        # Call the method under test
        progress_str = self.trainer.progress_string()

        # Assert that the progress string is correctly formatted
        self.assertTrue(isinstance(progress_str, str))
        self.assertIn("Epoch", progress_str)
        self.assertIn("GPU_mem", progress_str)

    def test_get_validator(self):
        """
        Test the get_validator method to ensure it returns an instance of ClassificationValidator for validation.
        """
        # Call the method under test
        validator = self.trainer.get_validator()

        # Assert that the validator is an instance of ClassificationValidator
        self.assertIsInstance(validator, yolo.classify.ClassificationValidator)

    def test_label_loss_items(self):
        """
        Test the label_loss_items method to ensure it returns a loss dict with labelled training loss items tensor.
        """
        # Mock data for testing
        mock_loss_items = [0.5]

        # Call the method under test
        loss_dict = self.trainer.label_loss_items(loss_items=mock_loss_items)

        # Assert that the loss dictionary is correctly formatted
        self.assertTrue(isinstance(loss_dict, dict))
        self.assertIn("train/loss", loss_dict)
        self.assertEqual(loss_dict["train/loss"], 0.5)

    def test_plot_metrics(self):
        """
        Test the plot_metrics method to ensure it plots metrics from a CSV file.
        """
        # Mock data for testing
        mock_csv_path = "path/to/metrics.csv"

        # Call the method under test
        self.trainer.plot_metrics()

        # Assert that the results.png file is created
        self.assertTrue(os.path.exists("results.png"))

    def test_final_eval(self):
        """
        Test the final_eval method to ensure it evaluates trained model and saves validation results.
        """
        # Mock data for testing
        mock_model_path = "path/to/model.pt"

        # Call the method under test
        self.trainer.final_eval()

        # Assert that the best model is stripped of optimizers and validated
        self.assertTrue(os.path.exists(mock_model_path))
        self.assertTrue(strip_optimizer.called)
        self.assertTrue(self.validator.validate.called)

    def test_plot_training_samples(self):
        """
        Test the plot_training_samples method to ensure it plots training samples with their annotations.
        """
        # Mock data for testing
        mock_batch = {
            "img": torch.randn(16, 3, 224, 224),
            "cls": torch.randint(0, 10, (16,))
        }
        mock_ni = 0

        # Call the method under test
        self.trainer.plot_training_samples(mock_batch, mock_ni)

        # Assert that the training samples are plotted correctly
        self.assertTrue(os.path.exists("training_sample_0.png"))

if __name__ == "__main__":
    unittest.main()
