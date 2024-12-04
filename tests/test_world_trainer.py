import unittest
from ultralytics.models.yolo.world.train_world import WorldTrainerFromScratch, DEFAULT_CFG

class TestWorldTrainerFromScratch(unittest.TestCase):
    def setUp(self):
        self.trainer = WorldTrainerFromScratch(cfg=DEFAULT_CFG)

    def test_build_dataset_train_mode_single_yolo_data(self):
        """
        Test the build_dataset method in train mode with a single YOLO data source.
        
        Steps:
        1. Set up the trainer with default configuration.
        2. Call the build_dataset method with a single YOLO data source and 'train' mode.
        3. Assert that the returned dataset is an instance of YOLODataset.
        """
        img_path = ["Objects365.yaml"]
        dataset = self.trainer.build_dataset(img_path, mode="train")
        self.assertIsInstance(dataset, YOLOConcatDataset)

    def test_build_dataset_train_mode_multiple_yolo_data(self):
        """
        Test the build_dataset method in train mode with multiple YOLO data sources.
        
        Steps:
        1. Set up the trainer with default configuration.
        2. Call the build_dataset method with multiple YOLO data sources and 'train' mode.
        3. Assert that the returned dataset is an instance of YOLOConcatDataset.
        """
        img_path = ["Objects365.yaml", "lvis.yaml"]
        dataset = self.trainer.build_dataset(img_path, mode="train")
        self.assertIsInstance(dataset, YOLOConcatDataset)

    def test_build_dataset_val_mode(self):
        """
        Test the build_dataset method in val mode.
        
        Steps:
        1. Set up the trainer with default configuration.
        2. Call the build_dataset method with a single YOLO data source and 'val' mode.
        3. Assert that the returned dataset is an instance of YOLODataset.
        """
        img_path = ["Objects365.yaml"]
        dataset = self.trainer.build_dataset(img_path, mode="val")
        self.assertIsInstance(dataset, YOLOConcatDataset)

    def test_get_dataset_valid_data_format(self):
        """
        Test the get_dataset method with a valid data format.
        
        Steps:
        1. Set up the trainer with default configuration.
        2. Call the get_dataset method and assert that it returns a tuple of train and val paths.
        """
        self.trainer.args.data = {
            "train": {"yolo_data": ["Objects365.yaml"]},
            "val": {"yolo_data": ["lvis.yaml"]}
        }
        train_paths, val_path = self.trainer.get_dataset()
        self.assertIsInstance(train_paths, list)
        self.assertIsInstance(val_path, str)

    def test_get_dataset_invalid_data_format(self):
        """
        Test the get_dataset method with an invalid data format.
        
        Steps:
        1. Set up the trainer with default configuration.
        2. Call the get_dataset method with an invalid data format and assert that it raises a ValueError.
        """
        self.trainer.args.data = {
            "train": {"yolo_data": ["Objects365.yaml"]},
            "val": {"yolo_data": []}
        }
        with self.assertRaises(ValueError):
            self.trainer.get_dataset()

if __name__ == '__main__':
    unittest.main()
