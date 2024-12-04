import unittest
from ultralytics.models.yolo.segment.train import SegmentationTrainer, DEFAULT_CFG

class TestSegmentationTrainer(unittest.TestCase):
    def setUp(self):
        self.args = dict(model='yolov8n-seg.pt', data='coco8-seg.yaml', epochs=3)
        self.trainer = SegmentationTrainer(overrides=self.args)

    def test_init(self):
        """
        Test the initialization of the SegmentationTrainer class.
        
        Steps:
        1. Create an instance of SegmentationTrainer with default arguments.
        2. Verify that the task is set to 'segment'.
        3. Verify that the model is an instance of SegmentationModel.
        """
        self.assertEqual(self.trainer.args["task"], "segment")
        self.assertIsInstance(self.trainer.model, SegmentationModel)

    def test_get_model(self):
        """
        Test the get_model method of the SegmentationTrainer class.
        
        Steps:
        1. Call the get_model method with default arguments.
        2. Verify that the returned model is an instance of SegmentationModel.
        3. Verify that the number of classes (nc) matches the data dictionary.
        """
        model = self.trainer.get_model()
        self.assertIsInstance(model, SegmentationModel)
        self.assertEqual(model.nc, self.trainer.data["nc"])

    def test_get_validator(self):
        """
        Test the get_validator method of the SegmentationTrainer class.
        
        Steps:
        1. Call the get_validator method.
        2. Verify that the returned validator is an instance of SegmentationValidator.
        3. Verify that the loss names are correctly set.
        """
        validator = self.trainer.get_validator()
        self.assertIsInstance(validator, yolo.segment.SegmentationValidator)
        self.assertEqual(self.trainer.loss_names, ("box_loss", "seg_loss", "cls_loss", "dfl_loss"))

    def test_plot_training_samples(self):
        """
        Test the plot_training_samples method of the SegmentationTrainer class.
        
        Steps:
        1. Create a mock batch dictionary with necessary keys and values.
        2. Call the plot_training_samples method with the mock batch.
        3. Verify that the plot_images function is called with the correct arguments.
        """
        from unittest.mock import patch
        batch = {
            "img": [None, None],
            "batch_idx": [0, 1],
            "cls": [[], []],
            "bboxes": [[], []],
            "masks": [[], []],
            "im_file": ["path/to/image1.jpg", "path/to/image2.jpg"]
        }
        ni = 0
        with patch('ultralytics.utils.plotting.plot_images') as mock_plot:
            self.trainer.plot_training_samples(batch, ni)
            mock_plot.assert_called_once_with(
                batch["img"],
                batch["batch_idx"],
                batch["cls"].squeeze(-1),
                batch["bboxes"],
                masks=batch["masks"],
                paths=batch["im_file"],
                fname=self.trainer.save_dir / f"train_batch{ni}.jpg",
                on_plot=self.trainer.on_plot
            )

    def test_plot_metrics(self):
        """
        Test the plot_metrics method of the SegmentationTrainer class.
        
        Steps:
        1. Call the plot_metrics method.
        2. Verify that the plot_results function is called with the correct arguments.
        """
        from unittest.mock import patch
        with patch('ultralytics.utils.plotting.plot_results') as mock_plot:
            self.trainer.plot_metrics()
            mock_plot.assert_called_once_with(file=self.trainer.csv, segment=True, on_plot=self.trainer.on_plot)

if __name__ == '__main__':
    unittest.main()
