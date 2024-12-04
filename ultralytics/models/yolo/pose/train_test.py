import unittest
from unittest.mock import patch, MagicMock
from ultralytics.models.yolo.pose.train import PoseTrainer

class TestPoseTrainer(unittest.TestCase):
    @patch('ultralytics.models.yolo.pose.train.DEFAULT_CFG')
    def test_init(self, mock_cfg):
        """Test the initialization of PoseTrainer."""
        args = {'model': 'yolov8n-pose.pt', 'data': 'coco8-pose.yaml', 'epochs': 3}
        trainer = PoseTrainer(overrides=args)
        self.assertEqual(trainer.args.model, 'yolov8n-pose.pt')
        self.assertEqual(trainer.args.data, 'coco8-pose.yaml')
        self.assertEqual(trainer.args.epochs, 3)
        mock_cfg.assert_called_once_with(task='pose')

    @patch('ultralytics.models.yolo.pose.train.PoseModel')
    def test_get_model(self, mock_model):
        """Test the get_model method."""
        trainer = PoseTrainer()
        model = trainer.get_model(cfg=None, weights=None, verbose=True)
        self.assertIsInstance(model, PoseModel)
        mock_model.assert_called_once_with(cfg=None, ch=3, nc=trainer.data['nc'], data_kpt_shape=trainer.data['kpt_shape'], verbose=True)

    @patch('ultralytics.models.yolo.pose.train.PoseValidator')
    def test_get_validator(self, mock_validator):
        """Test the get_validator method."""
        trainer = PoseTrainer()
        validator = trainer.get_validator()
        self.assertIsInstance(validator, PoseValidator)
        mock_validator.assert_called_once_with(trainer.test_loader, save_dir=trainer.save_dir, args=copy(trainer.args), _callbacks=trainer.callbacks)

    @patch('ultralytics.models.yolo.pose.train.plot_images')
    def test_plot_training_samples(self, mock_plot):
        """Test the plot_training_samples method."""
        trainer = PoseTrainer()
        batch = {
            'img': MagicMock(),
            'keypoints': MagicMock(),
            'cls': MagicMock(),
            'bboxes': MagicMock(),
            'im_file': MagicMock(),
            'batch_idx': MagicMock()
        }
        ni = 1
        trainer.plot_training_samples(batch, ni)
        mock_plot.assert_called_once_with(
            images=batch['img'],
            batch_idx=batch['batch_idx'],
            cls=batch['cls'].squeeze(-1),
            bboxes=batch['bboxes'],
            kpts=batch['keypoints'],
            paths=batch['im_file'],
            fname=trainer.save_dir / f"train_batch{ni}.jpg",
            on_plot=trainer.on_plot
        )

    @patch('ultralytics.models.yolo.pose.train.plot_results')
    def test_plot_metrics(self, mock_plot):
        """Test the plot_metrics method."""
        trainer = PoseTrainer()
        trainer.csv = 'results.csv'
        trainer.plot_metrics()
        mock_plot.assert_called_once_with(file='results.csv', pose=True, on_plot=trainer.on_plot)

if __name__ == '__main__':
    unittest.main()
