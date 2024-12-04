import unittest
from unittest.mock import patch, MagicMock
from ultralytics.models.yolo.world.train import WorldTrainer, on_pretrain_routine_end

class TestWorldTrainer(unittest.TestCase):
    @patch('ultralytics.utils.torch_utils.de_parallel')
    def test_on_pretrain_routine_end(self, mock_de_parallel):
        # Arrange
        trainer = MagicMock()
        trainer.ema.ema = MagicMock()
        trainer.test_loader.dataset.data = {'names': {'0': 'class1', '1': 'class2'}}
        trainer.text_model = MagicMock()

        # Act
        on_pretrain_routine_end(trainer)

        # Assert
        mock_de_parallel.assert_called_once_with(trainer.ema.ema)
        mock_de_parallel.return_value.set_classes.assert_called_once_with(['class1', 'class2'], cache_clip_model=False)
        trainer.text_model.load_state_dict.assert_called_once_with('ViT-B/32')
        for p in trainer.text_model.parameters():
            self.assertFalse(p.requires_grad_)

    def test_get_model(self):
        # Arrange
        args = {'model': 'yolov8s-world.pt', 'data': 'coco8.yaml', 'epochs': 3}
        trainer = WorldTrainer(overrides=args)
        cfg = {'yaml_file': 'coco8.yaml'}

        # Act
        model = trainer.get_model(cfg)

        # Assert
        self.assertIsInstance(model, WorldModel)
        self.assertEqual(model.nc, min(trainer.data['nc'], 80))
        self.assertTrue('on_pretrain_routine_end' in trainer.callbacks)

    def test_build_dataset(self):
        # Arrange
        args = {'model': 'yolov8s-world.pt', 'data': 'coco8.yaml', 'epochs': 3}
        trainer = WorldTrainer(overrides=args)
        img_path = 'path/to/images'
        batch_size = 16

        # Act
        dataset = trainer.build_dataset(img_path, mode='train', batch=batch_size)

        # Assert
        self.assertIsInstance(dataset, build_yolo_dataset)
        self.assertEqual(dataset.batch, batch_size)

    def test_preprocess_batch(self):
        # Arrange
        args = {'model': 'yolov8s-world.pt', 'data': 'coco8.yaml', 'epochs': 3}
        trainer = WorldTrainer(overrides=args)
        batch = {
            'img': torch.randn(2, 3, 640, 640),
            'texts': [['text1'], ['text2']]
        }
        trainer.clip.tokenize = MagicMock(return_value=torch.tensor([[1, 2], [3, 4]]))
        trainer.text_model.encode_text = MagicMock(return_value=torch.tensor([[[0.5, 0.5]], [[0.6, 0.6]]]))
        trainer.text_model.encode_text.return_value = trainer.text_model.encode_text.return_value.to(dtype=batch['img'].dtype)

        # Act
        processed_batch = trainer.preprocess_batch(batch)

        # Assert
        self.assertIn('txt_feats', processed_batch)
        self.assertEqual(processed_batch['txt_feats'].shape, (2, 1, 2))

if __name__ == '__main__':
    unittest.main()
