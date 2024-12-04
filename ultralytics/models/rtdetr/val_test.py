import unittest
from unittest.mock import patch, MagicMock
from ultralytics.models.rtdetr.val import RTDETRValidator, RTDETRDataset

class TestRTDETRValidator(unittest.TestCase):
    @patch('ultralytics.models.rtdetr.val.DetectionValidator')
    def test_build_dataset(self, mock_validator):
        validator = RTDETRValidator(args={'model': 'rtdetr-l.pt', 'data': 'coco8.yaml'})
        img_path = '/path/to/images'
        batch_size = 4
        dataset = validator.build_dataset(img_path, mode="val", batch=batch_size)
        self.assertIsInstance(dataset, RTDETRDataset)
        mock_validator.assert_called_once_with(args={'model': 'rtdetr-l.pt', 'data': 'coco8.yaml'}, imgsz=640, batch_size=batch_size, augment=False, hyp={'mosaic': 0.0, 'mixup': 0.0}, rect=False, cache=None, prefix='val: ', data={'nc': 80})

    @patch('ultralytics.models.rtdetr.val.DetectionValidator.postprocess')
    def test_postprocess(self, mock_postprocess):
        validator = RTDETRValidator(args={'model': 'rtdetr-l.pt', 'data': 'coco8.yaml'})
        preds = [torch.randn(1, 300, 6), None]
        outputs = validator.postprocess(preds)
        self.assertEqual(outputs[0].shape, (0, 6))
        mock_postprocess.assert_called_once_with(preds)

    @patch('ultralytics.models.rtdetr.val.DetectionValidator._prepare_batch')
    def test_prepare_batch(self, mock_prepare_batch):
        validator = RTDETRValidator(args={'model': 'rtdetr-l.pt', 'data': 'coco8.yaml'})
        batch = {
            "batch_idx": torch.tensor([0]),
            "cls": torch.randn(1, 300),
            "bboxes": torch.randn(1, 300, 4),
            "ori_shape": torch.tensor([[640, 640]]),
            "img": torch.randn(1, 3, 640, 640),
            "ratio_pad": torch.tensor([[[1.0, 0.0, 0.0, 0.0]]])
        }
        pbatch = validator._prepare_batch(0, batch)
        self.assertEqual(pbatch['cls'].shape, (300,))
        self.assertEqual(pbatch['bbox'].shape, (300, 4))
        mock_prepare_batch.assert_called_once_with(0, batch)

    @patch('ultralytics.models.rtdetr.val.DetectionValidator._prepare_pred')
    def test_prepare_pred(self, mock_prepare_pred):
        validator = RTDETRValidator(args={'model': 'rtdetr-l.pt', 'data': 'coco8.yaml'})
        pred = torch.randn(1, 300, 6)
        pbatch = {
            "ori_shape": torch.tensor([[640, 640]]),
            "imgsz": (640, 640)
        }
        predn = validator._prepare_pred(pred, pbatch)
        self.assertEqual(predn.shape, (300, 6))
        mock_prepare_pred.assert_called_once_with(pred, pbatch)

    def test_load_model(self):
        validator = RTDETRValidator(args={'model': 'rtdetr-l.pt', 'data': 'coco8.yaml'})
        model = validator.load_model()
        self.assertIsNotNone(model)

    def test_evaluate(self):
        validator = RTDETRValidator(args={'model': 'rtdetr-l.pt', 'data': 'coco8.yaml'})
        with patch('ultralytics.models.rtdetr.val.DetectionValidator.run'):
            validator.evaluate()
            validator.run.assert_called_once()

if __name__ == '__main__':
    unittest.main()
