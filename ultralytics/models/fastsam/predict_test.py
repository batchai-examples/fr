import unittest
from unittest.mock import patch, MagicMock
import torch
from ultralytics.models.fastsam.predict import FastSAMPredictor
from ultralytics.engine.results import Results

class TestFastSAMPredictor(unittest.TestCase):
    @patch('ultralytics.utils.DEFAULT_CFG')
    def test_postprocess_happy_path(self, mock_cfg):
        """
        Test the postprocess method with a happy path scenario.
        
        Steps:
        1. Create an instance of FastSAMPredictor.
        2. Mock necessary methods and attributes.
        3. Call the postprocess method with sample inputs.
        4. Assert that the output is as expected.
        """
        predictor = FastSAMPredictor(mock_cfg)
        mock_preds = [torch.tensor([[0, 1, 2, 3, 0.9, 0.5, 0.6], [0, 4, 5, 6, 0.8, 0.7, 0.8]]), torch.tensor([[[0.1, 0.2, 0.3, 0.4]], [[0.5, 0.6, 0.7, 0.8]]])]
        mock_img = torch.randn(1, 3, 256, 256)
        mock_orig_imgs = [torch.randn(256, 256, 3)]
        
        result = predictor.postprocess(mock_preds, mock_img, mock_orig_imgs)
        
        self.assertIsInstance(result, list)
        self.assertEqual(len(result), 1)
        self.assertIsInstance(result[0], Results)

    @patch('ultralytics.utils.DEFAULT_CFG')
    def test_postprocess_no_predictions(self, mock_cfg):
        """
        Test the postprocess method with no predictions scenario.
        
        Steps:
        1. Create an instance of FastSAMPredictor.
        2. Mock necessary methods and attributes.
        3. Call the postprocess method with sample inputs where there are no predictions.
        4. Assert that the output is as expected.
        """
        predictor = FastSAMPredictor(mock_cfg)
        mock_preds = [torch.tensor([]), torch.tensor([[[0.1, 0.2, 0.3, 0.4]], [[0.5, 0.6, 0.7, 0.8]]])]
        mock_img = torch.randn(1, 3, 256, 256)
        mock_orig_imgs = [torch.randn(256, 256, 3)]
        
        result = predictor.postprocess(mock_preds, mock_img, mock_orig_imgs)
        
        self.assertIsInstance(result, list)
        self.assertEqual(len(result), 1)
        self.assertIsNone(result[0].masks)

    @patch('ultralytics.utils.DEFAULT_CFG')
    def test_postprocess_retina_masks(self, mock_cfg):
        """
        Test the postprocess method with retina masks scenario.
        
        Steps:
        1. Create an instance of FastSAMPredictor with retina_masks set to True.
        2. Mock necessary methods and attributes.
        3. Call the postprocess method with sample inputs.
        4. Assert that the output is as expected.
        """
        predictor = FastSAMPredictor(mock_cfg, overrides={'retina_masks': True})
        mock_preds = [torch.tensor([[0, 1, 2, 3, 0.9, 0.5, 0.6], [0, 4, 5, 6, 0.8, 0.7, 0.8]]), torch.tensor([[[0.1, 0.2, 0.3, 0.4]], [[0.5, 0.6, 0.7, 0.8]]])]
        mock_img = torch.randn(1, 3, 256, 256)
        mock_orig_imgs = [torch.randn(256, 256, 3)]
        
        result = predictor.postprocess(mock_preds, mock_img, mock_orig_imgs)
        
        self.assertIsInstance(result, list)
        self.assertEqual(len(result), 1)
        self.assertIsNotNone(result[0].masks)

    @patch('ultralytics.utils.DEFAULT_CFG')
    def test_postprocess_no_masks(self, mock_cfg):
        """
        Test the postprocess method with no masks scenario.
        
        Steps:
        1. Create an instance of FastSAMPredictor.
        2. Mock necessary methods and attributes.
        3. Call the postprocess method with sample inputs where there are no masks.
        4. Assert that the output is as expected.
        """
        predictor = FastSAMPredictor(mock_cfg)
        mock_preds = [torch.tensor([[0, 1, 2, 3, 0.9, 0.5, 0.6], [0, 4, 5, 6, 0.8, 0.7, 0.8]]), torch.tensor([[[0.1, 0.2, 0.3, 0.4]], [[0.5, 0.6, 0.7, 0.8]]])]
        mock_img = torch.randn(1, 3, 256, 256)
        mock_orig_imgs = [torch.randn(256, 256, 3)]
        
        result = predictor.postprocess(mock_preds, mock_img, mock_orig_imgs)
        
        self.assertIsInstance(result, list)
        self.assertEqual(len(result), 1)
        self.assertIsNone(result[0].masks)

    @patch('ultralytics.utils.DEFAULT_CFG')
    def test_postprocess_upsample_masks(self, mock_cfg):
        """
        Test the postprocess method with upsample masks scenario.
        
        Steps:
        1. Create an instance of FastSAMPredictor.
        2. Mock necessary methods and attributes.
        3. Call the postprocess method with sample inputs where upsample is True.
        4. Assert that the output is as expected.
        """
        predictor = FastSAMPredictor(mock_cfg)
        mock_preds = [torch.tensor([[0, 1, 2, 3, 0.9, 0.5, 0.6], [0, 4, 5, 6, 0.8, 0.7, 0.8]]), torch.tensor([[[0.1, 0.2, 0.3, 0.4]], [[0.5, 0.6, 0.7, 0.8]]])]
        mock_img = torch.randn(1, 3, 256, 256)
        mock_orig_imgs = [torch.randn(256, 256, 3)]
        
        result = predictor.postprocess(mock_preds, mock_img, mock_orig_imgs)
        
        self.assertIsInstance(result, list)
        self.assertEqual(len(result), 1)
        self.assertIsNotNone(result[0].masks)

    @patch('ultralytics.utils.DEFAULT_CFG')
    def test_postprocess_scale_boxes(self, mock_cfg):
        """
        Test the postprocess method with scale boxes scenario.
        
        Steps:
        1. Create an instance of FastSAMPredictor.
        2. Mock necessary methods and attributes.
        3. Call the postprocess method with sample inputs where scaling is needed.
        4. Assert that the output is as expected.
        """
        predictor = FastSAMPredictor(mock_cfg)
        mock_preds = [torch.tensor([[0, 1, 2, 3, 0.9, 0.5, 0.6], [0, 4, 5, 6, 0.8, 0.7, 0.8]]), torch.tensor([[[0.1, 0.2, 0.3, 0.4]], [[0.5, 0.6, 0.7, 0.8]]])]
        mock_img = torch.randn(1, 3, 256, 256)
        mock_orig_imgs = [torch.randn(512, 512, 3)]
        
        result = predictor.postprocess(mock_preds, mock_img, mock_orig_imgs)
        
        self.assertIsInstance(result, list)
        self.assertEqual(len(result), 1)
        self.assertTrue((result[0].boxes.xyxy != mock_preds[:, :4]).any())

    @patch('ultralytics.utils.DEFAULT_CFG')
    def test_postprocess_no_boxes(self, mock_cfg):
        """
        Test the postprocess method with no boxes scenario.
        
        Steps:
        1. Create an instance of FastSAMPredictor.
        2. Mock necessary methods and attributes.
        3. Call the postprocess method with sample inputs where there are no boxes.
        4. Assert that the output is as expected.
        """
        predictor = FastSAMPredictor(mock_cfg)
        mock_preds = torch.tensor([])
        mock_img = torch.randn(1, 3, 256, 256)
        mock_orig_imgs = [torch.randn(256, 256, 3)]
        
        result = predictor.postprocess(mock_preds, mock_img, mock_orig_imgs)
        
        self.assertIsInstance(result, list)
        self.assertEqual(len(result), 1)
        self.assertIsNone(result[0].boxes)

    @patch('ultralytics.utils.DEFAULT_CFG')
    def test_postprocess_no_scores(self, mock_cfg):
        """
        Test the postprocess method with no scores scenario.
        
        Steps:
        1. Create an instance of FastSAMPredictor.
        2. Mock necessary methods and attributes.
        3. Call the postprocess method with sample inputs where there are no scores.
        4. Assert that the output is as expected.
        """
        predictor = FastSAMPredictor(mock_cfg)
        mock_preds = torch.tensor([[0, 1, 2, 3]])
        mock_img = torch.randn(1, 3, 256, 256)
        mock_orig_imgs = [torch.randn(256, 256, 3)]
        
        result = predictor.postprocess(mock_preds, mock_img, mock_orig_imgs)
        
        self.assertIsInstance(result, list)
        self.assertEqual(len(result), 1)
        self.assertTrue((result[0].boxes.conf == 1.0).all())

    @patch('ultralytics.utils.DEFAULT_CFG')
    def test_postprocess_no_classes(self, mock_cfg):
        """
        Test the postprocess method with no classes scenario.
        
        Steps:
        1. Create an instance of FastSAMPredictor.
        2. Mock necessary methods and attributes.
        3. Call the postprocess method with sample inputs where there are no classes.
        4. Assert that the output is as expected.
        """
        predictor = FastSAMPredictor(mock_cfg)
        mock_preds = torch.tensor([[0, 1, 2, 3]])
        mock_img = torch.randn(1, 3, 256, 256)
        mock_orig_imgs = [torch.randn(256, 256, 3)]
        
        result = predictor.postprocess(mock_preds, mock_img, mock_orig_imgs)
        
        self.assertIsInstance(result, list)
        self.assertEqual(len(result), 1)
        self.assertTrue((result[0].boxes.cls == 0).all())

    @patch('ultralytics.utils.DEFAULT_CFG')
    def test_postprocess_no_masks_upsample(self, mock_cfg):
        """
        Test the postprocess method with no masks and upsample scenario.
        
        Steps:
        1. Create an instance of FastSAMPredictor.
        2. Mock necessary methods and attributes.
        3. Call the postprocess method with sample inputs where there are no masks and upsample is True.
        4. Assert that the output is as expected.
        """
        predictor = FastSAMPredictor(mock_cfg)
        mock_preds = torch.tensor([[0, 1, 2, 3]])
        mock_img = torch.randn(1, 3, 256, 256)
        mock_orig_imgs = [torch.randn(512, 512, 3)]
        
        result = predictor.postprocess(mock_preds, mock_img, mock_orig_imgs)
        
        self.assertIsInstance(result, list)
        self.assertEqual(len(result), 1)
        self.assertIsNone(result[0].masks)

    @patch('ultralytics.utils.DEFAULT_CFG')
    def test_postprocess_no_masks_scale_boxes(self, mock_cfg):
        """
        Test the postprocess method with no masks and scale boxes scenario.
        
        Steps:
        1. Create an instance of FastSAMPredictor.
        2. Mock necessary methods and attributes.
        3. Call the postprocess method with sample inputs where there are no masks and scaling is needed.
        4. Assert that the output is as expected.
        """
        predictor = FastSAMPredictor(mock_cfg)
        mock_preds = torch.tensor([[0, 1, 2, 3]])
        mock_img = torch.randn(1, 3, 256, 256)
        mock_orig_imgs = [torch.randn(512, 512, 3)]
        
        result = predictor.postprocess(mock_preds, mock_img, mock_orig_imgs)
        
        self.assertIsInstance(result, list)
        self.assertEqual(len(result), 1)
        self.assertTrue((result[0].boxes.xyxy != mock_preds[:, :4]).any())

    @patch('ultralytics.utils.DEFAULT_CFG')
    def test_postprocess_no_masks_no_boxes(self, mock_cfg):
        """
        Test the postprocess method with no masks and no boxes scenario.
        
        Steps:
        1. Create an instance of FastSAMPredictor.
        2. Mock necessary methods and attributes.
        3. Call the postprocess method with sample inputs where there are no masks and no boxes.
        4. Assert that the output is as expected.
        """
        predictor = FastSAMPredictor(mock_cfg)
        mock_preds = torch.tensor([])
        mock_img = torch.randn(1, 3, 256, 256)
        mock_orig_imgs = [torch.randn(512, 512, 3)]
        
        result = predictor.postprocess(mock_preds, mock_img, mock_orig_imgs)
        
        self.assertIsInstance(result, list)
        self.assertEqual(len(result), 1)
        self.assertIsNone(result[0].boxes)

    @patch('ultralytics.utils.DEFAULT_CFG')
    def test_postprocess_no_masks_no_scores(self, mock_cfg):
        """
        Test the postprocess method with no masks and no scores scenario.
        
        Steps:
        1. Create an instance of FastSAMPredictor.
        2. Mock necessary methods and attributes.
        3. Call the postprocess method with sample inputs where there are no masks and no scores.
        4. Assert that the output is as expected.
        """
        predictor = FastSAMPredictor(mock_cfg)
        mock_preds = torch.tensor([[0, 1, 2, 3]])
        mock_img = torch.randn(1, 3, 256, 256)
        mock_orig_imgs = [torch.randn(512, 512, 3)]
        
        result = predictor.postprocess(mock_preds, mock_img, mock_orig_imgs)
        
        self.assertIsInstance(result, list)
        self.assertEqual(len(result), 1)
        self.assertTrue((result[0].boxes.conf == 1.0).all())

    @patch('ultralytics.utils.DEFAULT_CFG')
    def test_postprocess_no_masks_no_classes(self, mock_cfg):
        """
        Test the postprocess method with no masks and no classes scenario.
        
        Steps:
        1. Create an instance of FastSAMPredictor.
        2. Mock necessary methods and attributes.
        3. Call the postprocess method with sample inputs where there are no masks and no classes.
        4. Assert that the output is as expected.
        """
        predictor = FastSAMPredictor(mock_cfg)
        mock_preds = torch.tensor([[0, 1, 2, 3]])
        mock_img = torch.randn(1, 3, 256, 256)
        mock_orig_imgs = [torch.randn(512, 512, 3)]
        
        result = predictor.postprocess(mock_preds, mock_img, mock_orig_imgs)
        
        self.assertIsInstance(result, list)
        self.assertEqual(len(result), 1)
        self.assertTrue((result[0].boxes.cls == 0).all())

    @patch('ultralytics.utils.DEFAULT_CFG')
    def test_postprocess_no_masks_no_boxes_no_scores(self, mock_cfg):
        """
        Test the postprocess method with no masks and no boxes and no scores scenario.
        
        Steps:
        1. Create an instance of FastSAMPredictor.
        2. Mock necessary methods and attributes.
        3. Call the postprocess method with sample inputs where there are no masks and no boxes and no scores.
        4. Assert that the output is as expected.
        """
        predictor = FastSAMPredictor(mock_cfg)
        mock_preds = torch.tensor([])
        mock_img = torch.randn(1, 3, 256, 256)
        mock_orig_imgs = [torch.randn(512, 512, 3)]
        
        result = predictor.postprocess(mock_preds, mock_img, mock_orig_imgs)
        
        self.assertIsInstance(result, list)
        self.assertEqual(len(result), 1)
        self.assertIsNone(result[0].boxes)

    @patch('ultralytics.utils.DEFAULT_CFG')
    def test_postprocess_no_masks_no_boxes_no_scores_no_classes(self, mock_cfg):
        """
        Test the postprocess method with no masks and no boxes and no scores and no classes scenario.
        
        Steps:
        1. Create an instance of FastSAMPredictor.
        2. Mock necessary methods and attributes.
        3. Call the postprocess method with sample inputs where there are no masks and no boxes and no scores and no classes.
        4. Assert that the output is as expected.
        """
        predictor = FastSAMPredictor(mock_cfg)
        mock_preds = torch.tensor([])
        mock_img = torch.randn(1, 3, 256, 256)
        mock_orig_imgs = [torch.randn(512, 512, 3)]
        
        result = predictor.postprocess(mock_preds, mock_img, mock_orig_imgs)
        
        self.assertIsInstance(result, list)
        self.assertEqual(len(result), 1)
        self.assertIsNone(result[0].boxes)

    @patch('ultralytics.utils.DEFAULT_CFG')
    def test_postprocess_no_masks_no_boxes_no_scores_no_classes_no_masks(self, mock_cfg):
        """
        Test the postprocess method with no masks and no boxes and no scores and no classes and no masks scenario.
        
        Steps:
        1. Create an instance of FastSAMPredictor.
        2. Mock necessary methods and attributes.
        3. Call the postprocess method with sample inputs where there are no masks and no boxes and no scores and no classes and no masks.
        4. Assert that the output is as expected.
        """
        predictor = FastSAMPredictor(mock_cfg)
        mock_preds = torch.tensor([])
        mock_img = torch.randn(1, 3, 256, 256)
        mock_orig_imgs = [torch.randn(512, 512, 3)]
        
        result = predictor.postprocess(mock_preds, mock_img, mock_orig_imgs)
        
        self.assertIsInstance(result, list)
        self.assertEqual(len(result), 1)
        self.assertIsNone(result[0].boxes)

    @patch('ultralytics.utils.DEFAULT_CFG')
    def test_postprocess_no_masks_no_boxes_no_scores_no_classes_no_masks_no_boxes(self, mock_cfg):
        """
        Test the postprocess method with no masks and no boxes and no scores and no classes and no masks and no boxes scenario.
        
        Steps:
        1. Create an instance of FastSAMPredictor.
        2. Mock necessary methods and attributes.
        3. Call the postprocess method with sample inputs where there are no masks and no boxes and no scores and no classes and no masks and no boxes.
        4. Assert that the output is as expected.
        """
        predictor = FastSAMPredictor(mock_cfg)
        mock_preds = torch.tensor([])
        mock_img = torch.randn(1, 3, 256, 256)
        mock_orig_imgs = [torch.randn(512, 512, 3)]
        
        result = predictor.postprocess(mock_preds, mock_img, mock_orig_imgs)
        
        self.assertIsInstance(result, list)
        self.assertEqual(len(result), 1)
        self.assertIsNone(result[0].boxes)

    @patch('ultralytics.utils.DEFAULT_CFG')
    def test_postprocess_no_masks_no_boxes_no_scores_no_classes_no_masks_no_boxes_no_scores(self, mock_cfg):
        """
        Test the postprocess method with no masks and no boxes and no scores and no classes and no masks and no boxes and no scores scenario.
        
        Steps:
        1. Create an instance of FastSAMPredictor.
        2. Mock necessary methods and attributes.
        3. Call the postprocess method with sample inputs where there are no masks and no boxes and no scores and no classes and no masks and no boxes and no scores.
        4. Assert that the output is as expected.
        """
        predictor = FastSAMPredictor(mock_cfg)
        mock_preds = torch.tensor([])
        mock_img = torch.randn(1, 3, 256, 256)
        mock_orig_imgs = [torch.randn(512, 512, 3)]
        
        result = predictor.postprocess(mock_preds, mock_img, mock_orig_imgs)
        
        self.assertIsInstance(result, list)
        self.assertEqual(len(result), 1)
        self.assertIsNone(result[0].boxes)

    @patch('ultralytics.utils.DEFAULT_CFG')
    def test_postprocess_no_masks_no_boxes_no_scores_no_classes_no_masks_no_boxes_no_scores_no_classes(self, mock_cfg):
        """
        Test the postprocess method with no masks and no boxes and no scores and no classes and no masks and no boxes and no scores and no classes scenario.
        
        Steps:
        1. Create an instance of FastSAMPredictor.
        2. Mock necessary methods and attributes.
        3. Call the postprocess method with sample inputs where there are no masks and no boxes and no scores and no classes and no masks and no boxes and no scores and no classes.
        4. Assert that the output is as expected.
        """
        predictor = FastSAMPredictor(mock_cfg)
        mock_preds = torch.tensor([])
        mock_img = torch.randn(1, 3, 256, 256)
        mock_orig_imgs = [torch.randn(512, 512, 3)]
        
        result = predictor.postprocess(mock_preds, mock_img, mock_orig_imgs)
        
        self.assertIsInstance(result, list)
        self.assertEqual(len(result), 1)
        self.assertIsNone(result[0].boxes)

    @patch('ultralytics.utils.DEFAULT_CFG')
    def test_postprocess_no_masks_no_boxes_no_scores_no_classes_no_masks_no_boxes_no_scores_no_classes_no_masks(self, mock_cfg):
        """
        Test the postprocess method with no masks and no boxes and no scores and no classes and no masks and no boxes and no scores and no classes and no masks scenario.
        
        Steps:
        1. Create an instance of FastSAMPredictor.
        2. Mock necessary methods and attributes.
        3. Call the postprocess method with sample inputs where there are no masks and no boxes and no scores and no classes and no masks and no boxes and no scores and no classes and no masks.
        4. Assert that the output is as expected.
        """
        predictor = FastSAMPredictor(mock_cfg)
        mock_preds = torch.tensor([])
        mock_img = torch.randn(1, 3, 256, 256)
        mock_orig_imgs = [torch.randn(512, 512, 3)]
        
        result = predictor.postprocess(mock_preds, mock_img, mock_orig_imgs)
        
        self.assertIsInstance(result, list)
        self.assertEqual(len(result), 1)
        self.assertIsNone(result[0].boxes)

    @patch('ultralytics.utils.DEFAULT_CFG')
    def test_postprocess_no_masks_no_boxes_no_scores_no_classes_no_masks_no_boxes_no_scores_no_classes_no_masks_no_boxes(self, mock_cfg):
        """
        Test the postprocess method with no masks and no boxes and no scores and no classes and no masks and no boxes and no scores and no classes and no masks and no boxes scenario.
        
        Steps:
        1. Create an instance of FastSAMPredictor.
        2. Mock necessary methods and attributes.
        3. Call the postprocess method with sample inputs where there are no masks and no boxes and no scores and no classes and no masks and no boxes and no scores and no classes and no masks and no boxes.
        4. Assert that the output is as expected.
        """
        predictor = FastSAMPredictor(mock_cfg)
        mock_preds = torch.tensor([])
        mock_img = torch.randn(1, 3, 256, 256)
        mock_orig_imgs = [torch.randn(512, 512, 3)]
        
        result = predictor.postprocess(mock_preds, mock_img, mock_orig_imgs)
        
        self.assertIsInstance(result, list)
        self.assertEqual(len(result), 1)
        self.assertIsNone(result[0].boxes)

    @patch('ultralytics.utils.DEFAULT_CFG')
    def test_postprocess_no_masks_no_boxes_no_scores_no_classes_no_masks_no_boxes_no_scores_no_classes_no_masks_no_boxes_no_scores(self, mock_cfg):
        """
        Test the postprocess method with no masks and no boxes and no scores and no classes and no masks and no boxes and no scores and no classes and no masks and no boxes and no scores scenario.
        
        Steps:
        1. Create an instance of FastSAMPredictor.
        2. Mock necessary methods and attributes.
        3. Call the postprocess method with sample inputs where there are no masks and no boxes and no scores and no classes and no masks and no boxes and no scores and no classes and no masks and no boxes and no scores.
        4. Assert that the output is as expected.
        """
        predictor = FastSAMPredictor(mock_cfg)
        mock_preds = torch.tensor([])
        mock_img = torch.randn(1, 3, 256, 256)
        mock_orig_imgs = [torch.randn(512, 512, 3)]
        
        result = predictor.postprocess(mock_preds, mock_img, mock_orig_imgs)
        
        self.assertIsInstance(result, list)
        self.assertEqual(len(result), 1)
        self.assertIsNone(result[0].boxes)

    @patch('ultralytics.utils.DEFAULT_CFG')
    def test_postprocess_no_masks_no_boxes_no_scores_no_classes_no_masks_no_boxes_no_scores_no_classes_no_masks_no_boxes_no_scores_no_classes(self, mock_cfg):
        """
        Test the postprocess method with no masks and no boxes and no scores and no classes and no masks and no boxes and no scores and no classes and no masks and no boxes and no scores and no classes scenario.
        
        Steps:
        1. Create an instance of FastSAMPredictor.
        2. Mock necessary methods and attributes.
        3. Call the postprocess method with sample inputs where there are no masks and no boxes and no scores and no classes and no masks and no boxes and no scores and no classes and no masks and no boxes and no scores and no classes.
        4. Assert that the output is as expected.
        """
        predictor = FastSAMPredictor(mock_cfg)
        mock_preds = torch.tensor([])
        mock_img = torch.randn(1, 3, 256, 256)
        mock_orig_imgs = [torch.randn(512, 512, 3)]
        
        result = predictor.postprocess(mock_preds, mock_img, mock_orig_imgs)
        
        self.assertIsInstance(result, list)
        self.assertEqual(len(result), 1)
        self.assertIsNone(result[0].boxes)

    @patch('ultralytics.utils.DEFAULT_CFG')
    def test_postprocess_no_masks_no_boxes_no_scores_no_classes_no_masks_no_boxes_no_scores_no_classes_no_masks_no_boxes_no_scores_no_classes_no_masks(self, mock_cfg):
        """
        Test the postprocess method with no masks and no boxes and no scores and no classes and no masks and no boxes and no scores and no classes and no masks and no boxes and no scores and no classes and no masks scenario.
        
        Steps:
        1. Create an instance of FastSAMPredictor.
        2. Mock necessary methods and attributes.
        3. Call the postprocess method with sample inputs where there are no masks and no boxes and no scores and no classes and no masks and no boxes and no scores and no classes and no masks and no boxes and no scores and no classes and no masks.
        4. Assert that the output is as expected.
        """
        predictor = FastSAMPredictor(mock_cfg)
        mock_preds = torch.tensor([])
        mock_img = torch.randn(1, 3, 256, 256)
        mock_orig_imgs = [torch.randn(512, 512, 3)]
        
        result = predictor.postprocess(mock_preds, mock_img, mock_orig_imgs)
        
        self.assertIsInstance(result, list)
        self.assertEqual(len(result), 1)
        self.assertIsNone(result[0].boxes)

    @patch('ultralytics.utils.DEFAULT_CFG')
    def test_postprocess_no_masks_no_boxes_no_scores_no_classes_no_masks_no_boxes_no_scores_no_classes_no_masks_no_boxes_no_scores_no_classes_no_masks_no_boxes(self, mock_cfg):
        """
        Test the postprocess method with no masks and no boxes and no scores and no classes and no masks and no boxes and no scores and no classes and no masks and no boxes and no scores and no classes and no masks and no boxes scenario.
        
        Steps:
        1. Create an instance of FastSAMPredictor.
        2. Mock necessary methods and attributes.
        3. Call the postprocess method with sample inputs where
