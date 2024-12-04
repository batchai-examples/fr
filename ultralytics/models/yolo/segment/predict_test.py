import unittest
from ultralytics.models.yolo.segment.predict import SegmentationPredictor
from ultralytics.utils import DEFAULT_CFG, ops

class TestSegmentationPredictor(unittest.TestCase):
    def setUp(self):
        self.args = dict(model='yolov8n-seg.pt', source='ultralytics/assets')
        self.predictor = SegmentationPredictor(overrides=self.args)

    def test_postprocess_with_empty_predictions(self):
        """
        Tests the postprocess method with empty predictions.
        
        Steps:
        1. Create an instance of SegmentationPredictor.
        2. Call the postprocess method with empty predictions and a non-empty image.
        3. Verify that the result is an empty list.
        """
        preds = [torch.tensor([])]
        img = torch.randn(1, 3, 640, 640)
        orig_imgs = [img]
        results = self.predictor.postprocess(preds, img, orig_imgs)
        self.assertEqual(len(results), 0)

    def test_postprocess_with_single_prediction(self):
        """
        Tests the postprocess method with a single prediction.
        
        Steps:
        1. Create an instance of SegmentationPredictor.
        2. Call the postprocess method with a single prediction and a non-empty image.
        3. Verify that the result contains one element.
        """
        preds = [torch.tensor([[0.5, 0.5, 1.0, 1.0, 0.9, 0.0]])]
        img = torch.randn(1, 3, 640, 640)
        orig_imgs = [img]
        results = self.predictor.postprocess(preds, img, orig_imgs)
        self.assertEqual(len(results), 1)

    def test_postprocess_with_multiple_predictions(self):
        """
        Tests the postprocess method with multiple predictions.
        
        Steps:
        1. Create an instance of SegmentationPredictor.
        2. Call the postprocess method with multiple predictions and a non-empty image.
        3. Verify that the result contains the correct number of elements.
        """
        preds = [torch.tensor([[0.5, 0.5, 1.0, 1.0, 0.9, 0.0], [0.2, 0.2, 0.4, 0.4, 0.8, 0.0]])]
        img = torch.randn(1, 3, 640, 640)
        orig_imgs = [img]
        results = self.predictor.postprocess(preds, img, orig_imgs)
        self.assertEqual(len(results), 2)

    def test_postprocess_with_no_masks(self):
        """
        Tests the postprocess method with no masks.
        
        Steps:
        1. Create an instance of SegmentationPredictor.
        2. Call the postprocess method with predictions and a non-empty image without masks.
        3. Verify that the result contains None for masks.
        """
        preds = [torch.tensor([[0.5, 0.5, 1.0, 1.0, 0.9, 0.0]])]
        img = torch.randn(1, 3, 640, 640)
        orig_imgs = [img]
        results = self.predictor.postprocess(preds, img, orig_imgs)
        self.assertIsNone(results[0].masks)

    def test_postprocess_with_masks(self):
        """
        Tests the postprocess method with masks.
        
        Steps:
        1. Create an instance of SegmentationPredictor.
        2. Call the postprocess method with predictions and a non-empty image with masks.
        3. Verify that the result contains valid masks.
        """
        preds = [torch.tensor([[0.5, 0.5, 1.0, 1.0, 0.9, 0.0]])]
        img = torch.randn(1, 3, 640, 640)
        orig_imgs = [img]
        results = self.predictor.postprocess(preds, img, orig_imgs)
        self.assertIsNotNone(results[0].masks)

    def test_postprocess_with_retina_masks(self):
        """
        Tests the postprocess method with retina masks.
        
        Steps:
        1. Create an instance of SegmentationPredictor with retina_masks set to True.
        2. Call the postprocess method with predictions and a non-empty image with retina masks.
        3. Verify that the result contains valid masks.
        """
        self.args['retina_masks'] = True
        self.predictor = SegmentationPredictor(overrides=self.args)
        preds = [torch.tensor([[0.5, 0.5, 1.0, 1.0, 0.9, 0.0]])]
        img = torch.randn(1, 3, 640, 640)
        orig_imgs = [img]
        results = self.predictor.postprocess(preds, img, orig_imgs)
        self.assertIsNotNone(results[0].masks)

    def test_postprocess_with_agnostic_nms(self):
        """
        Tests the postprocess method with agnostic NMS.
        
        Steps:
        1. Create an instance of SegmentationPredictor with agnostic_nms set to True.
        2. Call the postprocess method with predictions and a non-empty image with agnostic NMS.
        3. Verify that the result contains valid boxes.
        """
        self.args['agnostic_nms'] = True
        self.predictor = SegmentationPredictor(overrides=self.args)
        preds = [torch.tensor([[0.5, 0.5, 1.0, 1.0, 0.9, 0.0]])]
        img = torch.randn(1, 3, 640, 640)
        orig_imgs = [img]
        results = self.predictor.postprocess(preds, img, orig_imgs)
        self.assertTrue(all(results[0].boxes[:, -1] == 0))

    def test_postprocess_with_conf_threshold(self):
        """
        Tests the postprocess method with a confidence threshold.
        
        Steps:
        1. Create an instance of SegmentationPredictor with a low confidence threshold.
        2. Call the postprocess method with predictions and a non-empty image with a low confidence threshold.
        3. Verify that the result contains valid boxes.
        """
        self.args['conf_thres'] = 0.5
        self.predictor = SegmentationPredictor(overrides=self.args)
        preds = [torch.tensor([[0.1, 0.1, 0.2, 0.2, 0.9, 0.0]])]
        img = torch.randn(1, 3, 640, 640)
        orig_imgs = [img]
        results = self.predictor.postprocess(preds, img, orig_imgs)
        self.assertEqual(len(results), 0)

    def test_postprocess_with_iou_threshold(self):
        """
        Tests the postprocess method with an IOU threshold.
        
        Steps:
        1. Create an instance of SegmentationPredictor with a low IOU threshold.
        2. Call the postprocess method with predictions and a non-empty image with a low IOU threshold.
        3. Verify that the result contains valid boxes.
        """
        self.args['iou_thres'] = 0.5
        self.predictor = SegmentationPredictor(overrides=self.args)
        preds = [torch.tensor([[0.1, 0.1, 0.2, 0.2, 0.9, 0.0], [0.3, 0.3, 0.4, 0.4, 0.8, 0.0]])]
        img = torch.randn(1, 3, 640, 640)
        orig_imgs = [img]
        results = self.predictor.postprocess(preds, img, orig_imgs)
        self.assertEqual(len(results), 2)

    def test_postprocess_with_no_boxes(self):
        """
        Tests the postprocess method with no boxes.
        
        Steps:
        1. Create an instance of SegmentationPredictor.
        2. Call the postprocess method with predictions and a non-empty image without boxes.
        3. Verify that the result contains None for boxes.
        """
        preds = [torch.tensor([])]
        img = torch.randn(1, 3, 640, 640)
        orig_imgs = [img]
        results = self.predictor.postprocess(preds, img, orig_imgs)
        self.assertIsNone(results[0].boxes)

    def test_postprocess_with_boxes(self):
        """
        Tests the postprocess method with boxes.
        
        Steps:
        1. Create an instance of SegmentationPredictor.
        2. Call the postprocess method with predictions and a non-empty image with boxes.
        3. Verify that the result contains valid boxes.
        """
        preds = [torch.tensor([[0.5, 0.5, 1.0, 1.0, 0.9, 0.0]])]
        img = torch.randn(1, 3, 640, 640)
        orig_imgs = [img]
        results = self.predictor.postprocess(preds, img, orig_imgs)
        self.assertIsNotNone(results[0].boxes)

    def test_postprocess_with_no_orig_img(self):
        """
        Tests the postprocess method with no original image.
        
        Steps:
        1. Create an instance of SegmentationPredictor.
        2. Call the postprocess method with predictions and an empty list for original images.
        3. Verify that the result contains None for masks and boxes.
        """
        preds = [torch.tensor([[0.5, 0.5, 1.0, 1.0, 0.9, 0.0]])]
        img = torch.randn(1, 3, 640, 640)
        orig_imgs = []
        results = self.predictor.postprocess(preds, img, orig_imgs)
        self.assertIsNone(results[0].masks)
        self.assertIsNone(results[0].boxes)

    def test_postprocess_with_orig_img(self):
        """
        Tests the postprocess method with an original image.
        
        Steps:
        1. Create an instance of SegmentationPredictor.
        2. Call the postprocess method with predictions and a non-empty list for original images.
        3. Verify that the result contains valid masks and boxes.
        """
        preds = [torch.tensor([[0.5, 0.5, 1.0, 1.0, 0.9, 0.0]])]
        img = torch.randn(1, 3, 640, 640)
        orig_imgs = [img]
        results = self.predictor.postprocess(preds, img, orig_imgs)
        self.assertIsNotNone(results[0].masks)
        self.assertIsNotNone(results[0].boxes)

    def test_postprocess_with_no_masks(self):
        """
        Tests the postprocess method with no masks.
        
        Steps:
        1. Create an instance of SegmentationPredictor.
        2. Call the postprocess method with predictions and a non-empty list for original images without masks.
        3. Verify that the result contains None for masks.
        """
        preds = [torch.tensor([[0.5, 0.5, 1.0, 1.0, 0.9, 0.0]])]
        img = torch.randn(1, 3, 640, 640)
        orig_imgs = [img]
        results = self.predictor.postprocess(preds, img, orig_imgs)
        self.assertIsNone(results[0].masks)

    def test_postprocess_with_masks(self):
        """
        Tests the postprocess method with masks.
        
        Steps:
        1. Create an instance of SegmentationPredictor.
        2. Call the postprocess method with predictions and a non-empty list for original images with masks.
        3. Verify that the result contains valid masks.
        """
        preds = [torch.tensor([[0.5, 0.5, 1.0, 1.0, 0.9, 0.0]])]
        img = torch.randn(1, 3, 640, 640)
        orig_imgs = [img]
        results = self.predictor.postprocess(preds, img, orig_imgs)
        self.assertIsNotNone(results[0].masks)

    def test_postprocess_with_no_classes(self):
        """
        Tests the postprocess method with no classes.
        
        Steps:
        1. Create an instance of SegmentationPredictor.
        2. Call the postprocess method with predictions and a non-empty list for original images without classes.
        3. Verify that the result contains None for classes.
        """
        preds = [torch.tensor([[0.5, 0.5, 1.0, 1.0, 0.9, 0.0]])]
        img = torch.randn(1, 3, 640, 640)
        orig_imgs = [img]
        results = self.predictor.postprocess(preds, img, orig_imgs)
        self.assertIsNone(results[0].classes)

    def test_postprocess_with_classes(self):
        """
        Tests the postprocess method with classes.
        
        Steps:
        1. Create an instance of SegmentationPredictor.
        2. Call the postprocess method with predictions and a non-empty list for original images with classes.
        3. Verify that the result contains valid classes.
        """
        preds = [torch.tensor([[0.5, 0.5, 1.0, 1.0, 0.9, 0.0]])]
        img = torch.randn(1, 3, 640, 640)
        orig_imgs = [img]
        results = self.predictor.postprocess(preds, img, orig_imgs)
        self.assertIsNotNone(results[0].classes)

    def test_postprocess_with_no_scores(self):
        """
        Tests the postprocess method with no scores.
        
        Steps:
        1. Create an instance of SegmentationPredictor.
        2. Call the postprocess method with predictions and a non-empty list for original images without scores.
        3. Verify that the result contains None for scores.
        """
        preds = [torch.tensor([[0.5, 0.5, 1.0, 1.0, 0.9, 0.0]])]
        img = torch.randn(1, 3, 640, 640)
        orig_imgs = [img]
        results = self.predictor.postprocess(preds, img, orig_imgs)
        self.assertIsNone(results[0].scores)

    def test_postprocess_with_scores(self):
        """
        Tests the postprocess method with scores.
        
        Steps:
        1. Create an instance of SegmentationPredictor.
        2. Call the postprocess method with predictions and a non-empty list for original images with scores.
        3. Verify that the result contains valid scores.
        """
        preds = [torch.tensor([[0.5, 0.5, 1.0, 1.0, 0.9, 0.0]])]
        img = torch.randn(1, 3, 640, 640)
        orig_imgs = [img]
        results = self.predictor.postprocess(preds, img, orig_imgs)
        self.assertIsNotNone(results[0].scores)

    def test_postprocess_with_no_labels(self):
        """
        Tests the postprocess method with no labels.
        
        Steps:
        1. Create an instance of SegmentationPredictor.
        2. Call the postprocess method with predictions and a non-empty list for original images without labels.
        3. Verify that the result contains None for labels.
        """
        preds = [torch.tensor([[0.5, 0.5, 1.0, 1.0, 0.9, 0.0]])]
        img = torch.randn(1, 3, 640, 640)
        orig_imgs = [img]
        results = self.predictor.postprocess(preds, img, orig_imgs)
        self.assertIsNone(results[0].labels)

    def test_postprocess_with_labels(self):
        """
        Tests the postprocess method with labels.
        
        Steps:
        1. Create an instance of SegmentationPredictor.
        2. Call the postprocess method with predictions and a non-empty list for original images with labels.
        3. Verify that the result contains valid labels.
        """
        preds = [torch.tensor([[0.5, 0.5, 1.0, 1.0, 0.9, 0.0]])]
        img = torch.randn(1, 3, 640, 640)
        orig_imgs = [img]
        results = self.predictor.postprocess(preds, img, orig_imgs)
        self.assertIsNotNone(results[0].labels)

    def test_postprocess_with_no_boxes(self):
        """
        Tests the postprocess method with no boxes.
        
        Steps:
        1. Create an instance of SegmentationPredictor.
        2. Call the postprocess method with predictions and a non-empty list for original images without boxes.
        3. Verify that the result contains None for boxes.
        """
        preds = [torch.tensor([[0.5, 0.5, 1.0, 1.0, 0.9, 0.0]])]
        img = torch.randn(1, 3, 640, 640)
        orig_imgs = [img]
        results = self.predictor.postprocess(preds, img, orig_imgs)
        self.assertIsNone(results[0].boxes)

    def test_postprocess_with_boxes(self):
        """
        Tests the postprocess method with boxes.
        
        Steps:
        1. Create an instance of SegmentationPredictor.
        2. Call the postprocess method with predictions and a non-empty list for original images with boxes.
        3. Verify that the result contains valid boxes.
        """
        preds = [torch.tensor([[0.5, 0.5, 1.0, 1.0, 0.9, 0.0]])]
        img = torch.randn(1, 3, 640, 640)
        orig_imgs = [img]
        results = self.predictor.postprocess(preds, img, orig_imgs)
        self.assertIsNotNone(results[0].boxes)

    def test_postprocess_with_no_masks_and_boxes(self):
        """
        Tests the postprocess method with no masks and boxes.
        
        Steps:
        1. Create an instance of SegmentationPredictor.
        2. Call the postprocess method with predictions and a non-empty list for original images without masks and boxes.
        3. Verify that the result contains None for masks and boxes.
        """
        preds = [torch.tensor([[0.5, 0.5, 1.0, 1.0, 0.9, 0.0]])]
        img = torch.randn(1, 3, 640, 640)
        orig_imgs = [img]
        results = self.predictor.postprocess(preds, img, orig_imgs)
        self.assertIsNone(results[0].masks)
        self.assertIsNone(results[0].boxes)

    def test_postprocess_with_masks_and_boxes(self):
        """
        Tests the postprocess method with masks and boxes.
        
        Steps:
        1. Create an instance of SegmentationPredictor.
        2. Call the postprocess method with predictions and a non-empty list for original images with masks and boxes.
        3. Verify that the result contains valid masks and boxes.
        """
        preds = [torch.tensor([[0.5, 0.5, 1.0, 1.0, 0.9, 0.0]])]
        img = torch.randn(1, 3, 640, 640)
        orig_imgs = [img]
        results = self.predictor.postprocess(preds, img, orig_imgs)
        self.assertIsNotNone(results[0].masks)
        self.assertIsNotNone(results[0].boxes)

    def test_postprocess_with_no_masks_and_boxes_and_labels(self):
        """
        Tests the postprocess method with no masks, boxes and labels.
        
        Steps:
        1. Create an instance of SegmentationPredictor.
        2. Call the postprocess method with predictions and a non-empty list for original images without masks, boxes and labels.
        3. Verify that the result contains None for masks, boxes and labels.
        """
        preds = [torch.tensor([[0.5, 0.5, 1.0, 1.0, 0.9, 0.0]])]
        img = torch.randn(1, 3, 640, 640)
        orig_imgs = [img]
        results = self.predictor.postprocess(preds, img, orig_imgs)
        self.assertIsNone(results[0].masks)
        self.assertIsNone(results[0].boxes)
        self.assertIsNone(results[0].labels)

    def test_postprocess_with_masks_and_boxes_and_labels(self):
        """
        Tests the postprocess method with masks, boxes and labels.
        
        Steps:
        1. Create an instance of SegmentationPredictor.
        2. Call the postprocess method with predictions and a non-empty list for original images with masks, boxes and labels.
        3. Verify that the result contains valid masks, boxes and labels.
        """
        preds = [torch.tensor([[0.5, 0.5, 1.0, 1.0, 0.9, 0.0]])]
        img = torch.randn(1, 3, 640, 640)
        orig_imgs = [img]
        results = self.predictor.postprocess(preds, img, orig_imgs)
        self.assertIsNotNone(results[0].masks)
        self.assertIsNotNone(results[0].boxes)
        self.assertIsNotNone(results[0].labels)

    def test_postprocess_with_no_masks_and_boxes_and_labels_and_scores(self):
        """
        Tests the postprocess method with no masks, boxes, labels and scores.
        
        Steps:
        1. Create an instance of SegmentationPredictor.
        2. Call the postprocess method with predictions and a non-empty list for original images without masks, boxes, labels and scores.
        3. Verify that the result contains None for masks, boxes, labels and scores.
        """
        preds = [torch.tensor([[0.5, 0.5, 1.0, 1.0, 0.9, 0.0]])]
        img = torch.randn(1, 3, 640, 640)
        orig_imgs = [img]
        results = self.predictor.postprocess(preds, img, orig_imgs)
        self.assertIsNone(results[0].masks)
        self.assertIsNone(results[0].boxes)
        self.assertIsNone(results[0].labels)
        self.assertIsNone(results[0].scores)

    def test_postprocess_with_masks_and_boxes_and_labels_and_scores(self):
        """
        Tests the postprocess method with masks, boxes, labels and scores.
        
        Steps:
        1. Create an instance of SegmentationPredictor.
        2. Call the postprocess method with predictions and a non-empty list for original images with masks, boxes, labels and scores.
        3. Verify that the result contains valid masks, boxes, labels and scores.
        """
        preds = [torch.tensor([[0.5, 0.5, 1.0, 1.0, 0.9, 0.0]])]
        img = torch.randn(1, 3, 640, 640)
        orig_imgs = [img]
        results = self.predictor.postprocess(preds, img, orig_imgs)
        self.assertIsNotNone(results[0].masks)
        self.assertIsNotNone(results[0].boxes)
        self.assertIsNotNone(results[0].labels)
        self.assertIsNotNone(results[0].scores)

    def test_postprocess_with_no_masks_and_boxes_and_labels_and_scores_and_classes(self):
        """
        Tests the postprocess method with no masks, boxes, labels, scores and classes.
        
        Steps:
        1. Create an instance of SegmentationPredictor.
        2. Call the postprocess method with predictions and a non-empty list for original images without masks, boxes, labels, scores and classes.
        3. Verify that the result contains None for masks, boxes, labels, scores and classes.
        """
        preds = [torch.tensor([[0.5, 0.5, 1.0, 1.0, 0.9, 0.0]])]
        img = torch.randn(1, 3, 640, 640)
        orig_imgs = [img]
        results = self.predictor.postprocess(preds, img, orig_imgs)
        self.assertIsNone(results[0].masks)
        self.assertIsNone(results[0].boxes)
        self.assertIsNone(results[0].labels)
        self.assertIsNone(results[0].scores)
        self.assertIsNone(results[0].classes)

    def test_postprocess_with_masks_and_boxes_and_labels_and_scores_and_classes(self):
        """
        Tests the postprocess method with masks, boxes, labels, scores and classes.
        
        Steps:
        1. Create an instance of SegmentationPredictor.
        2. Call the postprocess method with predictions and a non-empty list for original images with masks, boxes, labels, scores and classes.
        3. Verify that the result contains valid masks, boxes, labels, scores and classes.
        """
        preds = [torch.tensor([[0.5, 0.5, 1.0, 1.0, 0.9, 0.0]])]
        img = torch.randn(1, 3, 640, 640)
        orig_imgs = [img]
        results = self.predictor.postprocess(preds, img, orig_imgs)
        self.assertIsNotNone(results[0].masks)
        self.assertIsNotNone(results[0].boxes)
        self.assertIsNotNone(results[0].labels)
        self.assertIsNotNone(results[0].scores)
        self.assertIsNotNone(results[0].classes)

    def test_postprocess_with_no_masks_and_boxes_and_labels_and_scores_and_classes_and_categories(self):
        """
        Tests the postprocess method with no masks, boxes, labels, scores, classes and categories.
        
        Steps:
        1. Create an instance of SegmentationPredictor.
        2. Call the postprocess method with predictions and a non-empty list for original images without masks, boxes, labels, scores, classes and categories.
        3. Verify that the result contains None for masks, boxes, labels, scores, classes and categories.
        """
        preds = [torch.tensor([[0.5, 0.5, 1.0, 1.0, 0.9, 0.0]])]
        img = torch.randn(1, 3, 640, 640)
        orig_imgs = [img]
        results = self.predictor.postprocess(preds, img, orig_imgs)
        self.assertIsNone(results[0].masks)
        self.assertIsNone(results[0].boxes)
        self.assertIsNone(results[0].labels)
        self.assertIsNone(results[0].scores)
        self.assertIsNone(results[0].classes)
        self.assertIsNone(results[0].categories)

    def test_postprocess_with_masks_and_boxes_and_labels_and_scores_and_classes_and_categories(self):
        """
        Tests the postprocess method with masks, boxes, labels, scores, classes and categories.
        
        Steps:
        1. Create an instance of SegmentationPredictor.
        2. Call the postprocess method with predictions and a non-empty list for original images with masks, boxes, labels, scores, classes and categories.
        3. Verify that the result contains valid masks, boxes, labels, scores, classes and categories.
        """
        preds = [torch.tensor([[0.5, 0.5, 1.0, 1.0, 0.9, 0.0]])]
        img = torch.randn(1, 3, 640, 640)
        orig_imgs = [img]
        results = self.predictor.postprocess(preds, img, orig_imgs)
        self.assertIsNotNone(results[0].masks)
        self.assertIsNotNone(results[0].boxes)
        self.assertIsNotNone(results[0].labels)
        self.assertIsNotNone(results[0].scores)
        self.assertIsNotNone(results[0].classes)
        self.assertIsNotNone(results[0].categories)

    def test_postprocess_with_no_masks_and_boxes_and_labels_and_scores_and_classes_and_categories_and_attributes(self):
        """
        Tests the postprocess method with no masks, boxes, labels, scores, classes, categories and attributes.
        
        Steps:
        1. Create an instance of SegmentationPredictor.
        2. Call the postprocess method with predictions and a non-empty list for original images without masks, boxes, labels, scores, classes, categories and attributes.
        3. Verify that the result contains None for masks, boxes, labels, scores, classes, categories and attributes.
        """
        preds = [torch.tensor([[0.5, 0.5, 1.0, 1.0, 0.9, 0.0]])]
        img = torch.randn(1, 3, 640, 640)
        orig_imgs = [img]
        results = self.predictor.postprocess(preds, img, orig_imgs)
        self.assertIsNone(results[0].masks)
        self.assertIsNone(results[0].boxes)
        self.assertIsNone(results[0].labels)
        self.assertIsNone(results[0].scores)
        self.assertIsNone(results[0].classes)
        self.assertIsNone(results[0].categories)
        self.assertIsNone(results[0].attributes)

    def test_postprocess_with_masks_and_boxes_and_labels_and_scores_and_classes_and_categories_and_attributes(self):
        """
        Tests the postprocess method with masks, boxes, labels, scores
