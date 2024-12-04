import unittest
from ultralytics.models.yolo.detect.predict import DetectionPredictor
from ultralytics.utils import ops, ASSETS

class TestDetectionPredictor(unittest.TestCase):
    def setUp(self):
        self.args = dict(model='yolov8n.pt', source=ASSETS)
        self.predictor = DetectionPredictor(overrides=self.args)

    def test_postprocess_with_valid_input(self):
        """
        Tests the postprocess method with valid input.
        
        Steps:
            1. Create a mock prediction result.
            2. Call the postprocess method with the mock prediction and image data.
            3. Verify that the output is a list of Results objects.
        """
        # Mock prediction result
        preds = ops.non_max_suppression(
            [[0.5, 0.1, 0.2, 0.8, 0.9], [0.6, 0.3, 0.4, 0.7, 0.8]],
            self.args.conf,
            self.args.iou,
            agnostic=self.args.agnostic_nms,
            max_det=self.args.max_det,
            classes=self.args.classes,
        )

        # Mock image data
        img = ops.load_image(ASSETS[0])
        orig_imgs = [img]

        # Call the postprocess method
        results = self.predictor.postprocess(preds, img, orig_imgs)

        # Verify that the output is a list of Results objects
        self.assertIsInstance(results, list)
        for result in results:
            self.assertIsInstance(result, Results)

    def test_postprocess_with_empty_input(self):
        """
        Tests the postprocess method with empty input.
        
        Steps:
            1. Call the postprocess method with an empty prediction and image data.
            2. Verify that the output is an empty list.
        """
        # Empty prediction result
        preds = []

        # Mock image data
        img = ops.load_image(ASSETS[0])
        orig_imgs = [img]

        # Call the postprocess method
        results = self.predictor.postprocess(preds, img, orig_imgs)

        # Verify that the output is an empty list
        self.assertEqual(results, [])

    def test_postprocess_with_single_prediction(self):
        """
        Tests the postprocess method with a single prediction.
        
        Steps:
            1. Create a mock prediction result with a single detection.
            2. Call the postprocess method with the mock prediction and image data.
            3. Verify that the output is a list containing a single Results object.
        """
        # Mock prediction result
        preds = ops.non_max_suppression(
            [[0.5, 0.1, 0.2, 0.8, 0.9]],
            self.args.conf,
            self.args.iou,
            agnostic=self.args.agnostic_nms,
            max_det=self.args.max_det,
            classes=self.args.classes,
        )

        # Mock image data
        img = ops.load_image(ASSETS[0])
        orig_imgs = [img]

        # Call the postprocess method
        results = self.predictor.postprocess(preds, img, orig_imgs)

        # Verify that the output is a list containing a single Results object
        self.assertEqual(len(results), 1)
        self.assertIsInstance(results[0], Results)

    def test_postprocess_with_invalid_image_data(self):
        """
        Tests the postprocess method with invalid image data.
        
        Steps:
            1. Call the postprocess method with valid prediction and invalid image data (None).
            2. Verify that a TypeError is raised.
        """
        # Mock prediction result
        preds = ops.non_max_suppression(
            [[0.5, 0.1, 0.2, 0.8, 0.9]],
            self.args.conf,
            self.args.iou,
            agnostic=self.args.agnostic_nms,
            max_det=self.args.max_det,
            classes=self.args.classes,
        )

        # Invalid image data
        img = None
        orig_imgs = [img]

        # Call the postprocess method and verify that a TypeError is raised
        with self.assertRaises(TypeError):
            self.predictor.postprocess(preds, img, orig_imgs)

    def test_postprocess_with_multiple_images(self):
        """
        Tests the postprocess method with multiple images.
        
        Steps:
            1. Create mock prediction results for multiple images.
            2. Call the postprocess method with the mock predictions and image data.
            3. Verify that the output is a list of Results objects, one for each image.
        """
        # Mock prediction result for first image
        preds1 = ops.non_max_suppression(
            [[0.5, 0.1, 0.2, 0.8, 0.9]],
            self.args.conf,
            self.args.iou,
            agnostic=self.args.agnostic_nms,
            max_det=self.args.max_det,
            classes=self.args.classes,
        )

        # Mock prediction result for second image
        preds2 = ops.non_max_suppression(
            [[0.6, 0.3, 0.4, 0.7, 0.8]],
            self.args.conf,
            self.args.iou,
            agnostic=self.args.agnostic_nms,
            max_det=self.args.max_det,
            classes=self.args.classes,
        )

        # Mock image data
        img1 = ops.load_image(ASSETS[0])
        img2 = ops.load_image(ASSETS[1])
        orig_imgs = [img1, img2]

        # Call the postprocess method
        results = self.predictor.postprocess(preds1 + preds2, img1 + img2, orig_imgs)

        # Verify that the output is a list of Results objects, one for each image
        self.assertEqual(len(results), 2)
        for result in results:
            self.assertIsInstance(result, Results)


if __name__ == '__main__':
    unittest.main()
