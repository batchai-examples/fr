import unittest
from ultralytics.models.nas.predict import NASPredictor, Results
from ultralytics.utils.ops import xyxy2xywh, non_max_suppression, scale_boxes, convert_torch2numpy_batch

class TestNASPredictor(unittest.TestCase):
    def setUp(self):
        self.predictor = NASPredictor()
        self.predictor.args = Namespace(conf=0.5, iou=0.4, agnostic_nms=False, max_det=100, classes=None)

    def test_postprocess_happy_path(self):
        """
        Test the postprocess method with happy path data.
        
        Steps:
            1. Create mock input data for raw predictions, image, and original images.
            2. Call the postprocess method with the mock data.
            3. Verify that the output is a list of Results objects.
        """
        # Mock input data
        preds_in = [torch.tensor([[[0.1, 0.1, 0.5, 0.5, 0.9]]], dtype=torch.float32)]
        img = torch.zeros((640, 640, 3), dtype=torch.uint8)
        orig_imgs = [img]

        # Call the postprocess method
        results = self.predictor.postprocess(preds_in, img, orig_imgs)

        # Verify output
        self.assertIsInstance(results, list)
        self.assertEqual(len(results), 1)
        self.assertIsInstance(results[0], Results)

    def test_postprocess_negative_confidence(self):
        """
        Test the postprocess method with negative confidence threshold.
        
        Steps:
            1. Create mock input data for raw predictions, image, and original images.
            2. Set a negative confidence threshold in predictor args.
            3. Call the postprocess method with the mock data.
            4. Verify that no results are returned.
        """
        # Mock input data
        preds_in = [torch.tensor([[[0.1, 0.1, 0.5, 0.5, 0.9]]], dtype=torch.float32)]
        img = torch.zeros((640, 640, 3), dtype=torch.uint8)
        orig_imgs = [img]

        # Set negative confidence threshold
        self.predictor.args.conf = -1.0

        # Call the postprocess method
        results = self.predictor.postprocess(preds_in, img, orig_imgs)

        # Verify output
        self.assertIsInstance(results, list)
        self.assertEqual(len(results), 0)

    def test_postprocess_empty_predictions(self):
        """
        Test the postprocess method with empty predictions.
        
        Steps:
            1. Create mock input data for raw predictions, image, and original images.
            2. Set empty predictions in predictor args.
            3. Call the postprocess method with the mock data.
            4. Verify that no results are returned.
        """
        # Mock input data
        preds_in = [torch.tensor([], dtype=torch.float32)]
        img = torch.zeros((640, 640, 3), dtype=torch.uint8)
        orig_imgs = [img]

        # Call the postprocess method
        results = self.predictor.postprocess(preds_in, img, orig_imgs)

        # Verify output
        self.assertIsInstance(results, list)
        self.assertEqual(len(results), 0)

    def test_postprocess_single_image(self):
        """
        Test the postprocess method with a single image.
        
        Steps:
            1. Create mock input data for raw predictions, image, and original images.
            2. Call the postprocess method with the mock data.
            3. Verify that the output contains results for a single image.
        """
        # Mock input data
        preds_in = [torch.tensor([[[0.1, 0.1, 0.5, 0.5, 0.9]]], dtype=torch.float32)]
        img = torch.zeros((640, 640, 3), dtype=torch.uint8)
        orig_imgs = img

        # Call the postprocess method
        results = self.predictor.postprocess(preds_in, img, orig_imgs)

        # Verify output
        self.assertIsInstance(results, list)
        self.assertEqual(len(results), 1)
        self.assertIsInstance(results[0], Results)

if __name__ == '__main__':
    unittest.main()
