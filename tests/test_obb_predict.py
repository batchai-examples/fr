import unittest
from ultralytics.models.yolo.obb.predict import OBBPredictor
from ultralytics.utils import DEFAULT_CFG, ops

class TestOBBPredictor(unittest.TestCase):
    def setUp(self):
        self.args = dict(model='yolov8n-obb.pt', source=None)
        self.predictor = OBBPredictor(overrides=self.args)

    def test_postprocess_with_valid_input(self):
        """
        Tests the postprocess method with valid input.
        
        Steps:
        1. Create a mock prediction tensor.
        2. Call the postprocess method with the mock prediction, image, and original images.
        3. Assert that the output is a list of Results objects.
        """
        preds = torch.tensor([[[0.5, 0.5, 1.0, 1.0, 0.9, 0.0]]])
        img = torch.zeros((3, 640, 640))
        orig_imgs = [torch.zeros((3, 640, 640))]
        
        results = self.predictor.postprocess(preds, img, orig_imgs)
        
        self.assertIsInstance(results, list)
        self.assertEqual(len(results), 1)
        self.assertIsInstance(results[0], Results)

    def test_postprocess_with_empty_predictions(self):
        """
        Tests the postprocess method with empty predictions.
        
        Steps:
        1. Create an empty prediction tensor.
        2. Call the postprocess method with the empty prediction, image, and original images.
        3. Assert that the output is an empty list.
        """
        preds = torch.tensor([])
        img = torch.zeros((3, 640, 640))
        orig_imgs = [torch.zeros((3, 640, 640))]
        
        results = self.predictor.postprocess(preds, img, orig_imgs)
        
        self.assertIsInstance(results, list)
        self.assertEqual(len(results), 0)

    def test_postprocess_with_single_image(self):
        """
        Tests the postprocess method with a single image.
        
        Steps:
        1. Create a mock prediction tensor.
        2. Call the postprocess method with the mock prediction and a single image.
        3. Assert that the output is a list of Results objects.
        """
        preds = torch.tensor([[[0.5, 0.5, 1.0, 1.0, 0.9, 0.0]]])
        img = torch.zeros((3, 640, 640))
        
        results = self.predictor.postprocess(preds, img, None)
        
        self.assertIsInstance(results, list)
        self.assertEqual(len(results), 1)
        self.assertIsInstance(results[0], Results)

    def test_postprocess_with_multiple_images(self):
        """
        Tests the postprocess method with multiple images.
        
        Steps:
        1. Create a mock prediction tensor.
        2. Call the postprocess method with the mock prediction and multiple images.
        3. Assert that the output is a list of Results objects with the same length as the number of images.
        """
        preds = torch.tensor([[[0.5, 0.5, 1.0, 1.0, 0.9, 0.0]]])
        img = torch.zeros((3, 640, 640))
        orig_imgs = [torch.zeros((3, 640, 640)), torch.zeros((3, 640, 640))]
        
        results = self.predictor.postprocess(preds, img, orig_imgs)
        
        self.assertIsInstance(results, list)
        self.assertEqual(len(results), 2)
        for result in results:
            self.assertIsInstance(result, Results)

    def test_postprocess_with_invalid_image_shape(self):
        """
        Tests the postprocess method with an invalid image shape.
        
        Steps:
        1. Create a mock prediction tensor.
        2. Call the postprocess method with the mock prediction and an image with an invalid shape.
        3. Assert that the output is a list of Results objects with the same length as the number of images.
        """
        preds = torch.tensor([[[0.5, 0.5, 1.0, 1.0, 0.9, 0.0]]])
        img = torch.zeros((3, 640, 640))
        orig_imgs = [torch.zeros((3, 640, 640)), torch.zeros((3, 320, 320))]
        
        results = self.predictor.postprocess(preds, img, orig_imgs)
        
        self.assertIsInstance(results, list)
        self.assertEqual(len(results), 2)
        for result in results:
            self.assertIsInstance(result, Results)

    def test_postprocess_with_negative_confidence(self):
        """
        Tests the postprocess method with negative confidence scores.
        
        Steps:
        1. Create a mock prediction tensor with negative confidence scores.
        2. Call the postprocess method with the mock prediction and an image.
        3. Assert that the output is an empty list.
        """
        preds = torch.tensor([[[0.5, 0.5, 1.0, 1.0, -0.9, 0.0]]])
        img = torch.zeros((3, 640, 640))
        
        results = self.predictor.postprocess(preds, img, None)
        
        self.assertIsInstance(results, list)
        self.assertEqual(len(results), 0)

if __name__ == '__main__':
    unittest.main()
