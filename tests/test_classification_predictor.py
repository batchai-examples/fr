import unittest
from ultralytics.models.yolo.classify.predict import ClassificationPredictor
from ultralytics.utils import DEFAULT_CFG, ops

class TestClassificationPredictor(unittest.TestCase):
    def setUp(self):
        self.args = dict(model='yolov8n-cls.pt', source=ops.load_image('ultralytics/assets/bus.jpg'))
        self.predictor = ClassificationPredictor(overrides=self.args)

    def test_preprocess_with_torch_tensor(self):
        """
        Test the preprocess method with a torch tensor input.
        
        Steps:
            1. Create a sample image as a torch tensor.
            2. Call the preprocess method with the tensor.
            3. Assert that the output is a torch tensor and has the correct shape.
        """
        img = ops.load_image('ultralytics/assets/bus.jpg')
        img_tensor = torch.from_numpy(img)
        processed_img = self.predictor.preprocess(img_tensor)
        self.assertIsInstance(processed_img, torch.Tensor)
        self.assertEqual(processed_img.shape, (1, 3, 640, 640))

    def test_preprocess_with_cv2_image(self):
        """
        Test the preprocess method with a cv2 image input.
        
        Steps:
            1. Create a sample image as a cv2 image.
            2. Call the preprocess method with the image.
            3. Assert that the output is a torch tensor and has the correct shape.
        """
        img = ops.load_image('ultralytics/assets/bus.jpg')
        processed_img = self.predictor.preprocess(img)
        self.assertIsInstance(processed_img, torch.Tensor)
        self.assertEqual(processed_img.shape, (1, 3, 640, 640))

    def test_postprocess_with_valid_predictions(self):
        """
        Test the postprocess method with valid predictions.
        
        Steps:
            1. Create a sample image and predictions.
            2. Call the postprocess method with the predictions.
            3. Assert that the output is a list of Results objects.
        """
        img = ops.load_image('ultralytics/assets/bus.jpg')
        preds = torch.rand(1, self.predictor.model.nc)
        results = self.predictor.postprocess(preds, img, [img])
        self.assertIsInstance(results, list)
        self.assertEqual(len(results), 1)
        self.assertIsInstance(results[0], Results)

    def test_postprocess_with_empty_predictions(self):
        """
        Test the postprocess method with empty predictions.
        
        Steps:
            1. Create a sample image and empty predictions.
            2. Call the postprocess method with the predictions.
            3. Assert that the output is an empty list.
        """
        img = ops.load_image('ultralytics/assets/bus.jpg')
        preds = torch.tensor([])
        results = self.predictor.postprocess(preds, img, [img])
        self.assertIsInstance(results, list)
        self.assertEqual(len(results), 0)

    def test_postprocess_with_single_prediction(self):
        """
        Test the postprocess method with a single prediction.
        
        Steps:
            1. Create a sample image and a single prediction.
            2. Call the postprocess method with the predictions.
            3. Assert that the output is a list containing a single Results object.
        """
        img = ops.load_image('ultralytics/assets/bus.jpg')
        preds = torch.rand(1, self.predictor.model.nc)
        results = self.predictor.postprocess(preds, img, [img])
        self.assertIsInstance(results, list)
        self.assertEqual(len(results), 1)
        self.assertIsInstance(results[0], Results)

if __name__ == '__main__':
    unittest.main()
