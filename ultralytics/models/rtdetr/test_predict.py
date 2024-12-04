import unittest
from ultralytics.models.rtdetr.predict import RTDETRPredictor, LetterBox
from ultralytics.utils import ops

class TestRTDETRPredictor(unittest.TestCase):
    def setUp(self):
        self.imgsz = 640
        self.args = {
            "model": "rtdetr-l.pt",
            "source": "ultralytics/utils/assets/images/bus.jpg",
            "conf": 0.5,
            "classes": None
        }
        self.predictor = RTDETRPredictor(overrides=self.args)

    def test_postprocess_with_valid_predictions(self):
        """
        Test the postprocess method with valid predictions.
        """
        # Mock input data
        preds = torch.tensor([[[10, 20, 30, 40], [0.9, 0.8]], [[50, 60, 70, 80], [0.7, 0.6]]])
        img = torch.randn(1, 3, self.imgsz, self.imgsz)
        orig_imgs = ops.convert_torch2numpy_batch(img)

        # Call the method
        results = self.predictor.postprocess(preds, img, orig_imgs)

        # Assertions
        self.assertEqual(len(results), 1)
        result = results[0]
        self.assertEqual(result.boxes.xyxy.shape, (2, 4))
        self.assertEqual(result.boxes.conf.shape, (2, 1))
        self.assertEqual(result.boxes.cls.shape, (2, 1))

    def test_postprocess_with_empty_predictions(self):
        """
        Test the postprocess method with empty predictions.
        """
        # Mock input data
        preds = []
        img = torch.randn(1, 3, self.imgsz, self.imgsz)
        orig_imgs = ops.convert_torch2numpy_batch(img)

        # Call the method
        results = self.predictor.postprocess(preds, img, orig_imgs)

        # Assertions
        self.assertEqual(len(results), 0)

    def test_postprocess_with_low_confidence(self):
        """
        Test the postprocess method with low confidence predictions.
        """
        # Mock input data
        preds = torch.tensor([[[10, 20, 30, 40], [0.1, 0.2]], [[50, 60, 70, 80], [0.3, 0.4]]])
        img = torch.randn(1, 3, self.imgsz, self.imgsz)
        orig_imgs = ops.convert_torch2numpy_batch(img)

        # Call the method
        results = self.predictor.postprocess(preds, img, orig_imgs)

        # Assertions
        self.assertEqual(len(results), 0)

    def test_pre_transform_with_valid_images(self):
        """
        Test the pre_transform method with valid images.
        """
        # Mock input data
        im = [ops.read_image("ultralytics/utils/assets/images/bus.jpg")]

        # Call the method
        transformed_images = self.predictor.pre_transform(im)

        # Assertions
        self.assertEqual(len(transformed_images), 1)
        self.assertEqual(transformed_images[0].shape, (3, self.imgsz, self.imgsz))

    def test_pre_transform_with_empty_images(self):
        """
        Test the pre_transform method with empty images.
        """
        # Mock input data
        im = []

        # Call the method
        transformed_images = self.predictor.pre_transform(im)

        # Assertions
        self.assertEqual(len(transformed_images), 0)

    def test_pre_transform_with_invalid_image_shape(self):
        """
        Test the pre_transform method with invalid image shape.
        """
        # Mock input data
        im = [ops.read_image("ultralytics/utils/assets/images/bus.jpg")[:2, :, :]]

        # Call the method
        transformed_images = self.predictor.pre_transform(im)

        # Assertions
        self.assertEqual(len(transformed_images), 0)


if __name__ == "__main__":
    unittest.main()
