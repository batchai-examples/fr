import unittest
from ultralytics.models.yolo.pose.predict import PosePredictor
from ultralytics.utils import DEFAULT_CFG, ops

class TestPosePredictor(unittest.TestCase):
    def setUp(self):
        self.args = dict(model='yolov8n-pose.pt', source='path/to/test/image.jpg')
        self.predictor = PosePredictor(overrides=self.args)

    def test_postprocess_with_valid_input(self):
        """
        Tests the postprocess method with valid input.
        
        Steps:
        1. Create a mock prediction result.
        2. Call the postprocess method with the mock prediction result and an image.
        3. Verify that the output is a list of Results objects.
        """
        # Mock prediction result
        preds = ops.non_max_suppression(
            [[0.9, 10, 20, 30, 40, 50, 60, 70, 80, 90]],
            conf=0.5,
            iou=0.4,
            agnostic=False,
            max_det=1,
            classes=None,
            nc=1
        )
        
        # Mock image and orig_imgs
        img = ops.read_image('path/to/test/image.jpg')
        orig_imgs = [img]
        
        results = self.predictor.postprocess(preds, img, orig_imgs)
        self.assertIsInstance(results, list)
        self.assertEqual(len(results), 1)
        self.assertIsInstance(results[0], Results)

    def test_postprocess_with_empty_prediction(self):
        """
        Tests the postprocess method with an empty prediction result.
        
        Steps:
        1. Create an empty mock prediction result.
        2. Call the postprocess method with the empty mock prediction result and an image.
        3. Verify that the output is a list of Results objects with no elements.
        """
        preds = []
        img = ops.read_image('path/to/test/image.jpg')
        orig_imgs = [img]
        
        results = self.predictor.postprocess(preds, img, orig_imgs)
        self.assertIsInstance(results, list)
        self.assertEqual(len(results), 0)

    def test_postprocess_with_invalid_input(self):
        """
        Tests the postprocess method with invalid input.
        
        Steps:
        1. Create a mock prediction result with invalid format.
        2. Call the postprocess method with the mock prediction result and an image.
        3. Verify that the output is a list of Results objects with no elements.
        """
        preds = [[0.9, 10, 20, 30, 40, 50]]
        img = ops.read_image('path/to/test/image.jpg')
        orig_imgs = [img]
        
        results = self.predictor.postprocess(preds, img, orig_imgs)
        self.assertIsInstance(results, list)
        self.assertEqual(len(results), 0)

    def test_postprocess_with_device_warning(self):
        """
        Tests the postprocess method with a device warning.
        
        Steps:
        1. Set the device to 'mps' in the args.
        2. Create a mock prediction result.
        3. Call the postprocess method with the mock prediction result and an image.
        4. Verify that the output is a list of Results objects and a warning message is logged.
        """
        self.args['device'] = 'mps'
        LOGGER.warning = unittest.mock.MagicMock()
        
        preds = ops.non_max_suppression(
            [[0.9, 10, 20, 30, 40, 50, 60, 70, 80, 90]],
            conf=0.5,
            iou=0.4,
            agnostic=False,
            max_det=1,
            classes=None,
            nc=1
        )
        
        img = ops.read_image('path/to/test/image.jpg')
        orig_imgs = [img]
        
        results = self.predictor.postprocess(preds, img, orig_imgs)
        LOGGER.warning.assert_called_once_with(
            "WARNING ⚠️ Apple MPS known Pose bug. Recommend 'device=cpu' for Pose models. "
            "See https://github.com/ultralytics/ultralytics/issues/4031."
        )
        self.assertIsInstance(results, list)
        self.assertEqual(len(results), 1)
        self.assertIsInstance(results[0], Results)

if __name__ == '__main__':
    unittest.main()
