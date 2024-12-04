import unittest
from ultralytics.models.nas.val import NASValidator
import torch

class TestNASValidator(unittest.TestCase):
    def setUp(self):
        self.validator = NASValidator(args=Namespace(conf=0.5, iou=0.4, single_cls=False, max_det=100))

    def test_postprocess_happy_path(self):
        """
        Test the postprocess method with a typical input to ensure it returns expected output.
        """
        raw_preds = [
            torch.tensor([
                [0.1, 0.2, 0.3, 0.4, 0.9],
                [0.5, 0.6, 0.7, 0.8, 0.8]
            ])
        ]
        expected_output = torch.tensor([
            [0.1, 0.2, 0.3, 0.4, 0.9],
            [0.5, 0.6, 0.7, 0.8, 0.8]
        ])
        output = self.validator.postprocess(raw_preds)
        self.assertTrue(torch.equal(output, expected_output))

    def test_postprocess_negative_confidence(self):
        """
        Test the postprocess method with a low confidence threshold to ensure it filters out low-confidence boxes.
        """
        raw_preds = [
            torch.tensor([
                [0.1, 0.2, 0.3, 0.4, 0.1],
                [0.5, 0.6, 0.7, 0.8, 0.9]
            ])
        ]
        expected_output = torch.tensor([
            [0.5, 0.6, 0.7, 0.8, 0.9]
        ])
        output = self.validator.postprocess(raw_preds)
        self.assertTrue(torch.equal(output, expected_output))

    def test_postprocess_high_iou(self):
        """
        Test the postprocess method with a high IoU threshold to ensure it removes overlapping boxes.
        """
        raw_preds = [
            torch.tensor([
                [0.1, 0.2, 0.3, 0.4, 0.9],
                [0.25, 0.35, 0.45, 0.55, 0.8]
            ])
        ]
        expected_output = torch.tensor([
            [0.1, 0.2, 0.3, 0.4, 0.9]
        ])
        output = self.validator.postprocess(raw_preds)
        self.assertTrue(torch.equal(output, expected_output))

    def test_postprocess_empty_input(self):
        """
        Test the postprocess method with an empty input to ensure it returns an empty tensor.
        """
        raw_preds = []
        expected_output = torch.tensor([])
        output = self.validator.postprocess(raw_preds)
        self.assertTrue(torch.equal(output, expected_output))

if __name__ == '__main__':
    unittest.main()
