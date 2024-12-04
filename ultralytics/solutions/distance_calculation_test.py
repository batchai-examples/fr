import unittest
from unittest.mock import patch, MagicMock
import numpy as np
import cv2

from ultralytics.solutions.distance_calculation import DistanceCalculation, calculate_centroid, calculate_distance

class TestDistanceCalculation(unittest.TestCase):
    def setUp(self):
        self.names = {0: "person", 1: "car"}
        self.distance_calculation = DistanceCalculation(self.names)

    @patch('ultralytics.utils.checks.check_imshow')
    def test_init(self, mock_check_imshow):
        mock_check_imshow.return_value = True
        distance_calculation = DistanceCalculation(self.names)
        self.assertEqual(distance_calculation.view_img, False)
        self.assertEqual(distance_calculation.line_color, (255, 255, 0))
        self.assertEqual(distance_calculation.centroid_color, (255, 0, 255))

    def test_calculate_centroid(self):
        box = [10, 20, 30, 40]
        centroid = calculate_centroid(box)
        self.assertEqual(centroid, (20, 30))

    def test_calculate_distance(self):
        centroid1 = (10, 20)
        centroid2 = (50, 60)
        distance_m, distance_mm = calculate_distance(centroid1, centroid2)
        self.assertAlmostEqual(distance_m, 58.31, places=2)
        self.assertEqual(distance_mm, 58310)

    def test_start_process_no_tracks(self):
        im0 = np.zeros((480, 640, 3), dtype=np.uint8)
        tracks = [MagicMock(boxes=MagicMock(id=None))]
        result = self.distance_calculation.start_process(im0, tracks)
        self.assertIs(result, im0)

    def test_start_process_with_tracks(self):
        im0 = np.zeros((480, 640, 3), dtype=np.uint8)
        box1 = [10, 20, 30, 40]
        box2 = [50, 60, 70, 80]
        tracks = [
            MagicMock(boxes=MagicMock(id=[0])),
            MagicMock(boxes=MagicMock(id=[1]))
        ]
        self.distance_calculation.selected_boxes = {0: box1, 1: box2}
        result = self.distance_calculation.start_process(im0, tracks)
        self.assertIsNot(result, im0)

    def test_display_frames(self):
        with patch('cv2.imshow') as mock_imshow:
            with patch('cv2.waitKey') as mock_waitKey:
                mock_waitKey.return_value = 1
                self.distance_calculation.display_frames()
                mock_imshow.assert_called_once_with("Ultralytics Distance Estimation", self.distance_calculation.im0)
                mock_waitKey.assert_called_once()

if __name__ == "__main__":
    unittest.main()
