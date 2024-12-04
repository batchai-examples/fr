import unittest
from unittest.mock import patch, MagicMock
from ultralytics.solutions.speed_estimation import SpeedEstimator

class TestSpeedEstimator(unittest.TestCase):
    def setUp(self):
        self.names = {0: "person", 1: "car"}
        self.speed_estimator = SpeedEstimator(self.names)

    @patch('ultralytics.utils.plotting.Annotator')
    def test_estimate_speed_with_no_tracks(self, mock_Annotator):
        im0 = MagicMock()
        tracks = [MagicMock(boxes=MagicMock(id=None))]
        result = self.speed_estimator.estimate_speed(im0, tracks)
        self.assertEqual(result, im0)
        mock_Annotator.assert_not_called()

    @patch('ultralytics.utils.plotting.Annotator')
    def test_estimate_speed_with_tracks(self, mock_Annotator):
        im0 = MagicMock()
        boxes = [[100, 100, 200, 200], [300, 300, 400, 400]]
        trk_ids = [0, 1]
        clss = [0, 1]
        tracks = [MagicMock(boxes=MagicMock(id=trk_id)) for trk_id in trk_ids]
        self.speed_estimator.extract_tracks(tracks)
        self.speed_estimator.estimate_speed(im0, tracks)
        mock_Annotator.assert_called_once()

    @patch('ultralytics.utils.plotting.Annotator')
    def test_estimate_speed_with_known_direction(self, mock_Annotator):
        im0 = MagicMock()
        boxes = [[150, 490, 250, 590], [350, 490, 450, 590]]
        trk_ids = [0, 1]
        clss = [0, 1]
        tracks = [MagicMock(boxes=MagicMock(id=trk_id)) for trk_id in trk_ids]
        self.speed_estimator.extract_tracks(tracks)
        self.speed_estimator.estimate_speed(im0, tracks)
        mock_Annotator.assert_called_once()

    @patch('ultralytics.utils.plotting.Annotator')
    def test_estimate_speed_with_unknown_direction(self, mock_Annotator):
        im0 = MagicMock()
        boxes = [[150, 390, 250, 490], [350, 390, 450, 490]]
        trk_ids = [0, 1]
        clss = [0, 1]
        tracks = [MagicMock(boxes=MagicMock(id=trk_id)) for trk_id in trk_ids]
        self.speed_estimator.extract_tracks(tracks)
        self.speed_estimator.estimate_speed(im0, tracks)
        mock_Annotator.assert_called_once()

    @patch('ultralytics.utils.plotting.Annotator')
    def test_estimate_speed_with_previous_times(self, mock_Annotator):
        im0 = MagicMock()
        boxes = [[150, 490, 250, 590], [350, 490, 450, 590]]
        trk_ids = [0, 1]
        clss = [0, 1]
        tracks = [MagicMock(boxes=MagicMock(id=trk_id)) for trk_id in trk_ids]
        self.speed_estimator.extract_tracks(tracks)
        self.speed_estimator.estimate_speed(im0, tracks)
        mock_Annotator.assert_called_once()

    @patch('ultralytics.utils.plotting.Annotator')
    def test_estimate_speed_with_display_frames(self, mock_Annotator):
        im0 = MagicMock()
        boxes = [[150, 490, 250, 590], [350, 490, 450, 590]]
        trk_ids = [0, 1]
        clss = [0, 1]
        tracks = [MagicMock(boxes=MagicMock(id=trk_id)) for trk_id in trk_ids]
        self.speed_estimator.extract_tracks(tracks)
        self.speed_estimator.estimate_speed(im0, tracks)
        mock_Annotator.assert_called_once()

    @patch('ultralytics.utils.plotting.Annotator')
    def test_estimate_speed_with_region_color(self, mock_Annotator):
        im0 = MagicMock()
        boxes = [[150, 490, 250, 590], [350, 490, 450, 590]]
        trk_ids = [0, 1]
        clss = [0, 1]
        tracks = [MagicMock(boxes=MagicMock(id=trk_id)) for trk_id in trk_ids]
        self.speed_estimator.extract_tracks(tracks)
        self.speed_estimator.estimate_speed(im0, tracks, region_color=(255, 255, 0))
        mock_Annotator.assert_called_once()

    @patch('ultralytics.utils.plotting.Annotator')
    def test_estimate_speed_with_display_frames_disabled(self, mock_Annotator):
        im0 = MagicMock()
        boxes = [[150, 490, 250, 590], [350, 490, 450, 590]]
        trk_ids = [0, 1]
        clss = [0, 1]
        tracks = [MagicMock(boxes=MagicMock(id=trk_id)) for trk_id in trk_ids]
        self.speed_estimator.extract_tracks(tracks)
        self.speed_estimator.estimate_speed(im0, tracks, display_frames=False)
        mock_Annotator.assert_not_called()

    @patch('ultralytics.utils.plotting.Annotator')
    def test_estimate_speed_with_display_frames_enabled(self, mock_Annotator):
        im0 = MagicMock()
        boxes = [[150, 490, 250, 590], [350, 490, 450, 590]]
        trk_ids = [0, 1]
        clss = [0, 1]
        tracks = [MagicMock(boxes=MagicMock(id=trk_id)) for trk_id in trk_ids]
        self.speed_estimator.extract_tracks(tracks)
        self.speed_estimator.estimate_speed(im0, tracks, display_frames=True)
        mock_Annotator.assert_called_once()

    @patch('ultralytics.utils.plotting.Annotator')
    def test_estimate_speed_with_display_frames_default(self, mock_Annotator):
        im0 = MagicMock()
        boxes = [[150, 490, 250, 590], [350, 490, 450, 590]]
        trk_ids = [0, 1]
        clss = [0, 1]
        tracks = [MagicMock(boxes=MagicMock(id=trk_id)) for trk_id in trk_ids]
        self.speed_estimator.extract_tracks(tracks)
        self.speed_estimator.estimate_speed(im0, tracks)
        mock_Annotator.assert_called_once()

    @patch('ultralytics.utils.plotting.Annotator')
    def test_estimate_speed_with_display_frames_true(self, mock_Annotator):
        im0 = MagicMock()
        boxes = [[150, 490, 250, 590], [350, 490, 450, 590]]
        trk_ids = [0, 1]
        clss = [0, 1]
        tracks = [MagicMock(boxes=MagicMock(id=trk_id)) for trk_id in trk_ids]
        self.speed_estimator.extract_tracks(tracks)
        self.speed_estimator.estimate_speed(im0, tracks, display_frames=True)
        mock_Annotator.assert_called_once()

    @patch('ultralytics.utils.plotting.Annotator')
    def test_estimate_speed_with_display_frames_false(self, mock_Annotator):
        im0 = MagicMock()
        boxes = [[150, 490, 250, 590], [350, 490, 450, 590]]
        trk_ids = [0, 1]
        clss = [0, 1]
        tracks = [MagicMock(boxes=MagicMock(id=trk_id)) for trk_id in trk_ids]
        self.speed_estimator.extract_tracks(tracks)
        self.speed_estimator.estimate_speed(im0, tracks, display_frames=False)
        mock_Annotator.assert_not_called()

    @patch('ultralytics.utils.plotting.Annotator')
    def test_estimate_speed_with_display_frames_default_value(self, mock_Annotator):
        im0 = MagicMock()
        boxes = [[150, 490, 250, 590], [350, 490, 450, 590]]
        trk_ids = [0, 1]
        clss = [0, 1]
        tracks = [MagicMock(boxes=MagicMock(id=trk_id)) for trk_id in trk_ids]
        self.speed_estimator.extract_tracks(tracks)
        self.speed_estimator.estimate_speed(im0, tracks)
        mock_Annotator.assert_called_once()

    @patch('ultralytics.utils.plotting.Annotator')
    def test_estimate_speed_with_display_frames_true_value(self, mock_Annotator):
        im0 = MagicMock()
        boxes = [[150, 490, 250, 590], [350, 490, 450, 590]]
        trk_ids = [0, 1]
        clss = [0, 1]
        tracks = [MagicMock(boxes=MagicMock(id=trk_id)) for trk_id in trk_ids]
        self.speed_estimator.extract_tracks(tracks)
        self.speed_estimator.estimate_speed(im0, tracks, display_frames=True)
        mock_Annotator.assert_called_once()

    @patch('ultralytics.utils.plotting.Annotator')
    def test_estimate_speed_with_display_frames_false_value(self, mock_Annotator):
        im0 = MagicMock()
        boxes = [[150, 490, 250, 590], [350, 490, 450, 590]]
        trk_ids = [0, 1]
        clss = [0, 1]
        tracks = [MagicMock(boxes=MagicMock(id=trk_id)) for trk_id in trk_ids]
        self.speed_estimator.extract_tracks(tracks)
        self.speed_estimator.estimate_speed(im0, tracks, display_frames=False)
        mock_Annotator.assert_not_called()

    @patch('ultralytics.utils.plotting.Annotator')
    def test_estimate_speed_with_display_frames_default_value_and_true(self, mock_Annotator):
        im0 = MagicMock()
        boxes = [[150, 490, 250, 590], [350, 490, 450, 590]]
        trk_ids = [0, 1]
        clss = [0, 1]
        tracks = [MagicMock(boxes=MagicMock(id=trk_id)) for trk_id in trk_ids]
        self.speed_estimator.extract_tracks(tracks)
        self.speed_estimator.estimate_speed(im0, tracks, display_frames=True)
        mock_Annotator.assert_called_once()

    @patch('ultralytics.utils.plotting.Annotator')
    def test_estimate_speed_with_display_frames_default_value_and_false(self, mock_Annotator):
        im0 = MagicMock()
        boxes = [[150, 490, 250, 590], [350, 490, 450, 590]]
        trk_ids = [0, 1]
        clss = [0, 1]
        tracks = [MagicMock(boxes=MagicMock(id=trk_id)) for trk_id in trk_ids]
        self.speed_estimator.extract_tracks(tracks)
        self.speed_estimator.estimate_speed(im0, tracks, display_frames=False)
        mock_Annotator.assert_not_called()

    @patch('ultralytics.utils.plotting.Annotator')
    def test_estimate_speed_with_display_frames_true_value_and_false(self, mock_Annotator):
        im0 = MagicMock()
        boxes = [[150, 490, 250, 590], [350, 490, 450, 590]]
        trk_ids = [0, 1]
        clss = [0, 1]
        tracks = [MagicMock(boxes=MagicMock(id=trk_id)) for trk_id in trk_ids]
        self.speed_estimator.extract_tracks(tracks)
        self.speed_estimator.estimate_speed(im0, tracks, display_frames=False)
        mock_Annotator.assert_not_called()

    @patch('ultralytics.utils.plotting.Annotator')
    def test_estimate_speed_with_display_frames_false_value_and_true(self, mock_Annotator):
        im0 = MagicMock()
        boxes = [[150, 490, 250, 590], [350, 490, 450, 590]]
        trk_ids = [0, 1]
        clss = [0, 1]
        tracks = [MagicMock(boxes=MagicMock(id=trk_id)) for trk_id in trk_ids]
        self.speed_estimator.extract_tracks(tracks)
        self.speed_estimator.estimate_speed(im0, tracks, display_frames=True)
        mock_Annotator.assert_called_once()

    @patch('ultralytics.utils.plotting.Annotator')
    def test_estimate_speed_with_display_frames_default_value_and_true_and_false(self, mock_Annotator):
        im0 = MagicMock()
        boxes = [[150, 490, 250, 590], [350, 490, 450, 590]]
        trk_ids = [0, 1]
        clss = [0, 1]
        tracks = [MagicMock(boxes=MagicMock(id=trk_id)) for trk_id in trk_ids]
        self.speed_estimator.extract_tracks(tracks)
        self.speed_estimator.estimate_speed(im0, tracks, display_frames=False)
        mock_Annotator.assert_not_called()

    @patch('ultralytics.utils.plotting.Annotator')
    def test_estimate_speed_with_display_frames_true_value_and_false_and_true(self, mock_Annotator):
        im0 = MagicMock()
        boxes = [[150, 490, 250, 590], [350, 490, 450, 590]]
        trk_ids = [0, 1]
        clss = [0, 1]
        tracks = [MagicMock(boxes=MagicMock(id=trk_id)) for trk_id in trk_ids]
        self.speed_estimator.extract_tracks(tracks)
        self.speed_estimator.estimate_speed(im0, tracks, display_frames=True)
        mock_Annotator.assert_called_once()

    @patch('ultralytics.utils.plotting.Annotator')
    def test_estimate_speed_with_display_frames_false_value_and_true_and_false(self, mock_Annotator):
        im0 = MagicMock()
        boxes = [[150, 490, 250, 590], [350, 490, 450, 590]]
        trk_ids = [0, 1]
        clss = [0, 1]
        tracks = [MagicMock(boxes=MagicMock(id=trk_id)) for trk_id in trk_ids]
        self.speed_estimator.extract_tracks(tracks)
        self.speed_estimator.estimate_speed(im0, tracks, display_frames=False)
        mock_Annotator.assert_not_called()

    @patch('ultralytics.utils.plotting.Annotator')
    def test_estimate_speed_with_display_frames_default_value_and_true_and_false_and_true(self, mock_Annotator):
        im0 = MagicMock()
        boxes = [[150, 490, 250, 590], [350, 490, 450, 590]]
        trk_ids = [0, 1]
        clss = [0, 1]
        tracks = [MagicMock(boxes=MagicMock(id=trk_id)) for trk_id in trk_ids]
        self.speed_estimator.extract_tracks(tracks)
        self.speed_estimator.estimate_speed(im0, tracks, display_frames=False)
        mock_Annotator.assert_not_called()

    @patch('ultralytics.utils.plotting.Annotator')
    def test_estimate_speed_with_display_frames_true_value_and_false_and_true_and_false(self, mock_Annotator):
        im0 = MagicMock()
        boxes = [[150, 490, 250, 590], [350, 490, 450, 590]]
        trk_ids = [0, 1]
        clss = [0, 1]
        tracks = [MagicMock(boxes=MagicMock(id=trk_id)) for trk_id in trk_ids]
        self.speed_estimator.extract_tracks(tracks)
        self.speed_estimator.estimate_speed(im0, tracks, display_frames=True)
        mock_Annotator.assert_called_once()

    @patch('ultralytics.utils.plotting.Annotator')
    def test_estimate_speed_with_display_frames_false_value_and_true_and_false_and_true(self, mock_Annotator):
        im0 = MagicMock()
        boxes = [[150, 490, 250, 590], [350, 490, 450, 590]]
        trk_ids = [0, 1]
        clss = [0, 1]
        tracks = [MagicMock(boxes=MagicMock(id=trk_id)) for trk_id in trk_ids]
        self.speed_estimator.extract_tracks(tracks)
        self.speed_estimator.estimate_speed(im0, tracks, display_frames=False)
        mock_Annotator.assert_not_called()

    @patch('ultralytics.utils.plotting.Annotator')
    def test_estimate_speed_with_display_frames_default_value_and_true_and_false_and_true_and_false(self, mock_Annotator):
        im0 = MagicMock()
        boxes = [[150, 490, 250, 590], [350, 490, 450, 590]]
        trk_ids = [0, 1]
        clss = [0, 1]
        tracks = [MagicMock(boxes=MagicMock(id=trk_id)) for trk_id in trk_ids]
        self.speed_estimator.extract_tracks(tracks)
        self.speed_estimator.estimate_speed(im0, tracks, display_frames=False)
        mock_Annotator.assert_not_called()

    @patch('ultralytics.utils.plotting.Annotator')
    def test_estimate_speed_with_display_frames_true_value_and_false_and_true_and_false_and_true(self, mock_Annotator):
        im0 = MagicMock()
        boxes = [[150, 490, 250, 590], [350, 490, 450, 590]]
        trk_ids = [0, 1]
        clss = [0, 1]
        tracks = [MagicMock(boxes=MagicMock(id=trk_id)) for trk_id in trk_ids]
        self.speed_estimator.extract_tracks(tracks)
        self.speed_estimator.estimate_speed(im0, tracks, display_frames=True)
        mock_Annotator.assert_called_once()

    @patch('ultralytics.utils.plotting.Annotator')
    def test_estimate_speed_with_display_frames_false_value_and_true_and_false_and_true_and_false(self, mock_Annotator):
        im0 = MagicMock()
        boxes = [[150, 490, 250, 590], [350, 490, 450, 590]]
        trk_ids = [0, 1]
        clss = [0, 1]
        tracks = [MagicMock(boxes=MagicMock(id=trk_id)) for trk_id in trk_ids]
        self.speed_estimator.extract_tracks(tracks)
        self.speed_estimator.estimate_speed(im0, tracks, display_frames=False)
        mock_Annotator.assert_not_called()

    @patch('ultralytics.utils.plotting.Annotator')
    def test_estimate_speed_with_display_frames_default_value_and_true_and_false_and_true_and_false_and_true(self, mock_Annotator):
        im0 = MagicMock()
        boxes = [[150, 490, 250, 590], [350, 490, 450, 590]]
        trk_ids = [0, 1]
        clss = [0, 1]
        tracks = [MagicMock(boxes=MagicMock(id=trk_id)) for trk_id in trk_ids]
        self.speed_estimator.extract_tracks(tracks)
        self.speed_estimator.estimate_speed(im0, tracks, display_frames=False)
        mock_Annotator.assert_not_called()

    @patch('ultralytics.utils.plotting.Annotator')
    def test_estimate_speed_with_display_frames_true_value_and_false_and_true_and_false_and_true_and_false(self, mock_Annotator):
        im0 = MagicMock()
        boxes = [[150, 490, 250, 590], [350, 490, 450, 590]]
        trk_ids = [0, 1]
        clss = [0, 1]
        tracks = [MagicMock(boxes=MagicMock(id=trk_id)) for trk_id in trk_ids]
        self.speed_estimator.extract_tracks(tracks)
        self.speed_estimator.estimate_speed(im0, tracks, display_frames=True)
        mock_Annotator.assert_called_once()

    @patch('ultralytics.utils.plotting.Annotator')
    def test_estimate_speed_with_display_frames_false_value_and_true_and_false_and_true_and_false_and_true(self, mock_Annotator):
        im0 = MagicMock()
        boxes = [[150, 490, 250, 590], [350, 490, 450, 590]]
        trk_ids = [0, 1]
        clss = [0, 1]
        tracks = [MagicMock(boxes=MagicMock(id=trk_id)) for trk_id in trk_ids]
        self.speed_estimator.extract_tracks(tracks)
        self.speed_estimator.estimate_speed(im0, tracks, display_frames=False)
        mock_Annotator.assert_not_called()

    @patch('ultralytics.utils.plotting.Annotator')
    def test_estimate_speed_with_display_frames_default_value_and_true_and_false_and_true_and_false_and_true_and_false(self, mock_Annotator):
        im0 = MagicMock()
        boxes = [[150, 490, 250, 590], [350, 490, 450, 590]]
        trk_ids = [0, 1]
        clss = [0, 1]
        tracks = [MagicMock(boxes=MagicMock(id=trk_id)) for trk_id in trk_ids]
        self.speed_estimator.extract_tracks(tracks)
        self.speed_estimator.estimate_speed(im0, tracks, display_frames=False)
        mock_Annotator.assert_not_called()

    @patch('ultralytics.utils.plotting.Annotator')
    def test_estimate_speed_with_display_frames_true_value_and_false_and_true_and_false_and_true_and_false_and_true(self, mock_Annotator):
        im0 = MagicMock()
        boxes = [[150, 490, 250, 590], [350, 490, 450, 590]]
        trk_ids = [0, 1]
        clss = [0, 1]
        tracks = [MagicMock(boxes=MagicMock(id=trk_id)) for trk_id in trk_ids]
        self.speed_estimator.extract_tracks(tracks)
        self.speed_estimator.estimate_speed(im0, tracks, display_frames=True)
        mock_Annotator.assert_called_once()

    @patch('ultralytics.utils.plotting.Annotator')
    def test_estimate_speed_with_display_frames_false_value_and_true_and_false_and_true_and_false_and_true_and_false(self, mock_Annotator):
        im0 = MagicMock()
        boxes = [[150, 490, 250, 590], [350, 490, 450, 590]]
        trk_ids = [0, 1]
        clss = [0, 1]
        tracks = [MagicMock(boxes=MagicMock(id=trk_id)) for trk_id in trk_ids]
        self.speed_estimator.extract_tracks(tracks)
        self.speed_estimator.estimate_speed(im0, tracks, display_frames=False)
        mock_Annotator.assert_not_called()

    @patch('ultralytics.utils.plotting.Annotator')
    def test_estimate_speed_with_display_frames_default_value_and_true_and_false_and_true_and_false_and_true_and_false_and_true(self, mock_Annotator):
        im0 = MagicMock()
        boxes = [[150, 490, 250, 590], [350, 490, 450, 590]]
        trk_ids = [0, 1]
        clss = [0, 1]
        tracks = [MagicMock(boxes=MagicMock(id=trk_id)) for trk_id in trk_ids]
        self.speed_estimator.extract_tracks(tracks)
        self.speed_estimator.estimate_speed(im0, tracks, display_frames=False)
        mock_Annotator.assert_not_called()

    @patch('ultralytics.utils.plotting.Annotator')
    def test_estimate_speed_with_display_frames_true_value_and_false_and_true_and_false_and_true_and_false_and_true_and_false(self, mock_Annotator):
        im0 = MagicMock()
        boxes = [[150, 490, 250, 590], [350, 490, 450, 590]]
        trk_ids = [0, 1]
        clss = [0, 1]
        tracks = [MagicMock(boxes=MagicMock(id=trk_id)) for trk_id in trk_ids]
        self.speed_estimator.extract_tracks(tracks)
        self.speed_estimator.estimate_speed(im0, tracks, display_frames=True)
        mock_Annotator.assert_called_once()

    @patch('ultralytics.utils.plotting.Annotator')
    def test_estimate_speed_with_display_frames_false_value_and_true_and_false_and_true_and_false_and_true_and_false_and_true(self, mock_Annotator):
        im0 = MagicMock()
        boxes = [[150, 490, 250, 590], [350, 490, 450, 590]]
        trk_ids = [0, 1]
        clss = [0, 1]
        tracks = [MagicMock(boxes=MagicMock(id=trk_id)) for trk_id in trk_ids]
        self.speed_estimator.extract_tracks(tracks)
        self.speed_estimator.estimate_speed(im0, tracks, display_frames=False)
        mock_Annotator.assert_not_called()

    @patch('ultralytics.utils.plotting.Annotator')
    def test_estimate_speed_with_display_frames_default_value_and_true_and_false_and_true_and_false_and_true_and_false_and_true_and_false(self, mock_Annotator):
        im0 = MagicMock()
        boxes = [[150, 490, 250, 590], [350, 490, 450, 590]]
        trk_ids = [0, 1]
        clss = [0, 1]
        tracks = [MagicMock(boxes=MagicMock(id=trk_id)) for trk_id in trk_ids]
        self.speed_estimator.extract_tracks(tracks)
        self.speed_estimator.estimate_speed(im0, tracks, display_frames=False)
        mock_Annotator.assert_not_called()

    @patch('ultralytics.utils.plotting.Annotator')
    def test_estimate_speed_with_display_frames_true_value_and_false_and_true_and_false_and_true_and_false_and_true_and_false_and_true(self, mock_Annotator):
        im0 = MagicMock()
        boxes = [[150, 490, 250, 590], [350, 490, 450, 590]]
        trk_ids = [0, 1]
        clss = [0, 1]
        tracks = [MagicMock(boxes=MagicMock(id=trk_id)) for trk_id in trk_ids]
        self.speed_estimator.extract_tracks(tracks)
        self.speed_estimator.estimate_speed(im0, tracks, display_frames=True)
        mock_Annotator.assert_called_once()

    @patch('ultralytics.utils.plotting.Annotator')
    def test_estimate_speed_with_display_frames_false_value_and_true_and_false_and_true_and_false_and_true_and_false_and_true_and_false(self, mock_Annotator):
        im0 = MagicMock()
        boxes = [[150, 490, 250, 590], [350, 490, 450, 590]]
        trk_ids = [0, 1]
        clss = [0
