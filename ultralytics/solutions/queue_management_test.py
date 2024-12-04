import unittest
from ultralytics.solutions.queue_management import QueueManager, Polygon

class TestQueueManager(unittest.TestCase):
    def setUp(self):
        self.classes_names = {0: "person", 1: "car"}
        self.reg_pts = [(20, 60), (20, 680), (1120, 680), (1120, 60)]
        self.queue_manager = QueueManager(self.classes_names, reg_pts=self.reg_pts)

    def test_init(self):
        """
        Test the initialization of the QueueManager class.
        
        Steps:
        1. Create an instance of QueueManager with default parameters.
        2. Verify that the region_points are set correctly.
        3. Verify that the counting_region is a Polygon object.
        4. Verify that the view_img attribute is False by default.
        """
        queue_manager = QueueManager(self.classes_names)
        self.assertEqual(queue_manager.reg_pts, [(20, 60), (20, 680), (1120, 680), (1120, 60)])
        self.assertIsInstance(queue_manager.counting_region, Polygon)
        self.assertFalse(queue_manager.view_img)

    def test_extract_and_process_tracks(self):
        """
        Test the extract_and_process_tracks method.
        
        Steps:
        1. Create an instance of QueueManager with default parameters.
        2. Call the extract_and_process_tracks method with mock tracks data.
        3. Verify that the counts attribute is incremented when an object enters the counting region.
        """
        from ultralytics.nn.modules import DetectionModel
        from ultralytics.nn.predictor import Predictor

        class MockTracks:
            def __init__(self):
                self.boxes = MockBoxes()

        class MockBoxes:
            def __init__(self):
                self.xyxy = [[100, 100, 200, 200]]
                self.cls = [0]
                self.id = [1]

        mock_tracks = MockTracks()
        self.queue_manager.extract_and_process_tracks([mock_tracks])
        self.assertEqual(self.queue_manager.counts, 1)

    def test_display_frames(self):
        """
        Test the display_frames method.
        
        Steps:
        1. Create an instance of QueueManager with default parameters.
        2. Call the display_frames method.
        3. Verify that the cv2.imshow function is called once.
        """
        import cv2
        from unittest.mock import patch

        self.queue_manager.im0 = None
        with patch('cv2.imshow') as mock_imshow:
            self.queue_manager.display_frames()
            mock_imshow.assert_called_once()

    def test_process_queue(self):
        """
        Test the process_queue method.
        
        Steps:
        1. Create an instance of QueueManager with default parameters.
        2. Call the process_queue method with mock image and tracks data.
        3. Verify that the extract_and_process_tracks method is called once.
        """
        from ultralytics.nn.modules import DetectionModel
        from ultralytics.nn.predictor import Predictor

        class MockTracks:
            def __init__(self):
                self.boxes = MockBoxes()

        class MockBoxes:
            def __init__(self):
                self.xyxy = [[100, 100, 200, 200]]
                self.cls = [0]
                self.id = [1]

        mock_tracks = MockTracks()
        with patch.object(QueueManager, 'extract_and_process_tracks') as mock_extract:
            self.queue_manager.process_queue(None, [mock_tracks])
            mock_extract.assert_called_once()

    def test_init_with_custom_parameters(self):
        """
        Test the initialization of the QueueManager class with custom parameters.
        
        Steps:
        1. Create an instance of QueueManager with custom parameters.
        2. Verify that the region_points are set correctly.
        3. Verify that the counting_region is a Polygon object.
        4. Verify that the view_img attribute is True.
        """
        queue_manager = QueueManager(self.classes_names, reg_pts=self.reg_pts, view_img=True)
        self.assertEqual(queue_manager.reg_pts, [(20, 60), (20, 680), (1120, 680), (1120, 60)])
        self.assertIsInstance(queue_manager.counting_region, Polygon)
        self.assertTrue(queue_manager.view_img)

    def test_extract_and_process_tracks_with_no_tracks(self):
        """
        Test the extract_and_process_tracks method with no tracks.
        
        Steps:
        1. Create an instance of QueueManager with default parameters.
        2. Call the extract_and_process_tracks method with empty mock tracks data.
        3. Verify that the counts attribute remains unchanged.
        """
        from ultralytics.nn.modules import DetectionModel
        from ultralytics.nn.predictor import Predictor

        class MockTracks:
            def __init__(self):
                self.boxes = MockBoxes()

        class MockBoxes:
            def __init__(self):
                self.xyxy = []
                self.cls = []
                self.id = []

        mock_tracks = MockTracks()
        self.queue_manager.counts = 10
        self.queue_manager.extract_and_process_tracks([mock_tracks])
        self.assertEqual(self.queue_manager.counts, 10)

    def test_display_frames_with_no_image(self):
        """
        Test the display_frames method with no image.
        
        Steps:
        1. Create an instance of QueueManager with default parameters.
        2. Call the display_frames method with None as the image.
        3. Verify that the cv2.imshow function is not called.
        """
        import cv2
        from unittest.mock import patch

        self.queue_manager.im0 = None
        with patch('cv2.imshow') as mock_imshow:
            self.queue_manager.display_frames()
            mock_imshow.assert_not_called()

    def test_process_queue_with_no_tracks(self):
        """
        Test the process_queue method with no tracks.
        
        Steps:
        1. Create an instance of QueueManager with default parameters.
        2. Call the process_queue method with None as the image and empty mock tracks data.
        3. Verify that the extract_and_process_tracks method is called once.
        """
        from ultralytics.nn.modules import DetectionModel
        from ultralytics.nn.predictor import Predictor

        class MockTracks:
            def __init__(self):
                self.boxes = MockBoxes()

        class MockBoxes:
            def __init__(self):
                self.xyxy = []
                self.cls = []
                self.id = []

        mock_tracks = MockTracks()
        with patch.object(QueueManager, 'extract_and_process_tracks') as mock_extract:
            self.queue_manager.process_queue(None, [mock_tracks])
            mock_extract.assert_called_once()

if __name__ == '__main__':
    unittest.main()
