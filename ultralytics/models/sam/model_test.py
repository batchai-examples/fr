import unittest
from ultralytics.models.sam.model import SAM

class TestSAM(unittest.TestCase):
    def setUp(self):
        self.sam_model = SAM()

    def test_predict_with_image_path(self):
        """
        Tests the predict method with a valid image path.
        """
        # Arrange
        source = "path/to/image.jpg"
        
        # Act
        result = self.sam_model.predict(source)
        
        # Assert
        self.assertIsNotNone(result)

    def test_predict_with_streaming_enabled(self):
        """
        Tests the predict method with streaming enabled.
        """
        # Arrange
        source = "path/to/video.mp4"
        stream = True
        
        # Act
        result = self.sam_model.predict(source, stream=stream)
        
        # Assert
        self.assertIsNotNone(result)

    def test_predict_with_bounding_boxes(self):
        """
        Tests the predict method with bounding boxes.
        """
        # Arrange
        source = "path/to/image.jpg"
        bboxes = [[100, 100, 200, 200]]
        
        # Act
        result = self.sam_model.predict(source, bboxes=bboxes)
        
        # Assert
        self.assertIsNotNone(result)

    def test_predict_with_points(self):
        """
        Tests the predict method with points.
        """
        # Arrange
        source = "path/to/image.jpg"
        points = [[150, 150]]
        
        # Act
        result = self.sam_model.predict(source, points=points)
        
        # Assert
        self.assertIsNotNone(result)

    def test_predict_with_labels(self):
        """
        Tests the predict method with labels.
        """
        # Arrange
        source = "path/to/image.jpg"
        labels = ["person"]
        
        # Act
        result = self.sam_model.predict(source, labels=labels)
        
        # Assert
        self.assertIsNotNone(result)

    def test_predict_with_invalid_image_path(self):
        """
        Tests the predict method with an invalid image path.
        """
        # Arrange
        source = "path/to/nonexistent/image.jpg"
        
        # Act & Assert
        with self.assertRaises(FileNotFoundError):
            self.sam_model.predict(source)

    def test_predict_with_empty_bounding_boxes(self):
        """
        Tests the predict method with empty bounding boxes.
        """
        # Arrange
        source = "path/to/image.jpg"
        bboxes = []
        
        # Act
        result = self.sam_model.predict(source, bboxes=bboxes)
        
        # Assert
        self.assertIsNotNone(result)

    def test_predict_with_empty_points(self):
        """
        Tests the predict method with empty points.
        """
        # Arrange
        source = "path/to/image.jpg"
        points = []
        
        # Act
        result = self.sam_model.predict(source, points=points)
        
        # Assert
        self.assertIsNotNone(result)

    def test_predict_with_empty_labels(self):
        """
        Tests the predict method with empty labels.
        """
        # Arrange
        source = "path/to/image.jpg"
        labels = []
        
        # Act
        result = self.sam_model.predict(source, labels=labels)
        
        # Assert
        self.assertIsNotNone(result)

if __name__ == "__main__":
    unittest.main()
