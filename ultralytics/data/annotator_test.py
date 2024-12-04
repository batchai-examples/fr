import unittest
from pathlib import Path
from unittest.mock import patch, MagicMock

from ultralytics.data.annotator import auto_annotate, YOLO, SAM


class TestAutoAnnotate(unittest.TestCase):
    @patch('ultralytics.data.annotator.YOLO')
    @patch('ultralytics.data.annotator.SAM')
    def test_auto_annotate_happy_path(self, mock_sam, mock_yolo):
        """
        Test the auto_annotate function with happy path inputs.
        
        Steps:
        1. Mock the YOLO and SAM classes to return dummy objects.
        2. Call the auto_annotate function with valid input parameters.
        3. Verify that the output directory is created.
        4. Verify that the annotation file is written correctly.
        """
        data = 'ultralytics/assets'
        det_model = 'yolov8n.pt'
        sam_model = 'mobile_sam.pt'
        device = ''
        output_dir = Path(data).parent / f"{Path(data).stem}_auto_annotate_labels"

        mock_yolo_instance = MagicMock()
        mock_sam_instance = MagicMock()

        mock_yolo.return_value = mock_yolo_instance
        mock_sam.return_value = mock_sam_instance

        auto_annotate(data, det_model, sam_model, device)

        output_dir.mkdir.assert_called_once_with(exist_ok=True, parents=True)
        mock_yolo_instance.predict.assert_called_once_with(data=data, stream=True, device=device)
        mock_sam_instance.predict.assert_called_once_with(result.orig_img, bboxes=result.boxes.xyxy, verbose=False, save=False, device=device)

    @patch('ultralytics.data.annotator.YOLO')
    @patch('ultralytics.data.annotator.SAM')
    def test_auto_annotate_negative_output_dir(self, mock_sam, mock_yolo):
        """
        Test the auto_annotate function with a negative output directory.
        
        Steps:
        1. Mock the YOLO and SAM classes to return dummy objects.
        2. Call the auto_annotate function with valid input parameters but an invalid output directory.
        3. Verify that the output directory is not created.
        """
        data = 'ultralytics/assets'
        det_model = 'yolov8n.pt'
        sam_model = 'mobile_sam.pt'
        device = ''
        output_dir = Path('invalid_output_dir')

        mock_yolo_instance = MagicMock()
        mock_sam_instance = MagicMock()

        mock_yolo.return_value = mock_yolo_instance
        mock_sam.return_value = mock_sam_instance

        with self.assertRaises(FileNotFoundError):
            auto_annotate(data, det_model, sam_model, device, output_dir)

    @patch('ultralytics.data.annotator.YOLO')
    @patch('ultralytics.data.annotator.SAM')
    def test_auto_annotate_empty_class_ids(self, mock_sam, mock_yolo):
        """
        Test the auto_annotate function with empty class IDs.
        
        Steps:
        1. Mock the YOLO and SAM classes to return dummy objects.
        2. Call the auto_annotate function with valid input parameters but an image without any detections.
        3. Verify that no annotation file is written.
        """
        data = 'ultralytics/assets'
        det_model = 'yolov8n.pt'
        sam_model = 'mobile_sam.pt'
        device = ''

        mock_yolo_instance = MagicMock()
        mock_sam_instance = MagicMock()

        mock_yolo.return_value = mock_yolo_instance
        mock_sam.return_value = mock_sam_instance

        result = MagicMock(boxes=MagicMock(cls=MagicMock(int=MagicMock(return_value=[]))))
        mock_yolo_instance.predict.return_value = [result]

        auto_annotate(data, det_model, sam_model, device)

        self.assertFalse(mock_sam_instance.predict.called)
        self.assertFalse(output_dir.mkdir.called)

    @patch('ultralytics.data.annotator.YOLO')
    @patch('ultralytics.data.annotator.SAM')
    def test_auto_annotate_single_segment(self, mock_sam, mock_yolo):
        """
        Test the auto_annotate function with a single segment.
        
        Steps:
        1. Mock the YOLO and SAM classes to return dummy objects.
        2. Call the auto_annotate function with valid input parameters and an image with a single detection.
        3. Verify that the annotation file is written correctly with a single segment.
        """
        data = 'ultralytics/assets'
        det_model = 'yolov8n.pt'
        sam_model = 'mobile_sam.pt'
        device = ''

        mock_yolo_instance = MagicMock()
        mock_sam_instance = MagicMock()

        mock_yolo.return_value = mock_yolo_instance
        mock_sam.return_value = mock_sam_instance

        result = MagicMock(boxes=MagicMock(cls=MagicMock(int=MagicMock(return_value=[0])), xyxy=MagicMock(return_value=[[1, 2, 3, 4]])))
        mock_yolo_instance.predict.return_value = [result]

        segments = [[1, 2, 3, 4]]
        mock_sam_instance.predict.return_value = [MagicMock(masks=MagicMock(xyn=segments))]

        auto_annotate(data, det_model, sam_model, device)

        with open(f"{output_dir / Path(result.path).stem}.txt", "r") as f:
            content = f.read()
            self.assertEqual(content, "0 1.0 2.0 3.0 4.0\n")


if __name__ == '__main__':
    unittest.main()
