import unittest
from unittest.mock import patch, MagicMock
from ultralytics.data.build import build_yolo_dataset, build_grounding, build_dataloader, check_source, load_inference_source

class TestBuildYOLODataset(unittest.TestCase):
    @patch('ultralytics.data.build.YOLODataset')
    def test_build_yolo_dataset(self, mock_YOLODataset):
        dataset = MagicMock()
        batch_size = 16
        result = build_dataloader(dataset, batch_size, workers=4)
        self.assertIsInstance(result, InfiniteDataLoader)
        mock_YOLODataset.assert_called_once_with(dataset, batch_size=batch_size)

class TestBuildGrounding(unittest.TestCase):
    @patch('ultralytics.data.build.GroundingDataset')
    def test_build_grounding(self, mock_GroundingDataset):
        dataset = MagicMock()
        batch_size = 16
        result = build_dataloader(dataset, batch_size, workers=4)
        self.assertIsInstance(result, InfiniteDataLoader)
        mock_GroundingDataset.assert_called_once_with(dataset, batch_size=batch_size)

class TestBuildDataloader(unittest.TestCase):
    @patch('ultralytics.data.build.InfiniteDataLoader')
    def test_build_dataloader(self, mock_InfiniteDataLoader):
        dataset = MagicMock()
        batch_size = 16
        result = build_dataloader(dataset, batch_size, workers=4)
        self.assertIsInstance(result, InfiniteDataLoader)
        mock_InfiniteDataLoader.assert_called_once_with(
            dataset=dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=4,
            sampler=None,
            pin_memory=PIN_MEMORY,
            collate_fn=None,
            worker_init_fn=seed_worker,
            generator=torch.Generator().manual_seed(6148914691236517205 + RANK)
        )

class TestCheckSource(unittest.TestCase):
    def test_check_source_string(self):
        source = "path/to/image.jpg"
        result = check_source(source)
        self.assertEqual(result, ("path/to/image.jpg", False, False, True, False, False))

    def test_check_source_int(self):
        source = 0
        result = check_source(source)
        self.assertEqual(result, ("0", True, False, False, False, False))

    def test_check_source_url(self):
        source = "http://example.com/video.mp4"
        result = check_source(source)
        self.assertEqual(result, ("http://example.com/video.mp4", False, False, False, False, False))

class TestLoadInferenceSource(unittest.TestCase):
    @patch('ultralytics.data.build.LoadTensor')
    def test_load_inference_source_tensor(self, mock_LoadTensor):
        source = torch.Tensor()
        result = load_inference_source(source)
        self.assertIsInstance(result, LoadTensor)

    @patch('ultralytics.data.build.LoadStreams')
    def test_load_inference_source_stream(self, mock_LoadStreams):
        source = "path/to/stream"
        result = load_inference_source(source)
        self.assertIsInstance(result, LoadStreams)

    @patch('ultralytics.data.build.LoadScreenshots')
    def test_load_inference_source_screenshots(self, mock_LoadScreenshots):
        source = "screen"
        result = load_inference_source(source)
        self.assertIsInstance(result, LoadScreenshots)

    @patch('ultralytics.data.build.LoadPilAndNumpy')
    def test_load_inference_source_pil_and_numpy(self, mock_LoadPilAndNumpy):
        source = [Image.open("path/to/image.jpg")]
        result = load_inference_source(source)
        self.assertIsInstance(result, LoadPilAndNumpy)

    @patch('ultralytics.data.build.LoadImagesAndVideos')
    def test_load_inference_source_images_and_videos(self, mock_LoadImagesAndVideos):
        source = "path/to/video.mp4"
        result = load_inference_source(source)
        self.assertIsInstance(result, LoadImagesAndVideos)

if __name__ == '__main__':
    unittest.main()
