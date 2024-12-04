import unittest
from unittest.mock import patch, MagicMock
from ultralytics.models.yolo.classify.val import ClassificationValidator

class TestClassificationValidator(unittest.TestCase):

    @patch('ultralytics.models.yolo.classify.val.build_dataloader')
    def test_init(self, mock_build_dataloader):
        """
        Test the initialization of ClassificationValidator.
        
        Steps:
            1. Create an instance of ClassificationValidator with default arguments.
            2. Verify that the dataloader is initialized correctly.
            3. Verify that the args and save_dir are set correctly.
            4. Verify that the metrics are initialized correctly.
        """
        args = {'model': 'yolov8n-cls.pt', 'data': 'imagenet10'}
        validator = ClassificationValidator(args=args, dataloader=None, save_dir='test_save_dir')
        
        mock_build_dataloader.assert_not_called()
        self.assertEqual(validator.args, args)
        self.assertEqual(validator.save_dir, 'test_save_dir')
        self.assertIsInstance(validator.metrics, ClassifyMetrics)

    @patch('ultralytics.models.yolo.classify.val.build_dataloader')
    def test_get_desc(self, mock_build_dataloader):
        """
        Test the get_desc method of ClassificationValidator.
        
        Steps:
            1. Create an instance of ClassificationValidator with default arguments.
            2. Call the get_desc method and verify that it returns the correct string.
        """
        validator = ClassificationValidator(args={'model': 'yolov8n-cls.pt', 'data': 'imagenet10'})
        
        desc = validator.get_desc()
        self.assertEqual(desc, " classes top1_acc top5_acc")

    @patch('ultralytics.models.yolo.classify.val.build_dataloader')
    def test_init_metrics(self, mock_build_dataloader):
        """
        Test the init_metrics method of ClassificationValidator.
        
        Steps:
            1. Create an instance of ClassificationValidator with default arguments.
            2. Call the init_metrics method and verify that it initializes the confusion matrix, class names, and top-1 and top-5 accuracy correctly.
        """
        model = MagicMock()
        model.names = ['class1', 'class2']
        
        validator = ClassificationValidator(args={'model': 'yolov8n-cls.pt', 'data': 'imagenet10'})
        validator.init_metrics(model)
        
        self.assertEqual(validator.nc, 2)
        self.assertIsInstance(validator.confusion_matrix, ConfusionMatrix)
        self.assertEqual(validator.names, {'class1': 0, 'class2': 1})

    @patch('ultralytics.models.yolo.classify.val.build_dataloader')
    def test_preprocess(self, mock_build_dataloader):
        """
        Test the preprocess method of ClassificationValidator.
        
        Steps:
            1. Create an instance of ClassificationValidator with default arguments.
            2. Call the preprocess method and verify that it preprocesses the input batch correctly.
        """
        validator = ClassificationValidator(args={'model': 'yolov8n-cls.pt', 'data': 'imagenet10'})
        
        batch = {
            "img": torch.randn(1, 3, 224, 224),
            "cls": torch.tensor([0])
        }
        
        processed_batch = validator.preprocess(batch)
        
        self.assertEqual(processed_batch["img"].shape, (1, 3, 224, 224))
        self.assertEqual(processed_batch["cls"], torch.tensor([0]))

    @patch('ultralytics.models.yolo.classify.val.build_dataloader')
    def test_plot_val_samples(self, mock_build_dataloader):
        """
        Test the plot_val_samples method of ClassificationValidator.
        
        Steps:
            1. Create an instance of ClassificationValidator with default arguments.
            2. Call the plot_val_samples method and verify that it plots the validation image samples correctly.
        """
        validator = ClassificationValidator(args={'model': 'yolov8n-cls.pt', 'data': 'imagenet10'})
        
        batch = {
            "img": torch.randn(1, 3, 224, 224),
            "cls": torch.tensor([0])
        }
        
        with patch('ultralytics.models.yolo.classify.val.plot_images') as mock_plot:
            validator.plot_val_samples(batch, ni=0)
            
            mock_plot.assert_called_once_with(
                images=batch["img"],
                batch_idx=torch.arange(len(batch["img"])),
                cls=batch["cls"].view(-1),
                fname='test_save_dir/val_batch0_labels.jpg',
                names={'class1': 0},
                on_plot=None
            )

    @patch('ultralytics.models.yolo.classify.val.build_dataloader')
    def test_plot_predictions(self, mock_build_dataloader):
        """
        Test the plot_predictions method of ClassificationValidator.
        
        Steps:
            1. Create an instance of ClassificationValidator with default arguments.
            2. Call the plot_predictions method and verify that it plots the predicted bounding boxes on input images correctly.
        """
        validator = ClassificationValidator(args={'model': 'yolov8n-cls.pt', 'data': 'imagenet10'})
        
        batch = {
            "img": torch.randn(1, 3, 224, 224),
            "cls": torch.tensor([0])
        }
        
        preds = torch.tensor([[0.9, 0.1]])
        
        with patch('ultralytics.models.yolo.classify.val.plot_images') as mock_plot:
            validator.plot_predictions(batch, preds, ni=0)
            
            mock_plot.assert_called_once_with(
                images=batch["img"],
                batch_idx=torch.arange(len(batch["img"])),
                cls=torch.argmax(preds, dim=1),
                fname='test_save_dir/val_batch0_pred.jpg',
                names={'class1': 0},
                on_plot=None
            )

    @patch('ultralytics.models.yolo.classify.val.build_dataloader')
    def test_print_results(self, mock_build_dataloader):
        """
        Test the print_results method of ClassificationValidator.
        
        Steps:
            1. Create an instance of ClassificationValidator with default arguments.
            2. Call the print_results method and verify that it prints the evaluation metrics correctly.
        """
        validator = ClassificationValidator(args={'model': 'yolov8n-cls.pt', 'data': 'imagenet10'})
        
        with patch('ultralytics.models.yolo.classify.val.LOGGER.info') as mock_logger:
            validator.print_results()
            
            mock_logger.assert_called_once_with(" all 0.9 0.1")

if __name__ == '__main__':
    unittest.main()
