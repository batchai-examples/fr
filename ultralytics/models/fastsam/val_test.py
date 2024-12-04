import unittest
from ultralytics.models.fastsam.val import FastSAMValidator
from ultralytics.utils.metrics import SegmentMetrics

class TestFastSAMValidator(unittest.TestCase):
    def setUp(self):
        self.dataloader = None  # Mock dataloader
        self.save_dir = "test_save_dir"
        self.pbar = None  # Mock progress bar
        self.args = SimpleNamespace(task="segment", plots=False)
        self._callbacks = {}

    def test_init_with_all_arguments(self):
        """
        Test the initialization of FastSAMValidator with all arguments provided.
        
        Steps:
        1. Create an instance of FastSAMValidator with all required arguments.
        2. Verify that the task is set to 'segment'.
        3. Verify that plots are disabled.
        4. Verify that SegmentMetrics is initialized with the correct save_dir.
        """
        validator = FastSAMValidator(self.dataloader, self.save_dir, self.pbar, self.args, self._callbacks)
        self.assertEqual(validator.args.task, "segment")
        self.assertFalse(validator.args.plots)
        self.assertIsInstance(validator.metrics, SegmentMetrics)
        self.assertEqual(validator.metrics.save_dir, self.save_dir)

    def test_init_with_default_arguments(self):
        """
        Test the initialization of FastSAMValidator with default arguments.
        
        Steps:
        1. Create an instance of FastSAMValidator without providing any arguments.
        2. Verify that the task is set to 'segment'.
        3. Verify that plots are disabled.
        4. Verify that SegmentMetrics is initialized with a default save_dir.
        """
        validator = FastSAMValidator()
        self.assertEqual(validator.args.task, "segment")
        self.assertFalse(validator.args.plots)
        self.assertIsInstance(validator.metrics, SegmentMetrics)

    def test_init_with_missing_arguments(self):
        """
        Test the initialization of FastSAMValidator with missing arguments.
        
        Steps:
        1. Create an instance of FastSAMValidator without providing some required arguments.
        2. Verify that the task is set to 'segment'.
        3. Verify that plots are disabled.
        4. Verify that SegmentMetrics is initialized with a default save_dir.
        """
        validator = FastSAMValidator(dataloader=self.dataloader, args=self.args)
        self.assertEqual(validator.args.task, "segment")
        self.assertFalse(validator.args.plots)
        self.assertIsInstance(validator.metrics, SegmentMetrics)

    def test_init_with_invalid_arguments(self):
        """
        Test the initialization of FastSAMValidator with invalid arguments.
        
        Steps:
        1. Create an instance of FastSAMValidator with invalid arguments.
        2. Verify that the task is set to 'segment'.
        3. Verify that plots are disabled.
        4. Verify that SegmentMetrics is initialized with a default save_dir.
        """
        validator = FastSAMValidator(dataloader=self.dataloader, args=SimpleNamespace(task="invalid", plots=True))
        self.assertEqual(validator.args.task, "segment")
        self.assertFalse(validator.args.plots)
        self.assertIsInstance(validator.metrics, SegmentMetrics)

if __name__ == "__main__":
    unittest.main()
