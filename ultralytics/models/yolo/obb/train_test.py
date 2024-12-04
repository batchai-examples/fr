import unittest
from ultralytics.models.yolo.obb.train import OBBTrainer

class TestOBBTrainer(unittest.TestCase):
    def test_init_with_default_args(self):
        """
        Test the initialization of OBBTrainer with default arguments.
        
        Steps:
        1. Create an instance of OBBTrainer without passing any arguments.
        2. Verify that the task is set to 'obb'.
        3. Verify that the model configuration is set to DEFAULT_CFG.
        """
        trainer = OBBTrainer()
        self.assertEqual(trainer.args["task"], "obb")
        self.assertEqual(trainer.cfg, DEFAULT_CFG)

    def test_get_model_with_weights(self):
        """
        Test the get_model method with weights specified.
        
        Steps:
        1. Create an instance of OBBTrainer.
        2. Call the get_model method with a dummy weight path.
        3. Verify that the model is loaded with the specified weights.
        """
        trainer = OBBTrainer()
        model = trainer.get_model(weights="dummy_weights.pt")
        self.assertIsNotNone(model.state_dict())

    def test_get_validator(self):
        """
        Test the get_validator method.
        
        Steps:
        1. Create an instance of OBBTrainer.
        2. Call the get_validator method.
        3. Verify that the validator is an instance of OBBValidator.
        """
        trainer = OBBTrainer()
        validator = trainer.get_validator()
        self.assertIsInstance(validator, yolo.obb.OBBValidator)

    def test_train_with_invalid_args(self):
        """
        Test the train method with invalid arguments.
        
        Steps:
        1. Create an instance of OBBTrainer with invalid model path.
        2. Call the train method and expect a ValueError to be raised.
        """
        trainer = OBBTrainer(overrides={"model": "invalid_model.pt"})
        with self.assertRaises(ValueError):
            trainer.train()

if __name__ == "__main__":
    unittest.main()
