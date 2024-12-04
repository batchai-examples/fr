import unittest
from ultralytics.models.sam.modules.sam import Sam, ImageEncoderViT, PromptEncoder, MaskDecoder

class TestSam(unittest.TestCase):
    def setUp(self):
        self.image_encoder = ImageEncoderViT()
        self.prompt_encoder = PromptEncoder()
        self.mask_decoder = MaskDecoder()
        self.sam = Sam(self.image_encoder, self.prompt_encoder, self.mask_decoder)

    def test_init_with_default_values(self):
        """
        Test the initialization of Sam with default values.
        """
        sam = Sam(ImageEncoderViT(), PromptEncoder(), MaskDecoder())
        self.assertEqual(sam.mask_threshold, 0.0)
        self.assertEqual(sam.image_format, "RGB")

    def test_forward_with_valid_input(self):
        """
        Test the forward method with valid input.
        """
        image = torch.randn(1, 3, 256, 256)  # Example image tensor
        prompts = torch.randn(1, 4)  # Example prompt tensor
        output = self.sam(image, prompts)
        self.assertIsNotNone(output)

    def test_forward_with_invalid_image_shape(self):
        """
        Test the forward method with an invalid image shape.
        """
        image = torch.randn(1, 3, 256)  # Invalid image tensor (missing one dimension)
        prompts = torch.randn(1, 4)
        with self.assertRaises(ValueError):
            self.sam(image, prompts)

    def test_forward_with_invalid_prompt_shape(self):
        """
        Test the forward method with an invalid prompt shape.
        """
        image = torch.randn(1, 3, 256, 256)
        prompts = torch.randn(1, 3)  # Invalid prompt tensor (wrong number of dimensions)
        with self.assertRaises(ValueError):
            self.sam(image, prompts)

    def test_forward_with_empty_image(self):
        """
        Test the forward method with an empty image.
        """
        image = torch.empty((0, 3, 256, 256))  # Empty image tensor
        prompts = torch.randn(1, 4)
        with self.assertRaises(ValueError):
            self.sam(image, prompts)

    def test_forward_with_empty_prompts(self):
        """
        Test the forward method with empty prompts.
        """
        image = torch.randn(1, 3, 256, 256)
        prompts = torch.empty((0, 4))  # Empty prompt tensor
        output = self.sam(image, prompts)
        self.assertIsNotNone(output)

if __name__ == "__main__":
    unittest.main()
