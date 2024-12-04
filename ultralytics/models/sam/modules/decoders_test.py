import unittest
from typing import List, Tuple

from ultralytics.models.sam.modules.decoders import MLP, MaskDecoder


class TestMLP(unittest.TestCase):
    def test_forward(self):
        input_dim = 10
        hidden_dim = 20
        output_dim = 5
        num_layers = 3
        mlp = MLP(input_dim, hidden_dim, output_dim, num_layers)
        x = torch.randn(1, input_dim)
        y = mlp(x)
        self.assertEqual(y.shape, (1, output_dim))

    def test_sigmoid_output(self):
        input_dim = 10
        hidden_dim = 20
        output_dim = 5
        num_layers = 3
        mlp = MLP(input_dim, hidden_dim, output_dim, num_layers, sigmoid_output=True)
        x = torch.randn(1, input_dim)
        y = mlp(x)
        self.assertTrue((y >= 0).all() and (y <= 1).all())

    def test_num_layers(self):
        input_dim = 10
        hidden_dim = 20
        output_dim = 5
        num_layers = 1
        mlp = MLP(input_dim, hidden_dim, output_dim, num_layers)
        x = torch.randn(1, input_dim)
        y = mlp(x)
        self.assertEqual(y.shape, (1, output_dim))


class TestMaskDecoder(unittest.TestCase):
    def test_forward(self):
        image_embeddings = torch.randn(2, 256, 32, 32)
        image_pe = torch.randn(2, 256, 32, 32)
        sparse_prompt_embeddings = torch.randn(2, 10, 256)
        dense_prompt_embeddings = torch.randn(2, 10, 256)

        mask_decoder = MaskDecoder()
        masks, iou_pred = mask_decoder.predict_masks(
            image_embeddings=image_embeddings,
            image_pe=image_pe,
            sparse_prompt_embeddings=sparse_prompt_embeddings,
            dense_prompt_embeddings=dense_prompt_embeddings,
        )

        self.assertEqual(masks.shape, (2, 10, 32, 32))
        self.assertEqual(iou_pred.shape, (2, 10))

    def test_predict_masks(self):
        image_embeddings = torch.randn(2, 256, 32, 32)
        image_pe = torch.randn(2, 256, 32, 32)
        sparse_prompt_embeddings = torch.randn(2, 10, 256)
        dense_prompt_embeddings = torch.randn(2, 10, 256)

        mask_decoder = MaskDecoder()
        masks, iou_pred = mask_decoder.predict_masks(
            image_embeddings=image_embeddings,
            image_pe=image_pe,
            sparse_prompt_embeddings=sparse_prompt_embeddings,
            dense_prompt_embeddings=dense_prompt_embeddings,
        )

        self.assertEqual(masks.shape, (2, 10, 32, 32))
        self.assertEqual(iou_pred.shape, (2, 10))


if __name__ == "__main__":
    unittest.main()
