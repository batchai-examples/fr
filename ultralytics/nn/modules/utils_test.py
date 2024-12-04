import unittest
from ultralytics.nn.modules.utils import _get_clones, bias_init_with_prob, linear_init, inverse_sigmoid, multi_scale_deformable_attn_pytorch

class TestUtils(unittest.TestCase):

    def test_get_clones(self):
        """
        Test the `_get_clones` function to ensure it correctly clones a module.
        """
        # Arrange
        module = nn.Linear(10, 5)
        n = 3
        
        # Act
        cloned_modules = _get_clones(module, n)
        
        # Assert
        self.assertEqual(len(cloned_modules), n)
        for i in range(n):
            self.assertIsNotNone(cloned_modules[i])
            self.assertNotEqual(id(module), id(cloned_modules[i]))
            self.assertTrue(torch.equal(module.weight, cloned_modules[i].weight))
            self.assertTrue(torch.equal(module.bias, cloned_modules[i].bias))

    def test_bias_init_with_prob(self):
        """
        Test the `bias_init_with_prob` function to ensure it correctly calculates the bias initialization value.
        """
        # Arrange
        prior_prob = 0.01
        
        # Act
        bias_value = bias_init_with_prob(prior_prob)
        
        # Assert
        expected_bias_value = float(-np.log((1 - prior_prob) / prior_prob))
        self.assertAlmostEqual(bias_value, expected_bias_value)

    def test_linear_init(self):
        """
        Test the `linear_init` function to ensure it correctly initializes the weights and biases of a linear module.
        """
        # Arrange
        module = nn.Linear(10, 5)
        
        # Act
        linear_init(module)
        
        # Assert
        bound = 1 / math.sqrt(module.weight.shape[0])
        self.assertTrue(torch.all(module.weight.abs() <= bound))
        if hasattr(module, "bias") and module.bias is not None:
            self.assertTrue(torch.all(module.bias.abs() <= bound))

    def test_inverse_sigmoid(self):
        """
        Test the `inverse_sigmoid` function to ensure it correctly calculates the inverse sigmoid value.
        """
        # Arrange
        x = torch.tensor([0.1, 0.5, 0.9], dtype=torch.float32)
        
        # Act
        result = inverse_sigmoid(x)
        
        # Assert
        expected_result = torch.log(torch.tensor([0.1 / 0.9, 0.5 / 0.5, 0.9 / 0.1], dtype=torch.float32))
        self.assertTrue(torch.allclose(result, expected_result))

    def test_multi_scale_deformable_attn_pytorch_happy_path(self):
        """
        Test the `multi_scale_deformable_attn_pytorch` function with a happy path scenario.
        """
        # Arrange
        value = torch.randn(2, 16, 4, 32)
        value_spatial_shapes = torch.tensor([[4, 8], [2, 4]])
        sampling_locations = torch.randn(2, 10, 4, 2, 2)
        attention_weights = torch.randn(2, 10, 4, 2 * 2)
        
        # Act
        result = multi_scale_deformable_attn_pytorch(value, value_spatial_shapes, sampling_locations, attention_weights)
        
        # Assert
        self.assertEqual(result.shape, (2, 16 * 32, 10))

    def test_multi_scale_deformable_attn_pytorch_negative_case(self):
        """
        Test the `multi_scale_deformable_attn_pytorch` function with a negative case scenario.
        """
        # Arrange
        value = torch.randn(2, 16, 4, 32)
        value_spatial_shapes = torch.tensor([[4, 8], [2, 4]])
        sampling_locations = torch.randn(2, 10, 4, 2, 2)
        attention_weights = torch.randn(2, 10, 4, 2 * 2) - 1
        
        # Act
        result = multi_scale_deformable_attn_pytorch(value, value_spatial_shapes, sampling_locations, attention_weights)
        
        # Assert
        self.assertEqual(result.shape, (2, 16 * 32, 10))

    def test_multi_scale_deformable_attn_pytorch_corner_case(self):
        """
        Test the `multi_scale_deformable_attn_pytorch` function with a corner case scenario.
        """
        # Arrange
        value = torch.randn(2, 16, 4, 32)
        value_spatial_shapes = torch.tensor([[4, 8], [2, 4]])
        sampling_locations = torch.zeros(2, 10, 4, 2, 2)
        attention_weights = torch.ones(2, 10, 4, 2 * 2)
        
        # Act
        result = multi_scale_deformable_attn_pytorch(value, value_spatial_shapes, sampling_locations, attention_weights)
        
        # Assert
        self.assertEqual(result.shape, (2, 16 * 32, 10))

    def test_multi_scale_deformable_attn_pytorch_edge_case(self):
        """
        Test the `multi_scale_deformable_attn_pytorch` function with an edge case scenario.
        """
        # Arrange
        value = torch.randn(2, 16, 4, 32)
        value_spatial_shapes = torch.tensor([[4, 8], [2, 4]])
        sampling_locations = torch.ones(2, 10, 4, 2, 2) * 2
        attention_weights = torch.ones(2, 10, 4, 2 * 2)
        
        # Act
        result = multi_scale_deformable_attn_pytorch(value, value_spatial_shapes, sampling_locations, attention_weights)
        
        # Assert
        self.assertEqual(result.shape, (2, 16 * 32, 10))

if __name__ == '__main__':
    unittest.main()
