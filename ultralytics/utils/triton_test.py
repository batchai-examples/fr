import unittest
from ultralytics.utils.triton import TritonRemoteModel, np

class TestTritonRemoteModel(unittest.TestCase):
    def setUp(self):
        self.url = "http://localhost:8000"
        self.endpoint = "model_name"
        self.model = TritonRemoteModel(url=self.url, endpoint=self.endpoint)

    def test_init_with_url_and_endpoint(self):
        """
        Test the initialization of TritonRemoteModel with URL and endpoint.
        """
        # Arrange
        expected_endpoint = "model_name"
        expected_url = "http://localhost:8000"

        # Act
        model = TritonRemoteModel(url=self.url, endpoint=expected_endpoint)

        # Assert
        self.assertEqual(model.endpoint, expected_endpoint)
        self.assertEqual(model.url, expected_url)

    def test_init_with_url_only(self):
        """
        Test the initialization of TritonRemoteModel with URL only.
        """
        # Arrange
        url = "http://localhost:8000/model_name"
        expected_endpoint = "model_name"
        expected_url = "http://localhost:8000"

        # Act
        model = TritonRemoteModel(url=url)

        # Assert
        self.assertEqual(model.endpoint, expected_endpoint)
        self.assertEqual(model.url, expected_url)

    def test_call_with_valid_inputs(self):
        """
        Test the call method with valid inputs.
        """
        # Arrange
        input_data = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
        expected_output_shape = (2, 2)

        # Act
        output = self.model(input_data)

        # Assert
        self.assertEqual(output.shape, expected_output_shape)
        self.assertTrue(np.allclose(output, input_data))

    def test_call_with_invalid_input_dtype(self):
        """
        Test the call method with invalid input dtype.
        """
        # Arrange
        input_data = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.int32)
        expected_output_shape = (2, 2)

        # Act
        output = self.model(input_data)

        # Assert
        self.assertEqual(output.shape, expected_output_shape)
        self.assertTrue(np.allclose(output, input_data.astype(np.float32)))

    def test_call_with_empty_inputs(self):
        """
        Test the call method with empty inputs.
        """
        # Arrange
        input_data = np.array([], dtype=np.float32)

        # Act & Assert
        with self.assertRaises(ValueError):
            self.model(input_data)

    def test_call_with_incorrect_number_of_inputs(self):
        """
        Test the call method with incorrect number of inputs.
        """
        # Arrange
        input_data1 = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
        input_data2 = np.array([[5.0, 6.0], [7.0, 8.0]], dtype=np.float32)

        # Act & Assert
        with self.assertRaises(ValueError):
            self.model(input_data1, input_data2)

if __name__ == "__main__":
    unittest.main()
