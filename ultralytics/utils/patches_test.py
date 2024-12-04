import unittest
from ultralytics.utils.patches import imread, imwrite, imshow, torch_save

class TestImread(unittest.TestCase):
    """Test cases for the imread function."""

    def test_read_color_image(self):
        """
        Test reading a color image.
        
        Steps:
        1. Save a sample color image to a temporary file.
        2. Read the image using imread.
        3. Check if the read image is not None and has the correct shape.
        """
        filename = "test_color_image.jpg"
        cv2.imwrite(filename, np.array([[0, 0, 255], [0, 255, 0], [255, 0, 0]], dtype=np.uint8))
        img = imread(filename)
        self.assertIsNotNone(img)
        self.assertEqual(img.shape, (3, 3, 3))

    def test_read_gray_image(self):
        """
        Test reading a grayscale image.
        
        Steps:
        1. Save a sample grayscale image to a temporary file.
        2. Read the image using imread.
        3. Check if the read image is not None and has the correct shape.
        """
        filename = "test_gray_image.png"
        cv2.imwrite(filename, np.array([[0], [128], [255]], dtype=np.uint8))
        img = imread(filename)
        self.assertIsNotNone(img)
        self.assertEqual(img.shape, (3, 1))

    def test_read_nonexistent_file(self):
        """
        Test reading a non-existent file.
        
        Steps:
        1. Attempt to read a non-existent file using imread.
        2. Check if the function returns None.
        """
        filename = "non_existent_file.jpg"
        img = imread(filename)
        self.assertIsNone(img)

class TestImwrite(unittest.TestCase):
    """Test cases for the imwrite function."""

    def test_write_color_image(self):
        """
        Test writing a color image.
        
        Steps:
        1. Create a sample color image.
        2. Write the image to a temporary file using imwrite.
        3. Read the written image using imread and check if it matches the original image.
        """
        filename = "test_write_color_image.jpg"
        img = np.array([[0, 0, 255], [0, 255, 0], [255, 0, 0]], dtype=np.uint8)
        self.assertTrue(imwrite(filename, img))
        read_img = imread(filename)
        self.assertIsNotNone(read_img)
        self.assertEqual(img.shape, read_img.shape)

    def test_write_gray_image(self):
        """
        Test writing a grayscale image.
        
        Steps:
        1. Create a sample grayscale image.
        2. Write the image to a temporary file using imwrite.
        3. Read the written image using imread and check if it matches the original image.
        """
        filename = "test_write_gray_image.png"
        img = np.array([[0], [128], [255]], dtype=np.uint8)
        self.assertTrue(imwrite(filename, img))
        read_img = imread(filename)
        self.assertIsNotNone(read_img)
        self.assertEqual(img.shape, read_img.shape)

    def test_write_invalid_image(self):
        """
        Test writing an invalid image.
        
        Steps:
        1. Attempt to write a non-image object using imwrite.
        2. Check if the function returns False.
        """
        filename = "test_write_invalid_image.jpg"
        self.assertFalse(imwrite(filename, "not an image"))

class TestImshow(unittest.TestCase):
    """Test cases for the imshow function."""

    def test_show_color_image(self):
        """
        Test displaying a color image.
        
        Steps:
        1. Create a sample color image.
        2. Call imshow with the image and a window name.
        3. Check if no exception is raised (this is more of an integration test).
        """
        img = np.array([[0, 0, 255], [0, 255, 0], [255, 0, 0]], dtype=np.uint8)
        imshow("Test Window", img)

    def test_show_gray_image(self):
        """
        Test displaying a grayscale image.
        
        Steps:
        1. Create a sample grayscale image.
        2. Call imshow with the image and a window name.
        3. Check if no exception is raised (this is more of an integration test).
        """
        img = np.array([[0], [128], [255]], dtype=np.uint8)
        imshow("Test Window", img)

    def test_show_invalid_image(self):
        """
        Test displaying an invalid image.
        
        Steps:
        1. Attempt to call imshow with a non-image object.
        2. Check if no exception is raised (this is more of an integration test).
        """
        imshow("Test Window", "not an image")

class TestTorchSave(unittest.TestCase):
    """Test cases for the torch_save function."""

    def test_save_model(self):
        """
        Test saving a PyTorch model.
        
        Steps:
        1. Create a sample PyTorch model.
        2. Save the model using torch_save.
        3. Load the saved model and check if it matches the original model.
        """
        import torch
        model = torch.nn.Linear(10, 5)
        filename = "test_model.pth"
        self.assertTrue(torch_save(filename, model))
        loaded_model = torch.load(filename)
        self.assertEqual(model.state_dict(), loaded_model.state_dict())

    def test_save_invalid_object(self):
        """
        Test saving an invalid object.
        
        Steps:
        1. Attempt to save a non-PyTorch object using torch_save.
        2. Check if the function returns False.
        """
        filename = "test_save_invalid_object.pth"
        self.assertFalse(torch_save(filename, "not a model"))

if __name__ == "__main__":
    import unittest
    unittest.main()
