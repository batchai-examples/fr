import unittest
from ultralytics.utils.files import WorkingDirectory, spaces_in_path, increment_path, file_age, file_date, file_size, get_latest_run, update_models

class TestWorkingDirectory(unittest.TestCase):
    """Test the WorkingDirectory context manager."""

    def test_working_directory(self):
        """Test changing and restoring the working directory."""
        with WorkingDirectory("/tmp"):
            self.assertEqual(os.getcwd(), "/tmp")
        self.assertNotEqual(os.getcwd(), "/tmp")

class TestSpacesInPath(unittest.TestCase):
    """Test the spaces_in_path context manager."""

    def test_spaces_in_path_file(self):
        """Test copying a file with spaces in its name."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            original_path = Path(tmp_dir) / "file with spaces.txt"
            original_path.write_text("test")
            with spaces_in_path(original_path) as new_path:
                self.assertTrue(new_path.exists())
                self.assertEqual(new_path.read_text(), "test")

    def test_spaces_in_path_directory(self):
        """Test copying a directory with spaces in its name."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            original_path = Path(tmp_dir) / "directory with spaces"
            original_path.mkdir()
            (original_path / "file.txt").write_text("test")
            with spaces_in_path(original_path) as new_path:
                self.assertTrue(new_path.exists())
                self.assertEqual((new_path / "file.txt").read_text(), "test")

class TestIncrementPath(unittest.TestCase):
    """Test the increment_path function."""

    def test_increment_path_file(self):
        """Test incrementing a file path."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            original_path = Path(tmp_dir) / "file.txt"
            original_path.write_text("test")
            new_path = increment_path(original_path)
            self.assertTrue(new_path.exists())
            self.assertNotEqual(new_path, original_path)

    def test_increment_path_directory(self):
        """Test incrementing a directory path."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            original_path = Path(tmp_dir) / "directory"
            original_path.mkdir()
            new_path = increment_path(original_path)
            self.assertTrue(new_path.exists())
            self.assertNotEqual(new_path, original_path)

class TestFileAge(unittest.TestCase):
    """Test the file_age function."""

    def test_file_age(self):
        """Test getting the age of a file."""
        with tempfile.NamedTemporaryFile() as tmp_file:
            days = file_age(tmp_file.name)
            self.assertEqual(days, 0)

class TestFileDate(unittest.TestCase):
    """Test the file_date function."""

    def test_file_date(self):
        """Test getting the date of a file."""
        with tempfile.NamedTemporaryFile() as tmp_file:
            date_str = file_date(tmp_file.name)
            self.assertTrue(date_str.startswith(str(datetime.now().year)))

class TestFileSize(unittest.TestCase):
    """Test the file_size function."""

    def test_file_size(self):
        """Test getting the size of a file."""
        with tempfile.NamedTemporaryFile() as tmp_file:
            size = file_size(tmp_file.name)
            self.assertEqual(size, 0.0)

class TestGetLatestRun(unittest.TestCase):
    """Test the get_latest_run function."""

    def test_get_latest_run(self):
        """Test getting the latest run directory."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            (tmp_dir / "run1" / "last.pt").mkdir(parents=True)
            (tmp_dir / "run2" / "last.pt").mkdir(parents=True)
            latest_run = get_latest_run(tmp_dir)
            self.assertEqual(latest_run, str(Path(tmp_dir) / "run2"))

class TestUpdateModels(unittest.TestCase):
    """Test the update_models function."""

    def test_update_models(self):
        """Test updating models."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            model_name = "model.pt"
            (tmp_dir / model_name).write_text("test")
            update_models((model_name,), source_dir=Path(tmp_dir), update_names=False)
            updated_model_path = Path(tmp_dir) / "updated_models" / model_name
            self.assertTrue(updated_model_path.exists())
            self.assertEqual(updated_model_path.read_text(), "test")

if __name__ == "__main__":
    unittest.main()
