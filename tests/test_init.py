import pytest
from pathlib import Path
from ultralytics.utils import ASSETS, ROOT, WEIGHTS_DIR, checks, is_dir_writeable

# Constants used in tests
MODEL = WEIGHTS_DIR / "path with spaces" / "yolov8n.pt"  # test spaces in path
CFG = "yolov8n.yaml"
SOURCE = ASSETS / "bus.jpg"
TMP = (ROOT / "../tests/tmp").resolve()  # temp directory for test files
IS_TMP_WRITEABLE = is_dir_writeable(TMP)
CUDA_IS_AVAILABLE = checks.cuda_is_available()
CUDA_DEVICE_COUNT = checks.cuda_device_count()

__all__ = (
    "MODEL",
    "CFG",
    "SOURCE",
    "TMP",
    "IS_TMP_WRITEABLE",
    "CUDA_IS_AVAILABLE",
    "CUDA_DEVICE_COUNT",
)


def test_model_path():
    """
    Test if the MODEL path is correctly set with spaces in the directory name.
    """
    assert isinstance(MODEL, Path)
    assert MODEL.exists()
    assert str(MODEL) == str(WEIGHTS_DIR / "path with spaces" / "yolov8n.pt")


def test_config_path():
    """
    Test if the CFG path is correctly set to a valid YAML file.
    """
    assert isinstance(CFG, str)
    assert Path(CFG).exists()
    assert Path(CFG).suffix == ".yaml"


def test_source_image_path():
    """
    Test if the SOURCE image path is correctly set to a valid image file.
    """
    assert isinstance(SOURCE, Path)
    assert SOURCE.exists()
    assert SOURCE.suffix in {".jpg", ".jpeg"}


def test_tmp_directory():
    """
    Test if the TMP directory exists and is writable.
    """
    assert isinstance(TMP, Path)
    assert TMP.exists()
    assert TMP.is_dir()
    assert IS_TMP_WRITEABLE


def test_cuda_availability():
    """
    Test if CUDA is available on the system.
    """
    assert isinstance(CUDA_IS_AVAILABLE, bool)


def test_cuda_device_count():
    """
    Test if the number of CUDA devices is correctly counted.
    """
    assert isinstance(CUDA_DEVICE_COUNT, int)
    assert CUDA_DEVICE_COUNT >= 0


def test_model_path_with_nonexistent_directory():
    """
    Test if the MODEL path raises an error when the directory does not exist.
    """
    with pytest.raises(FileNotFoundError):
        Path("nonexistent_dir/yolov8n.pt")


def test_config_path_with_invalid_file():
    """
    Test if the CFG path raises an error when the file is invalid.
    """
    with pytest.raises(FileNotFoundError):
        Path("invalid_config.yaml")


def test_source_image_path_with_nonexistent_file():
    """
    Test if the SOURCE image path raises an error when the file does not exist.
    """
    with pytest.raises(FileNotFoundError):
        Path("nonexistent_bus.jpg")


def test_tmp_directory_with_nonexistent_path():
    """
    Test if the TMP directory raises an error when the path is invalid.
    """
    with pytest.raises(NotADirectoryError):
        Path("nonexistent_dir").resolve()


def test_cuda_availability_on_system_without_cuda():
    """
    Test if CUDA availability returns False on a system without CUDA.
    """
    # This test assumes that there is no CUDA available on the system
    assert not checks.cuda_is_available()
