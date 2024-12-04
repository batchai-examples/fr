import pytest

def test_checks_happy_path():
    """
    Test the happy path for the `test_checks` function.
    
    Steps:
    - Assert that torch.cuda.is_available() returns True if CUDA_IS_AVAILABLE is True.
    - Assert that torch.cuda.device_count() returns CUDA_DEVICE_COUNT if CUDA_IS_AVAILABLE is True.
    """
    from tests.test_cuda import test_checks
    test_checks()

def test_export_engine_matrix_happy_path():
    """
    Test the happy path for the `test_export_engine_matrix` function with default parameters.
    
    Steps:
    - Call the `test_export_engine_matrix` function with default parameters.
    - Assert that the exported model inference is successful.
    - Clean up the exported file and cache.
    """
    from tests.test_cuda import test_export_engine_matrix
    test_export_engine_matrix(task="detect", dynamic=True, int8=False, half=False, batch=2)

def test_train_happy_path():
    """
    Test the happy path for the `test_train` function with default parameters.
    
    Steps:
    - Call the `test_train` function with default parameters.
    - Assert that the model training is successful.
    """
    from tests.test_cuda import test_train
    test_train()

def test_predict_multiple_devices_happy_path():
    """
    Test the happy path for the `test_predict_multiple_devices` function.
    
    Steps:
    - Call the `test_predict_multiple_devices` function.
    - Assert that the model prediction is successful on both CPU and CUDA devices.
    """
    from tests.test_cuda import test_predict_multiple_devices
    test_predict_multiple_devices()

def test_autobatch_happy_path():
    """
    Test the happy path for the `test_autobatch` function.
    
    Steps:
    - Call the `test_autobatch` function.
    - Assert that the batch size check is successful.
    """
    from tests.test_cuda import test_autobatch
    test_autobatch()

def test_utils_benchmarks_happy_path():
    """
    Test the happy path for the `test_utils_benchmarks` function.
    
    Steps:
    - Call the `test_utils_benchmarks` function.
    - Assert that the model profiling is successful.
    """
    from tests.test_cuda import test_utils_benchmarks
    test_utils_benchmarks()

def test_predict_sam_happy_path():
    """
    Test the happy path for the `test_predict_sam` function with default parameters.
    
    Steps:
    - Call the `test_predict_sam` function with default parameters.
    - Assert that the SAM model prediction is successful.
    """
    from tests.test_cuda import test_predict_sam
    test_predict_sam()
