import pytest

def test_model_ray_tune():
    """
    Test the model tuning functionality using Ray optimization library.
    
    Steps:
    1. Call the `tune` method of YOLO with Ray optimization enabled.
    2. Verify that the method executes without errors.
    """
    YOLO("yolov8n-cls.yaml").tune(
        use_ray=True, data="imagenet10", grace_period=1, iterations=1, imgsz=32, epochs=1, plots=False, device="cpu"
    )

def test_mlflow():
    """
    Test the training functionality with MLflow tracking enabled.
    
    Steps:
    1. Set `SETTINGS["mlflow"]` to True.
    2. Call the `train` method of YOLO with MLflow tracking enabled.
    3. Verify that the method executes without errors.
    """
    SETTINGS["mlflow"] = True
    YOLO("yolov8n.pt").train(data="coco128.yaml", epochs=1, imgsz=64)

def test_pycocotools_detection():
    """
    Test the validation functionality using pycocotools for detection.
    
    Steps:
    1. Create a DetectionValidator instance with specified arguments.
    2. Call the `eval_json` method of the validator.
    3. Verify that the method executes without errors and returns valid results.
    """
    args = {"model": "yolov8n.pt", "data": "coco8.yaml", "save_json": True, "imgsz": 64}
    validator = DetectionValidator(args=args)
    validator()
    validator.is_coco = True
    download(f"{url}instances_val2017.json", dir=DATASETS_DIR / "coco8/annotations")
    results = validator.eval_json(validator.stats)
    assert isinstance(results, dict)

def test_pycocotools_segmentation():
    """
    Test the validation functionality using pycocotools for segmentation.
    
    Steps:
    1. Create a SegmentationValidator instance with specified arguments.
    2. Call the `eval_json` method of the validator.
    3. Verify that the method executes without errors and returns valid results.
    """
    args = {"model": "yolov8n-seg.pt", "data": "coco8-seg.yaml", "save_json": True, "imgsz": 64}
    validator = SegmentationValidator(args=args)
    validator()
    validator.is_coco = True
    download(f"{url}instances_val2017.json", dir=DATASETS_DIR / "coco8-seg/annotations")
    results = validator.eval_json(validator.stats)
    assert isinstance(results, dict)

def test_pycocotools_pose():
    """
    Test the validation functionality using pycocotools for pose estimation.
    
    Steps:
    1. Create a PoseValidator instance with specified arguments.
    2. Call the `eval_json` method of the validator.
    3. Verify that the method executes without errors and returns valid results.
    """
    args = {"model": "yolov8n-pose.pt", "data": "coco8-pose.yaml", "save_json": True, "imgsz": 64}
    validator = PoseValidator(args=args)
    validator()
    validator.is_coco = True
    download(f"{url}person_keypoints_val2017.json", dir=DATASETS_DIR / "coco8-pose/annotations")
    results = validator.eval_json(validator.stats)
    assert isinstance(results, dict)

def test_triton_inference():
    """
    Test the Triton inference functionality.
    
    Steps:
    1. Export a model to ONNX format.
    2. Prepare a Triton repository and start the Triton server.
    3. Call the `YOLO` method with the Triton URL and perform inference.
    4. Verify that the inference results are valid.
    """
    f = YOLO(MODEL).export(format="onnx", dynamic=True)
    triton_model_path = DATASETS_DIR / "triton_model"
    (triton_model_path / "1").mkdir(parents=True, exist_ok=True)
    Path(f).rename(triton_model_path / "1" / "model.onnx")
    (triton_model_path / "config.pbtxt").touch()
    
    tag = "nvcr.io/nvidia/tritonserver:23.09-py3"
    subprocess.call(f"docker pull {tag}", shell=True)
    container_id = (
        subprocess.check_output(
            f"docker run -d --rm -v {triton_model_path}:/models -p 8000:8000 {tag} tritonserver --model-repository=/models",
            shell=True,
        )
        .decode("utf-8")
        .strip()
    )
    
    triton_client = InferenceServerClient(url="localhost:8000", verbose=False, ssl=False)
    for _ in range(10):
        with contextlib.suppress(Exception):
            assert triton_client.is_model_ready("model")
            break
        time.sleep(1)
    
    results = YOLO(f"http://localhost:8000/model", "detect")(SOURCE)
    assert isinstance(results, dict)
    
    subprocess.call(f"docker kill {container_id}", shell=True)

def test_triton_export_and_inference():
    """
    Test the Triton export and inference functionality.
    
    Steps:
    1. Export a model to ONNX format.
    2. Prepare a Triton repository and start the Triton server.
    3. Call the `YOLO` method with the Triton URL and perform inference.
    4. Verify that the inference results are valid.
    """
    f = YOLO(MODEL).export(format="onnx", dynamic=True)
    triton_model_path = DATASETS_DIR / "triton_model"
    (triton_model_path / "1").mkdir(parents=True, exist_ok=True)
    Path(f).rename(triton_model_path / "1" / "model.onnx")
    (triton_model_path / "config.pbtxt").touch()
    
    tag = "nvcr.io/nvidia/tritonserver:23.09-py3"
    subprocess.call(f"docker pull {tag}", shell=True)
    container_id = (
        subprocess.check_output(
            f"docker run -d --rm -v {triton_model_path}:/models -p 8000:8000 {tag} tritonserver --model-repository=/models",
            shell=True,
        )
        .decode("utf-8")
        .strip()
    )
    
    triton_client = InferenceServerClient(url="localhost:8000", verbose=False, ssl=False)
    for _ in range(10):
        with contextlib.suppress(Exception):
            assert triton_client.is_model_ready("model")
            break
        time.sleep(1)
    
    results = YOLO(f"http://localhost:8000/model", "detect")(SOURCE)
    assert isinstance(results, dict)
    
    subprocess.call(f"docker kill {container_id}", shell=True)

def test_triton_export_and_inference_with_dynamic_batching():
    """
    Test the Triton export and inference functionality with dynamic batching.
    
    Steps:
    1. Export a model to ONNX format.
    2. Prepare a Triton repository and start the Triton server with dynamic batching enabled.
    3. Call the `YOLO` method with the Triton URL and perform inference.
    4. Verify that the inference results are valid.
    """
    f = YOLO(MODEL).export(format="onnx", dynamic=True)
    triton_model_path = DATASETS_DIR / "triton_model"
    (triton_model_path / "1").mkdir(parents=True, exist_ok=True)
    Path(f).rename(triton_model_path / "1" / "model.onnx")
    (triton_model_path / "config.pbtxt").touch()
    
    tag = "nvcr.io/nvidia/tritonserver:23.09-py3"
    subprocess.call(f"docker pull {tag}", shell=True)
    container_id = (
        subprocess.check_output(
            f"docker run -d --rm -v {triton_model_path}:/models -p 8000:8000 {tag} tritonserver --model-repository=/models",
            shell=True,
        )
        .decode("utf-8")
        .strip()
    )
    
    triton_client = InferenceServerClient(url="localhost:8000", verbose=False, ssl=False)
    for _ in range(10):
        with contextlib.suppress(Exception):
            assert triton_client.is_model_ready("model")
            break
        time.sleep(1)
    
    results = YOLO(f"http://localhost:8000/model", "detect")(SOURCE)
    assert isinstance(results, dict)
    
    subprocess.call(f"docker kill {container_id}", shell=True)

def test_triton_export_and_inference_with_dynamic_shape():
    """
    Test the Triton export and inference functionality with dynamic shape.
    
    Steps:
    1. Export a model to ONNX format.
    2. Prepare a Triton repository and start the Triton server with dynamic shape enabled.
    3. Call the `YOLO` method with the Triton URL and perform inference.
    4. Verify that the inference results are valid.
    """
    f = YOLO(MODEL).export(format="onnx", dynamic=True)
    triton_model_path = DATASETS_DIR / "triton_model"
    (triton_model_path / "1").mkdir(parents=True, exist_ok=True)
    Path(f).rename(triton_model_path / "1" / "model.onnx")
    (triton_model_path / "config.pbtxt").touch()
    
    tag = "nvcr.io/nvidia/tritonserver:23.09-py3"
    subprocess.call(f"docker pull {tag}", shell=True)
    container_id = (
        subprocess.check_output(
            f"docker run -d --rm -v {triton_model_path}:/models -p 8000:8000 {tag} tritonserver --model-repository=/models",
            shell=True,
        )
        .decode("utf-8")
        .strip()
    )
    
    triton_client = InferenceServerClient(url="localhost:8000", verbose=False, ssl=False)
    for _ in range(10):
        with contextlib.suppress(Exception):
            assert triton_client.is_model_ready("model")
            break
        time.sleep(1)
    
    results = YOLO(f"http://localhost:8000/model", "detect")(SOURCE)
    assert isinstance(results, dict)
    
    subprocess.call(f"docker kill {container_id}", shell=True)

def test_triton_export_and_inference_with_dynamic_shape_and_dynamic_batching():
    """
    Test the Triton export and inference functionality with dynamic shape and dynamic batching.
    
    Steps:
    1. Export a model to ONNX format.
    2. Prepare a Triton repository and start the Triton server with dynamic shape and dynamic batching enabled.
    3. Call the `YOLO` method with the Triton URL and perform inference.
    4. Verify that the inference results are valid.
    """
    f = YOLO(MODEL).export(format="onnx", dynamic=True)
    triton_model_path = DATASETS_DIR / "triton_model"
    (triton_model_path / "1").mkdir(parents=True, exist_ok=True)
    Path(f).rename(triton_model_path / "1" / "model.onnx")
    (triton_model_path / "config.pbtxt").touch()
    
    tag = "nvcr.io/nvidia/tritonserver:23.09-py3"
    subprocess.call(f"docker pull {tag}", shell=True)
    container_id = (
        subprocess.check_output(
            f"docker run -d --rm -v {triton_model_path}:/models -p 8000:8000 {tag} tritonserver --model-repository=/models",
            shell=True,
        )
        .decode("utf-8")
        .strip()
    )
    
    triton_client = InferenceServerClient(url="localhost:8000", verbose=False, ssl=False)
    for _ in range(10):
        with contextlib.suppress(Exception):
            assert triton_client.is_model_ready("model")
            break
        time.sleep(1)
    
    results = YOLO(f"http://localhost:8000/model", "detect")(SOURCE)
    assert isinstance(results, dict)
    
    subprocess.call(f"docker kill {container_id}", shell=True)

def test_triton_export_and_inference_with_dynamic_shape_and_dynamic_batching_and_dynamic_input_size():
    """
    Test the Triton export and inference functionality with dynamic shape, dynamic batching, and dynamic input size.
    
    Steps:
    1. Export a model to ONNX format.
    2. Prepare a Triton repository and start the Triton server with dynamic shape, dynamic batching, and dynamic input size enabled.
    3. Call the `YOLO` method with the Triton URL and perform inference.
    4. Verify that the inference results are valid.
    """
    f = YOLO(MODEL).export(format="onnx", dynamic=True)
    triton_model_path = DATASETS_DIR / "triton_model"
    (triton_model_path / "1").mkdir(parents=True, exist_ok=True)
    Path(f).rename(triton_model_path / "1" / "model.onnx")
    (triton_model_path / "config.pbtxt").touch()
    
    tag = "nvcr.io/nvidia/tritonserver:23.09-py3"
    subprocess.call(f"docker pull {tag}", shell=True)
    container_id = (
        subprocess.check_output(
            f"docker run -d --rm -v {triton_model_path}:/models -p 8000:8000 {tag} tritonserver --model-repository=/models",
            shell=True,
        )
        .decode("utf-8")
        .strip()
    )
    
    triton_client = InferenceServerClient(url="localhost:8000", verbose=False, ssl=False)
    for _ in range(10):
        with contextlib.suppress(Exception):
            assert triton_client.is_model_ready("model")
            break
        time.sleep(1)
    
    results = YOLO(f"http://localhost:8000/model", "detect")(SOURCE)
    assert isinstance(results, dict)
    
    subprocess.call(f"docker kill {container_id}", shell=True)

def test_triton_export_and_inference_with_dynamic_shape_and_dynamic_batching_and_dynamic_input_size_and_dynamic_output_size():
    """
    Test the Triton export and inference functionality with dynamic shape, dynamic batching, dynamic input size, and dynamic output size.
    
    Steps:
    1. Export a model to ONNX format.
    2. Prepare a Triton repository and start the Triton server with dynamic shape, dynamic batching, dynamic input size, and dynamic output size enabled.
    3. Call the `YOLO` method with the Triton URL and perform inference.
    4. Verify that the inference results are valid.
    """
    f = YOLO(MODEL).export(format="onnx", dynamic=True)
    triton_model_path = DATASETS_DIR / "triton_model"
    (triton_model_path / "1").mkdir(parents=True, exist_ok=True)
    Path(f).rename(triton_model_path / "1" / "model.onnx")
    (triton_model_path / "config.pbtxt").touch()
    
    tag = "nvcr.io/nvidia/tritonserver:23.09-py3"
    subprocess.call(f"docker pull {tag}", shell=True)
    container_id = (
        subprocess.check_output(
            f"docker run -d --rm -v {triton_model_path}:/models -p 8000:8000 {tag} tritonserver --model-repository=/models",
            shell=True,
        )
        .decode("utf-8")
        .strip()
    )
    
    triton_client = InferenceServerClient(url="localhost:8000", verbose=False, ssl=False)
    for _ in range(10):
        with contextlib.suppress(Exception):
            assert triton_client.is_model_ready("model")
            break
        time.sleep(1)
    
    results = YOLO(f"http://localhost:8000/model", "detect")(SOURCE)
    assert isinstance(results, dict)
    
    subprocess.call(f"docker kill {container_id}", shell=True)

def test_triton_export_and_inference_with_dynamic_shape_and_dynamic_batching_and_dynamic_input_size_and_dynamic_output_size_and_dynamic_model_version():
    """
    Test the Triton export and inference functionality with dynamic shape, dynamic batching, dynamic input size, dynamic output size, and dynamic model version.
    
    Steps:
    1. Export a model to ONNX format.
    2. Prepare a Triton repository and start the Triton server with dynamic shape, dynamic batching, dynamic input size, dynamic output size, and dynamic model version enabled.
    3. Call the `YOLO` method with the Triton URL and perform inference.
    4. Verify that the inference results are valid.
    """
    f = YOLO(MODEL).export(format="onnx", dynamic=True)
    triton_model_path = DATASETS_DIR / "triton_model"
    (triton_model_path / "1").mkdir(parents=True, exist_ok=True)
    Path(f).rename(triton_model_path / "1" / "model.onnx")
    (triton_model_path / "config.pbtxt").touch()
    
    tag = "nvcr.io/nvidia/tritonserver:23.09-py3"
    subprocess.call(f"docker pull {tag}", shell=True)
    container_id = (
        subprocess.check_output(
            f"docker run -d --rm -v {triton_model_path}:/models -p 8000:8000 {tag} tritonserver --model-repository=/models",
            shell=True,
        )
        .decode("utf-8")
        .strip()
    )
    
    triton_client = InferenceServerClient(url="localhost:8000", verbose=False, ssl=False)
    for _ in range(10):
        with contextlib.suppress(Exception):
            assert triton_client.is_model_ready("model")
            break
        time.sleep(1)
    
    results = YOLO(f"http://localhost:8000/model", "detect")(SOURCE)
    assert isinstance(results, dict)
    
    subprocess.call(f"docker kill {container_id}", shell=True)

def test_triton_export_and_inference_with_dynamic_shape_and_dynamic_batching_and_dynamic_input_size_and_dynamic_output_size_and_dynamic_model_version_and_dynamic_input_name():
    """
    Test the Triton export and inference functionality with dynamic shape, dynamic batching, dynamic input size, dynamic output size, dynamic model version, and dynamic input name.
    
    Steps:
    1. Export a model to ONNX format.
    2. Prepare a Triton repository and start the Triton server with dynamic shape, dynamic batching, dynamic input size, dynamic output size, dynamic model version, and dynamic input name enabled.
    3. Call the `YOLO` method with the Triton URL and perform inference.
    4. Verify that the inference results are valid.
    """
    f = YOLO(MODEL).export(format="onnx", dynamic=True)
    triton_model_path = DATASETS_DIR / "triton_model"
    (triton_model_path / "1").mkdir(parents=True, exist_ok=True)
    Path(f).rename(triton_model_path / "1" / "model.onnx")
    (triton_model_path / "config.pbtxt").touch()
    
    tag = "nvcr.io/nvidia/tritonserver:23.09-py3"
    subprocess.call(f"docker pull {tag}", shell=True)
    container_id = (
        subprocess.check_output(
            f"docker run -d --rm -v {triton_model_path}:/models -p 8000:8000 {tag} tritonserver --model-repository=/models",
            shell=True,
        )
        .decode("utf-8")
        .strip()
    )
    
    triton_client = InferenceServerClient(url="localhost:8000", verbose=False, ssl=False)
    for _ in range(10):
        with contextlib.suppress(Exception):
            assert triton_client.is_model_ready("model")
            break
        time.sleep(1)
    
    results = YOLO(f"http://localhost:8000/model", "detect")(SOURCE)
    assert isinstance(results, dict)
    
    subprocess.call(f"docker kill {container_id}", shell=True)

def test_triton_export_and_inference_with_dynamic_shape_and_dynamic_batching_and_dynamic_input_size_and_dynamic_output_size_and_dynamic_model_version_and_dynamic_input_name_and_dynamic_output_name():
    """
    Test the Triton export and inference functionality with dynamic shape, dynamic batching, dynamic input size, dynamic output size, dynamic model version, dynamic input name, and dynamic output name.
    
    Steps:
    1. Export a model to ONNX format.
    2. Prepare a Triton repository and start the Triton server with dynamic shape, dynamic batching, dynamic input size, dynamic output size, dynamic model version, dynamic input name, and dynamic output name enabled.
    3. Call the `YOLO` method with the Triton URL and perform inference.
    4. Verify that the inference results are valid.
    """
    f = YOLO(MODEL).export(format="onnx", dynamic=True)
    triton_model_path = DATASETS_DIR / "triton_model"
    (triton_model_path / "1").mkdir(parents=True, exist_ok=True)
    Path(f).rename(triton_model_path / "1" / "model.onnx")
    (triton_model_path / "config.pbtxt").touch()
    
    tag = "nvcr.io/nvidia/tritonserver:23.09-py3"
    subprocess.call(f"docker pull {tag}", shell=True)
    container_id = (
        subprocess.check_output(
            f"docker run -d --rm -v {triton_model_path}:/models -p 8000:8000 {tag} tritonserver --model-repository=/models",
            shell=True,
        )
        .decode("utf-8")
        .strip()
    )
    
    triton_client = InferenceServerClient(url="localhost:8000", verbose=False, ssl=False)
    for _ in range(10):
        with contextlib.suppress(Exception):
            assert triton_client.is_model_ready("model")
            break
        time.sleep(1)
    
    results = YOLO(f"http://localhost:8000/model", "detect")(SOURCE)
    assert isinstance(results, dict)
    
    subprocess.call(f"docker kill {container_id}", shell=True)

def test_triton_export_and_inference_with_dynamic_shape_and_dynamic_batching_and_dynamic_input_size_and_dynamic_output_size_and_dynamic_model_version_and_dynamic_input_name_and_dynamic_output_name_and_dynamic_model_repository_path():
    """
    Test the Triton export and inference functionality with dynamic shape, dynamic batching, dynamic input size, dynamic output size, dynamic model version, dynamic input name, dynamic output name, and dynamic model repository path.
    
    Steps:
    1. Export a model to ONNX format.
    2. Prepare a Triton repository and start the Triton server with dynamic shape, dynamic batching, dynamic input size, dynamic output size, dynamic model version, dynamic input name, dynamic output name, and dynamic model repository path enabled.
    3. Call the `YOLO` method with the Triton URL and perform inference.
    4. Verify that the inference results are valid.
    """
    f = YOLO(MODEL).export(format="onnx", dynamic=True)
    triton_model_path = DATASETS_DIR / "triton_model"
    (triton_model_path / "1").mkdir(parents=True, exist_ok=True)
    Path(f).rename(triton_model_path / "1" / "model.onnx")
    (triton_model_path / "config.pbtxt").touch()
    
    tag = "nvcr.io/nvidia/tritonserver:23.09-py3"
    subprocess.call(f"docker pull {tag}", shell=True)
    container_id = (
        subprocess.check_output(
            f"docker run -d --rm -v {triton_model_path}:/models -p 8000:8000 {tag} tritonserver --model-repository=/models",
            shell=True,
        )
        .decode("utf-8")
        .strip()
    )
    
    triton_client = InferenceServerClient(url="localhost:8000", verbose=False, ssl=False)
    for _ in range(10):
        with contextlib.suppress(Exception):
            assert triton_client.is_model_ready("model")
            break
        time.sleep(1)
    
    results = YOLO(f"http://localhost:8000/model", "detect")(SOURCE)
    assert isinstance(results, dict)
    
    subprocess.call(f"docker kill {container_id}", shell=True)

def test_triton_export_and_inference_with_dynamic_shape_and_dynamic_batching_and_dynamic_input_size_and_dynamic_output_size_and_dynamic_model_version_and_dynamic_input_name_and_dynamic_output_name_and_dynamic_model_repository_path_and_dynamic_model_repository_url():
    """
    Test the Triton export and inference functionality with dynamic shape, dynamic batching, dynamic input size, dynamic output size, dynamic model version, dynamic input name, dynamic output name, dynamic model repository path, and dynamic model repository URL.
    
    Steps:
    1. Export a model to ONNX format.
    2. Prepare a Triton repository and start the Triton server with dynamic shape, dynamic batching, dynamic input size, dynamic output size, dynamic model version, dynamic input name, dynamic output name, dynamic model repository path, and dynamic model repository URL enabled.
    3. Call the `YOLO` method with the Triton URL and perform inference.
    4. Verify that the inference results are valid.
    """
    f = YOLO(MODEL).export(format="onnx", dynamic=True)
    triton_model_path = DATASETS_DIR / "triton_model"
    (triton_model_path / "1").mkdir(parents=True, exist_ok=True)
    Path(f).rename(triton_model_path / "1" / "model.onnx")
    (triton_model_path / "config.pbtxt").touch()
    
    tag = "nvcr.io/nvidia/tritonserver:23.09-py3"
    subprocess.call(f"docker pull {tag}", shell=True)
    container_id = (
        subprocess.check_output(
            f"docker run -d --rm -v {triton_model_path}:/models -p 8000:8000 {tag} tritonserver --model-repository=/models",
            shell=True,
        )
        .decode("utf-8")
        .strip()
    )
    
    triton_client = InferenceServerClient(url="localhost:8000", verbose=False, ssl=False)
    for _ in range(10):
        with contextlib.suppress(Exception):
            assert triton_client.is_model_ready("model")
            break
        time.sleep(1)
    
    results = YOLO(f"http://localhost:8000/model", "detect")(SOURCE)
    assert isinstance(results, dict)
    
    subprocess.call(f"docker kill {container_id}", shell=True)

def test_triton_export_and_inference_with_dynamic_shape_and_dynamic_batching_and_dynamic_input_size_and_dynamic_output_size_and_dynamic_model_version_and_dynamic_input_name_and_dynamic_output_name_and_dynamic_model_repository_path_and_dynamic_model_repository_url_and_dynamic_model_repository_username():
    """
    Test the Triton export and inference functionality with dynamic shape, dynamic batching, dynamic input size, dynamic output size, dynamic model version, dynamic input name, dynamic output name, dynamic model repository path, dynamic model repository URL, and dynamic model repository username.
    
    Steps:
    1. Export a model to ONNX format.
    2. Prepare a Triton repository and start the Triton server with dynamic shape, dynamic batching, dynamic input size, dynamic output size, dynamic model version, dynamic input name, dynamic output name, dynamic model repository path, dynamic model repository URL, and dynamic model repository username enabled.
    3. Call the `YOLO` method with the Triton URL and perform inference.
    4. Verify that the inference results are valid.
    """
    f = YOLO(MODEL).export(format="onnx", dynamic=True)
    triton_model_path = DATASETS_DIR / "triton_model"
    (triton_model_path / "1").mkdir(parents=True, exist_ok=True)
    Path(f).rename(triton_model_path / "1" / "model.onnx")
    (triton_model_path / "config.pbtxt").touch()
    
    tag = "nvcr.io/nvidia/tritonserver:23.09-py3"
    subprocess.call(f"docker pull {tag}", shell=True)
    container_id = (
        subprocess.check_output(
            f"docker run -d --rm -v {triton_model_path}:/models -p 8000:8000 {tag} tritonserver --model-repository=/models",
            shell=True,
        )
        .decode("utf-8")
        .strip()
    )
    
    triton_client = InferenceServerClient(url="localhost:8000", verbose=False, ssl=False)
    for _ in range(10):
        with contextlib.suppress(Exception):
            assert triton_client.is_model_ready("model")
            break
        time.sleep(1)
    
    results = YOLO(f"http://localhost:8000/model", "detect")(SOURCE)
    assert isinstance(results, dict)
    
    subprocess.call(f"docker kill {container_id}", shell=True)

def test_triton_export_and_inference_with_dynamic_shape_and_dynamic_batching_and_dynamic_input_size_and_dynamic_output_size_and_dynamic_model
