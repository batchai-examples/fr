import pytest

def test_help():
    """Test the 'yolo help' command."""
    # Execute the command and check if it runs without errors
    run("yolo help")

def test_checks():
    """Test the 'yolo checks' command."""
    # Execute the command and check if it runs without errors
    run("yolo checks")

def test_version():
    """Test the 'yolo version' command."""
    # Execute the command and check if it runs without errors
    run("yolo version")

def test_settings_reset():
    """Test the 'yolo settings reset' command."""
    # Execute the command and check if it runs without errors
    run("yolo settings reset")

def test_cfg():
    """Test the 'yolo cfg' command."""
    # Execute the command and check if it runs without errors
    run("yolo cfg")

@pytest.mark.parametrize("task,model,data", TASK_MODEL_DATA)
def test_train_positive(task, model, data):
    """Test YOLO training for a given task, model, and data with positive parameters."""
    # Execute the command and check if it runs without errors
    run(f"yolo train {task} model={model} data={data} imgsz=32 epochs=1 cache=disk")

@pytest.mark.parametrize("task,model,data", TASK_MODEL_DATA)
def test_train_negative(task, model, data):
    """Test YOLO training for a given task, model, and data with negative parameters."""
    # Execute the command with invalid parameters and check if it raises an error
    with pytest.raises(subprocess.CalledProcessError):
        run(f"yolo train {task} model={model} data={data} imgsz=-1 epochs=0 cache=disk")

@pytest.mark.parametrize("task,model,data", TASK_MODEL_DATA)
def test_val_positive(task, model, data):
    """Test YOLO validation for a given task, model, and data with positive parameters."""
    # Execute the command and check if it runs without errors
    run(f"yolo val {task} model={model} data={data} imgsz=32 batch_size=1")

@pytest.mark.parametrize("task,model,data", TASK_MODEL_DATA)
def test_val_negative(task, model, data):
    """Test YOLO validation for a given task, model, and data with negative parameters."""
    # Execute the command with invalid parameters and check if it raises an error
    with pytest.raises(subprocess.CalledProcessError):
        run(f"yolo val {task} model={model} data={data} imgsz=-1 batch_size=0")

@pytest.mark.parametrize("task,model,data", TASK_MODEL_DATA)
def test_predict_positive(task, model, data):
    """Test YOLO prediction for a given task, model, and data with positive parameters."""
    # Execute the command and check if it runs without errors
    run(f"yolo predict {task} model={model} source=data/images/bus.jpg imgsz=32")

@pytest.mark.parametrize("task,model,data", TASK_MODEL_DATA)
def test_predict_negative(task, model, data):
    """Test YOLO prediction for a given task, model, and data with negative parameters."""
    # Execute the command with invalid parameters and check if it raises an error
    with pytest.raises(subprocess.CalledProcessError):
        run(f"yolo predict {task} model={model} source=data/images/bus.jpg imgsz=-1")

def test_train_gpu_positive():
    """Test YOLO training on GPU(s) for various tasks and models with positive parameters."""
    # Execute the command and check if it runs without errors
    run(f"yolo train task=detect model=yolov8n data=coco128 imgsz=32 epochs=1 device=0,1")

def test_train_gpu_negative():
    """Test YOLO training on GPU(s) for various tasks and models with negative parameters."""
    # Execute the command with invalid parameters and check if it raises an error
    with pytest.raises(subprocess.CalledProcessError):
        run(f"yolo train task=detect model=yolov8n data=coco128 imgsz=-1 epochs=0 device=0,1")
