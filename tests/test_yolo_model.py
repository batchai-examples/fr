import pytest
from ultralytics.models.yolo.model import YOLO, YOLOWorld

# Test case for initializing a regular YOLO model with default parameters
def test_init_regular_yolo():
    """
    Test the initialization of a regular YOLO model with default parameters.
    
    Steps:
    1. Create an instance of YOLO with default parameters.
    2. Verify that the task is set to 'detect'.
    3. Verify that the model path is correctly initialized.
    """
    yolo_model = YOLO()
    assert yolo_model.task == "detect"
    assert isinstance(yolo_model.model, DetectionModel)

# Test case for initializing a regular YOLO model with custom parameters
def test_init_regular_yolo_custom_params():
    """
    Test the initialization of a regular YOLO model with custom parameters.
    
    Steps:
    1. Create an instance of YOLO with a custom model path and task.
    2. Verify that the task is set to 'segment'.
    3. Verify that the model path is correctly initialized.
    """
    yolo_model = YOLO(model="custom_yolov8n.pt", task="segment")
    assert yolo_model.task == "segment"
    assert isinstance(yolo_model.model, SegmentationModel)

# Test case for initializing a YOLOWorld model with default parameters
def test_init_yoloworld():
    """
    Test the initialization of a YOLOWorld model with default parameters.
    
    Steps:
    1. Create an instance of YOLOWorld with default parameters.
    2. Verify that the task is set to 'detect'.
    3. Verify that the model path is correctly initialized.
    """
    yoloworld_model = YOLOWorld()
    assert yoloworld_model.task == "detect"
    assert isinstance(yoloworld_model.model, WorldModel)

# Test case for initializing a YOLOWorld model with custom parameters
def test_init_yoloworld_custom_params():
    """
    Test the initialization of a YOLOWorld model with custom parameters.
    
    Steps:
    1. Create an instance of YOLOWorld with a custom model path and task.
    2. Verify that the task is set to 'segment'.
    3. Verify that the model path is correctly initialized.
    """
    yoloworld_model = YOLOWorld(model="custom_yolov8s-world.pt", task="segment")
    assert yoloworld_model.task == "segment"
    assert isinstance(yoloworld_model.model, SegmentationModel)

# Test case for setting classes in a YOLO model
def test_set_classes_regular_yolo():
    """
    Test the set_classes method of a regular YOLO model.
    
    Steps:
    1. Create an instance of YOLO with default parameters.
    2. Call the set_classes method with custom class names.
    3. Verify that the model's names are updated correctly.
    """
    yolo_model = YOLO()
    classes = ["person", "car"]
    yolo_model.set_classes(classes)
    assert yolo_model.model.names == classes

# Test case for setting classes in a YOLOWorld model
def test_set_classes_yoloworld():
    """
    Test the set_classes method of a YOLOWorld model.
    
    Steps:
    1. Create an instance of YOLOWorld with default parameters.
    2. Call the set_classes method with custom class names.
    3. Verify that the model's names are updated correctly.
    """
    yoloworld_model = YOLOWorld()
    classes = ["person", "car"]
    yoloworld_model.set_classes(classes)
    assert yoloworld_model.model.names == classes

# Test case for initializing a regular YOLO model with a world model file
def test_init_regular_yolo_world_model():
    """
    Test the initialization of a regular YOLO model with a world model file.
    
    Steps:
    1. Create an instance of YOLO with a world model file.
    2. Verify that the task is set to 'detect'.
    3. Verify that the model path is correctly initialized.
    """
    yolo_model = YOLO(model="custom_yolov8n_world.pt")
    assert yolo_model.task == "detect"
    assert isinstance(yolo_model.model, WorldModel)

# Test case for initializing a regular YOLO model with an invalid task
def test_init_regular_yolo_invalid_task():
    """
    Test the initialization of a regular YOLO model with an invalid task.
    
    Steps:
    1. Create an instance of YOLO with an invalid task parameter.
    2. Verify that an exception is raised.
    """
    with pytest.raises(ValueError):
        yolo_model = YOLO(task="invalid_task")

# Test case for initializing a YOLOWorld model with an invalid task
def test_init_yoloworld_invalid_task():
    """
    Test the initialization of a YOLOWorld model with an invalid task.
    
    Steps:
    1. Create an instance of YOLOWorld with an invalid task parameter.
    2. Verify that an exception is raised.
    """
    with pytest.raises(ValueError):
        yoloworld_model = YOLOWorld(task="invalid_task")
