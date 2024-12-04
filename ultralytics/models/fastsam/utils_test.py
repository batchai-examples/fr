import torch
from ultralytics.models.fastsam.utils import adjust_bboxes_to_image_border, bbox_iou

def test_adjust_bboxes_to_image_border():
    """
    Test the adjust_bboxes_to_image_border function with various scenarios.
    """

    # Happy path: Boxes within threshold
    boxes = torch.tensor([[10, 10, 50, 50], [20, 20, 70, 70]])
    image_shape = (100, 100)
    adjusted_boxes = adjust_bboxes_to_image_border(boxes, image_shape)
    assert torch.equal(adjusted_boxes, torch.tensor([[10, 10, 50, 50], [20, 20, 70, 70]]))

    # Happy path: Boxes touching the border
    boxes = torch.tensor([[0, 0, 50, 50], [90, 90, 140, 140]])
    image_shape = (100, 100)
    adjusted_boxes = adjust_bboxes_to_image_border(boxes, image_shape)
    assert torch.equal(adjusted_boxes, torch.tensor([[0, 0, 50, 50], [90, 90, 100, 100]]))

    # Happy path: Boxes outside the border
    boxes = torch.tensor([[-10, -10, 60, 60], [110, 110, 150, 150]])
    image_shape = (100, 100)
    adjusted_boxes = adjust_bboxes_to_image_border(boxes, image_shape)
    assert torch.equal(adjusted_boxes, torch.tensor([[0, 0, 60, 60], [100, 100, 100, 100]]))

    # Negative case: Empty boxes
    boxes = torch.empty((0, 4))
    image_shape = (100, 100)
    adjusted_boxes = adjust_bboxes_to_image_border(boxes, image_shape)
    assert torch.equal(adjusted_boxes, torch.empty((0, 4)))

    # Negative case: Invalid image shape
    boxes = torch.tensor([[10, 10, 50, 50]])
    image_shape = (-1, -1)
    adjusted_boxes = adjust_bboxes_to_image_border(boxes, image_shape)
    assert torch.equal(adjusted_boxes, torch.tensor([[10, 10, 50, 50]]))

def test_bbox_iou():
    """
    Test the bbox_iou function with various scenarios.
    """

    # Happy path: IoU above threshold
    box1 = torch.tensor([20, 20, 40, 40])
    boxes = torch.tensor([[10, 10, 50, 50], [30, 30, 60, 60]])
    iou_thres = 0.9
    image_shape = (100, 100)
    high_iou_indices = bbox_iou(box1, boxes, iou_thres, image_shape)
    assert torch.equal(high_iou_indices, torch.tensor([1]))

    # Happy path: IoU below threshold
    box1 = torch.tensor([20, 20, 40, 40])
    boxes = torch.tensor([[10, 10, 50, 50], [30, 30, 60, 60]])
    iou_thres = 0.95
    image_shape = (100, 100)
    high_iou_indices = bbox_iou(box1, boxes, iou_thres, image_shape)
    assert torch.equal(high_iou_indices, torch.tensor([]))

    # Happy path: Empty boxes
    box1 = torch.empty((4,))
    boxes = torch.empty((0, 4))
    iou_thres = 0.9
    image_shape = (100, 100)
    high_iou_indices = bbox_iou(box1, boxes, iou_thres, image_shape)
    assert torch.equal(high_iou_indices, torch.tensor([]))

    # Negative case: Invalid box1 shape
    box1 = torch.empty((3,))
    boxes = torch.tensor([[10, 10, 50, 50]])
    iou_thres = 0.9
    image_shape = (100, 100)
    high_iou_indices = bbox_iou(box1, boxes, iou_thres, image_shape)
    assert torch.equal(high_iou_indices, torch.empty((0,)))

    # Negative case: Invalid boxes shape
    box1 = torch.tensor([20, 20, 40, 40])
    boxes = torch.empty((0,))
    iou_thres = 0.9
    image_shape = (100, 100)
    high_iou_indices = bbox_iou(box1, boxes, iou_thres, image_shape)
    assert torch.equal(high_iou_indices, torch.tensor([]))

if __name__ == "__main__":
    test_adjust_bboxes_to_image_border()
    test_bbox_iou()
