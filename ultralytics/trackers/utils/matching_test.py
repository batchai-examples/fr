import numpy as np
from ultralytics.trackers.utils.matching import linear_assignment, iou_distance, embedding_distance, fuse_score

def test_linear_assignment_happy_path():
    """
    Test the linear_assignment function with a happy path scenario.
    
    Steps:
        1. Create a cost matrix with valid values.
        2. Call the linear_assignment function with the cost matrix and threshold.
        3. Verify that the returned matches, unmatched_a, and unmatched_b are as expected.
    """
    cost_matrix = np.array([[0.1, 0.5], [0.4, 0.2]])
    thresh = 0.3
    matches, unmatched_a, unmatched_b = linear_assignment(cost_matrix, thresh)
    assert (matches == [[0, 1]]).all()
    assert (unmatched_a == [1]).all()
    assert (unmatched_b == [0]).all()

def test_linear_assignment_negative_threshold():
    """
    Test the linear_assignment function with a negative threshold.
    
    Steps:
        1. Create a cost matrix with valid values.
        2. Call the linear_assignment function with the cost matrix and a negative threshold.
        3. Verify that the returned matches, unmatched_a, and unmatched_b are as expected.
    """
    cost_matrix = np.array([[0.1, 0.5], [0.4, 0.2]])
    thresh = -0.1
    matches, unmatched_a, unmatched_b = linear_assignment(cost_matrix, thresh)
    assert (matches == [[0, 1]]).all()
    assert (unmatched_a == [1]).all()
    assert (unmatched_b == [0]).all()

def test_linear_assignment_empty_cost_matrix():
    """
    Test the linear_assignment function with an empty cost matrix.
    
    Steps:
        1. Create an empty cost matrix.
        2. Call the linear_assignment function with the empty cost matrix and a threshold.
        3. Verify that the returned matches, unmatched_a, and unmatched_b are as expected.
    """
    cost_matrix = np.array([])
    thresh = 0.3
    matches, unmatched_a, unmatched_b = linear_assignment(cost_matrix, thresh)
    assert (matches == []).all()
    assert (unmatched_a == []).all()
    assert (unmatched_b == []).all()

def test_iou_distance_happy_path():
    """
    Test the iou_distance function with a happy path scenario.
    
    Steps:
        1. Create bounding boxes for tracks and detections.
        2. Call the iou_distance function with the bounding boxes.
        3. Verify that the returned cost matrix is as expected.
    """
    tracks = [[0, 0, 1, 1], [2, 2, 3, 3]]
    detections = [[0.5, 0.5, 1.5, 1.5], [2.5, 2.5, 3.5, 3.5]]
    cost_matrix = iou_distance(tracks, detections)
    expected_cost_matrix = np.array([[0.5, 0.5], [0.5, 0.5]])
    assert (cost_matrix == expected_cost_matrix).all()

def test_embedding_distance_happy_path():
    """
    Test the embedding_distance function with a happy path scenario.
    
    Steps:
        1. Create embeddings for tracks and detections.
        2. Call the embedding_distance function with the embeddings.
        3. Verify that the returned cost matrix is as expected.
    """
    tracks = [[0.1, 0.2], [0.3, 0.4]]
    detections = [[0.5, 0.6], [0.7, 0.8]]
    cost_matrix = embedding_distance(tracks, detections)
    expected_cost_matrix = np.array([[0.44721359, 0.44721359], [0.44721359, 0.44721359]])
    assert (cost_matrix == expected_cost_matrix).all()

def test_fuse_score_happy_path():
    """
    Test the fuse_score function with a happy path scenario.
    
    Steps:
        1. Create a cost matrix and detection scores.
        2. Call the fuse_score function with the cost matrix and detection scores.
        3. Verify that the returned fused similarity matrix is as expected.
    """
    cost_matrix = np.array([[0.1, 0.5], [0.4, 0.2]])
    detections = [{'score': 0.8}, {'score': 0.6}]
    fused_similarity_matrix = fuse_score(cost_matrix, detections)
    expected_fused_similarity_matrix = np.array([[0.79200001, 0.5], [0.4, 0.32]])
    assert (fused_similarity_matrix == expected_fused_similarity_matrix).all()

def test_fuse_score_empty_cost_matrix():
    """
    Test the fuse_score function with an empty cost matrix.
    
    Steps:
        1. Create an empty cost matrix and detection scores.
        2. Call the fuse_score function with the empty cost matrix and detection scores.
        3. Verify that the returned fused similarity matrix is as expected.
    """
    cost_matrix = np.array([])
    detections = [{'score': 0.8}, {'score': 0.6}]
    fused_similarity_matrix = fuse_score(cost_matrix, detections)
    assert (fused_similarity_matrix == []).all()
