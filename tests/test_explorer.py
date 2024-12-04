import pytest
from ultralytics import Explorer
from ultralytics.utils import ASSETS

def test_similarity_happy_path():
    """Test similarity calculations and SQL queries for correctness and response length."""
    exp = Explorer(data="coco8.yaml")
    exp.create_embeddings_table()
    similar = exp.get_similar(idx=1)
    assert len(similar) == 4
    similar = exp.get_similar(img=ASSETS / "bus.jpg")
    assert len(similar) == 4
    similar = exp.get_similar(idx=[1, 2], limit=2)
    assert len(similar) == 2
    sim_idx = exp.similarity_index()
    assert len(sim_idx) == 4
    sql = exp.sql_query("WHERE labels LIKE '%zebra%'")
    assert len(sql) == 1

def test_similarity_negative_path():
    """Test similarity calculations with invalid input."""
    exp = Explorer(data="coco8.yaml")
    exp.create_embeddings_table()
    similar = exp.get_similar(idx=-1)
    assert len(similar) == 0
    similar = exp.get_similar(img=ASSETS / "nonexistent.jpg")
    assert len(similar) == 0
    similar = exp.get_similar(idx=[-1, -2], limit=2)
    assert len(similar) == 0

def test_det_happy_path():
    """Test detection functionalities and ensure the embedding table has bounding boxes."""
    exp = Explorer(data="coco8.yaml", model="yolov8n.pt")
    exp.create_embeddings_table(force=True)
    assert len(exp.table.head()["bboxes"]) > 0
    similar = exp.get_similar(idx=[1, 2], limit=10)
    assert len(similar) > 0
    # This is a loose test, just checks errors not correctness
    similar = exp.plot_similar(idx=[1, 2], limit=10)
    assert isinstance(similar, PIL.Image.Image)

def test_det_negative_path():
    """Test detection functionalities with invalid input."""
    exp = Explorer(data="coco8.yaml", model="yolov8n.pt")
    exp.create_embeddings_table(force=True)
    similar = exp.get_similar(idx=-1)
    assert len(similar) == 0
    # This is a loose test, just checks errors not correctness
    similar = exp.plot_similar(idx=[-1, -2], limit=10)
    assert isinstance(similar, PIL.Image.Image)

def test_seg_happy_path():
    """Test segmentation functionalities and verify the embedding table includes masks."""
    exp = Explorer(data="coco8-seg.yaml", model="yolov8n-seg.pt")
    exp.create_embeddings_table(force=True)
    assert len(exp.table.head()["masks"]) > 0
    similar = exp.get_similar(idx=[1, 2], limit=10)
    assert len(similar) > 0
    similar = exp.plot_similar(idx=[1, 2], limit=10)
    assert isinstance(similar, PIL.Image.Image)

def test_seg_negative_path():
    """Test segmentation functionalities with invalid input."""
    exp = Explorer(data="coco8-seg.yaml", model="yolov8n-seg.pt")
    exp.create_embeddings_table(force=True)
    similar = exp.get_similar(idx=-1)
    assert len(similar) == 0
    # This is a loose test, just checks errors not correctness
    similar = exp.plot_similar(idx=[-1, -2], limit=10)
    assert isinstance(similar, PIL.Image.Image)

def test_pose_happy_path():
    """Test pose estimation functionalities and check the embedding table for keypoints."""
    exp = Explorer(data="coco8-pose.yaml", model="yolov8n-pose.pt")
    exp.create_embeddings_table(force=True)
    assert len(exp.table.head()["keypoints"]) > 0
    similar = exp.get_similar(idx=[1, 2], limit=10)
    assert len(similar) > 0
    similar = exp.plot_similar(idx=[1, 2], limit=10)
    assert isinstance(similar, PIL.Image.Image)

def test_pose_negative_path():
    """Test pose estimation functionalities with invalid input."""
    exp = Explorer(data="coco8-pose.yaml", model="yolov8n-pose.pt")
    exp.create_embeddings_table(force=True)
    similar = exp.get_similar(idx=-1)
    assert len(similar) == 0
    # This is a loose test, just checks errors not correctness
    similar = exp.plot_similar(idx=[-1, -2], limit=10)
    assert isinstance(similar, PIL.Image.Image)

def test_similarity_edge_case():
    """Test similarity calculations with edge cases."""
    exp = Explorer(data="coco8.yaml")
    exp.create_embeddings_table()
    similar = exp.get_similar(idx=0)
    assert len(similar) == 4
    similar = exp.get_similar(img=ASSETS / "person.jpg")
    assert len(similar) == 4

def test_det_edge_case():
    """Test detection functionalities with edge cases."""
    exp = Explorer(data="coco8.yaml", model="yolov8n.pt")
    exp.create_embeddings_table(force=True)
    similar = exp.get_similar(idx=0, limit=1)
    assert len(similar) == 1
    # This is a loose test, just checks errors not correctness
    similar = exp.plot_similar(idx=0, limit=1)
    assert isinstance(similar, PIL.Image.Image)

def test_seg_edge_case():
    """Test segmentation functionalities with edge cases."""
    exp = Explorer(data="coco8-seg.yaml", model="yolov8n-seg.pt")
    exp.create_embeddings_table(force=True)
    similar = exp.get_similar(idx=0, limit=1)
    assert len(similar) == 1
    similar = exp.plot_similar(idx=0, limit=1)
    assert isinstance(similar, PIL.Image.Image)

def test_pose_edge_case():
    """Test pose estimation functionalities with edge cases."""
    exp = Explorer(data="coco8-pose.yaml", model="yolov8n-pose.pt")
    exp.create_embeddings_table(force=True)
    similar = exp.get_similar(idx=0, limit=1)
    assert len(similar) == 1
    similar = exp.plot_similar(idx=0, limit=1)
    assert isinstance(similar, PIL.Image.Image)

def test_similarity_large_input():
    """Test similarity calculations with large input."""
    exp = Explorer(data="coco8.yaml")
    exp.create_embeddings_table()
    similar = exp.get_similar(idx=[i for i in range(100)])
    assert len(similar) == 4

def test_det_large_input():
    """Test detection functionalities with large input."""
    exp = Explorer(data="coco8.yaml", model="yolov8n.pt")
    exp.create_embeddings_table(force=True)
    similar = exp.get_similar(idx=[i for i in range(100)], limit=50)
    assert len(similar) == 50
    # This is a loose test, just checks errors not correctness
    similar = exp.plot_similar(idx=[i for i in range(100)], limit=50)
    assert isinstance(similar, PIL.Image.Image)

def test_seg_large_input():
    """Test segmentation functionalities with large input."""
    exp = Explorer(data="coco8-seg.yaml", model="yolov8n-seg.pt")
    exp.create_embeddings_table(force=True)
    similar = exp.get_similar(idx=[i for i in range(100)], limit=50)
    assert len(similar) == 50
    similar = exp.plot_similar(idx=[i for i in range(100)], limit=50)
    assert isinstance(similar, PIL.Image.Image)

def test_pose_large_input():
    """Test pose estimation functionalities with large input."""
    exp = Explorer(data="coco8-pose.yaml", model="yolov8n-pose.pt")
    exp.create_embeddings_table(force=True)
    similar = exp.get_similar(idx=[i for i in range(100)], limit=50)
    assert len(similar) == 50
    similar = exp.plot_similar(idx=[i for i in range(100)], limit=50)
    assert isinstance(similar, PIL.Image.Image)

def test_similarity_empty_input():
    """Test similarity calculations with empty input."""
    exp = Explorer(data="coco8.yaml")
    exp.create_embeddings_table()
    similar = exp.get_similar(idx=[])
    assert len(similar) == 0

def test_det_empty_input():
    """Test detection functionalities with empty input."""
    exp = Explorer(data="coco8.yaml", model="yolov8n.pt")
    exp.create_embeddings_table(force=True)
    similar = exp.get_similar(idx=[], limit=10)
    assert len(similar) == 0
    # This is a loose test, just checks errors not correctness
    similar = exp.plot_similar(idx=[], limit=10)
    assert isinstance(similar, PIL.Image.Image)

def test_seg_empty_input():
    """Test segmentation functionalities with empty input."""
    exp = Explorer(data="coco8-seg.yaml", model="yolov8n-seg.pt")
    exp.create_embeddings_table(force=True)
    similar = exp.get_similar(idx=[], limit=10)
    assert len(similar) == 0
    similar = exp.plot_similar(idx=[], limit=10)
    assert isinstance(similar, PIL.Image.Image)

def test_pose_empty_input():
    """Test pose estimation functionalities with empty input."""
    exp = Explorer(data="coco8-pose.yaml", model="yolov8n-pose.pt")
    exp.create_embeddings_table(force=True)
    similar = exp.get_similar(idx=[], limit=10)
    assert len(similar) == 0
    similar = exp.plot_similar(idx=[], limit=10)
    assert isinstance(similar, PIL.Image.Image)

def test_similarity_null_input():
    """Test similarity calculations with null input."""
    exp = Explorer(data="coco8.yaml")
    exp.create_embeddings_table()
    similar = exp.get_similar(idx=None)
    assert len(similar) == 0

def test_det_null_input():
    """Test detection functionalities with null input."""
    exp = Explorer(data="coco8.yaml", model="yolov8n.pt")
    exp.create_embeddings_table(force=True)
    similar = exp.get_similar(idx=None, limit=10)
    assert len(similar) == 0
    # This is a loose test, just checks errors not correctness
    similar = exp.plot_similar(idx=None, limit=10)
    assert isinstance(similar, PIL.Image.Image)

def test_seg_null_input():
    """Test segmentation functionalities with null input."""
    exp = Explorer(data="coco8-seg.yaml", model="yolov8n-seg.pt")
    exp.create_embeddings_table(force=True)
    similar = exp.get_similar(idx=None, limit=10)
    assert len(similar) == 0
    similar = exp.plot_similar(idx=None, limit=10)
    assert isinstance(similar, PIL.Image.Image)

def test_pose_null_input():
    """Test pose estimation functionalities with null input."""
    exp = Explorer(data="coco8-pose.yaml", model="yolov8n-pose.pt")
    exp.create_embeddings_table(force=True)
    similar = exp.get_similar(idx=None, limit=10)
    assert len(similar) == 0
    similar = exp.plot_similar(idx=None, limit=10)
    assert isinstance(similar, PIL.Image.Image)

def test_similarity_invalid_input():
    """Test similarity calculations with invalid input."""
    exp = Explorer(data="coco8.yaml")
    exp.create_embeddings_table()
    similar = exp.get_similar(idx="invalid")
    assert len(similar) == 0

def test_det_invalid_input():
    """Test detection functionalities with invalid input."""
    exp = Explorer(data="coco8.yaml", model="yolov8n.pt")
    exp.create_embeddings_table(force=True)
    similar = exp.get_similar(idx="invalid", limit=10)
    assert len(similar) == 0
    # This is a loose test, just checks errors not correctness
    similar = exp.plot_similar(idx="invalid", limit=10)
    assert isinstance(similar, PIL.Image.Image)

def test_seg_invalid_input():
    """Test segmentation functionalities with invalid input."""
    exp = Explorer(data="coco8-seg.yaml", model="yolov8n-seg.pt")
    exp.create_embeddings_table(force=True)
    similar = exp.get_similar(idx="invalid", limit=10)
    assert len(similar) == 0
    similar = exp.plot_similar(idx="invalid", limit=10)
    assert isinstance(similar, PIL.Image.Image)

def test_pose_invalid_input():
    """Test pose estimation functionalities with invalid input."""
    exp = Explorer(data="coco8-pose.yaml", model="yolov8n-pose.pt")
    exp.create_embeddings_table(force=True)
    similar = exp.get_similar(idx="invalid", limit=10)
    assert len(similar) == 0
    similar = exp.plot_similar(idx="invalid", limit=10)
    assert isinstance(similar, PIL.Image.Image)

def test_similarity_duplicate_input():
    """Test similarity calculations with duplicate input."""
    exp = Explorer(data="coco8.yaml")
    exp.create_embeddings_table()
    similar = exp.get_similar(idx=[1, 2, 3, 3])
    assert len(similar) == 4

def test_det_duplicate_input():
    """Test detection functionalities with duplicate input."""
    exp = Explorer(data="coco8.yaml", model="yolov8n.pt")
    exp.create_embeddings_table(force=True)
    similar = exp.get_similar(idx=[1, 2, 3, 3], limit=5)
    assert len(similar) == 5
    # This is a loose test, just checks errors not correctness
    similar = exp.plot_similar(idx=[1, 2, 3, 3], limit=5)
    assert isinstance(similar, PIL.Image.Image)

def test_seg_duplicate_input():
    """Test segmentation functionalities with duplicate input."""
    exp = Explorer(data="coco8-seg.yaml", model="yolov8n-seg.pt")
    exp.create_embeddings_table(force=True)
    similar = exp.get_similar(idx=[1, 2, 3, 3], limit=5)
    assert len(similar) == 5
    similar = exp.plot_similar(idx=[1, 2, 3, 3], limit=5)
    assert isinstance(similar, PIL.Image.Image)

def test_pose_duplicate_input():
    """Test pose estimation functionalities with duplicate input."""
    exp = Explorer(data="coco8-pose.yaml", model="yolov8n-pose.pt")
    exp.create_embeddings_table(force=True)
    similar = exp.get_similar(idx=[1, 2, 3, 3], limit=5)
    assert len(similar) == 5
    similar = exp.plot_similar(idx=[1, 2, 3, 3], limit=5)
    assert isinstance(similar, PIL.Image.Image)

def test_similarity_negative_limit():
    """Test similarity calculations with negative limit."""
    exp = Explorer(data="coco8.yaml")
    exp.create_embeddings_table()
    similar = exp.get_similar(idx=[1, 2, 3], limit=-1)
    assert len(similar) == 0

def test_det_negative_limit():
    """Test detection functionalities with negative limit."""
    exp = Explorer(data="coco8.yaml", model="yolov8n.pt")
    exp.create_embeddings_table(force=True)
    similar = exp.get_similar(idx=[1, 2, 3], limit=-1)
    assert len(similar) == 0
    # This is a loose test, just checks errors not correctness
    similar = exp.plot_similar(idx=[1, 2, 3], limit=-1)
    assert isinstance(similar, PIL.Image.Image)

def test_seg_negative_limit():
    """Test segmentation functionalities with negative limit."""
    exp = Explorer(data="coco8-seg.yaml", model="yolov8n-seg.pt")
    exp.create_embeddings_table(force=True)
    similar = exp.get_similar(idx=[1, 2, 3], limit=-1)
    assert len(similar) == 0
    similar = exp.plot_similar(idx=[1, 2, 3], limit=-1)
    assert isinstance(similar, PIL.Image.Image)

def test_pose_negative_limit():
    """Test pose estimation functionalities with negative limit."""
    exp = Explorer(data="coco8-pose.yaml", model="yolov8n-pose.pt")
    exp.create_embeddings_table(force=True)
    similar = exp.get_similar(idx=[1, 2, 3], limit=-1)
    assert len(similar) == 0
    similar = exp.plot_similar(idx=[1, 2, 3], limit=-1)
    assert isinstance(similar, PIL.Image.Image)

def test_similarity_zero_limit():
    """Test similarity calculations with zero limit."""
    exp = Explorer(data="coco8.yaml")
    exp.create_embeddings_table()
    similar = exp.get_similar(idx=[1, 2, 3], limit=0)
    assert len(similar) == 0

def test_det_zero_limit():
    """Test detection functionalities with zero limit."""
    exp = Explorer(data="coco8.yaml", model="yolov8n.pt")
    exp.create_embeddings_table(force=True)
    similar = exp.get_similar(idx=[1, 2, 3], limit=0)
    assert len(similar) == 0
    # This is a loose test, just checks errors not correctness
    similar = exp.plot_similar(idx=[1, 2, 3], limit=0)
    assert isinstance(similar, PIL.Image.Image)

def test_seg_zero_limit():
    """Test segmentation functionalities with zero limit."""
    exp = Explorer(data="coco8-seg.yaml", model="yolov8n-seg.pt")
    exp.create_embeddings_table(force=True)
    similar = exp.get_similar(idx=[1, 2, 3], limit=0)
    assert len(similar) == 0
    similar = exp.plot_similar(idx=[1, 2, 3], limit=0)
    assert isinstance(similar, PIL.Image.Image)

def test_pose_zero_limit():
    """Test pose estimation functionalities with zero limit."""
    exp = Explorer(data="coco8-pose.yaml", model="yolov8n-pose.pt")
    exp.create_embeddings_table(force=True)
    similar = exp.get_similar(idx=[1, 2, 3], limit=0)
    assert len(similar) == 0
    similar = exp.plot_similar(idx=[1, 2, 3], limit=0)
    assert isinstance(similar, PIL.Image.Image)

def test_similarity_large_limit():
    """Test similarity calculations with large limit."""
    exp = Explorer(data="coco8.yaml")
    exp.create_embeddings_table()
    similar = exp.get_similar(idx=[1, 2, 3], limit=100)
    assert len(similar) == 4

def test_det_large_limit():
    """Test detection functionalities with large limit."""
    exp = Explorer(data="coco8.yaml", model="yolov8n.pt")
    exp.create_embeddings_table(force=True)
    similar = exp.get_similar(idx=[1, 2, 3], limit=100)
    assert len(similar) == 4
    # This is a loose test, just checks errors not correctness
    similar = exp.plot_similar(idx=[1, 2, 3], limit=100)
    assert isinstance(similar, PIL.Image.Image)

def test_seg_large_limit():
    """Test segmentation functionalities with large limit."""
    exp = Explorer(data="coco8-seg.yaml", model="yolov8n-seg.pt")
    exp.create_embeddings_table(force=True)
    similar = exp.get_similar(idx=[1, 2, 3], limit=100)
    assert len(similar) == 4
    similar = exp.plot_similar(idx=[1, 2, 3], limit=100)
    assert isinstance(similar, PIL.Image.Image)

def test_pose_large_limit():
    """Test pose estimation functionalities with large limit."""
    exp = Explorer(data="coco8-pose.yaml", model="yolov8n-pose.pt")
    exp.create_embeddings_table(force=True)
    similar = exp.get_similar(idx=[1, 2, 3], limit=100)
    assert len(similar) == 4
    similar = exp.plot_similar(idx=[1, 2, 3], limit=100)
    assert isinstance(similar, PIL.Image.Image)

def test_similarity_invalid_limit():
    """Test similarity calculations with invalid limit."""
    exp = Explorer(data="coco8.yaml")
    exp.create_embeddings_table()
    similar = exp.get_similar(idx=[1, 2, 3], limit="invalid")
    assert len(similar) == 0

def test_det_invalid_limit():
    """Test detection functionalities with invalid limit."""
    exp = Explorer(data="coco8.yaml", model="yolov8n.pt")
    exp.create_embeddings_table(force=True)
    similar = exp.get_similar(idx=[1, 2, 3], limit="invalid")
    assert len(similar) == 0
    # This is a loose test, just checks errors not correctness
    similar = exp.plot_similar(idx=[1, 2, 3], limit="invalid")
    assert isinstance(similar, PIL.Image.Image)

def test_seg_invalid_limit():
    """Test segmentation functionalities with invalid limit."""
    exp = Explorer(data="coco8-seg.yaml", model="yolov8n-seg.pt")
    exp.create_embeddings_table(force=True)
    similar = exp.get_similar(idx=[1, 2, 3], limit="invalid")
    assert len(similar) == 0
    similar = exp.plot_similar(idx=[1, 2, 3], limit="invalid")
    assert isinstance(similar, PIL.Image.Image)

def test_pose_invalid_limit():
    """Test pose estimation functionalities with invalid limit."""
    exp = Explorer(data="coco8-pose.yaml", model="yolov8n-pose.pt")
    exp.create_embeddings_table(force=True)
    similar = exp.get_similar(idx=[1, 2, 3], limit="invalid")
    assert len(similar) == 0
    similar = exp.plot_similar(idx=[1, 2, 3], limit="invalid")
    assert isinstance(similar, PIL.Image.Image)

def test_similarity_empty_idx():
    """Test similarity calculations with empty idx."""
    exp = Explorer(data="coco8.yaml")
    exp.create_embeddings_table()
    similar = exp.get_similar(idx=[], limit=1)
    assert len(similar) == 0

def test_det_empty_idx():
    """Test detection functionalities with empty idx."""
    exp = Explorer(data="coco8.yaml", model="yolov8n.pt")
    exp.create_embeddings_table(force=True)
    similar = exp.get_similar(idx=[], limit=1)
    assert len(similar) == 0
    # This is a loose test, just checks errors not correctness
    similar = exp.plot_similar(idx=[], limit=1)
    assert isinstance(similar, PIL.Image.Image)

def test_seg_empty_idx():
    """Test segmentation functionalities with empty idx."""
    exp = Explorer(data="coco8-seg.yaml", model="yolov8n-seg.pt")
    exp.create_embeddings_table(force=True)
    similar = exp.get_similar(idx=[], limit=1)
    assert len(similar) == 0
    similar = exp.plot_similar(idx=[], limit=1)
    assert isinstance(similar, PIL.Image.Image)

def test_pose_empty_idx():
    """Test pose estimation functionalities with empty idx."""
    exp = Explorer(data="coco8-pose.yaml", model="yolov8n-pose.pt")
    exp.create_embeddings_table(force=True)
    similar = exp.get_similar(idx=[], limit=1)
    assert len(similar) == 0
    similar = exp.plot_similar(idx=[], limit=1)
    assert isinstance(similar, PIL.Image.Image)

def test_similarity_nonexistent_idx():
    """Test similarity calculations with nonexistent idx."""
    exp = Explorer(data="coco8.yaml")
    exp.create_embeddings_table()
    similar = exp.get_similar(idx=[999], limit=1)
    assert len(similar) == 0

def test_det_nonexistent_idx():
    """Test detection functionalities with nonexistent idx."""
    exp = Explorer(data="coco8.yaml", model="yolov8n.pt")
    exp.create_embeddings_table(force=True)
    similar = exp.get_similar(idx=[999], limit=1)
    assert len(similar) == 0
    # This is a loose test, just checks errors not correctness
    similar = exp.plot_similar(idx=[999], limit=1)
    assert isinstance(similar, PIL.Image.Image)

def test_seg_nonexistent_idx():
    """Test segmentation functionalities with nonexistent idx."""
    exp = Explorer(data="coco8-seg.yaml", model="yolov8n-seg.pt")
    exp.create_embeddings_table(force=True)
    similar = exp.get_similar(idx=[999], limit=1)
    assert len(similar) == 0
    similar = exp.plot_similar(idx=[999], limit=1)
    assert isinstance(similar, PIL.Image.Image)

def test_pose_nonexistent_idx():
    """Test pose estimation functionalities with nonexistent idx."""
    exp = Explorer(data="coco8-pose.yaml", model="yolov8n-pose.pt")
    exp.create_embeddings_table(force=True)
    similar = exp.get_similar(idx=[999], limit=1)
    assert len(similar) == 0
    similar = exp.plot_similar(idx=[999], limit=1)
    assert isinstance(similar, PIL.Image.Image)

def test_similarity_duplicate_idx():
    """Test similarity calculations with duplicate idx."""
    exp = Explorer(data="coco8.yaml")
    exp.create_embeddings_table()
    similar = exp.get_similar(idx=[0, 0], limit=2)
    assert len(similar) == 1

def test_det_duplicate_idx():
    """Test detection functionalities with duplicate idx."""
    exp = Explorer(data="coco8.yaml", model="yolov8n.pt")
    exp.create_embeddings_table(force=True)
    similar = exp.get_similar(idx=[0, 0], limit=2)
    assert len(similar) == 1
    # This is a loose test, just checks errors not correctness
    similar = exp.plot_similar(idx=[0, 0], limit=2)
    assert isinstance(similar, PIL.Image.Image)

def test_seg_duplicate_idx():
    """Test segmentation functionalities with duplicate idx."""
    exp = Explorer(data="coco8-seg.yaml", model="yolov8n-seg.pt")
    exp.create_embeddings_table(force=True)
    similar = exp.get_similar(idx=[0, 0], limit=2)
    assert len(similar) == 1
    similar = exp.plot_similar(idx=[0, 0], limit=2)
    assert isinstance(similar, PIL.Image.Image)

def test_pose_duplicate_idx():
    """Test pose estimation functionalities with duplicate idx."""
    exp = Explorer(data="coco8-pose.yaml", model="yolov8n-pose.pt")
    exp.create_embeddings_table(force=True)
    similar = exp.get_similar(idx=[0, 0], limit=2)
    assert len(similar) == 1
    similar = exp.plot_similar(idx=[0, 0], limit=2)
    assert isinstance(similar, PIL.Image.Image)

def test_similarity_large_limit():
    """Test similarity calculations with large limit."""
    exp = Explorer(data="coco8.yaml")
    exp.create_embeddings_table()
    similar = exp.get_similar(idx=[0], limit=1000)
    assert len(similar) == 999

def test_det_large_limit():
    """Test detection functionalities with large limit."""
    exp = Explorer(data="coco8.yaml", model="yolov8n.pt")
    exp.create_embeddings_table(force=True)
    similar = exp.get_similar(idx=[0], limit=1000)
    assert len(similar) == 999
    # This is a loose test, just checks errors not correctness
    similar = exp.plot_similar(idx=[0], limit=1000)
    assert isinstance(similar, PIL.Image.Image)

def test_seg_large_limit():
    """Test segmentation functionalities with large limit."""
    exp = Explorer(data="coco8-seg.yaml", model="yolov8n-seg.pt")
    exp.create_embeddings_table(force=True)
    similar = exp.get_similar(idx=[0], limit=1000)
    assert len(similar) == 999
    similar = exp.plot_similar(idx=[0], limit=1000)
    assert isinstance(similar, PIL.Image.Image)

def test_pose_large_limit():
    """Test pose estimation functionalities with large limit."""
    exp = Explorer(data="coco8-pose.yaml", model="yolov8n-pose.pt")
    exp.create_embeddings_table(force=True)
    similar = exp.get_similar(idx=[0], limit=1000)
    assert len(similar) == 999
    similar = exp.plot_similar(idx=[0], limit=1000)
    assert isinstance(similar, PIL.Image.Image)

def test_similarity_negative_limit():
    """Test similarity calculations with negative limit."""
    exp = Explorer(data="coco8.yaml")
    exp.create_embeddings_table()
    similar = exp.get_similar(idx=[0], limit=-1)
    assert len(similar) == 0

def test_det_negative_limit():
    """Test detection functionalities with negative limit."""
    exp = Explorer(data="coco8.yaml", model="yolov8n.pt")
    exp.create_embeddings_table(force=True)
    similar = exp.get_similar(idx=[0], limit=-1)
    assert len(similar) == 0
    # This is a loose test, just checks errors not correctness
    similar = exp.plot_similar(idx=[0], limit=-1)
    assert isinstance(similar, PIL.Image.Image)

def test_seg_negative_limit():
    """Test segmentation functionalities with negative limit."""
    exp = Explorer(data
