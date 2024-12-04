import unittest
from ultralytics.models import YOLO, RTDETR, SAM, FastSAM, NAS, YOLOWorld

class TestModels(unittest.TestCase):
    def test_yolo_creation(self):
        """
        Test the creation of a YOLO model.
        
        Steps:
        1. Create an instance of YOLO.
        2. Verify that the instance is not None.
        """
        yolo = YOLO()
        self.assertIsNotNone(yolo)

    def test_rtdetr_creation(self):
        """
        Test the creation of a RTDETR model.
        
        Steps:
        1. Create an instance of RTDETR.
        2. Verify that the instance is not None.
        """
        rtdetr = RTDETR()
        self.assertIsNotNone(rtdetr)

    def test_sam_creation(self):
        """
        Test the creation of a SAM model.
        
        Steps:
        1. Create an instance of SAM.
        2. Verify that the instance is not None.
        """
        sam = SAM()
        self.assertIsNotNone(sam)

    def test_fastsam_creation(self):
        """
        Test the creation of a FastSAM model.
        
        Steps:
        1. Create an instance of FastSAM.
        2. Verify that the instance is not None.
        """
        fastsam = FastSAM()
        self.assertIsNotNone(fastsam)

    def test_nas_creation(self):
        """
        Test the creation of a NAS model.
        
        Steps:
        1. Create an instance of NAS.
        2. Verify that the instance is not None.
        """
        nas = NAS()
        self.assertIsNotNone(nas)

    def test_yoloworld_creation(self):
        """
        Test the creation of a YOLOWorld model.
        
        Steps:
        1. Create an instance of YOLOWorld.
        2. Verify that the instance is not None.
        """
        yoloworld = YOLOWorld()
        self.assertIsNotNone(yoloworld)

if __name__ == '__main__':
    unittest.main()
