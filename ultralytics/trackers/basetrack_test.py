import unittest
from ultralytics.trackers.basetrack import TrackState, BaseTrack

class TestBaseTrack(unittest.TestCase):
    def setUp(self):
        self.track = BaseTrack()

    def test_initialization(self):
        """
        Test the initialization of a BaseTrack object.
        
        Steps:
        1. Create an instance of BaseTrack.
        2. Verify that track_id is set to 0.
        3. Verify that is_activated is False.
        4. Verify that state is TrackState.New.
        5. Verify that history is an empty OrderedDict.
        6. Verify that features is an empty list.
        7. Verify that curr_feature is None.
        8. Verify that score is 0.
        9. Verify that start_frame is 0.
        10. Verify that frame_id is 0.
        11. Verify that time_since_update is 0.
        12. Verify that location is (np.inf, np.inf).
        """
        self.assertEqual(self.track.track_id, 0)
        self.assertFalse(self.track.is_activated)
        self.assertEqual(self.track.state, TrackState.New)
        self.assertTrue(isinstance(self.track.history, OrderedDict))
        self.assertEqual(len(self.track.history), 0)
        self.assertEqual(self.track.features, [])
        self.assertIsNone(self.track.curr_feature)
        self.assertEqual(self.track.score, 0)
        self.assertEqual(self.track.start_frame, 0)
        self.assertEqual(self.track.frame_id, 0)
        self.assertEqual(self.track.time_since_update, 0)
        self.assertEqual(self.track.location, (np.inf, np.inf))

    def test_next_id(self):
        """
        Test the next_id method of BaseTrack.
        
        Steps:
        1. Call next_id method twice.
        2. Verify that the second call returns a value greater than the first call.
        """
        id1 = BaseTrack.next_id()
        id2 = BaseTrack.next_id()
        self.assertTrue(id2 > id1)

    def test_activate(self):
        """
        Test the activate method of BaseTrack.
        
        Steps:
        1. Call activate method with some arguments.
        2. Verify that is_activated is True.
        """
        self.track.activate(1, 2, 3)
        self.assertTrue(self.track.is_activated)

    def test_predict(self):
        """
        Test the predict method of BaseTrack.
        
        Steps:
        1. Call predict method.
        2. Verify that an exception is raised (since it's abstract).
        """
        with self.assertRaises(NotImplementedError):
            self.track.predict()

    def test_update(self):
        """
        Test the update method of BaseTrack.
        
        Steps:
        1. Call update method with some arguments.
        2. Verify that an exception is raised (since it's abstract).
        """
        with self.assertRaises(NotImplementedError):
            self.track.update(1, 2, 3)

    def test_mark_lost(self):
        """
        Test the mark_lost method of BaseTrack.
        
        Steps:
        1. Call mark_lost method.
        2. Verify that state is TrackState.Lost.
        """
        self.track.mark_lost()
        self.assertEqual(self.track.state, TrackState.Lost)

    def test_mark_removed(self):
        """
        Test the mark_removed method of BaseTrack.
        
        Steps:
        1. Call mark_removed method.
        2. Verify that state is TrackState.Removed.
        """
        self.track.mark_removed()
        self.assertEqual(self.track.state, TrackState.Removed)

    def test_reset_id(self):
        """
        Test the reset_id method of BaseTrack.
        
        Steps:
        1. Call next_id method once to get an ID.
        2. Call reset_id method.
        3. Call next_id method again and verify that it returns 1 (default value).
        """
        BaseTrack.next_id()
        BaseTrack.reset_id()
        self.assertEqual(BaseTrack.next_id(), 1)

if __name__ == '__main__':
    unittest.main()
