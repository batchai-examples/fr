import unittest
from ultralytics.trackers import register_tracker, BOTSORT, BYTETracker

class TestInit(unittest.TestCase):
    def test_register_tracker(self):
        """
        Test the register_tracker function to ensure it correctly registers a tracker.
        """
        # Arrange
        tracker_name = "test_tracker"
        tracker_instance = BOTSORT()

        # Act
        register_tracker(tracker_name, tracker_instance)

        # Assert
        self.assertIn(tracker_name, register_tracker._trackers)
        self.assertIs(register_tracker._trackers[tracker_name], tracker_instance)

    def test_register_existing_tracker(self):
        """
        Test the register_tracker function to ensure it raises an error when registering a tracker with an existing name.
        """
        # Arrange
        tracker_name = "test_tracker"
        tracker_instance = BOTSORT()

        # Act & Assert
        register_tracker(tracker_name, tracker_instance)
        with self.assertRaises(ValueError):
            register_tracker(tracker_name, BYTETracker())

    def test_get_registered_tracker(self):
        """
        Test the get_registered_tracker function to ensure it correctly retrieves a registered tracker.
        """
        # Arrange
        tracker_name = "test_tracker"
        tracker_instance = BOTSORT()
        register_tracker(tracker_name, tracker_instance)

        # Act
        retrieved_tracker = register_tracker.get_registered_tracker(tracker_name)

        # Assert
        self.assertEqual(retrieved_tracker, tracker_instance)

    def test_get_non_existent_tracker(self):
        """
        Test the get_registered_tracker function to ensure it raises an error when retrieving a non-existent tracker.
        """
        # Arrange
        tracker_name = "non_existent_tracker"

        # Act & Assert
        with self.assertRaises(KeyError):
            register_tracker.get_registered_tracker(tracker_name)

if __name__ == '__main__':
    unittest.main()
