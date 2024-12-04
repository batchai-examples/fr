import unittest
from ultralytics.data.explorer import plot_query_result

class TestPlotQueryResult(unittest.TestCase):
    """
    Test cases for the plot_query_result function.
    """

    def test_plot_query_result_happy_path(self):
        """
        Test case to verify that the plot_query_result function works correctly with valid input.
        """
        # Arrange
        query_result = {"data": [1, 2, 3], "labels": ["A", "B", "C"]}
        
        # Act
        result = plot_query_result(query_result)
        
        # Assert
        self.assertIsNotNone(result)

    def test_plot_query_result_empty_data(self):
        """
        Test case to verify that the plot_query_result function handles empty data correctly.
        """
        # Arrange
        query_result = {"data": [], "labels": []}
        
        # Act
        result = plot_query_result(query_result)
        
        # Assert
        self.assertIsNotNone(result)

    def test_plot_query_result_missing_data(self):
        """
        Test case to verify that the plot_query_result function handles missing data correctly.
        """
        # Arrange
        query_result = {"labels": ["A", "B", "C"]}
        
        # Act
        result = plot_query_result(query_result)
        
        # Assert
        self.assertIsNotNone(result)

    def test_plot_query_result_missing_labels(self):
        """
        Test case to verify that the plot_query_result function handles missing labels correctly.
        """
        # Arrange
        query_result = {"data": [1, 2, 3]}
        
        # Act
        result = plot_query_result(query_result)
        
        # Assert
        self.assertIsNotNone(result)

if __name__ == '__main__':
    unittest.main()
