import unittest
from unittest.mock import patch, MagicMock
from ultralytics.utils.dist import find_free_network_port, generate_ddp_file, generate_ddp_command, ddp_cleanup

class TestDistUtils(unittest.TestCase):

    @patch('socket.socket')
    def test_find_free_network_port(self, mock_socket):
        """
        Tests the `find_free_network_port` function to ensure it returns a free port.
        """
        # Mocking socket behavior
        mock_socket.return_value.getsockname.return_value = ('127.0.0.1', 8080)
        
        # Call the function
        port = find_free_network_port()
        
        # Assert that the returned port is as expected
        self.assertEqual(port, 8080)

    @patch('ultralytics.utils.dist.generate_ddp_file')
    def test_generate_ddp_file(self, mock_generate_ddp_file):
        """
        Tests the `generate_ddp_file` function to ensure it generates a DDP file correctly.
        """
        # Mocking trainer object
        trainer = MagicMock()
        
        # Call the function
        generate_ddp_file(trainer)
        
        # Assert that the function was called with the correct arguments
        mock_generate_ddp_file.assert_called_once_with(trainer)

    @patch('ultralytics.utils.dist.generate_ddp_command')
    def test_generate_ddp_command(self, mock_generate_ddp_command):
        """
        Tests the `generate_ddp_command` function to ensure it generates a DDP command correctly.
        """
        # Mocking trainer object
        trainer = MagicMock()
        
        # Call the function
        generate_ddp_command(2, trainer)
        
        # Assert that the function was called with the correct arguments
        mock_generate_ddp_command.assert_called_once_with(2, trainer)

    @patch('ultralytics.utils.dist.ddp_cleanup')
    def test_ddp_cleanup(self, mock_ddp_cleanup):
        """
        Tests the `ddp_cleanup` function to ensure it cleans up the DDP file correctly.
        """
        # Mocking trainer object and file
        trainer = MagicMock()
        file = "path/to/temp_file.py"
        
        # Call the function
        ddp_cleanup(trainer, file)
        
        # Assert that the function was called with the correct arguments
        mock_ddp_cleanup.assert_called_once_with(trainer, file)

if __name__ == '__main__':
    unittest.main()
