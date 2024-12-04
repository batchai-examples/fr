import unittest
from unittest.mock import patch, MagicMock

from ultralytics.hub.auth import Auth, API_KEY_URL

class TestAuth(unittest.TestCase):
    def setUp(self):
        self.auth = Auth(api_key="test_api_key", verbose=True)

    @patch('ultralytics.utils.SETTINGS.get')
    def test_init_with_api_key(self, mock_get):
        mock_get.return_value = "test_api_key"
        auth = Auth()
        self.assertEqual(auth.api_key, "test_api_key")

    @patch('ultralytics.hub.auth.request_with_credentials')
    @patch('ultralytics.utils.SETTINGS.update')
    def test_authenticate_success(self, mock_update, mock_request):
        mock_request.return_value = {"success": True}
        auth = Auth(api_key="test_api_key")
        self.assertTrue(auth.authenticate())

    @patch('ultralytics.hub.auth.request_with_credentials')
    def test_authenticate_failure(self, mock_request):
        mock_request.return_value = {"success": False}
        auth = Auth(api_key="test_api_key")
        self.assertFalse(auth.authenticate())

    @patch('ultralytics.utils.getpass.getpass')
    def test_request_api_key_success(self, mock_getpass):
        mock_getpass.return_value = "test_api_key"
        auth = Auth()
        self.assertTrue(auth.request_api_key())

    @patch('ultralytics.utils.getpass.getpass')
    def test_request_api_key_failure(self, mock_getpass):
        mock_getpass.return_value = ""
        auth = Auth()
        with self.assertRaises(ConnectionError):
            auth.request_api_key()

    @patch('ultralytics.hub.auth.requests.post')
    def test_auth_with_cookies_success(self, mock_post):
        mock_post.return_value.json.return_value = {"success": True}
        auth = Auth()
        self.assertTrue(auth.auth_with_cookies())

    @patch('ultralytics.hub.auth.requests.post')
    def test_auth_with_cookies_failure(self, mock_post):
        mock_post.return_value.json.return_value = {"success": False}
        auth = Auth()
        self.assertFalse(auth.auth_with_cookies())

    @patch('ultralytics.hub.auth.Auth.get_auth_header')
    def test_get_auth_header_success(self, mock_get_header):
        mock_get_header.return_value = {"authorization": "Bearer test_token"}
        auth = Auth(api_key="test_api_key")
        self.assertEqual(auth.get_auth_header(), {"authorization": "Bearer test_token"})

    @patch('ultralytics.hub.auth.Auth.get_auth_header')
    def test_get_auth_header_failure(self, mock_get_header):
        mock_get_header.return_value = None
        auth = Auth(api_key="test_api_key")
        self.assertIsNone(auth.get_auth_header())

if __name__ == '__main__':
    unittest.main()
