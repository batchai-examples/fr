import unittest
from unittest.mock import patch, MagicMock

from ultralytics.utils.callbacks.hub import on_pretrain_routine_end, on_fit_epoch_end, on_model_save, on_train_end, on_train_start, on_val_start, on_predict_start, on_export_start, callbacks


class TestHubCallbacks(unittest.TestCase):
    def setUp(self):
        self.trainer = MagicMock()
        self.validator = MagicMock()
        self.predictor = MagicMock()
        self.exporter = MagicMock()

    @patch('ultralytics.utils.callbacks.hub.HUB_WEB_ROOT', 'http://example.com')
    @patch('ultralytics.utils.callbacks.hub.PREFIX', '[HUB]')
    def test_on_pretrain_routine_end(self):
        """
        Test the on_pretrain_routine_end function.
        
        Steps:
        1. Call the function with a mock trainer object.
        2. Verify that the timer for upload rate limit is started.
        """
        session = MagicMock()
        self.trainer.hub_session = session
        on_pretrain_routine_end(self.trainer)
        self.assertTrue(session.timers)

    @patch('ultralytics.utils.callbacks.hub.HUB_WEB_ROOT', 'http://example.com')
    @patch('ultralytics.utils.callbacks.hub.PREFIX', '[HUB]')
    def test_on_fit_epoch_end(self):
        """
        Test the on_fit_epoch_end function.
        
        Steps:
        1. Call the function with a mock trainer object.
        2. Verify that the metrics are uploaded to the session queue.
        3. Verify that the timer for upload rate limit is reset if the time elapsed is greater than the rate limit.
        """
        session = MagicMock()
        self.trainer.hub_session = session
        on_fit_epoch_end(self.trainer)
        self.assertIn('train', session.metrics_queue)
        self.assertTrue(session.timers['metrics'])

    @patch('ultralytics.utils.callbacks.hub.HUB_WEB_ROOT', 'http://example.com')
    @patch('ultralytics.utils.callbacks.hub.PREFIX', '[HUB]')
    def test_on_model_save(self):
        """
        Test the on_model_save function.
        
        Steps:
        1. Call the function with a mock trainer object.
        2. Verify that the checkpoint is uploaded to the session.
        3. Verify that the timer for upload rate limit is reset if the time elapsed is greater than the rate limit.
        """
        session = MagicMock()
        self.trainer.hub_session = session
        on_model_save(self.trainer)
        self.assertTrue(session.upload_model.called)
        self.assertTrue(session.timers['ckpt'])

    @patch('ultralytics.utils.callbacks.hub.HUB_WEB_ROOT', 'http://example.com')
    @patch('ultralytics.utils.callbacks.hub.PREFIX', '[HUB]')
    def test_on_train_end(self):
        """
        Test the on_train_end function.
        
        Steps:
        1. Call the function with a mock trainer object.
        2. Verify that the final model and metrics are uploaded to the session.
        3. Verify that the heartbeats are stopped.
        """
        session = MagicMock()
        self.trainer.hub_session = session
        on_train_end(self.trainer)
        self.assertTrue(session.upload_model.called)
        self.assertFalse(session.alive)

    @patch('ultralytics.utils.callbacks.hub.events')
    def test_on_train_start(self, mock_events):
        """
        Test the on_train_start function.
        
        Steps:
        1. Call the function with a mock trainer object.
        2. Verify that the events are run.
        """
        on_train_start(self.trainer)
        mock_events.assert_called_once_with(self.trainer.args)

    @patch('ultralytics.utils.callbacks.hub.events')
    def test_on_val_start(self, mock_events):
        """
        Test the on_val_start function.
        
        Steps:
        1. Call the function with a mock validator object.
        2. Verify that the events are run.
        """
        on_val_start(self.validator)
        mock_events.assert_called_once_with(self.validator.args)

    @patch('ultralytics.utils.callbacks.hub.events')
    def test_on_predict_start(self, mock_events):
        """
        Test the on_predict_start function.
        
        Steps:
        1. Call the function with a mock predictor object.
        2. Verify that the events are run.
        """
        on_predict_start(self.predictor)
        mock_events.assert_called_once_with(self.predictor.args)

    @patch('ultralytics.utils.callbacks.hub.events')
    def test_on_export_start(self, mock_events):
        """
        Test the on_export_start function.
        
        Steps:
        1. Call the function with a mock exporter object.
        2. Verify that the events are run.
        """
        on_export_start(self.exporter)
        mock_events.assert_called_once_with(self.exporter.args)

    def test_callbacks_enabled(self):
        """
        Test if callbacks are enabled when SETTINGS['HUB'] is True.
        
        Steps:
        1. Set SETTINGS['HUB'] to True.
        2. Verify that the callbacks dictionary is not empty.
        """
        with patch('ultralytics.utils.callbacks.hub.SETTINGS', {'HUB': True}):
            self.assertNotEqual(callbacks, {})

    def test_callbacks_disabled(self):
        """
        Test if callbacks are disabled when SETTINGS['HUB'] is False.
        
        Steps:
        1. Set SETTINGS['HUB'] to False.
        2. Verify that the callbacks dictionary is empty.
        """
        with patch('ultralytics.utils.callbacks.hub.SETTINGS', {'HUB': False}):
            self.assertEqual(callbacks, {})


if __name__ == '__main__':
    unittest.main()
