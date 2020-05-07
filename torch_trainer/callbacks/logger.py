from ..callbacks.base import Callback


class NeptuneLogger(Callback):

    def __init__(self, logger):
        super().__init__()
        self.logger = logger

    def on_batch_end(self, batch, logs=None):
        logs = logs or {}
        self._send(logs, 'on_batch_end_')

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        self._send(logs, 'on_epoch_end_')

    def on_train_end(self, logs=None):
        self.logger.stop()

    def _send(self, logs, prefix):
        for k in self.params['metrics']:
            if k in logs:
                self.logger.send_metric(prefix + k, logs[k])
