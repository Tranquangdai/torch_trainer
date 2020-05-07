import numpy as np

import torch

from ..callbacks import callbacks as cbks
from ..utils.generic_utils import Progbar, to_list, to_numpy, unpack_singleton
from .network import NetWork


class FitLoop(object):

    def train_step(self, batch):
        x, y = self.transfer_input(batch)

        output_dict = dict()
        y_pred = self(x)
        output_dict['loss'] = self.loss(y_pred, y)

        if self.metrics is not None:
            for metric in self.metrics:
                output_dict[metric.__name__] = metric(y_pred, y)
        return output_dict

    def train_on_batch(self, batch):
        self.train()
        output_dict = self.train_step(batch)
        output_dict['loss'].backward()
        if self.clipnorm > 0:
            self.clip_gradient(self.clipnorm)
        self.optimizer.step()
        self.optimizer.zero_grad()
        return to_numpy(output_dict)

    def fit(self,
            train_dataloader,
            epochs=1,
            val_dataloader=None,
            callbacks=None):
        num_train_samples = len(train_dataloader.dataset)

        do_validation = bool(val_dataloader)
        steps_per_epoch = len(train_dataloader)

        self.history = cbks.History()
        _callbacks = [cbks.BaseLogger(stateful_metrics=self.metrics_names[1:])]
        _callbacks.append(cbks.ProgbarLogger(
            'steps', stateful_metrics=self.metrics_names[1:]))
        _callbacks += (callbacks or []) + [self.history]
        callbacks = cbks.CallbackList(_callbacks)

        callback_model = self._get_callback_model()
        callbacks.set_model(callback_model)

        callbacks_metrics = self.metrics_names
        if do_validation:
            callbacks_metrics += ['val_' + m for m in callbacks_metrics]

        callbacks.set_params({
            'batch_size': train_dataloader.batch_size,
            'epochs': epochs,
            'num_train_samples': num_train_samples,
            'verbose': 1,
            'steps': steps_per_epoch,
            'do_validation': do_validation,
            'metrics': callbacks_metrics,
            'train_dataloader': train_dataloader,
        })

        callbacks._call_begin_hook('train')
        epoch_logs = {}
        for epoch in range(epochs):
            epoch_logs['epoch_count'] = epoch + 1
            callbacks.on_epoch_begin(epoch)

            for batch_index, batch in enumerate(train_dataloader):
                batch_logs = {'batch': batch_index,
                              'size': len(self.parse_input(batch)[0]),
                              'epoch_count': epoch
                              }
                callbacks._call_batch_hook(
                    'train', 'begin', batch_index, batch_logs)

                outs = self.train_on_batch(batch)
                batch_logs.update(outs)
                callbacks._call_batch_hook(
                    'train', 'end', batch_index, batch_logs)

            if do_validation:
                val_outs = self.evaluate(
                    val_dataloader, callbacks=callbacks, verbose=0)
                val_outs = to_list(val_outs)
                for l, o in zip(self.metrics_names, val_outs):
                    epoch_logs['val_' + l] = o
            callbacks.on_epoch_end(epoch, epoch_logs)
        callbacks._call_end_hook('train')
        return self.history

    def clip_gradient(self, max_norm=5.0):
        torch.nn.utils.clip_grad_norm_(self.parameters(),
                                       max_norm=max_norm,
                                       norm_type=2)


class TestLoop(object):

    def eval_step(self, batch):
        x, y = self.transfer_input(batch)
        output_dict = dict()
        with torch.no_grad():
            y_pred = self(x)
            output_dict['loss'] = self.loss(y_pred, y)

        if self.metrics is not None:
            for metric in self.metrics:
                output_dict[metric.__name__] = metric(y_pred, y)
        return output_dict

    def eval_on_batch(self, batch):
        output_dict = self.eval_step(batch)
        return to_numpy(output_dict)

    def evaluate(self, dataloader, callbacks=None, verbose=0):
        self.eval()
        num_samples = len(dataloader.dataset)
        steps = len(dataloader)

        if verbose == 1:
            progbar = Progbar(target=steps)

        # Check if callbacks have not been already configured
        if not isinstance(callbacks, cbks.CallbackList):
            callbacks = cbks.CallbackList(callbacks)
            callback_model = self._get_callback_model()
            callbacks.set_model(callback_model)
            callback_metrics = list(self.metrics_names)

            callback_params = {
                'batch_size': dataloader.batch_size,
                'steps': steps,
                'samples': num_samples,
                'verbose': verbose,
                'metrics': callback_metrics,
            }
            callbacks.set_params(callback_params)
        callbacks._call_begin_hook('test')

        outs = []
        for batch_index, batch in enumerate(dataloader):
            batch_logs = {'batch': batch_index, 'size': 1}
            callbacks._call_batch_hook(
                'test', 'begin', batch_index, batch_logs)
            batch_outs = self.eval_on_batch(batch)
            if isinstance(batch_outs, dict):
                if batch_index == 0:
                    outs.extend([0.] * len(batch_outs))
                for i, batch_out in enumerate(batch_outs.values()):
                    outs[i] += float(batch_out)
            elif isinstance(batch_outs, list):
                if batch_index == 0:
                    outs.extend([0.] * len(batch_outs))
                for i, batch_out in enumerate(batch_outs):
                    outs[i] += float(batch_out)
            else:
                if batch_index == 0:
                    outs.append(0.)
                outs[0] += float(batch_outs)
            batch_logs.update(batch_outs)
            callbacks._call_batch_hook('test', 'end', batch_index, batch_logs)
            if verbose == 1:
                progbar.update(batch_index + 1)
        for i in range(len(outs)):
            outs[i] /= steps
        callbacks._call_end_hook('test')
        return unpack_singleton(outs)


class PredictLoop(object):

    def predict_step(self, batch):
        x, y = self.transfer_input(batch)
        with torch.no_grad():
            y_pred = self(x)
        return y_pred

    def predict_on_batch(self, batch):
        return self.predict_step(batch).cpu().numpy()

    def predict(self, dataloader, callbacks=None, verbose=0):
        self.eval()
        num_samples = len(dataloader.dataset)
        steps = len(dataloader)

        if verbose == 1:
            progbar = Progbar(target=steps)

        # Check if callbacks have not been already configured
        if not isinstance(callbacks, cbks.CallbackList):
            callbacks = cbks.CallbackList(callbacks)
            callback_model = self._get_callback_model()
            callbacks.set_model(self)
            callback_metrics = list(self.metrics_names)

            callback_params = {
                'batch_size': dataloader.batch_size,
                'steps': steps,
                'samples': num_samples,
                'verbose': verbose,
                'metrics': callback_metrics,
            }
            callbacks.set_params(callback_params)
        callbacks._call_begin_hook('predict')

        unconcatenated_outs = []
        for batch_index, batch in enumerate(dataloader):
            batch_logs = {'batch': batch_index, 'size': 1}
            callbacks._call_batch_hook(
                'predict', 'begin', batch_index, batch_logs)
            batch_outs = self.predict_on_batch(batch)
            batch_outs = to_list(batch_outs)
            if batch_index == 0:
                for batch_out in batch_outs:
                    unconcatenated_outs.append([])
            for i, batch_out in enumerate(batch_outs):
                unconcatenated_outs[i].append(batch_out)

            batch_logs['outputs'] = batch_outs
            callbacks._call_batch_hook(
                'predict', 'end', batch_index, batch_logs)
            if verbose == 1:
                progbar.update(batch_index + 1)
        callbacks.on_predict_end()
        if len(unconcatenated_outs) == 1:
            return np.concatenate(unconcatenated_outs[0], axis=0)
        return [np.concatenate(unconcatenated_outs[i], axis=0)
                for i in range(len(unconcatenated_outs))]

    def predict_classes(self, dataloader, verbose=0):
        proba = self.predict(dataloader, verbose=verbose)
        if proba.shape[-1] > 1:
            return proba.argmax(axis=-1)
        else:
            return (proba > 0.5).astype('int32')


class Looper(FitLoop, TestLoop, PredictLoop):
    pass


class DeviceTransfer(object):

    @property
    def device(self):
        return str(torch.device("cuda:0" if torch.cuda.is_available() else "cpu"))

    def _transfer_data_to_single_gpu(self, data):
        if isinstance(data, tuple) or isinstance(data, list):
            data = list(data)
            for i, item in enumerate(data):
                data[i] = item.cuda()
            return data

    def auto_transfer(self, data):
        if self.device == 'cpu':
            return data
        elif self.device == 'cuda:0':
            return self._transfer_data_to_single_gpu(data)


class Model(Looper, NetWork, DeviceTransfer):

    def compile(self, optimizer=None, loss=None, metrics=None, clipnorm=0.0):
        self.optimizer = optimizer
        self.loss = loss
        self.metrics = metrics
        self.clipnorm = clipnorm

        self._is_compiled = True
        self.to(self.device)

    @property
    def metrics_names(self):
        metrics_names = ['loss']
        metrics_names += [m.__name__ for m in self.metrics]
        return metrics_names

    @staticmethod
    def parse_input(batch):
        return (batch[0], batch[1])

    def transfer_input(self, batch):
        return self.auto_transfer(self.parse_input(batch))

    def _get_callback_model(self):
        """Returns the Callback Model for this Model."""
        if hasattr(self, 'callback_model') and self.callback_model:
            return self.callback_model
        return self
