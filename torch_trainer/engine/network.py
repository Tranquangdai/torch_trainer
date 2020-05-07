import torch

from ..utils.layer_utils import summary


class NetWork(torch.nn.Module):

    def save(self, filepath):
        torch.save({
            'model_state_dict': self.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
        }, filepath)

    def save_weights(self, filepath):
        torch.save(self.state_dict(), filepath)

    def load_model(self, filepath):
        checkpoint = torch.load(filepath)
        self.load_state_dict(checkpoint['model_state_dict'])
        if not self._is_compiled:
            raise ValueError(
                'You must compile the model before loading optimizer weights')
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.train()

    def load_weights(self, filepath):
        self.load_state_dict(torch.load(filepath))
        self.eval()

    def summary(self, input_shape):
        summary(self, input_shape)

    @property
    def layers(self):
        return list(self.named_children())

    def get_layer_weight_by_name(self, layer_name=None):
        params = []
        for name, param in self.named_parameters():
            if name.split('.')[0] == layer_name:
                param.requires_grad = False
                params.append((name, param))
        return params
