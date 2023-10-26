from transformers import PreTrainedModel
import torch
import torch.nn as nn
import os
from HyenaDNA import HyenaDNAPreTrainedModel
from constants import DIR_PATH


class MyHyenaDNA(PreTrainedModel):
    def __init__(self, pretrained_model_name, device, num_of_layers=2, hidden_size=128, dropout=0.3):
        super(PreTrainedModel, self).__init__()
        self.only_hyena = num_of_layers == 0
        self.hynedaDNA = HyenaDNAPreTrainedModel.from_pretrained(f'{DIR_PATH}/checkpoints', pretrained_model_name, use_head=self.only_hyena, device=device)
        dim = 256  # hyenaDNA embeddings size
        self.classification_head = None

        if not self.only_hyena:
            layers = [nn.Dropout(p=dropout)]
            for i in range(num_of_layers):
                layers.append(nn.Linear(dim, hidden_size))
                layers.append(nn.LeakyReLU()) # ReLU # LeakyReLU
                layers.append(nn.Dropout(p=dropout))
                dim = hidden_size
                hidden_size = int(hidden_size / 2)
            layers.append(nn.Linear(dim, 2))  # 2 since we are using binary classification
            self.classification_head = nn.Sequential(*layers)

    def forward(self, x, **kwargs):
        if self.only_hyena:
            outputs = self.hynedaDNA(x)
        else:
            hidden_state = torch.mean(self.hynedaDNA(x), dim=1)
            outputs = self.classification_head(hidden_state)
        return outputs

    @staticmethod
    def load_pretrained(path, pretrained_model_name, device):
        hidden, layers, dropout = os.path.basename(path).split('__')[2:5]
        model = MyHyenaDNA(pretrained_model_name, device, int(layers), int(hidden.strip('_')), dropout=float(dropout))
        state_dict = torch.load(path, map_location=torch.device('cpu'))
        model.load_state_dict(state_dict)
        return model

