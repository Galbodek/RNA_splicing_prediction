from transformers import PreTrainedModel
import torch.nn as nn
from HyenaDNA import HyenaDNAPreTrainedModel

dir_path = '/sternadi/nobackup/volume1/ellarannon/splicing'

class MyHyenaDNA(PreTrainedModel):
    def __init__(self, pretrained_model_name, device, num_of_layers=2, hidden_size=128, dropout=0.3):
        super(PreTrainedModel, self).__init__()
        self.hynedaDNA = HyenaDNAPreTrainedModel.from_pretrained(f'{dir_path}/checkpoints', pretrained_model_name, use_head=False, device=device)
        layers = [nn.Dropout(p=dropout)]
        dim = 256  # hyenaDNA embeddings size
        for i in range(num_of_layers):
            layers.append(nn.Linear(dim, hidden_size))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(p=dropout))
            dim = hidden_size
            hidden_size = int(hidden_size / 2)
        layers.append(nn.Linear(dim, 2))  # 2 since we are using binary classification
        self.classification_head = nn.Sequential(*layers)

    def forward(self, x, **kwargs):
        hidden_state = self.hynedaDNA(x)
        outputs = self.classification_head(hidden_state)
        return outputs