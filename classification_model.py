from transformers import PreTrainedModel
import torch
import torch.nn as nn
from HyenaDNA import HyenaDNAPreTrainedModel

dir_path = '/davidb/ellarannon/splicing/' # '/sternadi/nobackup/volume1/ellarannon/splicing'


class MyHyenaDNA(PreTrainedModel):
    def __init__(self, pretrained_model_name, device, num_of_layers=2, hidden_size=128, dropout=0.3):
        super(PreTrainedModel, self).__init__()
        self.hynedaDNA = HyenaDNAPreTrainedModel.from_pretrained(f'{dir_path}/checkpoints', pretrained_model_name, use_head=False, device=device)
        dim = 256  # hyenaDNA embeddings size
        # self.dropout = nn.Dropout(p=dropout)
        # self.lstm = nn.LSTM(dim, hidden_size, batch_first=True, bidirectional=True, num_layers=num_of_layers, dropout=dropout)
        # self.proj = nn.Linear(2*hidden_size, 2)  # 2 since we are using binary classification
        layers = [nn.Dropout(p=dropout)]
        for i in range(num_of_layers):
            layers.append(nn.Linear(dim, hidden_size))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(p=dropout))
            dim = hidden_size
            hidden_size = int(hidden_size / 2)
        layers.append(nn.Linear(dim, 2))  # 2 since we are using binary classification
        self.classification_head = nn.Sequential(*layers)

    def forward(self, x, **kwargs):
        # with torch.no_grad():
        hidden_state = torch.mean(self.hynedaDNA(x), dim=1)
        # h = self.lstm(self.dropout(hidden_state))[0]
        # outputs = self.proj(h)
        outputs = self.classification_head(hidden_state)
        return outputs

    @staticmethod
    def load_pretrained(path, pretrained_model_name, device):
        _, nout, hidden, layers, dropout, _ = os.path.basename(path).split('__')
        model = MyHyenaDNA(pretrained_model_name, device, int(layers), int(hidden), dropout=float(dropout))
        state_dict = torch.load(path, map_location=torch.device('cpu'))
        model.load_state_dict(state_dict)
        return model

