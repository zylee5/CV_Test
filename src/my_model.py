import torch.nn as nn
import torch
import constants

class MyModel(nn.Module):
    def __init__(self, input_size=constants.NUM_FEATURES, hidden_size=64, num_classes=constants.NUM_CLASSES):
        super(MyModel, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers=2, batch_first=True, dropout=0.3)
        self.dropout = nn.Dropout(0.5)
        self.fc1 = nn.Linear(hidden_size, 64)
        self.output_layer = nn.Linear(64, num_classes)

    def forward(self, x):
        out, _ = self.lstm(x)
        out = out[:, -1, :]
        out = self.dropout(out)
        out = torch.relu(self.fc1(out))
        out = self.output_layer(out)
        return out