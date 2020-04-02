import torch
import torch.nn as nn
import torch.nn.functional as F

class charGen(nn.Module):
    
    def __init__(self, n_letters, lstm_size, lstm_layers=3, lstm_dropout=0, dropout=0, hidden_dim=128):
        super(charGen, self).__init__()
        
        self.n_letters = n_letters
        self.lstm_size = lstm_size
        self.lstm_layers = lstm_layers
        self.lstm_dropout = lstm_dropout
        self.dropout = dropout
        self.hidden_dim = hidden_dim

        self.lstm = nn.LSTM(n_letters,
                            lstm_size,
                            num_layers=lstm_layers,
                            batch_first=False,
                            dropout=lstm_dropout)
        self.dropout = nn.Dropout(p=dropout)
        self.relu = nn.ReLU()
        self.fc = nn.Sequential(
                        nn.Linear(self.lstm_size, self.hidden_dim),
                        self.relu,
                        self.dropout,
                        nn.Linear(self.hidden_dim, self.hidden_dim),
                        self.relu,
                        self.dropout,
                        nn.Linear(self.hidden_dim, self.hidden_dim),
                        self.relu,
                        self.dropout,
                        nn.Linear(self.hidden_dim, self.n_letters)
                    )
        
    def forward(self, x, prev_states):
        out, state = self.lstm(x, prev_states)
        fc_in = out.view(-1, out.size(2))
        fc_out = self.fc(fc_in)
        return fc_out, state

    def zero_state(self, batch_size):
        return [torch.zeros(self.lstm_layers, batch_size, self.lstm_size), 
                torch.zeros(self.lstm_layers, batch_size, self.lstm_size)]

def get_model(n_letters=96):

    # Model Parameters
    # Size of LSTM layers
    lstm_size = 512
    # Number of LSTM layers
    lstm_layers = 3 
    # Dropout in LSTM layers
    lstm_dropout = 0 
    # Dropout in fully connected layers
    dropout = 0
    # Dimension of fully connected layers
    hidden_dim = 512

    return charGen( n_letters, 
                    lstm_size=lstm_size, 
                    lstm_layers=lstm_layers,
                    lstm_dropout=lstm_dropout,
                    dropout=dropout,
                    hidden_dim=hidden_dim)