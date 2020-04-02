import numpy as np
import string
import random

import torch
import torch.nn as nn
import torch.nn.functional as F
torch.manual_seed(42)

import model as md
from utils import predict
from utils import decode_letter
from utils import decode_seq

import argparse

parser = argparse.ArgumentParser(description="Generate text using charLSTM.")

parser.add_argument('--initial_letters', type=str, default="Give me some ",
                    help='Initial letters to start with. Default is "Give me some "')

parser.add_argument('--num_letters', type=int, default=100,
                    help='Number of letters to generate. Default is 100')

args = parser.parse_args()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print("Running on " + str(device))

model = md.get_model().to(device)
model.load_state_dict(torch.load("model/model_state_dict"))

print(predict(model, device, initial_letters=args.initial_letters, num_letters=args.num_letters, top_k=5))

