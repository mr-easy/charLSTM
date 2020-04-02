import numpy as np
import re
import os
import string
import random

import torch
import torch.nn as nn
import torch.nn.functional as F
torch.manual_seed(42)

import model as md
import utils
from utils import predict
from utils import decode_letter
from utils import decode_seq

# Hyperparameters
seq_len = 1000
lr = 0.0005
batch_size = 32
num_epochs = 200

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print("Running on " + str(device))

model = md.get_model().to(device)

initial_letters = ["Hi brother", "I am", "Go now", "Mark Twain", "reached an end", "red colored", "thirsty, ", "He wouldn't"]

data_files = os.listdir("data/")
data = []
for data_file in data_files:
    with open("data/" + data_file, 'r', encoding='utf-8-sig') as f:
        text = f.read()
    print("Reading from " + data_file, end="... ")
    text = re.sub("\n\n+", "<br>", text)
    text = text.replace("\n", " ")
    text = text.replace("<br>", "\n")
    text = re.sub(' +', ' ', text)
    text = re.sub('-+', '-', text)
    text = "\n".join([t.strip() for t in text.split("\n") if len(t.strip())>0])
    #first_word = str.upper(data_file).split(".")[0].split("-")[0]
    #start_index = text.find(first_word)
    #text = text[start_index:]   # For starting with book name
    data.append(text)
    print("Done.")
    unk = [i for i in text if(i not in utils.all_letters)]
    if(len(unk) > 0):
        print(data_file, "Contains some unknown characters: ", end="")
        print(unk)

def prepare_data(data, seq_len):
    d_y = data[0][1:] + "\n"
    num_seqs = len(data[0])//seq_len
    x = np.array(list(data[0])[:seq_len*num_seqs]).reshape(-1, seq_len)
    y = np.array(list(d_y)[:seq_len*num_seqs]).reshape(-1, seq_len)
    #print(x.shape)
    for d in data[1:]:
        d_y = d[1:] + "\n"
        num_seqs = len(d)//seq_len
        x = np.append(x, np.array(list(d)[:seq_len*num_seqs]).reshape(-1, seq_len), axis=0)
        y = np.append(y, np.array(list(d_y)[:seq_len*num_seqs]).reshape(-1, seq_len), axis=0)
        #print(x.shape)
    return x, y

def get_batches(x, y, batch_size):
    num_batches = x.shape[0]//batch_size
    last_batch_size = x.shape[0] - num_batches*batch_size
    for b in range(0, x.shape[0], batch_size):
        encoded_x = np.zeros((seq_len, batch_size, utils.n_letters))
        encoded_y = np.zeros((seq_len, batch_size, utils.n_letters))
        for i, seq in enumerate(x[b:b+batch_size]):
            for j, letter in enumerate(seq):
                encoded_x[j,i,utils.letter_to_index[letter]] = 1
                encoded_y[j,i,utils.letter_to_index[y[b+i][j]]] = 1
        if(b + batch_size <= x.shape[0]):
            yield encoded_x, encoded_y
        else:
            yield encoded_x[:, :last_batch_size-1, :], encoded_y[:, :last_batch_size-1, :]

x, y = prepare_data(data, seq_len)

print("Random sentence without any training:")
print(predict(model, device, initial_letters=random.choice(initial_letters), top_k=9))    

loss_function = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=lr)

# TRAINING LOOP
epoch_losses = []
iteration = 0
for e in range(1, num_epochs+1):
    running_losses = []
    
    for batch_x,batch_y in get_batches(x, y, batch_size):

        iteration += 1
        
        model.train()
        
        batch_x = torch.Tensor(batch_x).to(device)
        batch_y = torch.Tensor(batch_y).to(device)
        
        state = model.zero_state(batch_y.shape[1])
        state[0] = state[0].to(device)
        state[1] = state[1].to(device)
        
        # Reset all gradients
        optimizer.zero_grad()
        
        # Forward propogation
        out, _ = model(batch_x, state)
        
        batch_y = batch_y.reshape(-1, utils.n_letters)
        loss = loss_function(out, torch.max(batch_y, dim=1)[1])
        running_losses.append(loss.item())
    
        # Backpropogation
        loss.backward()

        # Optimization step
        optimizer.step()

        if(iteration%50 == 0):
            print('Epoch: {}/{}\t'.format(e, num_epochs),
                  'Iteration: {}\t'.format(iteration),
                  'Loss: {}\t'.format(running_losses[-1]))
            print(predict(model, device, initial_letters=random.choice(initial_letters), top_k=3))
            
    epoch_losses.append(np.mean(running_losses))
    print("---------------------------------------------------------------------------")
    print('Epoch: {}/{}\t'.format(e, num_epochs),
          'Loss: {}\t'.format(epoch_losses[-1]))
    for start in initial_letters:
        print(predict(model, device, initial_letters=start, top_k=3))
    print("---------------------------------------------------------------------------")

#save epoch losses
with open("epoch_losses.txt", "w") as f:
	for l in epoch_losses:
		f.write(str(l) + "\n")

print("Saving model...")
torch.save(model.state_dict(), "model/model_state_dict")
print("Model saved.")