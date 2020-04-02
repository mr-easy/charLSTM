from scipy.special import softmax
import numpy as np
import torch
import string

all_letters = string.ascii_letters + " " + "\n" + string.punctuation + string.digits
n_letters = len(all_letters)
letter_to_index = {v:k for k, v in enumerate(all_letters)}
index_to_letter = {v:k for k, v in letter_to_index.items()}

def decode_letter(onehot_letter, top_k=1):
    choices = np.argpartition(onehot_letter, -top_k)[-top_k:]
    probs = onehot_letter[choices]
    probs = softmax(probs)
    probs = probs / np.sum(probs)  # normalize probabilities
    choice = np.random.choice(choices, 1, p=probs)[0]
    return index_to_letter[choice]

def decode_seq(onehot_seq):
    return "".join([decode_letter(onehot_letter) for onehot_letter in onehot_seq])

def predict(model, device, n_letters=96, initial_letters="Hello", num_letters=100, top_k=3):
    model.eval()
    states = model.zero_state(1)
    states[0] = states[0].to(device)
    states[1] = states[1].to(device)
    
    for c in initial_letters:
        # encode c
        enc_c = np.zeros((1, 1, n_letters))
        enc_c[0, 0, letter_to_index[c]] = 1
        enc_c = torch.Tensor(enc_c).to(device)
        out, states = model(enc_c, states)
    initial_letters = "[" + initial_letters + "]"
    
    initial_letters += decode_letter(out[0].cpu().data.numpy(), top_k=top_k)
    if(initial_letters[-1] in [' ', "\n", ",", "."]):
        word_started = True
    else:
        word_started = False
    for i in range(num_letters):
        enc_c = np.zeros((1, 1, n_letters))
        enc_c[0, 0, letter_to_index[initial_letters[-1]]] = 1
        enc_c = torch.Tensor(enc_c).to(device)
        out, states = model(enc_c, states)
        if(word_started):
            initial_letters += decode_letter(out[0].cpu().data.numpy(), top_k=top_k)
        else:
            initial_letters += decode_letter(out[0].cpu().data.numpy(), top_k=1)
        if(initial_letters[-1] in [' ', "\n", ",", "."]):
            word_started = True
        else:
            word_started = False
            
    return initial_letters