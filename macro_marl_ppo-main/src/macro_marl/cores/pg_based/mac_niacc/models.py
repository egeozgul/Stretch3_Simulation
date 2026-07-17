import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence, pad_packed_sequence
from transformers import DistilBertModel, DistilBertTokenizer

def Linear(input_dim, output_dim, act_fn='leaky_relu', init_weight_uniform=True):
    gain = torch.nn.init.calculate_gain(act_fn)
    fc = torch.nn.Linear(input_dim, output_dim)
    if init_weight_uniform:
        nn.init.xavier_uniform_(fc.weight, gain=gain)
    else:
        nn.init.xavier_normal_(fc.weight, gain=gain)
    nn.init.constant_(fc.bias, 0.00)
    return fc

class Actor(nn.Module):

    def __init__(self, input_dim, output_dim, mlp_layer_size=[32,32], rnn_layer_size=32):
        super(Actor, self).__init__()

        # Load pre-trained DistilBERT model and tokenizer
        self.distilbert = DistilBertModel.from_pretrained('distilbert-base-uncased')
        self.tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')

        # Adjust input_dim to account for the DistilBERT embedding
        self.fc1 = Linear(input_dim + self.distilbert.config.dim, mlp_layer_size[0], act_fn='leaky_relu')
        self.fc2 = Linear(mlp_layer_size[0], mlp_layer_size[0], act_fn='leaky_relu')
        self.gru = nn.GRU(mlp_layer_size[0], hidden_size=rnn_layer_size, num_layers=1, batch_first=True)
        self.fc3 = Linear(rnn_layer_size, mlp_layer_size[1], act_fn='leaky_relu')
        self.fc4 = Linear(mlp_layer_size[1], output_dim, act_fn='linear')

    def forward(self, x, h=None, eps=0.0, test_mode=False, sentence=None):
        # Move inputs to the same device as the network
        device = self.fc1.weight.device
        x = x.to(device)
        if h is not None:
            h = h.to(device)

        # Encode the sentence using DistilBERT
        if sentence is not None:
            inputs = self.tokenizer(sentence, return_tensors="pt", padding=True, truncation=True).to(device)
            with torch.no_grad():
                outputs = self.distilbert(**inputs)
            # Use the [CLS] token embedding as the sentence representation
            sentence_embedding = outputs.last_hidden_state[:, 0, :]
        else:
            # Use zero embedding when no sentence is provided
            batch_size = x.size(0)
            seq_len = x.size(1)
            sentence_embedding = torch.zeros(batch_size, seq_len, self.distilbert.config.dim, device=device)
        
        # Concatenate the sentence embedding with the input
        x = torch.cat([x, sentence_embedding], dim=-1)

        x = F.leaky_relu(self.fc1(x))
        x = F.leaky_relu(self.fc2(x))
        x, h = self.gru(x, h)
        x = F.leaky_relu(self.fc3(x))
        x = self.fc4(x)

        action_logits = F.log_softmax(x, dim=-1)

        if not test_mode:
            logits_1 = action_logits + torch.log(torch.tensor(1 - eps, device=device))
            logits_2 = torch.full_like(action_logits, torch.log(torch.tensor(eps, device=device)) - torch.log(torch.tensor(action_logits.size(-1), device=device)))
            logits = torch.stack([logits_1, logits_2])
            action_logits = torch.logsumexp(logits, dim=0)

        return action_logits, h

    def init_hidden(self, batch_size):
        # Initialize hidden state to zero, ensure it's on the correct device
        return torch.zeros(1, batch_size, self.gru.hidden_size).to(self.fc1.weight.device)

class Critic(nn.Module):

    def __init__(self, input_dim, output_dim=1, mlp_layer_size=[32,32], rnn_layer_size=32):
        super(Critic, self).__init__()

        # Load pre-trained DistilBERT model and tokenizer
        self.distilbert = DistilBertModel.from_pretrained('distilbert-base-uncased')
        self.tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')

        # Adjust input_dim to account for the DistilBERT embedding
        self.fc1 = Linear(input_dim + self.distilbert.config.dim, mlp_layer_size[0], act_fn='leaky_relu')
        self.fc2 = Linear(mlp_layer_size[0], mlp_layer_size[0], act_fn='leaky_relu')
        self.gru = nn.GRU(mlp_layer_size[0], hidden_size=rnn_layer_size, num_layers=1, batch_first=True)
        self.fc3 = Linear(rnn_layer_size, mlp_layer_size[1], act_fn='leaky_relu')
        self.fc4 = Linear(mlp_layer_size[1], output_dim, act_fn='linear')

    def forward(self, x, h=None, sentence=None):
        # Move inputs to the same device as the network
        device = self.fc1.weight.device
        x = x.to(device)
        if h is not None:
            h = h.to(device)

        # Encode the sentence using DistilBERT
        if sentence is not None:
            inputs = self.tokenizer(sentence, return_tensors="pt", padding=True, truncation=True).to(device)
            with torch.no_grad():
                outputs = self.distilbert(**inputs)
            # Use the [CLS] token embedding as the sentence representation
            sentence_embedding = outputs.last_hidden_state[:, 0, :]
        else:
            # Use zero embedding when no sentence is provided
            batch_size = x.size(0)
            seq_len = x.size(1)
            sentence_embedding = torch.zeros(batch_size, seq_len, self.distilbert.config.dim, device=device)
        
        # Concatenate the sentence embedding with the input
        x = torch.cat([x, sentence_embedding], dim=-1)

        x = F.leaky_relu(self.fc1(x))
        x = F.leaky_relu(self.fc2(x))
        x, h = self.gru(x, h)
        x = F.leaky_relu(self.fc3(x))
        state_value = self.fc4(x)
        return state_value, h

    def init_hidden(self, batch_size):
        # Initialize hidden state to zero, ensure it's on the correct device
        return torch.zeros(1, batch_size, self.gru.hidden_size).to(self.fc1.weight.device)