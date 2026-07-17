import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence, pad_packed_sequence

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

        self.fc1 = Linear(input_dim, mlp_layer_size[0], act_fn='leaky_relu')
        self.fc2 = Linear(mlp_layer_size[0], mlp_layer_size[0], act_fn='leaky_relu')
        self.gru = nn.GRU(mlp_layer_size[0], hidden_size=rnn_layer_size, num_layers=1, batch_first=True)
        self.fc3 = Linear(rnn_layer_size, mlp_layer_size[1], act_fn='leaky_relu')
        self.fc4 = Linear(mlp_layer_size[1], output_dim, act_fn='linear')

    def forward(self, x, h=None, eps=0.0, test_mode=False):

        # Handle empty sequences to avoid RNN error
        if x.shape[1] == 0:  # sequence length is 0
            # Return dummy output with correct shape
            batch_size = x.shape[0]
            output_shape = (batch_size, 1, self.fc4.out_features)
            dummy_output = torch.zeros(output_shape, device=x.device, dtype=x.dtype)
            dummy_h = torch.zeros(1, batch_size, self.gru.hidden_size, device=x.device, dtype=x.dtype) if h is None else h
            return F.log_softmax(dummy_output, dim=-1), dummy_h

        x = F.leaky_relu(self.fc1(x))
        x = F.leaky_relu(self.fc2(x))
        x, h = self.gru(x, h)
        x = F.leaky_relu(self.fc3(x))
        x = self.fc4(x)

        action_logits = F.log_softmax(x, dim=-1)

        if not test_mode:
            eps_clamped = float(max(0.0, min(1.0 - 1e-6, eps)))
            if eps_clamped > 0.0:
                logits_1 = action_logits + np.log(1 - eps_clamped)
                logits_2 = torch.full_like(action_logits, np.log(eps_clamped) - np.log(action_logits.size(-1)))
                logits = torch.stack([logits_1, logits_2])
                action_logits = torch.logsumexp(logits, axis=0)

        return action_logits, h

class Critic(nn.Module):

    def __init__(self, input_dim, output_dim=1, mlp_layer_size=[32,32], rnn_layer_size=32):
        super(Critic, self).__init__()

        self.fc1 = Linear(input_dim, mlp_layer_size[0], act_fn='leaky_relu')
        self.fc2 = Linear(mlp_layer_size[0], mlp_layer_size[0], act_fn='leaky_relu')
        self.gru = nn.GRU(mlp_layer_size[0], hidden_size=rnn_layer_size, num_layers=1, batch_first=True)
        self.fc3 = Linear(rnn_layer_size, mlp_layer_size[1], act_fn='leaky_relu')
        self.fc4 = Linear(mlp_layer_size[1], output_dim, act_fn='linear')

    def forward(self, x, h=None):

        # Handle empty sequences to avoid RNN error
        if x.shape[1] == 0:  # sequence length is 0
            # Return dummy output with correct shape
            batch_size = x.shape[0]
            output_shape = (batch_size, 1, 1)  # critic outputs single value
            dummy_output = torch.zeros(output_shape, device=x.device, dtype=x.dtype)
            dummy_h = torch.zeros(1, batch_size, self.gru.hidden_size, device=x.device, dtype=x.dtype) if h is None else h
            return dummy_output, dummy_h

        x = F.leaky_relu(self.fc1(x))
        x = F.leaky_relu(self.fc2(x))
        x, h = self.gru(x, h)
        x = F.leaky_relu(self.fc3(x))
        state_value = self.fc4(x)
        return state_value, h

class GRUEncoder(nn.Module):

    def __init__(self, input_dim, mlp_layer_size=[32,32], rnn_layer_size=32):
        super(GRUEncoder, self).__init__()

        self.fc1 = Linear(input_dim, mlp_layer_size[0], act_fn='leaky_relu')
        self.fc2 = Linear(mlp_layer_size[0], mlp_layer_size[0], act_fn='leaky_relu')
        self.gru = nn.GRU(mlp_layer_size[0], hidden_size=rnn_layer_size, num_layers=1, batch_first=True)

    def forward(self, x, h=None, eps=0.0, test_mode=False):
        # Handle empty sequences to avoid RNN error
        if x.shape[1] == 0:  # sequence length is 0
            # Return dummy output with correct shape
            batch_size = x.shape[0]
            dummy_output = torch.zeros(batch_size, 1, self.gru.hidden_size, device=x.device, dtype=x.dtype)
            dummy_h = torch.zeros(1, batch_size, self.gru.hidden_size, device=x.device, dtype=x.dtype) if h is None else h
            return dummy_output, dummy_h

        x = F.leaky_relu(self.fc1(x))
        x = F.leaky_relu(self.fc2(x))
        x, h = self.gru(x, h)
        return x, h

class AgentCentricGRUCritic(nn.Module):

    def __init__(self, input_dim, output_dim=1, mlp_layer_size=[32,32], rnn_layer_size=32, n_agent=3, encoders=None):
        super(AgentCentricGRUCritic, self).__init__()
        
        if encoders is None:
            # NOTE: each agent obs dim can differ
            if type(input_dim) is int or type(input_dim) == np.int64:
                input_dim = [input_dim for _ in range(n_agent)]
            self.encoders = nn.ModuleList([GRUEncoder(input_dim[i], mlp_layer_size, rnn_layer_size) for i in range(n_agent)])
        else:
            self.encoders = nn.ModuleList(encoders)
        self.n_agent = n_agent
        
        # Simple concatenation approach (no attention)
        self.fc = Linear(rnn_layer_size * n_agent, mlp_layer_size[1], act_fn='leaky_relu')
        self.value = Linear(mlp_layer_size[1], output_dim, act_fn='linear')

    def recover_joint_feature(self, x_agent, v):
        # x_agent should be a tensor of size n_batch x max_squ_epi_length x dim_feature
        # v should be a tensor of size n_batch x trace_len
        idx = (torch.cumsum(v, dim=1) - 1).to(torch.int64).unsqueeze(-1).tile((1, 1, x_agent.size()[2])) # n_batch x trace_len x dim_feature
        j_x = torch.gather(x_agent, 1, idx) # n_batch x trace_len x dim_feature
        return j_x

    def forward(self, x, v, h=None):
        # x should be a tensor of size n_batch x max_j_squ_epi_length (=trace_len) x (dim_obs * n_agent)
        # v should be a tensor of size n_batch x trace_len x n_agent
        # h should be a tensor of size n_hidden_layer x (dim_h * n_agent)
        
        assert x.size()[0] == v.size()[0], f'The length of x ({x.size()[0]}) should be matched to the length of v ({v.size()[0]}). Both values are number of batch.'
        assert x.size()[1] == v.size()[1], f'The width of x ({x.size()[1]}) should be matched to the width of v ({v.size()[1]}). Both values are the maximum squeezed joint episode length.'
        n_batch, trace_len, n_agent = v.size()[0], v.size()[1], self.n_agent
        device = x.device
        
        # For agent centric processing
        x = x.reshape((n_batch, trace_len, n_agent, -1)).permute(2,0,1,3).contiguous() # n_agent x n_batch x trace_len x dim_obs
        v = v.permute(2,0,1).contiguous() # n_agent x n_batch x trace_len
            
        j_v = torch.amax(v, dim=0).bool() # n_batch x trace_len
        if h is not None:
            n_layers = h.size()[0]
            h = h.reshape((n_layers, n_agent, -1)).permute(1,0,2).contiguous() # n_agent x n_layers x dim_h
        else:
            n_layers = 0

        feature = []
        h_next = []
        for agent_idx in range(n_agent):
            v_agent = v[agent_idx].bool()  # Ensure it's boolean
            squ_epi_len_agent = v_agent.sum(1)
            x_agent = torch.split_with_sizes(x[agent_idx][v_agent], list(squ_epi_len_agent))
            x_agent = pad_sequence(x_agent, padding_value=torch.tensor(0.0), batch_first=True).to(device) # n_batch x max_squ_epi_length_agent x dim_obs

            if h is not None:
                h_agent = h[agent_idx]
            else:
                h_agent = None
            
            gru_outputs_agent, h_next_agent = self.encoders[agent_idx](x_agent, h_agent)

            j_outputs_agent = self.recover_joint_feature(gru_outputs_agent, v_agent) * j_v.unsqueeze(-1) # n_batch x trace_len x dim_feature

            feature.append(j_outputs_agent.unsqueeze(0))
            if h_next_agent is not None:
                h_next.append(h_next_agent) 

        feature = torch.concat(feature, dim=0) # n_agent x n_batch x trace_len x dim_feature
        feature = feature.permute(1, 2, 0, 3).reshape(n_batch, trace_len, -1).contiguous() # n_batch x trace_len x (n_agent * dim_feature)
        if len(h_next):
            h_next = torch.concat(h_next, dim=0) # n_agent x n_layers x dim_h
            n_layers = h_next.size()[1]
            h_next = h_next.permute(1, 0, 2).reshape(n_layers, -1).contiguous() # n_layers x (n_agent * dim_h)
        else:
            h_next = None

        squ_j_epi_len = j_v.sum(1)
        feature = feature[j_v]

        # Simple concatenation approach (no attention)
        feature = F.leaky_relu(self.fc(feature))
        state_value = self.value(feature)

        state_value = torch.split_with_sizes(state_value, list(squ_j_epi_len))
        state_value = pad_sequence(state_value, padding_value=torch.tensor(0.0), batch_first=True).reshape(n_batch, -1, 1).to(device) # n_batch x trace_len x 1

        return state_value, h_next