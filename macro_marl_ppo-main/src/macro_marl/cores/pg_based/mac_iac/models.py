import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from transformers import AutoModel, AutoTokenizer

# Global shared instruction encoder to avoid loading BERT multiple times
_SHARED_INSTRUCTION_ENCODER = None
_SHARED_TOKENIZER = None

def get_shared_instruction_encoder(device='cpu'):
    """Get or create the shared instruction encoder (singleton pattern)"""
    global _SHARED_INSTRUCTION_ENCODER, _SHARED_TOKENIZER
    if _SHARED_INSTRUCTION_ENCODER is None:
        _SHARED_INSTRUCTION_ENCODER = AutoModel.from_pretrained('sentence-transformers/all-MiniLM-L6-v2')
        _SHARED_TOKENIZER = AutoTokenizer.from_pretrained('sentence-transformers/all-MiniLM-L6-v2')
        # Freeze by default
        for p in _SHARED_INSTRUCTION_ENCODER.parameters():
            p.requires_grad = False
        _SHARED_INSTRUCTION_ENCODER = _SHARED_INSTRUCTION_ENCODER.to(device)
        _SHARED_INSTRUCTION_ENCODER.eval()
    return _SHARED_INSTRUCTION_ENCODER, _SHARED_TOKENIZER

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

    def __init__(self, input_dim, output_dim, mlp_layer_size=[32,32], rnn_layer_size=32,
                 use_instructions=False, instruction_fusion='concat', instruction_emb_size=8,
                 freeze_bert=True, shared_encoder=None, shared_tokenizer=None):
        super(Actor, self).__init__()

        self.use_instructions = use_instructions
        self.instruction_fusion = instruction_fusion
        self.base_input_dim = input_dim

        final_input_dim = input_dim

        if use_instructions:
            # Use shared BERT sentence-transformer encoder
            if shared_encoder is not None and shared_tokenizer is not None:
                self.instruction_encoder = shared_encoder
                self.tokenizer = shared_tokenizer
            else:
                self.instruction_encoder, self.tokenizer = get_shared_instruction_encoder()

            # Project BERT 384-dim CLS output to instruction_emb_size (default 8)
            self.instruction_projection = Linear(384, instruction_emb_size, act_fn='leaky_relu')
            self.instruction_dim = instruction_emb_size

            if instruction_fusion == 'concat':
                final_input_dim += instruction_emb_size
            elif instruction_fusion == 'film':
                self.film_gamma = Linear(self.instruction_dim, mlp_layer_size[0])
                self.film_beta = Linear(self.instruction_dim, mlp_layer_size[0])
            elif instruction_fusion == 'attention':
                self.cross_attention = nn.MultiheadAttention(embed_dim=mlp_layer_size[0], num_heads=4, batch_first=True)
                self.instruction_to_mlp = Linear(self.instruction_dim, mlp_layer_size[0])

        self.fc1 = Linear(final_input_dim, mlp_layer_size[0], act_fn='leaky_relu')
        self.fc2 = Linear(mlp_layer_size[0], mlp_layer_size[0], act_fn='leaky_relu')
        self.gru = nn.GRU(mlp_layer_size[0], hidden_size=rnn_layer_size, num_layers=1, batch_first=True)
        self.fc3 = Linear(rnn_layer_size, mlp_layer_size[1], act_fn='leaky_relu')
        self.fc4 = Linear(mlp_layer_size[1], output_dim, act_fn='linear')

        # Cache for efficiency - avoids re-encoding same instruction text
        self._cached_instruction = None
        self._cached_instruction_emb = None

    def contrastive_loss(self, instruction_texts, margin=4.0):
        """Compute contrastive loss to push apart embeddings of different instructions.

        For each pair of distinct instructions, penalizes if their projected
        embeddings are closer than `margin` in L2 distance. This ensures
        the 8-dim projection separates semantically similar but functionally
        different instructions (e.g. 'left' vs 'right' cutting board).

        Args:
            instruction_texts: List of instruction text strings.
            margin: Minimum L2 distance between different instruction embeddings.
        Returns:
            Scalar contrastive loss (0 if fewer than 2 instructions).
        """
        if not self.use_instructions or len(instruction_texts) < 2:
            return torch.tensor(0.0, device=next(self.parameters()).device)

        embeddings = []
        for text in instruction_texts:
            emb = self.encode_instruction(text)  # (1, instruction_dim)
            embeddings.append(emb.squeeze(0))    # (instruction_dim,)
        embeddings = torch.stack(embeddings)      # (N, instruction_dim)

        loss = torch.tensor(0.0, device=embeddings.device)
        n_pairs = 0
        for i in range(len(embeddings)):
            for j in range(i + 1, len(embeddings)):
                dist = torch.norm(embeddings[i] - embeddings[j], p=2)
                loss = loss + torch.clamp(margin - dist, min=0.0)
                n_pairs += 1
        if n_pairs > 0:
            loss = loss / n_pairs
        return loss

    def encode_instruction(self, instruction_text):
        """Encode instruction text to embedding using BERT sentence-transformer and cache them for easy access.

        Args:
            instruction_text: A string or list of strings.
        Returns:
            Embedding tensor of shape (B, instruction_dim) i.e. (B, rnn_layer_size).
        """
        device = next(self.parameters()).device

        if isinstance(instruction_text, str):
            instruction_text = [instruction_text]

        # Check cache for single instruction
        if len(instruction_text) == 1 and instruction_text[0] == self._cached_instruction:
            return self._cached_instruction_emb.to(device)

        with torch.no_grad():
            tokens = self.tokenizer(instruction_text, return_tensors='pt',
                                    padding=True, truncation=True, max_length=64)
            tokens = {k: v.to(device) for k, v in tokens.items()}

            # Move encoder to correct device if needed
            if next(self.instruction_encoder.parameters()).device != device:
                self.instruction_encoder = self.instruction_encoder.to(device)

            outputs = self.instruction_encoder(**tokens)
            bert_emb = outputs.last_hidden_state[:, 0, :]  # CLS token (B, 384)

        instruction_emb = self.instruction_projection(bert_emb)  # (B, rnn_layer_size)

        # Cache single instruction
        if len(instruction_text) == 1:
            self._cached_instruction = instruction_text[0]
            self._cached_instruction_emb = instruction_emb.detach()

        return instruction_emb

    def forward(self, x, h=None, eps=0.0, test_mode=False, instruction=None, instruction_emb=None):
        batch_size = x.shape[0]
        device = x.device

        if self.use_instructions:
            has_instruction = instruction is not None or instruction_emb is not None
            
            if has_instruction:
                # If text instruction provided, encode it via BERT
                if instruction_emb is None and instruction is not None:
                    instruction_emb = self.encode_instruction(instruction)
                # Handle batch size expansion for 2D embeddings only
                if instruction_emb is not None and instruction_emb.dim() == 2:
                    if instruction_emb.shape[0] == 1 and batch_size > 1:
                        instruction_emb = instruction_emb.expand(batch_size, -1)
                    elif instruction_emb.shape[0] != batch_size:
                        raise ValueError("Instruction embedding batch size mismatch")
            else:
                # No instruction provided - create zero embedding vector for concat fusion
                instruction_emb = torch.zeros(batch_size, self.instruction_dim, device=device)
            
            if self.instruction_fusion == 'concat':
                if x.dim() == 3:
                    seq_len = x.size(1)
                    # Handle both 2D and 3D instruction embeddings
                    if instruction_emb.dim() == 2:
                        # 2D: (batch, emb_dim) -> expand to (batch, seq, emb_dim)
                        instruction_exp = instruction_emb.unsqueeze(1).expand(-1, seq_len, -1)
                    elif instruction_emb.dim() == 3:
                        # 3D: (batch, inst_seq, emb_dim) -> align with x's seq_len
                        if instruction_emb.size(1) == seq_len:
                            instruction_exp = instruction_emb
                        elif instruction_emb.size(1) == 1:
                            instruction_exp = instruction_emb.expand(-1, seq_len, -1)
                        else:
                            # Truncate or pad to match seq_len
                            instruction_exp = instruction_emb[:, :seq_len, :]
                    else:
                        raise ValueError(f"Unexpected instruction_emb dim: {instruction_emb.dim()}")
                    x = torch.cat([x, instruction_exp], dim=-1)
                else:
                    # x is 2D
                    if instruction_emb.dim() == 3:
                        # Take first timestep
                        instruction_emb = instruction_emb[:, 0, :]
                    x = torch.cat([x, instruction_emb], dim=-1)

        x = F.leaky_relu(self.fc1(x))

        if self.use_instructions and self.instruction_fusion == 'film' and instruction_emb is not None:
            # For FiLM, reduce 3D instruction to 2D if needed
            inst_for_film = instruction_emb[:, 0, :] if instruction_emb.dim() == 3 else instruction_emb
            gamma = self.film_gamma(inst_for_film)
            beta = self.film_beta(inst_for_film)
            if x.dim() == 3:
                gamma = gamma.unsqueeze(1)
                beta = beta.unsqueeze(1)
            x = gamma * x + beta

        x = F.leaky_relu(self.fc2(x))

        if self.use_instructions and self.instruction_fusion == 'attention' and instruction_emb is not None:
            # For attention, reduce 3D instruction to 2D if needed for projection
            inst_for_attn = instruction_emb[:, 0, :] if instruction_emb.dim() == 3 else instruction_emb
            instruction_key = self.instruction_to_mlp(inst_for_attn)
            if x.dim() == 3:
                instruction_key = instruction_key.unsqueeze(1).expand(-1, x.size(1), -1)
            else:
                instruction_key = instruction_key.unsqueeze(1)
            x_unsqueezed = x.unsqueeze(1) if x.dim() == 2 else x
            x_attended, _ = self.cross_attention(x_unsqueezed, instruction_key, instruction_key)
            if x.dim() == 2:
                x_attended = x_attended.squeeze(1)
            x = x + x_attended

        if x.dim() == 2:
            x = x.unsqueeze(1)
            x, h = self.gru(x, h)
            x = x.squeeze(1)
        else:
            x, h = self.gru(x, h)
        x = F.leaky_relu(self.fc3(x))
        x = self.fc4(x)

        action_logits = F.log_softmax(x, dim=-1)

        if not test_mode:
            logits_1 = action_logits + np.log(max(1-eps, 1e-10))
            logits_2 = torch.full_like(action_logits, np.log(max(eps, 1e-10)) - np.log(action_logits.size(-1)))
            logits = torch.stack([logits_1, logits_2])
            action_logits = torch.logsumexp(logits, axis=0)

        return action_logits, h

class Critic(nn.Module):

    def __init__(self, input_dim, output_dim=1, mlp_layer_size=[32,32], rnn_layer_size=32,
                 use_instructions=False, instruction_fusion='concat', instruction_emb_size=8,
                 freeze_bert=True, shared_encoder=None, shared_tokenizer=None):
        super(Critic, self).__init__()

        self.use_instructions = use_instructions
        self.instruction_fusion = instruction_fusion

        final_input_dim = input_dim

        if use_instructions:
            if shared_encoder is not None and shared_tokenizer is not None:
                self.instruction_encoder = shared_encoder
                self.tokenizer = shared_tokenizer
            else:
                self.instruction_encoder, self.tokenizer = get_shared_instruction_encoder()

            self.instruction_projection = Linear(384, instruction_emb_size, act_fn='leaky_relu')
            self.instruction_dim = instruction_emb_size

            if instruction_fusion == 'concat':
                final_input_dim += instruction_emb_size
            elif instruction_fusion == 'film':
                self.film_gamma = Linear(self.instruction_dim, mlp_layer_size[0])
                self.film_beta = Linear(self.instruction_dim, mlp_layer_size[0])
            elif instruction_fusion == 'attention':
                self.cross_attention = nn.MultiheadAttention(embed_dim=mlp_layer_size[0], num_heads=4, batch_first=True)
                self.instruction_to_mlp = Linear(self.instruction_dim, mlp_layer_size[0])

        self.fc1 = Linear(final_input_dim, mlp_layer_size[0], act_fn='leaky_relu')
        self.fc2 = Linear(mlp_layer_size[0], mlp_layer_size[0], act_fn='leaky_relu')
        self.gru = nn.GRU(mlp_layer_size[0], hidden_size=rnn_layer_size, num_layers=1, batch_first=True)
        self.fc3 = Linear(rnn_layer_size, mlp_layer_size[1], act_fn='leaky_relu')
        self.fc4 = Linear(mlp_layer_size[1], output_dim, act_fn='linear')

        self._cached_instruction = None
        self._cached_instruction_emb = None

    def encode_instruction(self, instruction_text):
        """Encode instruction text to embedding using BERT sentence-transformer."""
        device = next(self.parameters()).device

        if isinstance(instruction_text, str):
            instruction_text = [instruction_text]

        if len(instruction_text) == 1 and instruction_text[0] == self._cached_instruction:
            return self._cached_instruction_emb.to(device)

        with torch.no_grad():
            tokens = self.tokenizer(instruction_text, return_tensors='pt',
                                    padding=True, truncation=True, max_length=64)
            tokens = {k: v.to(device) for k, v in tokens.items()}

            if next(self.instruction_encoder.parameters()).device != device:
                self.instruction_encoder = self.instruction_encoder.to(device)

            outputs = self.instruction_encoder(**tokens)
            bert_emb = outputs.last_hidden_state[:, 0, :]

        instruction_emb = self.instruction_projection(bert_emb)

        if len(instruction_text) == 1:
            self._cached_instruction = instruction_text[0]
            self._cached_instruction_emb = instruction_emb.detach()

        return instruction_emb

    def forward(self, x, h=None, instruction=None, instruction_emb=None):
        batch_size = x.shape[0]
        device = x.device

        if self.use_instructions:
            has_instruction = instruction is not None or instruction_emb is not None
            
            if has_instruction:
                if instruction_emb is None and instruction is not None:
                    instruction_emb = self.encode_instruction(instruction)
                if instruction_emb is not None and instruction_emb.dim() == 2:
                    if instruction_emb.shape[0] == 1 and batch_size > 1:
                        instruction_emb = instruction_emb.expand(batch_size, -1)
                    elif instruction_emb.shape[0] != batch_size:
                        raise ValueError("Instruction embedding batch size mismatch")
            else:
                # No instruction provided - create zero embedding vector for concat fusion
                instruction_emb = torch.zeros(batch_size, self.instruction_dim, device=device)
            
            if self.instruction_fusion == 'concat':
                if x.dim() == 3:
                    seq_len = x.size(1)
                    # Handle both 2D and 3D instruction embeddings
                    if instruction_emb.dim() == 2:
                        # 2D: (batch, emb_dim) -> expand to (batch, seq, emb_dim)
                        instruction_exp = instruction_emb.unsqueeze(1).expand(-1, seq_len, -1)
                    elif instruction_emb.dim() == 3:
                        # 3D: (batch, inst_seq, emb_dim) -> align with x's seq_len
                        if instruction_emb.size(1) == seq_len:
                            instruction_exp = instruction_emb
                        elif instruction_emb.size(1) == 1:
                            instruction_exp = instruction_emb.expand(-1, seq_len, -1)
                        else:
                            # Truncate or pad to match seq_len
                            instruction_exp = instruction_emb[:, :seq_len, :]
                    else:
                        raise ValueError(f"Unexpected instruction_emb dim: {instruction_emb.dim()}")
                    x = torch.cat([x, instruction_exp], dim=-1)
                else:
                    # x is 2D
                    if instruction_emb.dim() == 3:
                        # Take first timestep
                        instruction_emb = instruction_emb[:, 0, :]
                    x = torch.cat([x, instruction_emb], dim=-1)

        x = F.leaky_relu(self.fc1(x))

        if self.use_instructions and self.instruction_fusion == 'film' and instruction_emb is not None:
            # For FiLM, reduce 3D instruction to 2D if needed
            inst_for_film = instruction_emb[:, 0, :] if instruction_emb.dim() == 3 else instruction_emb
            gamma = self.film_gamma(inst_for_film)
            beta = self.film_beta(inst_for_film)
            if x.dim() == 3:
                gamma = gamma.unsqueeze(1)
                beta = beta.unsqueeze(1)
            x = gamma * x + beta

        x = F.leaky_relu(self.fc2(x))

        if self.use_instructions and self.instruction_fusion == 'attention' and instruction_emb is not None:
            # For attention, reduce 3D instruction to 2D if needed for projection
            inst_for_attn = instruction_emb[:, 0, :] if instruction_emb.dim() == 3 else instruction_emb
            instruction_key = self.instruction_to_mlp(inst_for_attn)
            if x.dim() == 3:
                instruction_key = instruction_key.unsqueeze(1).expand(-1, x.size(1), -1)
            else:
                instruction_key = instruction_key.unsqueeze(1)
            x_unsqueezed = x.unsqueeze(1) if x.dim() == 2 else x
            x_attended, _ = self.cross_attention(x_unsqueezed, instruction_key, instruction_key)
            if x.dim() == 2:
                x_attended = x_attended.squeeze(1)
            x = x + x_attended

        if x.dim() == 2:
            x = x.unsqueeze(1)
            x, h = self.gru(x, h)
            x = x.squeeze(1)
        else:
            x, h = self.gru(x, h)
        x = F.leaky_relu(self.fc3(x))
        state_value = self.fc4(x)
        return state_value, h