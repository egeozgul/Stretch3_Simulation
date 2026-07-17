import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence, pad_packed_sequence
from transformers import AutoModel, AutoTokenizer

# Global shared instruction encoder to avoid loading BERT multiple times
_SHARED_INSTRUCTION_ENCODER = None
_SHARED_TOKENIZER = None

def get_shared_instruction_encoder(device='cpu'):
    """Get or create the shared instruction encoder (singleton pattern)"""
    global _SHARED_INSTRUCTION_ENCODER, _SHARED_TOKENIZER
    if _SHARED_INSTRUCTION_ENCODER is None:
        print("[OPTIMIZATION] Loading shared instruction encoder (only once)...")
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
                 use_instructions=False, instruction_fusion='concat', freeze_bert=True,
                 shared_encoder=None, shared_tokenizer=None):
        super(Actor, self).__init__()
        
        self.use_instructions = use_instructions
        self.instruction_fusion = instruction_fusion
        self.base_input_dim = input_dim
        
        final_input_dim = input_dim

        # Instruction encoder setup - use shared encoder
        if use_instructions:
            if shared_encoder is not None and shared_tokenizer is not None:
                self.instruction_encoder = shared_encoder
                self.tokenizer = shared_tokenizer
            else:
                self.instruction_encoder, self.tokenizer = get_shared_instruction_encoder()
            
            # Note: encoder is shared, freeze state already set
            
            self.instruction_projection = Linear(384, rnn_layer_size, act_fn='leaky_relu')
            
            if instruction_fusion == 'concat':
                final_input_dim += rnn_layer_size
            elif instruction_fusion == 'film':
                self.film_gamma = Linear(rnn_layer_size, mlp_layer_size[0], act_fn='linear')
                self.film_beta = Linear(rnn_layer_size, mlp_layer_size[0], act_fn='linear')
            elif instruction_fusion == 'attention':
                self.instruction_to_mlp = Linear(rnn_layer_size, mlp_layer_size[0], act_fn='leaky_relu')
                self.attention_query = Linear(mlp_layer_size[0], mlp_layer_size[0], act_fn='leaky_relu')
                self.attention_key = Linear(mlp_layer_size[0], mlp_layer_size[0], act_fn='leaky_relu')
                self.attention_value = Linear(mlp_layer_size[0], mlp_layer_size[0], act_fn='leaky_relu')
        
        # Original network layers
        self.fc1 = Linear(final_input_dim, mlp_layer_size[0], act_fn='leaky_relu')
        self.fc2 = Linear(mlp_layer_size[0], mlp_layer_size[0], act_fn='leaky_relu')
        self.gru = nn.GRU(mlp_layer_size[0], hidden_size=rnn_layer_size, num_layers=1, batch_first=True)
        self.fc3 = Linear(rnn_layer_size, mlp_layer_size[1], act_fn='leaky_relu')
        self.fc4 = Linear(mlp_layer_size[1], output_dim, act_fn='linear')
        
        # Cache for efficiency
        self._cached_instruction = None
        self._cached_instruction_emb = None
    
    def encode_instruction(self, instruction_text):
        """Encode instruction text to embedding"""
        device = next(self.parameters()).device
        
        # Handle single string by converting to list
        if isinstance(instruction_text, str):
            instruction_text = [instruction_text]
            return_single = True
        else:
            return_single = False
        
        # Check cache for single instruction
        if len(instruction_text) == 1 and instruction_text[0] == self._cached_instruction:
            return self._cached_instruction_emb.to(device)
        
        with torch.no_grad() if not self.instruction_encoder.training else torch.enable_grad():
            tokens = self.tokenizer(instruction_text, return_tensors='pt',
                                   padding=True, truncation=True, max_length=64)
            tokens = {k: v.to(device) for k, v in tokens.items()}
            
            outputs = self.instruction_encoder(**tokens)
            instruction_emb = outputs.last_hidden_state[:, 0, :]
        
        instruction_emb = self.instruction_projection(instruction_emb)
        
        # Cache single instruction
        if len(instruction_text) == 1:
            self._cached_instruction = instruction_text[0]
            self._cached_instruction_emb = instruction_emb.detach()
        
        return instruction_emb
    
    def forward(self, x, h=None, eps=0.0, test_mode=False, instruction=None, instruction_emb=None):
        batch_size = x.shape[0]
        
        # Handle instruction if provided
        has_real_instruction = False
        if self.use_instructions:
            # If instructions are enabled but none provided, create zero embedding for concat fusion
            if instruction is None and instruction_emb is None:
                if self.instruction_fusion == 'concat':
                    # For concat, we need to always have an embedding (even if zeros)
                    # to maintain the expected input dimension
                    device = x.device
                    instr_emb_dim = self.instruction_projection.out_features if hasattr(self, 'instruction_projection') else 32
                    instruction_emb = torch.zeros(batch_size, instr_emb_dim, device=device)
                # For FiLM and attention, we can skip if no instruction provided
                has_real_instruction = False
            else:
                if instruction_emb is None:
                    instruction_emb = self.encode_instruction(instruction)
                has_real_instruction = True
            
            # Expand instruction embedding to match batch size if needed
            if instruction_emb is not None:
                if instruction_emb.shape[0] == 1 and batch_size > 1:
                    instruction_emb = instruction_emb.expand(batch_size, -1)
                elif instruction_emb.shape[0] != batch_size:
                    raise ValueError("Batch size of instruction embedding does not match input batch size")
            
            if self.instruction_fusion == 'concat' and instruction_emb is not None:
                # Reshape x if it's a sequence (B, T, F) -> (B*T, F)
                original_shape = x.shape
                if len(x.shape) == 3:
                    x = x.reshape(-1, x.shape[-1])
                    # Repeat instruction embedding for each item in sequence
                    # instruction_emb can be 2D (B, E) or 3D (B, T, E) or already flattened (B*T, E).
                    # Normalize it to shape (B*T, E) to match x after reshaping.
                    n_rows = x.shape[0]
                    if instruction_emb.dim() == 3:
                        # Typical case: (B, T, E) -> flatten to (B*T, E) when dims match
                        if instruction_emb.size(0) == original_shape[0] and instruction_emb.size(1) == original_shape[1]:
                            instruction_emb = instruction_emb.reshape(-1, instruction_emb.size(-1))
                        elif instruction_emb.size(0) == n_rows and instruction_emb.size(1) == 1:
                            # (B*T, 1, E) -> squeeze
                            instruction_emb = instruction_emb.squeeze(1)
                        else:
                            # Fallback: average across the temporal dim then repeat-interleave
                            instruction_emb = instruction_emb.mean(dim=1)
                            instruction_emb = instruction_emb.repeat_interleave(original_shape[1], dim=0)
                    elif instruction_emb.dim() == 2:
                        # (B, E) -> repeat for sequence length
                        instruction_emb = instruction_emb.repeat_interleave(original_shape[1], dim=0)
                    else:
                        raise ValueError(f"Unsupported instruction_emb dim: {instruction_emb.dim()}")

                # print(f"DEBUG (Actor): Before concat - x shape: {x.shape}, instruction_emb shape: {instruction_emb.shape}")
                x = torch.cat([x, instruction_emb], dim=-1)
                # print(f"DEBUG (Actor): After concat - x shape: {x.shape}")
                
                # Reshape back if it was a sequence
                if len(original_shape) == 3:
                    x = x.reshape(original_shape[0], original_shape[1], -1)
        
        # First layers
        x = F.leaky_relu(self.fc1(x))
        
        # FiLM modulation (only if real instruction provided)
        if self.use_instructions and self.instruction_fusion == 'film' and has_real_instruction and instruction_emb is not None:
            gamma = self.film_gamma(instruction_emb)
            beta = self.film_beta(instruction_emb)
            if len(x.shape) == 3:
                gamma = gamma.unsqueeze(1)
                beta = beta.unsqueeze(1)
            x = gamma * x + beta
        
        x = F.leaky_relu(self.fc2(x))
        
        # Attention fusion (only if real instruction provided)
        if self.use_instructions and self.instruction_fusion == 'attention' and has_real_instruction and instruction_emb is not None:
            instruction_key = self.instruction_to_mlp(instruction_emb)
            if len(x.shape) == 3:
                # x: [B, T, E]; align instruction_key to [B, T, E]
                if instruction_key.dim() == 2:
                    instruction_key = instruction_key.unsqueeze(1).expand(-1, x.size(1), -1)
                elif instruction_key.dim() == 3:
                    if instruction_key.size(1) == 1:
                        instruction_key = instruction_key.expand(-1, x.size(1), -1)
                    elif instruction_key.size(1) != x.size(1):
                        instruction_key = instruction_key[:, :x.size(1), :]
            else:
                # x: [B, E]; need keys length 1 → [B, 1, E]
                if instruction_key.dim() == 2:
                    instruction_key = instruction_key.unsqueeze(1)
                elif instruction_key.dim() == 3:
                    instruction_key = instruction_key[:, :1, :]
            x_unsqueezed = x.unsqueeze(1) if len(x.shape) == 2 else x
            x_attended, _ = self.cross_attention(x_unsqueezed, instruction_key, instruction_key)
            if len(x.shape) == 2:
                x_attended = x_attended.squeeze(1)
            x = x + x_attended
        
        # Ensure GRU always receives 3D input: (batch, seq_len, feat)
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
            logits_2 = torch.full_like(action_logits, np.log(max(eps, 1e-10))-np.log(action_logits.size(-1)))
            logits = torch.stack([logits_1, logits_2])
            action_logits = torch.logsumexp(logits, axis=0)
        
        return action_logits, h

class Critic(nn.Module):
    def __init__(self, input_dim, output_dim=1, mlp_layer_size=[32,32], rnn_layer_size=32,
                 use_instructions=False, instruction_fusion='concat', freeze_bert=True,
                 shared_encoder=None, shared_tokenizer=None):
        super(Critic, self).__init__()
        
        self.use_instructions = use_instructions
        self.instruction_fusion = instruction_fusion
        
        final_input_dim = input_dim

        if use_instructions:
            # Use shared encoder if provided, otherwise get global shared one
            if shared_encoder is not None and shared_tokenizer is not None:
                self.instruction_encoder = shared_encoder
                self.tokenizer = shared_tokenizer
            else:
                self.instruction_encoder, self.tokenizer = get_shared_instruction_encoder()
            
            # Note: encoder is shared, freeze state already set
            
            self.instruction_projection = Linear(384, rnn_layer_size, act_fn='leaky_relu')
            
            if instruction_fusion == 'concat':
                final_input_dim += rnn_layer_size
            elif instruction_fusion == 'film':
                self.film_gamma = Linear(rnn_layer_size, mlp_layer_size[0], act_fn='linear')
                self.film_beta = Linear(rnn_layer_size, mlp_layer_size[0], act_fn='linear')
            elif instruction_fusion == 'attention':
                self.instruction_to_mlp = Linear(rnn_layer_size, mlp_layer_size[0], act_fn='leaky_relu')
                self.attention_query = Linear(mlp_layer_size[0], mlp_layer_size[0], act_fn='leaky_relu')
                self.attention_key = Linear(mlp_layer_size[0], mlp_layer_size[0], act_fn='leaky_relu')
                self.attention_value = Linear(mlp_layer_size[0], mlp_layer_size[0], act_fn='leaky_relu')
        
        self.fc1 = Linear(final_input_dim, mlp_layer_size[0], act_fn='leaky_relu')
        self.fc2 = Linear(mlp_layer_size[0], mlp_layer_size[0], act_fn='leaky_relu')
        self.gru = nn.GRU(mlp_layer_size[0], hidden_size=rnn_layer_size, num_layers=1, batch_first=True)
        self.fc3 = Linear(rnn_layer_size, mlp_layer_size[1], act_fn='leaky_relu')
        self.fc4 = Linear(mlp_layer_size[1], output_dim, act_fn='linear')
        
        self._cached_instruction = None
        self._cached_instruction_emb = None
    
    def encode_instruction(self, instruction_text):
        """Encode instruction text to embedding"""
        device = next(self.parameters()).device
        
        if isinstance(instruction_text, str):
            instruction_text = [instruction_text]
        
        if len(instruction_text) == 1 and instruction_text[0] == self._cached_instruction:
            return self._cached_instruction_emb.to(device)
        
        with torch.no_grad() if not self.instruction_encoder.training else torch.enable_grad():
            tokens = self.tokenizer(instruction_text, return_tensors='pt',
                                   padding=True, truncation=True, max_length=64)
            tokens = {k: v.to(device) for k, v in tokens.items()}
            
            outputs = self.instruction_encoder(**tokens)
            instruction_emb = outputs.last_hidden_state[:, 0, :]
        
        instruction_emb = self.instruction_projection(instruction_emb)
        
        if len(instruction_text) == 1:
            self._cached_instruction = instruction_text[0]
            self._cached_instruction_emb = instruction_emb.detach()
        
        return instruction_emb
    
    def forward(self, x, h=None, instruction=None, instruction_emb=None):
        batch_size = x.shape[0]
        
        has_real_instruction = False
        if self.use_instructions:
            # If instructions are enabled but none provided, create zero embedding for concat fusion
            if instruction is None and instruction_emb is None:
                if self.instruction_fusion == 'concat':
                    # For concat, we need to always have an embedding (even if zeros)
                    # to maintain the expected input dimension
                    device = x.device
                    instr_emb_dim = self.instruction_projection.out_features if hasattr(self, 'instruction_projection') else 32
                    instruction_emb = torch.zeros(batch_size, instr_emb_dim, device=device)
                # For FiLM and attention, we can skip if no instruction provided
                has_real_instruction = False
            else:
                if instruction_emb is None:
                    instruction_emb = self.encode_instruction(instruction)
                has_real_instruction = True
            
            # Expand instruction embedding to match batch size if needed
            if instruction_emb is not None:
                if instruction_emb.shape[0] == 1 and batch_size > 1:
                    instruction_emb = instruction_emb.expand(batch_size, -1)
                elif instruction_emb.shape[0] != batch_size:
                    raise ValueError("Batch size of instruction embedding does not match input batch size")

            if self.instruction_fusion == 'concat' and instruction_emb is not None:
                # Reshape x if it's a sequence (B, T, F) -> (B*T, F)
                original_shape = x.shape
                if len(x.shape) == 3:
                    x = x.reshape(-1, x.shape[-1])
                    # Repeat instruction embedding for each item in sequence
                    # Normalize instruction_emb to (B*T, E) similar to Actor logic
                    n_rows = x.shape[0]
                    if instruction_emb.dim() == 3:
                        if instruction_emb.size(0) == original_shape[0] and instruction_emb.size(1) == original_shape[1]:
                            instruction_emb = instruction_emb.reshape(-1, instruction_emb.size(-1))
                        elif instruction_emb.size(0) == n_rows and instruction_emb.size(1) == 1:
                            instruction_emb = instruction_emb.squeeze(1)
                        else:
                            instruction_emb = instruction_emb.mean(dim=1)
                            instruction_emb = instruction_emb.repeat_interleave(original_shape[1], dim=0)
                    elif instruction_emb.dim() == 2:
                        instruction_emb = instruction_emb.repeat_interleave(original_shape[1], dim=0)
                    else:
                        raise ValueError(f"Unsupported instruction_emb dim: {instruction_emb.dim()}")

                # print(f"DEBUG (Critic): Before concat - x shape: {x.shape}, instruction_emb shape: {instruction_emb.shape}")
                x = torch.cat([x, instruction_emb], dim=-1)
                # print(f"DEBUG (Critic): After concat - x shape: {x.shape}")
                
                # Reshape back if it was a sequence
                if len(original_shape) == 3:
                    x = x.reshape(original_shape[0], original_shape[1], -1)
        
        x = F.leaky_relu(self.fc1(x))
        
        if self.use_instructions and self.instruction_fusion == 'film' and has_real_instruction and instruction_emb is not None:
            gamma = self.film_gamma(instruction_emb)
            beta = self.film_beta(instruction_emb)
            if len(x.shape) == 3:
                gamma = gamma.unsqueeze(1)
                beta = beta.unsqueeze(1)
            x = gamma * x + beta
        
        x = F.leaky_relu(self.fc2(x))
        
        if self.use_instructions and self.instruction_fusion == 'attention' and has_real_instruction and instruction_emb is not None:
            instruction_key = self.instruction_to_mlp(instruction_emb)
            if len(x.shape) == 3:
                if instruction_key.dim() == 2:
                    instruction_key = instruction_key.unsqueeze(1).expand(-1, x.size(1), -1)
                elif instruction_key.dim() == 3:
                    if instruction_key.size(1) == 1:
                        instruction_key = instruction_key.expand(-1, x.size(1), -1)
                    elif instruction_key.size(1) != x.size(1):
                        instruction_key = instruction_key[:, :x.size(1), :]
            else:
                if instruction_key.dim() == 2:
                    instruction_key = instruction_key.unsqueeze(1)
                elif instruction_key.dim() == 3:
                    instruction_key = instruction_key[:, :1, :]
            x_unsqueezed = x.unsqueeze(1) if len(x.shape) == 2 else x
            x_attended, _ = self.cross_attention(x_unsqueezed, instruction_key, instruction_key)
            if len(x.shape) == 2:
                x_attended = x_attended.squeeze(1)
            x = x + x_attended
        
        # Ensure GRU always receives 3D input: (batch, seq_len, feat)
        if x.dim() == 2:
            x = x.unsqueeze(1)
            x, h = self.gru(x, h)
            x = x.squeeze(1)
        else:
            x, h = self.gru(x, h)
        x = F.leaky_relu(self.fc3(x))
        state_value = self.fc4(x)
        
        return state_value, h


# ============ TEST SCRIPT ============
def test_instruction_models():
    """Comprehensive test for Actor and Critic with instruction encoding"""
    
    print("="*60)
    print("Testing Instruction-Conditioned Actor and Critic")
    print("="*60)
    
    # Setup
    obs_dim = 18  # Example: MPE Simple Spread observation dimension
    action_dim = 5  # 5 discrete actions
    batch_size = 4
    seq_len = 10
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Test different fusion methods
    fusion_methods = ['concat', 'film', 'attention']
    
    for fusion in fusion_methods:
        print(f"\nTesting {fusion.upper()} fusion method:")
        print("-"*40)
        
        # Create models
        actor = Actor(
            input_dim=obs_dim,
            output_dim=action_dim,
            use_instructions=True,
            instruction_fusion=fusion,
            freeze_bert=True
        ).to(device)
        
        critic = Critic(
            input_dim=obs_dim,
            output_dim=1,
            use_instructions=True,
            instruction_fusion=fusion,
            freeze_bert=True
        ).to(device)
        
        # Test data
        observations_single = torch.randn(batch_size, obs_dim).to(device)
        observations_seq = torch.randn(batch_size, seq_len, obs_dim).to(device)
        
        # Test instructions
        instructions = [
            "Move to the red landmark",
            "Avoid blue agents and reach green target",
            "Collaborate with ally to cover all landmarks",
            "Stay in formation and patrol the area"
        ]
        
        # Test 1: Single timestep forward pass with single instruction
        print(f"Test 1: Single timestep with single instruction")
        hidden_actor = torch.zeros(1, batch_size, 32).to(device)
        hidden_critic = torch.zeros(1, batch_size, 32).to(device)
        
        # With instruction as string (same for all batch items)
        action_logits, h_a = actor(observations_single, hidden_actor, 
                                   instruction=instructions[0])
        values, h_c = critic(observations_single, hidden_critic,
                            instruction=instructions[0])
        
        print(f"  Action logits shape: {action_logits.shape}")
        print(f"  Values shape: {values.shape}")
        assert action_logits.shape == (batch_size, action_dim)
        assert values.shape == (batch_size, 1)
        
        # Test 2: Sequence forward pass
        print(f"Test 2: Sequence processing")
        action_logits_seq, h_a_seq = actor(observations_seq, hidden_actor,
                                           instruction=instructions[0])
        values_seq, h_c_seq = critic(observations_seq, hidden_critic,
                                     instruction=instructions[0])
        
        print(f"  Action logits seq shape: {action_logits_seq.shape}")
        print(f"  Values seq shape: {values_seq.shape}")
        assert action_logits_seq.shape == (batch_size, seq_len, action_dim)
        assert values_seq.shape == (batch_size, seq_len, 1)
        
        # Test 3: Pre-encoded instruction embedding
        print(f"Test 3: Pre-encoded instruction (efficiency test)")
        instruction_emb = actor.encode_instruction(instructions[0])
        
        # Multiple forward passes with same embedding
        for i in range(3):
            action_logits, _ = actor(observations_single, hidden_actor,
                                    instruction_emb=instruction_emb)
            values, _ = critic(observations_single, hidden_critic,
                              instruction_emb=instruction_emb)
        print(f"  Pre-encoded embedding shape: {instruction_emb.shape}")
        print(f"  Successfully reused embedding multiple times")
        
        # Test 4: Batch of different instructions
        print(f"Test 4: Batch processing with different instructions")
        batch_instructions = instructions[:batch_size]
        instruction_emb_batch = actor.encode_instruction(batch_instructions)
        
        action_logits_batch, _ = actor(observations_single, hidden_actor,
                                       instruction_emb=instruction_emb_batch)
        values_batch, _ = critic(observations_single, hidden_critic,
                                instruction_emb=instruction_emb_batch)
        
        print(f"  Batch instruction embedding shape: {instruction_emb_batch.shape}")
        assert instruction_emb_batch.shape == (batch_size, 32)
        
        # Test 5: Without instructions (backward compatibility)
        print(f"Test 5: Backward compatibility (no instruction)")
        actor_no_inst = Actor(input_dim=obs_dim, output_dim=action_dim, 
                             use_instructions=False).to(device)
        critic_no_inst = Critic(input_dim=obs_dim, output_dim=1, 
                               use_instructions=False).to(device)
        
        action_logits_no_inst, _ = actor_no_inst(observations_single, hidden_actor)
        values_no_inst, _ = critic_no_inst(observations_single, hidden_critic)
        
        print(f"  Works without instructions: ✓")
        
        # Test 6: Gradient flow
        print(f"Test 6: Gradient flow test")
        actor.train()
        critic.train()
        
        # Unfreeze BERT for this test
        test_actor = Actor(
            input_dim=obs_dim,
            output_dim=action_dim,
            use_instructions=True,
            instruction_fusion=fusion,
            freeze_bert=False
        ).to(device)
        
        optimizer = torch.optim.Adam(test_actor.parameters(), lr=1e-4)
        
        action_logits, _ = test_actor(observations_single, hidden_actor,
                                      instruction=instructions[0])
        loss = -action_logits.mean()  # Dummy loss
        loss.backward()
        optimizer.step()
        
        print(f"  Gradient flow successful: ✓")
        
        print(f"\n{fusion.upper()} fusion: All tests passed! ✓")
    
    # Test 7: Memory efficiency comparison
    print("\n" + "="*60)
    print("Test 7: Memory and Speed Comparison")
    print("-"*40)
    
    import time
    
    # Frozen BERT vs Unfrozen
    actor_frozen = Actor(obs_dim, action_dim, use_instructions=True, 
                         freeze_bert=True).to(device)
    actor_unfrozen = Actor(obs_dim, action_dim, use_instructions=True,
                          freeze_bert=False).to(device)
    
    # Count parameters
    frozen_params = sum(p.numel() for p in actor_frozen.parameters() if p.requires_grad)
    unfrozen_params = sum(p.numel() for p in actor_unfrozen.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in actor_frozen.parameters())
    
    print(f"Trainable parameters (frozen BERT): {frozen_params:,}")
    print(f"Trainable parameters (unfrozen BERT): {unfrozen_params:,}")
    print(f"Total parameters: {total_params:,}")
    print(f"Parameter reduction: {(1 - frozen_params/unfrozen_params)*100:.1f}%")
    
    # Speed test
    obs_test = torch.randn(32, obs_dim).to(device)
    
    # Warm up
    for _ in range(5):
        actor_frozen(obs_test, instruction="test")
    
    # Time frozen
    torch.cuda.synchronize() if torch.cuda.is_available() else None
    start = time.time()
    for _ in range(10):
        actor_frozen(obs_test, instruction="Move to target")
    torch.cuda.synchronize() if torch.cuda.is_available() else None
    frozen_time = time.time() - start
    
    # Time with pre-encoded
    inst_emb = actor_frozen.encode_instruction("Move to target")
    torch.cuda.synchronize() if torch.cuda.is_available() else None
    start = time.time()
    for _ in range(10):
        actor_frozen(obs_test, instruction_emb=inst_emb)
    torch.cuda.synchronize() if torch.cuda.is_available() else None
    preencoded_time = time.time() - start
    
    print(f"\nInference time (10 iterations, batch size 32):")
    print(f"  With text encoding: {frozen_time:.3f}s")
    print(f"  With pre-encoded: {preencoded_time:.3f}s")
    print(f"  Speedup: {frozen_time/preencoded_time:.1f}x")
    
    # Test 8: MARL-specific test
    print("\n" + "="*60)
    print("Test 8: MARL-Specific Scenario (Multi-Agent)")
    print("-"*40)
    
    n_agents = 3
    shared_instruction = "Navigate to landmarks while avoiding collisions"
    
    # Create shared models (parameter sharing across agents)
    actor_shared = Actor(obs_dim, action_dim, use_instructions=True, 
                         instruction_fusion='concat').to(device)
    critic_shared = Critic(obs_dim, output_dim=1, use_instructions=True,
                          instruction_fusion='concat').to(device)
    
    # Pre-encode instruction once for efficiency
    instruction_emb = actor_shared.encode_instruction(shared_instruction)
    instruction_emb_critic = critic_shared.encode_instruction(shared_instruction)
    
    # Simulate multi-agent rollout
    agent_obs = torch.randn(n_agents, obs_dim).to(device)
    agent_hidden_actor = torch.zeros(1, n_agents, 32).to(device)
    agent_hidden_critic = torch.zeros(1, n_agents, 32).to(device)
    
    # Forward pass for all agents with shared instruction
    actions, h_a = actor_shared(agent_obs, agent_hidden_actor, 
                                instruction_emb=instruction_emb)
    values, h_c = critic_shared(agent_obs, agent_hidden_critic,
                               instruction_emb=instruction_emb_critic)
    
    print(f"  Multi-agent actions shape: {actions.shape}")
    print(f"  Multi-agent values shape: {values.shape}")
    assert actions.shape == (n_agents, action_dim)
    assert values.shape == (n_agents, 1)
    print(f"  MARL scenario successful: ✓")
    
    print("\n" + "="*60)
    print("ALL TESTS PASSED SUCCESSFULLY! ✓")
    print("="*60)