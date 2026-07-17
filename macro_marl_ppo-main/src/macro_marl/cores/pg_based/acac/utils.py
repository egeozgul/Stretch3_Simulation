import pickle
import torch
import os
import random
import numpy as np
import wandb

class Agent:

    def __init__(self):
        self.idx = None
        self.encoder = None
        self.encoder_tgt = None
        self.actor_net = None
        self.actor_optimizer = None
        self.actor_loss = None
        self.critic_net = None
        self.critic_tgt_net = None
        self.critic_optimizer = None
        self.critic_loss = None

class Linear_Decay(object):

    def __init__ (self, total_steps, init_value, end_value):
        self.total_steps = total_steps
        self.init_value = init_value
        self.end_value = end_value

    def get_value(self, step):
        frac = min(float(step) / self.total_steps, 1.0)
        return self.init_value + frac * (self.end_value-self.init_value)

def save_policies(run_id, agents, save_dir):
    if os.environ.get("MARC_DISABLE_POLICY_SAVE", "0") == "1":
        return
    # Get wandb run name if available
    try:
        wandb_run_name = wandb.run.name if wandb.run is not None else None
        if wandb_run_name:
            filename_suffix = f"_{wandb_run_name}"
        else:
            filename_suffix = ""
    except:
        filename_suffix = ""

    for agent in agents:
        # Save state dict for visualization (compatible with test scripts)
        PATH = (os.environ.get("MARC_ARTIFACT_ROOT", ".") + "/policy_nns/") + save_dir + '/' + str(run_id) + '_agent_' + str(agent.idx) + filename_suffix + '.pt'
        torch.save(agent.actor_net.state_dict(), PATH)

    # Save critic state dict
    critic_path = (os.environ.get("MARC_ARTIFACT_ROOT", ".") + "/policy_nns/") + save_dir + '/' + str(run_id) + '_agent_critic' + filename_suffix + '.pt'
    torch.save(agent.critic_net.state_dict(), critic_path)

def save_train_data(run_id, data, save_dir):
    with open((os.environ.get("MARC_ARTIFACT_ROOT", ".") + "/performance/") + save_dir + '/train/train_perform' + str(run_id) + '.pickle', 'wb') as handle:
        pickle.dump(data, handle)

def save_test_data(run_id, data, save_dir):
    with open((os.environ.get("MARC_ARTIFACT_ROOT", ".") + "/performance/") + save_dir + '/test/test_perform' + str(run_id) + '.pickle', 'wb') as handle:
        pickle.dump(data, handle)
