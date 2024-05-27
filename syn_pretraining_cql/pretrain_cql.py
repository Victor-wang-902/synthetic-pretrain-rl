import os
import sys
import joblib
import argparse
from tqdm import tqdm
from copy import deepcopy

import random
import numpy as np
import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F

CUDA_AVAILABLE = torch.cuda.is_available()
if CUDA_AVAILABLE:
    DEVICE = 'cuda'
else:
    DEVICE = 'cpu'


def pretrain_cql_args():
    parser = argparse.ArgumentParser()
    # General setup
    parser.add_argument("--exp_prefix", type=str, default="pretrain_cql_test")
    parser.add_argument("--data_dir", type=str, default='./mdpdata')
    parser.add_argument("--model_dir", type=str, default='./pretrained_cql_critic')
    parser.add_argument("--generate_data_only", action='store_true', default=False)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", type=str, default="cpu")

    # Q networks (critics) architecture
    parser.add_argument("--critic_hidden_layer", type=int, default=2)
    parser.add_argument("--critic_hidden_size", type=int, default=256)
    parser.add_argument("--orthogonal_init", action="store_true", default=False)

    # Training hyperparameters
    parser.add_argument("--total_pretrain_steps", "-it", type=int, default=int(1e5))
    parser.add_argument("--batch_size", "-B", type=int, default=256)
    parser.add_argument("--learning_rate", "-lr", type=float, default=3e-4)

    # Synthetic MDP data setup
    parser.add_argument("--mdp_n_traj", '-n', type=int, default=1000)
    parser.add_argument("--mdp_max_len", '-len', type=int, default=1000)
    parser.add_argument("--mdp_n_state", '-nS', type=int, default=1000)
    parser.add_argument("--mdp_n_action", '-nA', type=int, default=1000)
    parser.add_argument("--mdp_policy_temp", '-pt', type=float, default=1)
    parser.add_argument("--mdp_transition_temp", '-tt', type=float, default=1)
    parser.add_argument("--mdp_random_start", '-rs', action='store_true', default=False)
    parser.add_argument("--mdp_state_dim", '-dS', type=int, default=17)
    parser.add_argument("--mdp_action_dim", '-dA', type=int, default=6)

    args = parser.parse_args()

    return args


def set_random_seed(seed):
    np.random.seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.manual_seed(seed)
    random.seed(seed)


def check_input(value):
    if isinstance(value, (float, int)):
        assert value > 0, "Numerical input must be a positive real number."
    elif isinstance(value, str):
        assert value == 'inf', "String input must be 'inf'."
    else:
        raise TypeError("Input must be a positive real number or the string 'inf'.")


def softmax_with_torch(x, temperature):
    return F.softmax(Tensor(x / temperature), dim=0).numpy()


def generate_mdp_data(n_traj, max_length, n_state, n_action, policy_temperature, transition_temperature,
                      random_start=False, save_dir='./mdpdata'):
    # Check if temperatures are valid inputs
    check_input(policy_temperature)
    check_input(transition_temperature)

    # Set data file name and path
    data_file_name = 'mdp_traj%d_len%d_ns%d_na%d_pt%s_tt%s_rs%s.pkl' % (n_traj, max_length, n_state, n_action,
                                                                        str(policy_temperature),
                                                                        str(transition_temperature),
                                                                        str(random_start))
    data_save_path = os.path.join(save_dir, data_file_name)

    # Return if the MDP dataset is already generated
    if os.path.exists(data_save_path):
        dataset = joblib.load(data_save_path)
        print("Synthetic MDP data has already been generated. Loaded from:", data_save_path)
        return dataset

    # Set total number of synthetic MDP data
    n_data = n_traj * max_length

    #  Infinite policy and transition temperature is equivalent to generating i.i.d synthetic MDP data.
    if str(policy_temperature) == str(transition_temperature) == 'inf':
        states = np.random.randint(n_state, size=n_data)
        next_states = np.concatenate((states[1:], np.random.randint(n_state, size=1)))
        actions = np.random.randint(n_action, size=n_data)

    #  Otherwise, generate synthetic data according to MDP probability distributions.
    else:
        states = np.zeros(n_data, dtype=int)
        actions = np.zeros(n_data, dtype=int)
        next_states = np.zeros(n_data, dtype=int)
        i = 0
        for j_traj in tqdm(range(n_traj)):
            if random_start:
                np.random.seed(j_traj)
            else:
                np.random.seed(n_traj)

            state = np.random.randint(n_state)
            for t in range(max_length):
                states[i] = state
                # For each step, an action is taken, and a next state is decided. We assume that a fixed policy is
                # generating the data, so the action distribution only depends on the state.
                if str(policy_temperature) == 'inf':
                    action_probs = None
                else:
                    np.random.seed(state)
                    action_probs = softmax_with_torch(np.random.rand(n_action), policy_temperature)

                np.random.seed(42 + j_traj * 1000 + t * 333)
                action = np.random.choice(n_action, p=action_probs)
                actions[i] = action

                if str(transition_temperature) == 'inf':
                    next_state_probs = None
                else:
                    np.random.seed(state * 888 + action * 777)
                    next_state_probs = softmax_with_torch(np.random.rand(n_state), transition_temperature)

                np.random.seed(666 + j_traj * 1000 + t * 333)
                next_state = np.random.choice(n_state, p=next_state_probs)
                next_states[i] = next_state

                state = next_state
                i += 1

    data_dict = {'observations': states,
                 'actions': actions,
                 'next_observations': next_states}

    joblib.dump(data_dict, data_save_path)
    print("Synthetic MDP data generation finished. Saved to:", data_save_path)
    return data_dict


class FullyConnectedQFunction(nn.Module):
    def __init__(self, obs_dim, action_dim, arch='256-256', orthogonal_init=False):
        super().__init__()
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.arch = arch
        self.orthogonal_init = orthogonal_init

        self.hidden_layers = nn.ModuleList()
        self.hidden_activation = F.relu

        d = obs_dim + action_dim
        hidden_sizes = [int(h) for h in arch.split('-')]
        for hidden_size in hidden_sizes:
            fc = nn.Linear(d, hidden_size)
            if orthogonal_init:
                nn.init.orthogonal_(fc.weight, gain=np.sqrt(2))
                nn.init.constant_(fc.bias, 0.0)
            self.hidden_layers.append(fc)
            d = hidden_size

        self.last_fc_layer = nn.Linear(d, 1)
        # The projection layer maps features to next observations
        self.hidden_to_next_obs = nn.Linear(d, self.obs_dim)

        if orthogonal_init:
            nn.init.orthogonal_(self.last_fc_layer.weight, gain=1e-2)
            nn.init.orthogonal_(self.hidden_to_next_obs.weight, gain=1e-2)
        else:
            nn.init.xavier_uniform_(self.last_fc_layer.weight, gain=1e-2)
            nn.init.xavier_uniform_(self.hidden_to_next_obs.weight, gain=1e-2)

        nn.init.constant_(self.last_fc_layer.bias, 0.0)
        nn.init.constant_(self.hidden_to_next_obs.bias, 0.0)

    def forward(self, observations, actions):
        h = self.get_feature(observations, actions)
        return torch.squeeze(self.last_fc_layer(h), dim=-1)

    def get_feature(self, observations, actions):
        h = torch.cat([observations, actions], dim=-1)
        for fc_layer in self.hidden_layers:
            h = self.hidden_activation(fc_layer(h))
        return h

    def predict_next_obs(self, observations, actions):
        h = self.get_feature(observations, actions)
        return self.hidden_to_next_obs(h)


def subsample_batch(dataset, size):
    indices = np.random.randint(dataset['observations'].shape[0], size=size)
    return {key: dataset[key][indices, ...] for key in dataset.keys()}


def batch_to_torch(batch, device):
    return {
        k: torch.from_numpy(v).to(device=device, non_blocking=True)
        for k, v in batch.items()
    }


def pretrain_with_batch(args, batch, optimizer, critic1, critic2):
    observations = batch['observations']
    actions = batch['actions']
    next_observations = batch['next_observations']

    obs_next_q1 = critic1.predict_next_obs(observations, actions)
    obs_next_q2 = critic2.predict_next_obs(observations, actions)
    pretrain_loss1 = F.mse_loss(obs_next_q1, next_observations)
    pretrain_loss2 = F.mse_loss(obs_next_q2, next_observations)
    pretrain_loss = pretrain_loss1 + pretrain_loss2

    optimizer.zero_grad()
    pretrain_loss.backward()
    optimizer.step()


def main(args):
    # Load MDP data
    mdp_dataset = generate_mdp_data(args.mdp_n_traj,
                                    args.mdp_max_len,
                                    args.mdp_n_state,
                                    args.mdp_n_action,
                                    args.mdp_policy_temp,
                                    args.mdp_transition_temp,
                                    random_start=args.mdp_random_start,
                                    save_dir=args.data_dir
                                    )
    if args.generate_data_only:
        return

    # We reset seed to map each MDP state/action index to a particular vector fixed across different experiments
    np.random.seed(0)
    index2state = 2 * np.random.rand(args.mdp_n_state, args.mdp_state_dim) - 1
    index2action = 2 * np.random.rand(args.mdp_n_action, args.mdp_action_dim) - 1
    index2state, index2action = index2state.astype(np.float32), index2action.astype(np.float32)

    # Reset seed according to users' choice for subsequent pre-training and fine-tuning
    set_random_seed(args.seed)

    # Initialize two Q networks (critics) and their target networks
    critic_arch = '-'.join([str(args.critic_hidden_size) for _ in range(args.critic_hidden_layer)])
    critic1 = FullyConnectedQFunction(
        obs_dim=args.mdp_state_dim,
        action_dim=args.mdp_action_dim,
        arch=critic_arch,
        orthogonal_init=args.orthogonal_init
    ).to(args.device)
    critic2 = FullyConnectedQFunction(
        obs_dim=args.mdp_state_dim,
        action_dim=args.mdp_action_dim,
        arch=critic_arch,
        orthogonal_init=args.orthogonal_init
    ).to(args.device)

    # Set file name for pretrained CQL critics
    pretrain_model_name = '%s_l%d_hs%d_dimS%d_dimA%d_traj%d_len%d_nS%d_nA%d_pt%s_tt%s_rs%s.pth' % \
                          (args.exp_prefix,
                           args.critic_hidden_layer,
                           args.critic_hidden_size,
                           args.mdp_state_dim,
                           args.mdp_action_dim,
                           args.mdp_n_traj,
                           args.mdp_max_len,
                           args.mdp_n_state,
                           args.mdp_n_action,
                           args.mdp_policy_temp,
                           args.mdp_transition_temp,
                           args.mdp_random_start)

    pretrain_model_path = os.path.join(args.model_dir, pretrain_model_name)
    if os.path.exists(pretrain_model_path):
        pretrain_dict = torch.load(pretrain_model_path, map_location=torch.device(args.device))

        critic1.load_state_dict(pretrain_dict['critic1'])
        critic2.load_state_dict(pretrain_dict['critic2'])
        print("Pretrained model loaded from:", pretrain_model_path)

    else:
        print("Pretrained model does not exist:", pretrain_model_path)
        print("================== Start Synthetic Pre-training ===================")
        optimizer = torch.optim.Adam(
            list(critic1.parameters()) + list(critic2.parameters()), args.learning_rate
        )
        for step in tqdm(range(args.total_pretrain_steps), desc='Pre-training CQL critics'):
            batch = subsample_batch(mdp_dataset, args.batch_size)
            batch['observations'] = index2state[batch['observations']]
            batch['next_observations'] = index2state[batch['next_observations']]
            batch['actions'] = index2action[batch['actions']]
            batch = batch_to_torch(batch, args.device)

            # Forward Dynamics Pretrain
            pretrain_with_batch(args, batch, optimizer, critic1, critic2)

        pretrain_dict = {'critic1': critic1.state_dict(),
                         'critic2': critic2.state_dict(),
                         'config': vars(args)
                         }
        torch.save(pretrain_dict, pretrain_model_path)
        print("Save pretrained model to:", pretrain_model_path)

    target_critic1 = deepcopy(critic1)
    target_critic2 = deepcopy(critic2)

    return critic1, critic2, target_critic1, target_critic2


if __name__ == '__main__':
    args = pretrain_cql_args()
    os.makedirs(args.data_dir, exist_ok=True)
    os.makedirs(args.model_dir, exist_ok=True)
    main(pretrain_cql_args())
