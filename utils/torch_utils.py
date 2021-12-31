import torch
import numpy as np
from collections import namedtuple


def soft_update(target_model, model, smooth_factor):
    """
    Soft update target model.
    """
    with torch.no_grad():
        for target_params, params in zip(target_model.parameters(), model.parameters()):
            target_params.data.copy_(smooth_factor * params + (1 - smooth_factor) * target_params.data)


def replay_entry(state, action, reward, next_state, done, dtype=torch.float32, device='cpu'):
    experience = namedtuple('experience', ['state', 'action', 'reward', 'next_state', 'done'])
    entry = experience(state=torch.as_tensor(state, dtype=dtype, device=torch.device(device)),
                       action=torch.as_tensor(action, dtype=dtype, device=torch.device(device)),
                       reward=torch.as_tensor(reward, dtype=dtype, device=torch.device(device)),
                       next_state=torch.as_tensor(next_state, dtype=dtype, device=torch.device(device)),
                       done=torch.as_tensor(done, dtype=dtype, device=torch.device(device)))
    return entry


def to_tensor(xin):
    if isinstance(xin, np.ndarray):
        return torch.from_numpy(xin).float()
    elif isinstance(xin, torch.Tensor):
        return xin
    elif isinstance(xin, list):
        entry = xin
        state_tensor = torch.vstack([e.state for e in entry])
        action_tensor = torch.vstack([e.action for e in entry])
        reward_tensor = torch.vstack([e.reward for e in entry])
        next_state_tensor = torch.vstack([e.next_state for e in entry])
        done_tensor = torch.vstack([e.done for e in entry])
        return state_tensor, action_tensor, reward_tensor, next_state_tensor, done_tensor
    else:
        raise TypeError("Input must be a numpy array, torch Tensor, or a list of experience tuples.")
