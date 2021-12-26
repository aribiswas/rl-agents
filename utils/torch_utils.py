import torch


def soft_update(target_model, model, smooth_factor):
    """
    Soft update target model.
    """
    with torch.no_grad():
        for target_params, params in zip(target_model.parameters(), model.parameters()):
            target_params.data.copy_(smooth_factor * params + (1 - smooth_factor) * target_params.data)


def to_torch_tensor(state, action, reward, next_state, done, device='cpu'):
    state_tensor = torch.from_numpy(state).float().to(device)
    action_tensor = torch.from_numpy(action).float().to(device)
    reward_tensor = torch.from_numpy(reward).float().to(device)
    next_state_tensor = torch.from_numpy(next_state).float().to(device)
    done_tensor = torch.from_numpy(done).float().to(device)
    return state_tensor, action_tensor, reward_tensor, next_state_tensor, done_tensor

