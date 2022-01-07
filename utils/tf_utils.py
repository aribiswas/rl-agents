import tensorflow as tf
import numpy as np
from collections import namedtuple


def soft_update(target_model, model, smooth_factor):
    """
    Soft update target model.
    """
    for v, tv in zip(model.trainable_variables, target_model.trainable_variables):
        tv.assign((1 - smooth_factor) * tv + smooth_factor * v)


def replay_entry(state, action, reward, next_state, done, dtype=tf.float32, device='/CPU:0'):
    experience = namedtuple('experience', ['state', 'action', 'reward', 'next_state', 'done'])
    with tf.device(device):
        entry = experience(state=tf.convert_to_tensor(state, dtype=dtype),
                           action=tf.convert_to_tensor(action, dtype=dtype),
                           reward=tf.convert_to_tensor(reward, dtype=dtype),
                           next_state=tf.convert_to_tensor(next_state, dtype=dtype),
                           done=tf.convert_to_tensor(done, dtype=dtype))
    return entry


def to_tensor(xin, batch_size=1):
    if isinstance(xin, np.ndarray):
        return tf.reshape(tf.convert_to_tensor(xin), shape=(batch_size,)+xin.shape)
    elif isinstance(xin, tf.Tensor):
        return xin
    elif isinstance(xin, list):
        entry = xin
        state_tensor = tf.convert_to_tensor([e.state for e in entry])
        action_tensor = tf.convert_to_tensor([e.action for e in entry])
        reward_tensor = tf.convert_to_tensor([e.reward for e in entry])
        next_state_tensor = tf.convert_to_tensor([e.next_state for e in entry])
        done_tensor = tf.convert_to_tensor([e.done for e in entry])
        return state_tensor, action_tensor, reward_tensor, next_state_tensor, done_tensor
    else:
        raise TypeError("Input must be a numpy array, tf.Tensor, or a list of experience tuples.")