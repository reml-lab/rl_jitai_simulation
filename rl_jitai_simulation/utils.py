
import matplotlib.pyplot as plt
import numpy as np
import torch
from tqdm import tqdm

def run_RL_loop_DQN(env, agent, n_episodes, b_train, b_plot=False, plot_title='', y_lim=(-200,3500), color='C0'):
  return_values = []
  for i in tqdm(range(n_episodes), desc ="DQN {} σ={} duration".format(env.chosen_obs_names_str, env.sigma)):
    obs = env.reset()
    return_value = 0
    done = False
    while not done:
      action = agent.choose_action(obs)
      obs_, reward, done, info = env.step(action)
      agent.observe(obs, action, reward, obs_, done)
      if b_train: agent.update()
      obs = obs_
      return_value += reward
    return_values.append(return_value)
  if b_plot:
    plt.figure(figsize=(3,2))
    plt.plot(return_values, color=color)
    if len(plot_title) < 1:
      plot_title = ' (σ={})'.format(env.sigma)
    plot_detail = 'train' if b_train else 'perf'
    plt.title('DQN {} {}{}'.format(env.chosen_obs_names_str, plot_detail, plot_title))
    plt.ylim(y_lim); plt.xlabel('episode'); plt.ylabel('return'); plt.grid(); plt.show()
  return return_values

def run_RL_loop_REINFORCE(env, agent, M, n_episodes, b_train, b_plot=False, plot_title='', y_lim=(-200,3500)):
  return_values = []
  for i in tqdm(range(n_episodes), desc ="REINFORCE {} σ={} duration".format(env.chosen_obs_names_str, env.sigma)):
    if b_train: agent.init_grad()
    return_samples = []
    for m in range(M):
      obs = env.reset()
      return_sample = 0
      done = False
      while not done:
        action = agent.choose_action(obs)
        obs_, reward, done, info = env.step(action)
        agent.store_rewards(reward)
        obs = obs_
        return_sample += reward
      return_samples.append(return_sample)
      if b_train: agent.compute_grad()
    if b_train: agent.take_step(env.max_episode_length)
    return_values.append(np.mean(return_samples))
  if b_plot:
    plt.figure(figsize=(3,2))
    plt.plot(return_values)
    if len(plot_title) < 1:
      plot_title = ' (σ={})'.format(env.sigma)
    plt.title('REINFORCE {} learning{}'.format(env.chosen_obs_names_str, plot_title))
    plt.ylim(y_lim); plt.xlabel('episode'); plt.ylabel('return'); plt.grid(); plt.show()
  return return_values

def set_random_seed(seed):
  np.random.seed(seed)
  torch.manual_seed(seed)

def  get_key(sigma, δd, εd, seed, chosen_obs_name):
  str_detail = 'σ={} δd={} εd={} seed={} obs={}'.format(sigma, δd, εd, seed, '-'.join(chosen_obs_name))
  str_key = str_detail.replace('.','').replace(' ','_').replace('-','').replace('=','')
  return str_key

