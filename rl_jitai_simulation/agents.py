import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

class PolicyNetwork(nn.Module):
  def __init__(self, lr, input_dims, n_actions, fc1_dim, fc2_dim):
    super(PolicyNetwork, self).__init__()    
    self.fc2_dim = fc2_dim
    self.fc1 = nn.Linear(*input_dims, fc1_dim)
    if self.fc2_dim is None:
      self.a_logits = nn.Linear(fc1_dim, n_actions)
    else:
      self.fc2 = nn.Linear( fc1_dim, fc2_dim)
      self.a_logits = nn.Linear(fc2_dim, n_actions)      
    self.optimizer = optim.Adam(self.parameters(), lr=lr)
    self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    self.to(self.device)

  def forward(self, state):
    h = F.relu(self.fc1(state))
    if self.fc2_dim is not None:
      h = F.relu(self.fc2(h))
    a_logits = self.a_logits(h)
    return a_logits

class ReinforceAgent():
  def __init__(self, lr, gamma, input_dims, fc1_dim, fc2_dim, n_actions=4):
    self.fc1_dim = fc1_dim
    self.gamma = gamma
    self.lr = lr
    self.reward_list = []
    self.log_prob_action_list = []
    self.policy = PolicyNetwork(self.lr, input_dims, n_actions, fc1_dim=fc1_dim, fc2_dim=fc2_dim)
    if fc2_dim is None:
      self.config = 'REINFORCE lr={} fc1={}'.format(lr, fc1_dim)
    else:
      self.config = 'REINFORCE lr={} fc1={} fc2={}'.format(lr, fc1_dim, fc2_dim)    

  def choose_action(self, observation, env_trajectory=None):
    state = torch.tensor(observation, dtype=torch.float).to(self.policy.device)
    action_probs = F.softmax(self.policy.forward(state), dim=0)
    action_dist = torch.distributions.Categorical(action_probs)
    action = action_dist.sample()
    log_probs = action_dist.log_prob(action)
    self.log_prob_action_list.append(log_probs)
    chosen_action = action.item()  
    return chosen_action

  def store_rewards(self, reward):
    self.reward_list.append(reward)

  def init_grad(self):
    self.policy.optimizer.zero_grad()
    self.loss = 0

  def compute_grad(self):
    G = np.zeros_like(self.reward_list, dtype=np.float64)
    for t in range(len(self.reward_list)):
      G_sum = 0
      discount = 1
      for k in range(t, len(self.reward_list)):
        G_sum += self.reward_list[k] * discount
        discount *= self.gamma
      G[t] = G_sum
    G = torch.tensor(G, dtype=torch.float).to(self.policy.device)    
    for g, logprob in zip(G, self.log_prob_action_list):
      self.loss += -g * logprob
    self.log_prob_action_list = []
    self.reward_list = []

  def take_step(self, chosen_M):
    self.policy.optimizer.zero_grad()
    self.loss = self.loss / chosen_M
    self.loss.backward()
    self.policy.optimizer.step()

class ReplayBuffer():
  def __init__(self, max_size, input_shape, n_actions):
    self.mem_size = max_size
    self.mem_counter = 0
    self.state_memory     = np.zeros((self.mem_size, *input_shape), dtype=np.float32)
    self.new_state_memory = np.zeros((self.mem_size, *input_shape), dtype=np.float32)
    self.action_memory    = np.zeros(self.mem_size, dtype=np.int64)
    self.reward_memory    = np.zeros(self.mem_size, dtype=np.float32)
    self.done_memory      = np.zeros(self.mem_size, dtype=bool)

  def insert_buffer(self, state, action, reward, state_, done):
    index = self.mem_counter % self.mem_size
    self.state_memory[index]     = state
    self.new_state_memory[index] = state_
    self.action_memory[index]    = action
    self.reward_memory[index]    = reward
    self.done_memory[index]      = done
    self.mem_counter += 1

  def sample_buffer(self, batch_size):
    max_mem = min(self.mem_counter, self.mem_size)
    batch   = np.random.choice(max_mem, batch_size, replace=False)
    states  = self.state_memory[batch]
    actions = self.action_memory[batch]
    rewards = self.reward_memory[batch]
    states_ = self.new_state_memory[batch]
    dones = self.done_memory[batch]
    return states, actions, rewards, states_, dones

class DDQNetwork(nn.Module):
  def __init__(self, lr, n_actions, input_dims, fc1_dim, fc2_dim):
    super(DDQNetwork, self).__init__()
    self.fc2_dim = fc2_dim   
    self.fc1 = nn.Linear(*input_dims, fc1_dim)
    if self.fc2_dim is None:
      self.V   = nn.Linear(fc1_dim, 1)
      self.A   = nn.Linear(fc1_dim, n_actions)
    else:    
      self.fc2 = nn.Linear(fc1_dim, fc2_dim)
      self.V   = nn.Linear(fc2_dim, 1)
      self.A   = nn.Linear(fc2_dim, n_actions)
    self.optimizer = optim.Adam(self.parameters(), lr=lr)
    self.loss = nn.MSELoss()
    self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    self.to(self.device)

  def forward(self, state):
    l1 = F.relu(self.fc1(state))    
    if self.fc2_dim is None:    
      V = self.V(l1)
      A = self.A(l1)
    else:      
      l2 = F.relu(self.fc2(l1))
      V = self.V(l2)
      A = self.A(l2)
    return V, A

class DQNAgent():
  def __init__(self, gamma, batch_size, n_actions, input_dims, lr, fc1_dim, fc2_dim,
               epsilon=1, eps_min=0.01, eps_dec=0.001, replace=1000, mem_size=1000000):
    self.gamma = gamma
    self.epsilon = epsilon
    self.lr = lr
    self.n_actions = n_actions
    self.input_dims = input_dims
    self.batch_size = batch_size
    self.fc1_dim = fc1_dim
    self.fc2_dim = fc2_dim
    self.eps_min = eps_min
    self.eps_dec = eps_dec
    self.replace = replace
    self.learn_step_counter = 0
    self.action_space = [i for i in range(self.n_actions)]
    self.buffer = ReplayBuffer(mem_size, input_dims, n_actions)
    self.q_eval = DDQNetwork(self.lr, self.n_actions, self.input_dims, self.fc1_dim, self.fc2_dim)
    self.q_next = DDQNetwork(self.lr, self.n_actions, self.input_dims, self.fc1_dim, self.fc2_dim)
    self.q_next.load_state_dict(self.q_eval.state_dict())
    self.config = 'DQN lr={} batch={} h={}-{} dec={} rep={}'.format(
                  self.lr, self.batch_size, self.fc1_dim, self.fc2_dim, self.eps_dec, self.replace)
  
  def choose_action(self, observation):    
    if np.random.random() > self.epsilon:
      state = torch.tensor(np.array([observation]),dtype=torch.float).to(self.q_eval.device)
      _, advantage = self.q_eval.forward(state)   
      action = torch.argmax(advantage).item()
    else:
      action_choices = self.action_space  
      action = np.random.choice(action_choices)      
    return action

  def observe(self, state, action, reward, state_, done):
    self.buffer.insert_buffer(state, action, reward, state_, done)

  def replace_target_network(self):
    if self.learn_step_counter % self.replace == 0:
      self.q_next.load_state_dict(self.q_eval.state_dict())

  def decrement_epsilon(self):
    self.epsilon = self.epsilon - self.eps_dec if self.epsilon > self.eps_min else self.eps_min

  def update(self):
    if self.buffer.mem_counter < self.batch_size:
      return
    self.q_eval.optimizer.zero_grad()
    self.replace_target_network()
    states, actions, rewards, states_, dones = self.buffer.sample_buffer(self.batch_size)
    states  = torch.tensor(states).to(self.q_eval.device)
    rewards = torch.tensor(rewards).to(self.q_eval.device)
    dones   = torch.tensor(dones).to(self.q_eval.device)
    actions = torch.tensor(actions).to(self.q_eval.device)
    states_ = torch.tensor(states_).to(self.q_eval.device)
    indices = np.arange(self.batch_size)
    V_s, A_s   = self.q_eval.forward(states)
    V_s_, A_s_ = self.q_next.forward(states_)
    q_pred = torch.add(V_s,  (A_s  - A_s.mean(dim=1,  keepdim=True)))[indices, actions]
    q_next = torch.add(V_s_, (A_s_ - A_s_.mean(dim=1, keepdim=True)))
    q_target = rewards + self.gamma*torch.max(q_next, dim=1)[0].detach()
    q_target[dones] = 0.0
    loss = self.q_eval.loss(q_target, q_pred).to(self.q_eval.device)
    loss.backward()
    self.q_eval.optimizer.step()
    self.learn_step_counter += 1
    self.decrement_epsilon()


