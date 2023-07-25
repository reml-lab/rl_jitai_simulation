import numpy as np
from scipy.stats import norm
from sklearn.preprocessing import OneHotEncoder
from gym import spaces
import gym
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=UserWarning)

class JITAI_env(gym.Env):
  def __init__(self, sigma, chosen_obs_names, seed=0, max_episode_length=50, n_version=0, n_context=2, n_actions=4,
               δh=0.1, εh=0.05, δd=0.1, εd=0.4, µs=[0.1,0.1], ρ1=50., ρ2=200., ρ3=0., D_threshold=1, 
               b_add_time_inhomogeneous=True, b_display=True):
    '''This is the class for JITAI environment. The possible obs names for chosen_obs_names are: 'C', 'P', 'L', 'H', 'D', 'T', 
    for example: chosen_obs_names=['C','H','D'] or ['P', 'T'], where C is for true contex, P is for probability of context=0, 
    L is for inferred context, H is for habituation, D is for disengagement and T is the binary indicator.'''
    super(JITAI_env, self).__init__()
    self.max_episode_length = max_episode_length
    self.n_version = n_version
    self.C = n_context
    self.sigma = sigma    
    self.chosen_obs_names = chosen_obs_names
    self.chosen_obs_names_str = '-'.join(chosen_obs_names)    
    self.seed = seed
    self.rng = np.random.default_rng(self.seed)
    self.b_add_time_inhomogeneous = b_add_time_inhomogeneous
    self.δh = δh
    self.εh = εh
    self.δd = δd
    self.εd = εd
    self.µs = µs
    self.ρ1 = ρ1
    self.ρ2 = ρ2
    self.ρ3 = ρ3
    self.D_threshold = D_threshold
    self.init_c_true = 0
    self.init_probs = []
    for i in range(self.C):
      self.init_probs.append(1/self.C)
    self.init_c_infer = 0
    self.init_h = 0.1
    self.init_d = 0.1
    self.init_s = 0.1
    self.current_state = self.reset()   
    min_obs, max_obs = self.extract_min_max(self.chosen_obs_names, b_add_time_inhomogeneous)
    min_obs = np.array(min_obs)
    max_obs = np.array(max_obs)
    self.observation_space = spaces.Box(low=min_obs, high=max_obs, dtype = np.float32)
    self.action_space      = spaces.Discrete(n_actions,)
    self.config = {'obs':self.chosen_obs_names_str, 'σ':sigma, 'δh':self.δh, 'εh':self.εh, 'δd':self.δd, 'εd':self.εd,
                   'µs':self.µs, 'ρ1':self.ρ1, 'ρ2':self.ρ2 , 'ρ3':self.ρ3, 'D_threshold':self.D_threshold}
    if b_display:
      str_config = ' '.join([k + '='+ str(v) for k,v in self.config.items()])
      print('env config:', str_config)
    assert(len(µs)==n_context), 'error: length of µs must match the number of contexts.'    

  def sample_context(self, chosen_rng, input_sigma, class_balance=.5):
    '''This generates a context sample (with 2 contexts). The output is (c_true, p, c_infer), where: 
    c_true is the true context, p is the context probabilities, and c_infer is the inferred context.'''
    mu  = np.array([0,1])
    sigma_values = np.array([input_sigma, input_sigma])
    pc_true = np.array([class_balance, 1-class_balance])
    c_true = chosen_rng.choice(2, p=pc_true)
    x = mu[c_true] + sigma_values[c_true] * chosen_rng.standard_normal()
    if input_sigma > 0.1:
      p_num = norm.pdf((x-mu)/sigma_values) * pc_true
      p = p_num/(np.sum(p_num))
    else:
      p  = np.array([0,0])
      if c_true == 0:
        p[0] = 1
      else:
        p[1] = 1
    c_infer = np.argmax(p)
    return (c_true, p, c_infer)

  def create_obs_array(self, input_c_true, input_probs, input_c_infer, input_h, input_d, input_s):
    obs_array = [input_c_true]
    for i in range(self.C):
      obs_array.append(input_probs[i])
    obs_array.append(input_c_infer)
    obs_array.append(input_h)
    obs_array.append(input_d)
    obs_array.append(input_s)
    return np.array(obs_array)

  def unpack_obs_array(self,obs):
    input_c_true = int(obs[0])
    input_probs = obs[1:self.C+1]
    input_c_infer = int(obs[self.C+1])
    input_h = obs[self.C+2]
    input_d = obs[self.C+3]
    input_s = obs[self.C+4]
    return({"c_true":input_c_true, "probs":input_probs, "c_infer":input_c_infer,"h":input_h, "d":input_d,"s":input_s})

  def get_current_state(self):
    return self.current_state

  def get_C(self):
    return self.unpack_obs_array(self.current_state)['c_true']
      
  def get_H(self):
    return self.unpack_obs_array(self.current_state)['h']

  def get_P(self):
    return self.unpack_obs_array(self.current_state)['probs'][0]
  
  def get_L(self):
    return self.unpack_obs_array(self.current_state)['c_infer']
  
  def get_D(self):
    return self.unpack_obs_array(self.current_state)['d']
  
  def get_S(self):
    return self.unpack_obs_array(self.current_state)['s']

  def get_T(self):
    indicator_value = 0 if ((self.current_t % 2) == 0) else 1
    return indicator_value
  
  def extract_obs(self, chosen_obs_names, b_add_time_inhomogeneous):
    '''This extracts the obs in chosen_obs_names from the full state. If b_add_time_inhomogeneous is True, 
    then the full state is augmented with the time inhomogeneous (one hot vector).'''
    obs_only = []
    for check_obs in chosen_obs_names:
      if check_obs == 'C':
        obs_only.append(self.get_C())
      elif check_obs == 'P':
        obs_only.append(self.get_P())
      elif check_obs == 'L':
        obs_only.append(self.get_L())
      elif check_obs == 'H':
        obs_only.append(self.get_H())
      elif check_obs == 'D':
        obs_only.append(self.get_D())
      elif check_obs == 'S':
        obs_only.append(self.get_S())
      elif check_obs == 'T':
        obs_only.append(self.get_T())
      else:
        str_message = 'error in check_obs. obs name {} does not exist.'.format(check_obs)
        print(str_message)
        assert(1==2), str_message
    if b_add_time_inhomogeneous:
      enc = OneHotEncoder(handle_unknown='ignore')
      X = np.arange(self.max_episode_length)
      X = X.reshape(-1, 1)        
      enc.fit(X)        
      one_hot  = list(enc.transform([[int(self.current_t)]]).toarray()[0])
      obs_only = obs_only + one_hot
    return obs_only
  
  def extract_min_max(self, chosen_obs_names, b_add_time_inhomogeneous):
    '''This extracts the min and max values for the obs in chosen_obs_names. If b_add_time_inhomogeneous is True, 
    then the results are augmented with the min and max values of the time inhomogeneous (one hot vector).'''
    min_values = []; max_values = []
    for check_obs in chosen_obs_names:
      if check_obs == 'C':
        min_values.append(0.)
        max_values.append(1.)
      elif check_obs == 'P':
        min_values.append(0.)
        max_values.append(1.)
      elif check_obs == 'L':
        min_values.append(0.)
        max_values.append(1.)
      elif check_obs == 'H':
        min_values.append(0.)
        max_values.append(1.)
      elif check_obs == 'D':
        min_values.append(0.)
        max_values.append(1.)
      elif check_obs == 'S':
        min_values.append(0.)
        max_values.append(300.)
      elif check_obs == 'T':
        min_values.append(0.)
        max_values.append(1.)
      else:
        str_message = 'error in check_obs. obs name {} does not exist.'.format(check_obs)
        print(str_message)
        assert(1==2), str_message
    if b_add_time_inhomogeneous:     
      min_values = min_values + list(np.zeros(self.max_episode_length))
      max_values = max_values + list(np.ones(self.max_episode_length))
    return min_values, max_values
  
  def reset(self):
    self.current_t = 0
    self.current_state = self.create_obs_array(self.init_c_true, self.init_probs, self.init_c_infer, self.init_h, self.init_d, self.init_s)
    self.obs = self.extract_obs(self.chosen_obs_names, self.b_add_time_inhomogeneous)
    return self.obs
  
  def step(self, agent_action: int):
    a  = int(agent_action)
    ht = float(self.current_state[self.C+2])
    dt = float(self.current_state[self.C+3])
    st = float(self.current_state[self.C+4])
    obs_dict = self.unpack_obs_array(self.current_state)
    c_true   = obs_dict["c_true"]
    if a == 0: 
      h_next_mu = (1-self.δh) * ht
    else:
      h_next_mu = float(min(1, ht + self.εh))
    x  = 2 + c_true    
    if a == 0:
      d_next_mu = dt
    elif (a == 1) or (a == x):
      d_next_mu = (1-self.δd) * dt
    else:
      d_next_mu = float(min(1, dt + self.εd))    
    if   a == 0: 
      s_next_mu = self.µs[c_true]
    elif a == 1: 
      s_next_mu = self.µs[c_true] + (1 - h_next_mu) * self.ρ1
    elif a == x: 
      s_next_mu = self.µs[c_true] + (1 - h_next_mu) * self.ρ2
    else:
      s_next_mu = self.µs[c_true] + (1 - h_next_mu) * self.ρ3    

    c_true, probs_next_mu, c_infer_next_mu = self.sample_context(self.rng, self.sigma)    
    if self.n_version == 0:
      probs_next = probs_next_mu
      c_infer_next = c_infer_next_mu
      h_next = h_next_mu
      d_next = d_next_mu
      s_next = s_next_mu
    if self.n_version == 1:
      probs_next = probs_next_mu
      c_infer_next = c_infer_next_mu
      h_next = float(self.rng.normal(h_next_mu,0.25,1))
      d_next = float(self.rng.normal(d_next_mu,0.25,1))
      s_next = float(self.rng.normal(s_next_mu,25,1))
      c_infer_next = min(1,c_infer_next); c_infer_next = max(0,c_infer_next)
      for i in range(self.C):
        probs_next[i] = float(min(1,probs_next[i])); probs_next[i] = float(max(0,probs_next[i]))
      h_next = float(min(1,h_next)); h_next = float(max(0,h_next))
      d_next = float(min(1,d_next)); d_next = float(max(0,d_next))
    step_reward = s_next
    current_state = self.create_obs_array(c_true, probs_next, c_infer_next, h_next, d_next, s_next)
    self.current_state = current_state
    self.obs = self.extract_obs(self.chosen_obs_names, self.b_add_time_inhomogeneous)
    self.current_t += 1

    condition1 = (self.current_t >= self.max_episode_length)
    condition2 = (d_next >= self.D_threshold)
    if condition1 or condition2:
      info = {}
      if condition1:
        info['done'] = 'found t={} (> max episode length)'.format(self.current_t)
      if condition2:
        info['done'] = 'found d={:.1f} (> disengage threshold)'.format(d_next, self.D_threshold) 
      done = True
      return self.obs, step_reward, done, info
    else:      
      done = False
      return self.obs, step_reward, done, {}

  def get_inferred_error(self, chosen_sigma, b_default=True, sample_seed=0, N_data=1000):
    '''This computes the corresponding inferred error, given the uncertainty sigma.
    If b_default is True then this uses some default values. If b_default is False then this computes the inferred error, 
    using N_data samples and sample_seed.'''
    if b_default:
      map_sigma_error = {'0':0., '0.4':0.096, '0.6':.175, '0.8':.273, '1':.311, '2':.414}
      inferred_error = map_sigma_error[str(chosen_sigma)]
    else:
      c_true  = np.zeros((N_data,),dtype=int)
      c_infer = np.zeros((N_data,),dtype=int)
      chosen_rng = np.random.default_rng(sample_seed)
      for i in range(N_data):
        c_true[i], _ ,c_infer[i] = self.sample_context(chosen_rng, chosen_sigma)
      inferred_error = np.mean(c_true!=c_infer)
    return inferred_error

  def get_current_state_length(self):
    return len(self.current_state)  
  
  def get_obs_length(self):
    return len(self.obs)  

  def render(self):
    pass
