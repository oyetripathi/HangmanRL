import numpy as np
import tensorflow as tf
import string
from keras.preprocessing.sequence import pad_sequences
from network import QNetwork
from env import Hangman
from tqdm import tqdm

class Agent(object):
   
   def __init__(self,
                vocab_size,
                embed_size,
                num_heads,
                key_dim,
                maxlen=29,
                base_model_wts = None,
                attention_axes=(1,2)):
      self.letter_dict = self.build_letter_dict()
      self.maxlen = maxlen
      self.model = QNetwork(vocab_size=vocab_size, embed_size=embed_size,
                            num_heads=num_heads, key_dim=key_dim, maxlen=maxlen, 
                            base_model_wts=base_model_wts, attention_axes=attention_axes)
      self.target_model = QNetwork(vocab_size=vocab_size, embed_size=embed_size,
                            num_heads=num_heads, key_dim=key_dim, maxlen=maxlen, 
                            attention_axes=attention_axes)
      
   def update_target_model(self):
      self.target_model.set_weights(self.model.get_weights())
   
   def save_target_model(self, path):
      self.target_model.save_weights(path)

   def load_target_model(self, path):
      state, guessed = np.random.rand(200, self.maxlen), np.random.randint(0,2, size=(200, 26))
      state, guessed = tf.convert_to_tensor(state, dtype=tf.float32), tf.convert_to_tensor(guessed, dtype=tf.bool)
      self.target_model(state, guessed)
      self.target_model.load_weights(path)

   def select_random_action(self, num):
      return np.random.choice(26, size=num)
   
   
   def select_action(self, state, guessed):
      action_probs = self.model(state, guessed, training=False)
      action = tf.argmax(action_probs, axis=-1).numpy()
      return action
   
   
   def action_as_char(self, action):
      return [chr(ord('a')+i) for i in action.tolist()]

   def build_letter_dict(self):
      set_letters = set(string.ascii_lowercase)
      letters = list(set_letters)
      letters.sort()
      letter_dict = {l : i+1 for i, l in enumerate(letters)}
      letter_dict['_'] = 27
      letter_dict['.'] = 0
      return letter_dict
   
   def letter_to_num(self, c):
      return self.letter_dict[c]
   
   def process_input(self, words):
      seq = [list(map(self.letter_to_num, word)) for word in words]
      return pad_sequences(seq, maxlen = 29, padding="post", value=0)
      
   

class Trainer(object):
   
   def __init__(self,
                gamma = 0.99,
                epsilon  = 1.0,
                epsilon_min = 0.1,
                epsilon_max = 1.0,
                epsilon_random_frames = 50000,
                epsilon_greedy_frames = 1000000.0,
                max_memory_length = 100000):
      
      self.gamma = gamma  # Discount factor for past rewards
      self.epsilon = epsilon  # Epsilon greedy parameter
      self.epsilon_min = epsilon_min  # Minimum epsilon greedy parameter
      self.epsilon_max = epsilon_max # Maximum epsilon greedy parameter
      self.epsilon_interval = (self.epsilon_max - self.epsilon_min)  # Rate at which to reduce chance of random action being taken
      self.epsilon_random_frames = epsilon_random_frames
      self.epsilon_greedy_frames = epsilon_greedy_frames
      self.max_memory_length = max_memory_length

      self.batch_size = 4
      self.action_history = []
      self.state_history = []
      self.state_next_history = []
      self.guessed_history = []
      self.guessed_next_history = []
      self.rewards_history = []
      self.done_history = []
      self.frame_count = 0

      self.loss_function = tf.losses.Huber()
      self.optimizer = tf.optimizers.Adam()
   
   def store(self, history, new_arr):
      if history is None:
         return new_arr
      else:
         return np.append(history, new_arr, axis = 0)
    
   def update_history(self, action, state, guessed, state_next, guessed_next, done, reward):
      self.action_history.append(action)
      self.state_history.append(state)
      self.state_next_history.append(state_next)
      self.guessed_history.append(guessed)
      self.guessed_next_history.append(guessed_next)
      self.done_history.append(done)
      self.rewards_history.append(reward)

      if (len(self.rewards_history) - self.max_memory_length) > 0:
         extra = len(self.rewards_history) - self.max_memory_length
         del self.action_history[:extra]
         del self.state_history[:extra]
         del self.guessed_history[:extra]
         del self.state_next_history[:extra]
         del self.guessed_next_history[:extra]
         del self.done_history[:extra]
         del self.rewards_history[:extra]

   def record(self, agent : Agent, env: Hangman):
      (state, guessed) = env.reset()
      episode_reward = []
      while np.any(np.logical_not(env.done)):
        state = agent.process_input(state)
        self.frame_count += 1
        if (self.frame_count < self.epsilon_random_frames) or (self.epsilon > np.random.rand(1)[0]):
           action = agent.select_random_action(state.shape[0])
        else:
           action = agent.select_action(state, guessed)
        
        self.epsilon -= (self.epsilon_interval / self.epsilon_greedy_frames)
        self.epsilon = max(self.epsilon, self.epsilon_min)

        (state_next, guessed_next), reward, done, _ = env.step(agent.action_as_char(action))
        episode_reward.append(np.mean(reward))
        self.update_history(action, state, guessed, agent.process_input(state_next), guessed_next, done, reward)
        state, guessed = state_next, guessed_next
      return np.mean(episode_reward), env.winners.sum()

   def train_step(self, agent : Agent, env: Hangman):
      indices = np.random.choice(range(len(self.done_history)), size=self.batch_size)
      state_sample = np.array([self.state_history[i] for i in indices]).reshape(-1, agent.maxlen)
      state_next_sample = np.array([self.state_next_history[i] for i in indices]).reshape(-1, agent.maxlen)
      guesssed_sample = np.array([self.guessed_history[i] for i in indices]).reshape(-1, 26)
      guesssed_next_sample = np.array([self.guessed_next_history[i] for i in indices]).reshape(-1, 26)
      rewards_sample = np.array([self.rewards_history[i] for i in indices]).ravel()
      action_sample = np.array([self.action_history[i] for i in indices]).ravel()
      done_sample = tf.convert_to_tensor(np.array([self.done_history[i] for i in indices]).ravel(), dtype=tf.float32)

      future_rewards = agent.target_model(state_next_sample, guesssed_next_sample)
      updated_q_values = rewards_sample + self.gamma * tf.reduce_max(
                future_rewards, axis=-1)
      updated_q_values = updated_q_values * (1 - done_sample) - done_sample

      masks = tf.one_hot(action_sample, env.num_actions)

      with tf.GradientTape() as tape:
         q_values = agent.model(state_sample, guesssed_sample)
         q_action = tf.reduce_sum(tf.multiply(q_values, masks), axis = 1)
         loss = self.loss_function(updated_q_values, q_action)
      grads = tape.gradient(loss, agent.model.trainable_variables)
      self.optimizer.apply_gradients(zip(grads, agent.model.trainable_variables))

   def train(self, agent: Agent, env: Hangman, record_steps=100, epochs=10):
      print("Recording Episodes...")
      for _ in tqdm(range(record_steps)):
         _, _ = self.record(agent, env)
      print("Training Network...")
      for _ in tqdm(range(epochs)):
         self.train_step(agent, env)
      agent.update_target_model()

   def eval(self, agent: Agent, env: Hangman, record_steps=100):
      reward = 0
      wins = 0
      print("Evaluating Agent...")
      for _ in tqdm(range(record_steps)):
         eps_reward, eps_wins = self.record(agent, env)
         reward += eps_reward
         wins += eps_wins

      return reward/record_steps, wins/(env.num_env*record_steps)
      
           

# word_src = "words_250000_train.txt"
# max_lives = 5
# num_env = 4
# env = Hangman(word_src, num_env=num_env, max_lives=max_lives, verbose = True)
# agent = Agent(vocab_size = 28, maxlen=29,embed_size = 128, num_heads = 16, key_dim = 2)
# trainer = Trainer()
# trainer.record(agent, env)
# trainer.record(agent, env)
# trainer.train_step(agent, env)
# rew, wins = trainer.eval(agent, env)
# print(rew, wins)