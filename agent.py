import numpy as np
from collections import deque
from collections import namedtuple
from dqnetwork import make_dqnet, optimize

class DQNAgent():
    rewards = []
    total_reward = 0
    iter_done = 0
    games_done = 0
    Memory = namedtuple('Memory', ['obs', 'action', 'reward', 'next_obs', 'done'])

    def __init__(self, hyperparameters):
        self.epsilon = hyperparameters['epsilon']
        self.epsilon_final = hyperparameters['epsilon_final']
        self.decay = hyperparameters['decay']
        self.lr = hyperparameters['learning_rate']
        self.gamma = hyperparameters['gamma']
        self.iter_update_target = hyperparameters['iter_update_target']
        self.buffer = deque(maxlen=hyperparameters['buffer_size'])
        self.buffer_start_size = hyperparameters['buffer_start_size']
        self.accumulated_loss = []
        self.target_net = make_dqnet()
        self.moving_net = make_dqnet()

    def act_greedy(self, obs):
        state_value = np.array([obs])
        q_values = self.moving_net(state_value, training = False)
        return np.argmax(q_values, axis=-1)

    def act_eps_greedy(self, obs):
        if np.random.random() < self.epsilon:
            return np.random.random_integers(0, 5)
        else:
            return self.act_greedy(obs)
    
    def store_memory(self, obs, action, reward, next_obs, done):
        new_memory = self.Memory(obs = obs, action = action, reward = reward, next_obs = next_obs, done = done)
        self.buffer.append(new_memory)

        self.iter_done +=1

        if self.epsilon>self.epsilon_final:
            self.epsilon*=self.decay
        
        self.total_reward+=reward
    
    def sample_and_optimize(self, batch_size):
        if len(self.buffer) > self.buffer_start_size:
            indices = np.random.choice(len(self.buffer), batch_size, replace=False)
            states = []
            actions = []
            rewards = []
            next_states = []
            for i in indices:
                states.append(self.buffer[i].obs)
                actions.append(self.buffer[i].action)
                rewards.append(self.buffer[i].reward)
                next_states.append(self.buffer[i].next_obs)
            minibatch = (np.asarray(states, 'float32'), np.asarray(actions, 'int32'), np.asarray(rewards, 'float32'), np.asarray(next_states, 'float32'))
            loss = optimize(self.moving_net, self.target_net, minibatch, self.gamma)
            self.accumulated_loss.append(loss)

        if self.iter_done % self.iter_update_target:
            self.target_net.set_weights(self.moving_net.get_weights())
    
    def get_info(self):
        self.rewards.append(self.total_reward)
        self.games_done+=1
        return(self.games_done, self.total_reward, np.mean(self.rewards[-40:]), self.epsilon, np.mean(self.accumulated_loss))

    def reset(self):
        self.total_reward=0
        self.accumulated_loss=[]
        if self.games_done%50 == 0:
            self.moving_net.save("saved_models/model_games_done_{}.h5".format(self.games_done))