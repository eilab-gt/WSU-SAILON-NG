import gym
from gym import spaces
from util import vectorize_state
from objects import objects
import wandb


class VizDoomEnv(gym.Env):

    def __init__(self, state_queue, action_queue, terminal_queue, performance_queue, log):
        super(VizDoomEnv, self).__init__()
        self.step_count = 0
        self.total_reward = 0
        self.last_state_dict = None
        self.state_queue = state_queue
        self.action_queue = action_queue
        self.performance_queue = performance_queue
        self.terminal_queue = terminal_queue
        self.log = log
        self.action_space = spaces.Discrete(8)
        self.observation_space = spaces.Box(-512, 512, shape=(90,))
        self.enemy_initial_health = None
        self.player_initial_health = None
        self.actions = ['nothing', 'left', 'right', 'forward', 
            'backward', 'shoot', 'turn_left', 'turn_right']
        self.action_freq = {i: 0 for i in range(8)}


    def reset(self):
        self.step_count = 0
        self.total_reward = 0
        self.action_freq = {i: 0 for i in range(8)}
        # if in the middle of timesteps use last state o/w grab state from TA2
        if not self.last_state_dict:
            self.last_state_dict = self.state_queue.get()
            self.enemy_initial_health = sum(map(lambda x: x['health'], 
                self.last_state_dict['enemies']))
            self.player_initial_health = self.last_state_dict['player']['health']
        # checks whether training has ended
        if self.last_state_dict == {}:
            raise RuntimeError()
        state = vectorize_state(self.last_state_dict)
        self.log.debug('Resetting State')
        return state

    def step(self, action):
        # passes action to TA2
        self.action_freq[action] += 1
        self.action_queue.put(action)
        # grabs next state from TA2
        next_state_dict = self.state_queue.get()
        # checks whether training has ended
        if next_state_dict == {}:
            raise RuntimeError()
        # if next state is None, uses last state
        next_state_dict = next_state_dict or self.last_state_dict
        next_state = vectorize_state(next_state_dict)
        # grabs terminal flag and performance from TA2
        done = self.terminal_queue.get()
        performance = self.performance_queue.get()

        def compute_reward(state, next_state):
            enemy_kill = len(state['enemies']) - len(next_state['enemies'])
            enemy_damage = 0.001 * (sum(map(lambda x: x['health'], state['enemies'])) 
                - sum(map(lambda x: x['health'], next_state['enemies']))) / self.enemy_initial_health
            player_damage = -0.001 * (state['player']['health'] - next_state['player']['health']) / self.player_initial_health
            return enemy_kill + enemy_damage + player_damage

        reward = (5 if performance else 0) if done \
            else compute_reward(self.last_state_dict, next_state_dict)

        # increment logging variables
        self.step_count += 1
        self.total_reward += reward
        # force reset() to grab a state from TA2 if episode has concluded
        # o/w give reset() the option to use last state
        self.last_state_dict = None if done else next_state_dict
        if done:
            # log total reward accumulated during episode
            self.log.info(f'Training Episode End: reward={self.total_reward} steps={self.step_count}')
            log_dict = {'reward': self.total_reward, 'length': self.step_count}
            for i in range(8):
                log_dict[self.actions[i]] = self.action_freq[i] / self.step_count
            wandb.log(log_dict)
        return next_state, reward, done, {}

    def render(self):
        raise NotImplementedError()
