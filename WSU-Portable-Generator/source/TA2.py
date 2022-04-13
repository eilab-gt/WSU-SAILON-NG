#!/usr/bin/env python3
# ************************************************************************************************ #
# **                                                                                            ** #
# **    AIQ-SAIL-ON TA2 Agent Example                                                           ** #
# **                                                                                            ** #
# **        Brian L Thomas, 2020                                                                ** #
# **                                                                                            ** #
# **  Tools by the AI Lab - Artificial Intelligence Quotient (AIQ) in the School of Electrical  ** #
# **  Engineering and Computer Science at Washington State University.                          ** #
# **                                                                                            ** #
# **  Copyright Washington State University, 2020                                               ** #
# **  Copyright Brian L. Thomas, 2020                                                           ** #
# **                                                                                            ** #
# **  All rights reserved                                                                       ** #
# **  Modification, distribution, and sale of this work is prohibited without permission from   ** #
# **  Washington State University.                                                              ** #
# **                                                                                            ** #
# **  Contact: Brian L. Thomas (bthomas1@wsu.edu)                                               ** #
# **  Contact: Larry Holder (holder@wsu.edu)                                                    ** #
# **  Contact: Diane J. Cook (djcook@wsu.edu)                                                   ** #
# ************************************************************************************************ #

import queue
import random
import threading
import time
import math
import numpy as np
from gym import spaces, Env
from stable_baselines3 import PPO
from vizdoom_simple_novelty_detector import VizdoomSimpleNoveltyDetector
from objects.TA2_logic import TA2Logic


class VizDoomEnv(Env):

    def __init__(self):
        self.observation_space = spaces.Box(-1, 1, shape=(22,))
        self.action_space = spaces.Discrete(8)

    def reset(self):
        pass

    def step(self, action):
        pass

    def render(self, mode="human"):
        pass


class ThreadedProcessingExample(threading.Thread):
    def __init__(self, processing_object: list, response_queue: queue.Queue):
        threading.Thread.__init__(self)
        self.processing_object = processing_object
        self.response_queue = response_queue
        self.is_done = False
        return

    def run(self):
        """All work tasks should happen or be called from within this function.
        """
        is_novel = False
        message = ''

        # Do some fake work here.
        for work in self.processing_object:
            sum = 0
            for i in range(work):
                sum += i
            message += 'Did sum of {}. '.format(work)

        self.response_queue.put((is_novel, message))
        return

    def stop(self):
        self.is_done = True
        return


class TA2Agent(TA2Logic):

    def __init__(self):
        super().__init__()

        self.possible_answers = list()
        # This variable can be set to true and the system will attempt to end training at the
        # completion of the current episode, or sooner if possible.
        self.end_training_early = False
        # This variable is checked only during the evaluation phase.  If set to True the system
        # will attempt to cleanly end the experiment at the conclusion of the current episode,
        # or sooner if possible.
        self.end_experiment_early = False
        self.novelty_detector = VizdoomSimpleNoveltyDetector()
        self.env = VizDoomEnv()
        self.model = PPO('MlpPolicy', self.env)
        self.model.set_parameters('vizdoom_baseline.zip')
        self.map_norm = 1000
        self.health_norm = 100
        self.ammo_norm = 100
        self.rooms = [
            {'x': (-512, -384), 'y': (-320, 320), 'index': [0, 5]},
            {'x': (-320, 320), 'y': (-512, -384), 'index': [1, 7]},
            {'x': (-320, 320), 'y': (384, 512), 'index': [2, 8]},
            {'x': (384, 512), 'y': (-320, 320), 'index': [3, 6]},
            {'x': (-320, 320), 'y': (-320, 320), 'index': [4, 5, 6, 7, 8]},
            {'x': (-384, -320), 'y': (-64, 64), 'index': [5, 0, 4]},
            {'x': (320, 384), 'y': (-64, 64), 'index': [6, 3, 4]},
            {'x': (-64, 64), 'y': (-384, -320), 'index': [7, 1, 4]},
            {'x': (-64, 64), 'y': (320, 384), 'index': [8, 2, 4]}
        ]
        self.item_types = {'health': 1, 'ammo': 0.5, 'trap': 0, 'obstacle': -0.5}
        self.possible_answers = [{'action': 'nothing'}, {'action': 'left'}, {'action': 'right'},
                                 {'action': 'forward'}, {'action': 'backward'}, {'action': 'shoot'},
                                 {'action': 'turn_left'}, {'action': 'turn_right'}]
        return

    def _vectorize_state(self, obs):

        def get_room(obj):
            x, y = obj['x_position'], obj['y_position']
            for room in self.rooms:
                if room['x'][0] <= x <= room['x'][1] \
                        and room['y'][0] <= y <= room['y'][1]:
                    return room['index']

        def normalize(value, offset, norm):
            return (value + offset) / (norm or 1)

        def calculate_relative_angle(a, b):
            if abs(a - b) < abs(a - b - 360):
                return a - b
            else:
                return a - b - 360

        def normalize_angle(angle, norm):
            if angle > 180:
                return (360 - angle) / norm
            elif angle < -180:
                return (angle + 360) / norm
            else:
                return angle / norm

        vector = np.zeros(22) - 1

        vector[0] = normalize(obs['player']['x_position'], self.map_norm / 2, self.map_norm)
        vector[1] = normalize(obs['player']['y_position'], self.map_norm / 2, self.map_norm)
        vector[2] = normalize(obs['player']['angle'], 0, 360)
        vector[3] = normalize(obs['player']['health'], 0, self.health_norm)
        vector[4] = normalize(obs['player']['ammo'], 0, self.ammo_norm)

        player = obs['player']
        player_room = get_room(player)
        player_coordinate = np.array([player['x_position'], player['y_position']])

        relevant_enemies = []
        for enemy in obs['enemies']:
            enemy_coordinate = np.array([enemy['x_position'], enemy['y_position']])
            dist = np.linalg.norm(player_coordinate - enemy_coordinate)
            enemy['relative_distance'] = dist
            relevant_enemies.append(enemy)
        relevant_enemies = sorted(relevant_enemies, key=lambda x: x['relative_distance'])[:2]

        pointer = 5
        for i in range(len(relevant_enemies)):
            offset = i * 4
            enemy = relevant_enemies[i]
            enemy_coordinate = np.array([enemy['x_position'], enemy['y_position']])
            angle = math.atan2(enemy_coordinate[1] - player_coordinate[1],
                               enemy_coordinate[0] - player_coordinate[0]) * 180 / math.pi
            relative_angle = calculate_relative_angle(player['angle'], angle)
            relative_aim_angle = calculate_relative_angle(enemy['angle'], angle)
            vector[pointer + offset] = normalize(enemy['health'], 0, self.health_norm)
            vector[pointer + offset + 1] = normalize(enemy['relative_distance'], 0, self.map_norm)
            vector[pointer + offset + 2] = normalize_angle(relative_angle, 180)
            vector[pointer + offset + 3] = normalize_angle(relative_aim_angle, 180)

        relevant_items = []
        for item_type in self.item_types:
            for item in obs['items'][item_type]:
                item_room = get_room(item)
                if item_room[0] not in player_room:
                    continue
                item_coordinate = np.array([item['x_position'], item['y_position']])
                dist = np.linalg.norm(player_coordinate - item_coordinate)
                item['type'] = item_type
                item['relative_distance'] = dist
                relevant_items.append(item)
        relevant_items = sorted(relevant_items, key=lambda x: x['relative_distance'])[:3]

        pointer = 13
        for i in range(len(relevant_items)):
            offset = i * 3
            item = relevant_items[i]
            item_coordinate = np.array([item['x_position'], item['y_position']])
            angle = math.atan2(item_coordinate[1] - player_coordinate[1],
                               item_coordinate[0] - player_coordinate[0]) * 180 / math.pi
            relative_angle = calculate_relative_angle(player['angle'], angle)
            vector[pointer + offset] = self.item_types[item['type']]
            vector[pointer + offset + 1] = normalize(item['relative_distance'], 0, self.map_norm)
            vector[pointer + offset + 2] = normalize_angle(relative_angle, 180)

        return vector

    def experiment_start(self):
        """This function is called when this TA2 has connected to a TA1 and is ready to begin
        the experiment.
        """
        self.log.info('Experiment Start')
        return

    def training_start(self):
        """This function is called when we are about to begin training on episodes of data in
        your chosen domain.
        """
        self.log.info('Training Start')
        return

    def training_episode_start(self, episode_number: int):
        """This function is called at the start of each training episode, with the current episode
        number (0-based) that you are about to begin.

        Parameters
        ----------
        episode_number : int
            This identifies the 0-based episode number you are about to begin training on.
        """
        self.log.info('Training Episode Start: #{}'.format(episode_number))
        return

    def training_instance(self, feature_vector: dict, feature_label: dict) -> dict:
        """Process a training

        Parameters
        ----------
        feature_vector : dict
            The dictionary of the feature vector.  Domain specific feature vector formats are
            defined on the github (https://github.com/holderlb/WSU-SAILON-NG).
        feature_label : dict
            The dictionary of the label for this feature vector.  Domain specific feature labels
            are defined on the github (https://github.com/holderlb/WSU-SAILON-NG). This will always
            be in the format of {'action': label}.  Some domains that do not need an 'oracle' label
            on training data will receive a valid action chosen at random.

        Returns
        -------
        dict
            A dictionary of your label prediction of the format {'action': label}.  This is
                strictly enforced and the incorrect format will result in an exception being thrown.
        """
        label_prediction = random.choice(self.possible_answers)
        return label_prediction

    def training_performance(self, performance: float, feedback: dict = None):
        """Provides the current performance on training after each instance.

        Parameters
        ----------
        performance : float
            The normalized performance score.
        feedback : dict, optional
            A dictionary that may provide additional feedback on your prediction based on the
            budget set in the TA1. If there is no feedback, the object will be None.
        """
        self.log.debug('Training Performance: {}'.format(performance))
        return

    def training_episode_end(self, performance: float, feedback: dict = None):
        """Provides the final performance on the training episode and indicates that the training
        episode has ended.

        Parameters
        ----------
        performance : float
            The final normalized performance score of the episode.
        feedback : dict, optional
            A dictionary that may provide additional feedback on your prediction based on the
            budget set in the TA1. If there is no feedback, the object will be None.

        Returns
        -------
        float, float, int, dict
            A float of the probability of there being novelty.
            A float of the probability threshold for this to evaluate as novelty detected.
            Integer representing the predicted novelty level.
            A JSON-valid dict characterizing the novelty.
        """
        self.log.info('Training Episode End: performance={}'.format(performance))

        novelty_probability = random.random()
        novelty_threshold = 0.8
        novelty = 0
        novelty_characterization = dict()

        return novelty_probability, novelty_threshold, novelty, novelty_characterization

    def training_end(self):
        """This function is called when we have completed the training episodes.
        """
        self.log.info('Training End')
        return

    def train_model(self):
        """Train your model here if needed.  If you don't need to train, just leave the function
        empty.  After this completes, the logic calls save_model() and reset_model() as needed
        throughout the rest of the experiment.
        """
        self.log.info('Train the model here if needed.')

        # Simulate training the model by sleeping.
        self.log.info('Simulating training with a 5 second sleep.')
        time.sleep(5)

        return

    def save_model(self, filename: str):
        """Saves the current model in memory to disk so it may be loaded back to memory again.

        Parameters
        ----------
        filename : str
            The filename to save the model to.
        """
        self.log.info('Save model to disk.')
        return

    def reset_model(self, filename: str):
        """Loads the model from disk to memory.

        Parameters
        ----------
        filename : str
            The filename where the model was stored.
        """
        self.log.info('Load model from disk.')
        return

    def trial_start(self, trial_number: int, novelty_description: dict):
        """This is called at the start of a trial with the current 0-based number.

        Parameters
        ----------
        trial_number : int
            This is the 0-based trial number in the novelty group.
        novelty_description : dict
            A dictionary that will have a description of the trial's novelty.
        """
        self.log.info('Trial Start: #{}  novelty_desc: {}'.format(trial_number,
                                                                  str(novelty_description)))
        return

    def testing_start(self):
        """This is called after a trial has started but before we begin going through the
        episodes.
        """
        self.log.info('Testing Start')
        return

    def testing_episode_start(self, episode_number: int):
        """This is called at the start of each testing episode in a trial, you are provided the
        0-based episode number.

        Parameters
        ----------
        episode_number : int
            This is the 0-based episode number in the current trial.
        """
        self.log.info('Testing Episode Start: #{}'.format(episode_number))

        # Throw something together to create work.
        # work_list = list([5000])
        # for i in range(3 + episode_number):
        #     work_list.append(50000)
        # response_queue = queue.Queue()
        # response = None
        # # Initialize the example of doing work safely outside the main thread.
        # # Remember, in Python all objects beyond int, float, bool, and str are passed
        # # by reference.
        # threaded_work = ThreadedProcessingExample(processing_object=work_list,
        #                                           response_queue=response_queue)
        # # Start the work in a separate thread.
        # threaded_work.start()
        #
        # while response is None:
        #     try:
        #         # Try to get the response from the queue for 5 seconds before we let the AMQP
        #         # network event loop do any required work (such as sending heartbeats) for
        #         # 0.5 seconds.  By having the get(block=True) we ensure that there is basically
        #         # no wait for the result once it is put in the queue.
        #         response = response_queue.get(block=True, )
        #     except queue.Empty:
        #         # Process any amqp events for 0.5 seconds before we try to get the results again.
        #         self.process_amqp_events()
        #
        # # Safely end and clean up the threaded work object.
        # threaded_work.stop()
        # threaded_work.join()
        #
        # self.log.info('message from threaded work: {}'.format(response[1]))
        # self.log.warning('Please remove this sample threaded work object from '
        #                  'testing_episode_start() before running actual experiments.')
        return

    def testing_instance(self, feature_vector: dict, novelty_indicator: bool = None) -> dict:
        """Evaluate a testing instance.  Returns the predicted label or action, if you believe
        this episode is novel, and what novelty level you beleive it to be.

        Parameters
        ----------
        feature_vector : dict
            The dictionary containing the feature vector.  Domain specific feature vectors are
            defined on the github (https://github.com/holderlb/WSU-SAILON-NG).
        novelty_indicator : bool, optional
            An indicator about the "big red button".
                - True == novelty has been introduced.
                - False == novelty has not been introduced.
                - None == no information about novelty is being provided.

        Returns
        -------
        dict
            A dictionary of your label prediction of the format {'action': label}.  This is
                strictly enforced and the incorrect format will result in an exception being thrown.
        """
        ##model funcs
        observation = self._vectorize_state(feature_vector)
        action, _ = self.model.predict(observation)
        label_prediction = self.possible_answers[action]
        
        #novelty
        
        state_valid_info = self.novelty_detector.on_step_result(feature_vector)
        if state_valid_info:
            #if there's info that we've returned
            self.novelty_probability, self.threshold, self.novelty, self.characterization = state_valid_info
        else:
            #no novelty
            self.novelty_probability, self.threshold, self.novelty, self.characterization = 0, 0, 0, {}
        
        return label_prediction

    def testing_performance(self, performance: float, feedback: dict = None):
        """Provides the current performance on training after each instance.

        Parameters
        ----------
        performance : float
            The normalized performance score.
        feedback : dict, optional
            A dictionary that may provide additional feedback on your prediction based on the
            budget set in the TA1. If there is no feedback, the object will be None.
        """
        return

    def testing_episode_end(self, performance: float, feedback: dict = None):
        """Provides the final performance on the testing episode.

        Parameters
        ----------
        performance : float
            The final normalized performance score of the episode.
        feedback : dict, optional
            A dictionary that may provide additional feedback on your prediction based on the
            budget set in the TA1. If there is no feedback, the object will be None.

        Returns
        -------
        float, float, int, dict
            A float of the probability of there being novelty.
            A float of the probability threshold for this to evaluate as novelty detected.
            Integer representing the predicted novelty level.
            A JSON-valid dict characterizing the novelty.
        """
        self.log.info(
            'Testing Episode End: performance={}'.format(performance))

        # novelty_probability = random.random()
        # novelty_threshold = 0.8
        # novelty = random.choice(list(range(4)))
        # novelty_characterization = dict()
        self.log.info("NOVELTY INFO: %s, %s, %s, %s", (self.novelty_probability, self.threshold, self.novelty, self.characterization))
        return self.novelty_probability, self.threshold, self.novelty, self.characterization

    def testing_end(self):
        """This is called after the last episode of a trial has completed, before trial_end().
        """
        self.log.info('Testing End')
        return

    def trial_end(self):
        """This is called at the end of each trial.
        """
        self.log.info('Trial End')
        return

    def experiment_end(self):
        """This is called when the experiment is done.
        """
        self.log.info('Experiment End')
        return


if __name__ == "__main__":
    agent = TA2Agent()
    agent.run()
