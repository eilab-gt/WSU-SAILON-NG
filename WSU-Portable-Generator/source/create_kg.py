"""
create_kg.py
Class for knowledge graph creation & associated operations.
"""
from __future__ import annotations
from abc import abstractmethod, ABC
from typing import Type
from typing import TYPE_CHECKING
import os
from distutils.command.clean import clean
import networkx as nx
import matplotlib.pyplot as plt
from scipy.spatial import distance
#import pandas as pd
#import csv
from math import floor
import numpy as np
import copy
#if TYPE_CHECKING:
#    from framework.envs.base_env import BaseEnv
class KG():
    """
    Class for knowledge graph.
    """
    def __init__(self, state_mask=['angle', 'relative_distance'], rule_learner_mask=['x_position', 'y_position', 'z_position','id','name','value'], image_path = 'kg_img/', **kwargs):
        """
        Initialize the knowledge graph constructor
        Arguments:
            state_mask: list of strings of keys to remove from observation JSON on ingest
            rule_learner_mask: list of strings of keys to remove from node dictionaries when creating a list of dictionaries to pass to the rule learner
            image_path: path to save images to
        """
        self.state_mask = state_mask
        self.rule_learner_mask = rule_learner_mask
        self.image_path = image_path
        if not os.path.exists(self.image_path):
            os.makedirs(image_path)
    def state_to_graph(self, state):
        """
        Arguments:
            state: JSON state representation
        Returns: a networkx graph of the state observation.
        """
        #TODO: update this to expect to call NoveltyDetector.__get_states_from_env() as the state
        #append all objects in the environment, add any extra graph processing here
        data_list = []
        for enemy_dict in state['enemies']:
            data_dict, id = self.clean_dict_vd('enemies', enemy_dict)
            data_list.append(data_dict)
        for item_type in state['items'].keys():
            for item in state['items'][item_type]:
                data_dict, id = self.clean_dict_vd(item_type, item)
                data_list.append(data_dict)
        player_id = 0 #state['player']['id'] #TODO: this is vizdoom-specific
        state['player']['type'] = 'player'
        state['player']['value'] = 'player'
        state['player']['uid'] = player_id
        state['player']['label'] = 'player'
        state['player']['id'] = 'player'
        for key in self.state_mask:
            try: del state['player'][key]
            except: pass
        data = {'data': state['player']}
        data_list.append(data)
        graph_dict = {
            'data': [],
            'directed': False,
            'multigraph': False,
            'elements': {'nodes': data_list,
            'edges': []}
        }
        G = nx.cytoscape_graph(graph_dict)
        return G
    
    def enrich_graph(self, G):
        """
        Enrich the observation graph with e.g. relative distances
        """
        # get iterable to calculate relative distances over
        #zombies = [node for node, attribute in G.nodes(data=True) if 'ZombieMan' in attribute['value']]
        not_player = [node for node, attribute in G.nodes(data=True) if attribute['value'] not in ['player']]
        for ob in not_player:
            self.append_distance(G, 'player', ob)
        return G
    def graph_to_dict_list(self, G, list_len=100):
        """
        Convert graph representation to a list of dictionary objects
        Each object should be in the same index in the list no matter which state observation is passed in
        objects will be indexed based on their 'uid' field in the node dictionary, and will be unique for every entity
        Arguments:
            kg_size: Int. List size for rule learner pipeline. Size may need to adjust based on future novelties or observation changes. Represents total number of objects tracked in KG
        """
        state_dict_list = np.array([{}]*list_len)
        # for each node in the graph, pull the dictionary associated with that node
        # mask these dictionaries before passing to the rule learner
        # first get the player node (do not append relative distance to player to this node)
        node_dict = G.nodes['player']
        list_position = node_dict['uid']
        for key in self.rule_learner_mask:
            try: del node_dict[key]
            except: pass
        state_dict_list[list_position] = node_dict
        # then get the other nodes (append relative distance to player)
        not_player = [node for node, attribute in G.nodes(data=True) if attribute['type'] not in ['player']]
        _, player_neighbor_dictionary = [(n, nbrdict) for n, nbrdict in G.adjacency() if n == 'player'][0]
        for npc in not_player:
            node_dict = G.nodes[npc]
            list_position = node_dict['uid']
            node_dict['relative_distance_to_player'] = player_neighbor_dictionary[npc]['distance']
            for key in self.rule_learner_mask:
                try: del node_dict[key]
                except: pass
            state_dict_list[list_position] = node_dict
        return state_dict_list
    def visualize_graph(self,G,figname='enriched_graph'):
        """
        matplotlib plot of graph
        """
        pos = nx.spring_layout(G)
        plt.figure()
        nx.draw_networkx(G)
        edge_labels = {}
        for edge in list(G.edges):
            edge_labels[edge] = floor(G.edges[edge]['distance'])
        nx.draw_networkx_edge_labels(
            G, pos,
            edge_labels=edge_labels,
            font_color='red'
        )
        plt.axis('off')
        plt.savefig(self.image_path + figname + '.png')
        plt.close()
        return
    #######################################
    ## VIZDOOM-SPECIFIC HELPER FUNCTIONS ##
    #######################################
    
    def clean_dict_vd(self, obs_key, dict):
        # DISCLAIMER: Vizdoom states are weird and items are structured slightly differently, and some have the 
        # type field and some do not (ex. traps), which is why this is slightly ugly
        #TODO: This function is pretty brittle
        dict = copy.deepcopy(dict) #just precautionary if we need to use the raw state later
        id = dict['id']
        if obs_key == 'enemies':
            dict['type'] = dict['name']
        keys = dict.keys()
        if 'type' in keys:
            dict['value'] = dict['type'] + '_' + str(id)
            dict['uid'] = dict['id']
            dict['id'] = dict['type'] + '_' + str(id) # <------ This is purely for readability when observing what the rule learner sees, not necessary
        elif 'name' in keys:
            dict['value'] = dict['name'] + '-' + str(id)
            dict['uid'] = dict['id']
            dict['id'] = dict['name'] + '_' + str(id)
        else:
            dict['value'] = str(obs_key) + '_' + str(id)
            dict['uid'] = dict['id']
            dict['id'] = str(obs_key) + '_' + str(id)
            dict['label'] = str(id)
            dict['type'] = str(obs_key)

        dict['label'] = str(dict['value']) # need this for nice dash plotting

        for key in self.state_mask:
            try: del dict[key]
            except: pass
        
        data_dict = {'data': dict}
        return data_dict, id
            
    ##############################
    ## GENERIC HELPER FUNCTIONS ##
    ##############################
    def calculate_distance(self, obj1, obj2):
        pos1 = (obj1['x_position'], obj1['y_position'], obj1['z_position'])
        pos2 = (obj2['x_position'], obj2['y_position'], obj2['z_position'])
        return distance.euclidean(pos1, pos2)
    def append_distance(self, G, node1, node2):
        dist = self.calculate_distance(G.nodes[node1], G.nodes[node2])
        G.add_edge(node1, node2, distance=dist, name='distance', label='distance: %s' %(floor(dist),))
        return G