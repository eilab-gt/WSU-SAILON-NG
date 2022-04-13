"""
vizdoom_novelty_detector.py
Novelty detector for vizdoom domain. Uses the KG repo as well as the EngineLearning repo
This will load in the rule model, and use its novelty detection components to determine if there
is a novelty present
"""
from __future__ import annotations
from asyncio.format_helpers import extract_stack

from create_kg import KG
#from EngineLearning import *     # change this, placeholder for now

import json
import copy
#from test_csv_trajectory import get_test_states_actions_rewards

#from .objects.make_vizdoom_json_functions import add_state_to_json_dict
import copy
import math

class VizdoomSimpleNoveltyDetector():
    def __init__(self, state_mask=['angle', 'relative_distance'], rule_mask=['name','label','value','id', 'x_position', 'y_position', 'z_position', 'uid', 'relative_distance_to_player'], JSON_path="vizdoom_json.json", **kwargs):
        self.KG = KG(state_mask=state_mask, rule_learner_mask=rule_mask)
        with open(JSON_path, 'r') as json_file:
            self.json = json.load(json_file)
        #self.states = get_test_states_actions_rewards()
        #self.rule_model = GameCloneLearning.RuleModel(rule_model_path)  #load in rule pickl file here? Need to figure this out here

        self.discrete_properties = ['type']
        self.continuous_properties = ['health', 'relative_distance_to_player', 'ammo', 'uid', 'x_position', 'y_position', 'z_position']

        self.empty_json_dict = {
            "prop_continuous":{
            },
            "prop_discrete":{
            },
            "max_num_entities":None,
            "hashables":[
                "uid"
            ]
            }

        self.empty_discrete_property_dict = {
                    "min_set": None,
                    "mapping":{
                    }
                }

        self.empty_continuous_property_dict = {
                    "dtype":None,
                    "min":None,
                    "max":None,
                    "max_diff":None
                }

        
        self.json_dump_path = "vizdoom_json_dump.json"
        try:
            with open(self.json_dump_path, 'r') as filename:
                self.json_dict = json.load(filename)
            #print("loaded json")
        except:
            self.json_dict = {}
            #print("did not load json")
        self.count = 0
        self.last_state = None

        """
        initialize KG and rule model params as well as anything else we need
        """

    def on_reset(self):
        pass
    #stub for now

    def on_step_result(self, state=None, update_JSON = False): ##TODO for TA2: make sure this state is as expected and this function still runs
        """
        Takes in the current time step state, then returns the rule models output for novelty detection
        Steps:
            1. construct enriched KG from state info
            2. pass it into rule model 
            3. report boolean is_novelty
        """
        if state == None:
            state = copy.deepcopy(self.env.get_state('pos'))
        else:
            pass
        KG_graph = self.KG.state_to_graph(state)
        KG_enriched = self.KG.enrich_graph(KG_graph)
        KG_dict = self.KG.graph_to_dict_list(KG_enriched)
        KG_dict_filtered = list(filter(None, KG_dict))
        if update_JSON:
            self.update_JSON_from_observation(state) # update the json of expected observations
        else:
            pass
        state_validated = self.validate_state(KG_dict_filtered) # return boolean is_novelty
        self.last_state = state
        return not state_validated


    #helper function to get all occurences of entities
    def extract_entities(self, entity_name, dict_list):
        return [entity for entity in dict_list if entity_name in entity['id']]

    
    def update_JSON_from_observation(self, state):
        new_json_dict = self.add_state_to_json_dict(state)
        self.json_dict = copy.deepcopy(new_json_dict)
        self.count += 1
        if self.count % 1000 == 0:
            #print("dumping json")
            #print(self.json_dict)
            #print("")
            #print(new_json_dict)
            with open(self.json_dump_path, 'w') as outfile:
                json.dump(self.json_dict, outfile, indent=4)

        return


    def add_entity_to_json(self, entity, entity_type_list, obs_entity_types, new_json_dict):
        if entity['type'] in new_json_dict:
            # existing entity type
            entity_dict = copy.deepcopy(new_json_dict[entity['type']])
            # assume all entity properties are static so if we have seen an entity type once then all the properties for that entity already exist #TODO: is this a valid assumption?
            for prop in entity:
                if prop in self.continuous_properties:
                    prop_dict = copy.deepcopy(entity_dict['prop_continuous'][prop])
                    #step_diff = abs(entity[prop] - self.last_state[])
                    # check if the new observed value is outside of the defined range and adjust if needed
                    if prop_dict['min'] > entity[prop]:
                        prop_dict['min'] = math.floor(entity[prop]) -1# round a little
                        entity_dict['prop_continuous'][prop] = copy.deepcopy(prop_dict)
                    elif prop_dict['max'] < entity[prop]:
                        prop_dict['max'] = math.ceil(entity[prop]) +1# round a little
                        entity_dict['prop_continuous'][prop] = copy.deepcopy(prop_dict)
                    else:
                        pass

                elif prop in self.discrete_properties:
                    prop_dict = copy.deepcopy(entity_dict['prop_discrete'][prop])
                    # check if the observed discrete value is already in the mapping
                    if entity[prop] in prop_dict['mapping'].values():
                        # check if the observed value is in the min set:
                        # get a list of the keys (should be 1) for the observed discrete value
                        mapping_key = [key for key in prop_dict['mapping'] if prop_dict['mapping'][key] == entity[prop]][0]
                        if mapping_key in prop_dict['min_set']:
                            pass
                        else:
                            prop_dict['min_set'] = prop_dict['min_set'] + [mapping_key]
                            entity_dict['prop_discrete'][prop] = copy.deepcopy(prop_dict)
                    else:
                        # if not add a new mapping entry
                        next_key = max(prop_dict['mapping'].keys()) + 1
                        prop_dict['mapping'][next_key] = entity[prop]
                        # and also add this key to the min set
                        prop_dict['min_set'] = prop_dict['min_set'] + [next_key]
                        entity_dict['prop_discrete'][prop] = copy.deepcopy(prop_dict)
                else:
                    print("Warning: Unknown property type %s. Please assign to either discrete or continuous properties." % prop)
            # check if we need to increase the max number of observed entities of this type
            # we don't need to check this every time but I am lazy #TODO
            num_entities = entity_type_list.count(entity['type'])
            if num_entities > entity_dict['max_num_entities']:
                entity_dict['max_num_entities'] = num_entities
            new_json_dict[entity['type']] = copy.deepcopy(entity_dict)
        else:
            # new entity type
            entity_dict = copy.deepcopy(self.empty_json_dict)
            for prop in entity:
                if prop in self.continuous_properties:
                    prop_dict = copy.deepcopy(self.empty_continuous_property_dict)
                    prop_dict['dtype'] = type(entity[prop]).__name__
                    # we only have one observation of the value of this property so that observation is both the max and min value
                    prop_dict['min'] = entity[prop]-1
                    prop_dict['max'] = entity[prop]+1
                    entity_dict['prop_continuous'][prop] = copy.deepcopy(prop_dict)

                elif prop in self.discrete_properties:
                    prop_dict = copy.deepcopy(self.empty_discrete_property_dict)
                    # the mapping contains one entry
                    prop_dict['min_set'] = [0]
                    prop_dict['mapping'] = {0:entity[prop]}
                    entity_dict['prop_discrete'][prop] = copy.deepcopy(prop_dict)
                else:
                    print("Warning: Unknown property type %s. Please assign to either discrete or continuous properties." % prop)
            entity_dict['max_num_entities'] = entity_type_list.count(entity['type'])
            new_json_dict[entity['type']] = copy.deepcopy(entity_dict)
        return new_json_dict


    def add_state_to_json_dict(self, state):
        KG_graph = self.KG.state_to_graph(state)
        KG_enriched = self.KG.enrich_graph(KG_graph)
        KG_dict = self.KG.graph_to_dict_list(KG_enriched)
        KG_dict_filtered = list(filter(None, KG_dict))
        # enable counting the number of entities of each type for the whole observation
        entity_type_list = [obs['type'] for obs in KG_dict_filtered]
        obs_entity_types = list(set([obs['type'] for obs in KG_dict_filtered])) # list of unique entity types

        # for each dict (entity) in dict_list
        #entity = KG_dict_filtered[2]
        new_json_dict = copy.deepcopy(self.json_dict)
        for entity in KG_dict_filtered:
            new_json_dict = self.add_entity_to_json(entity, entity_type_list, obs_entity_types, new_json_dict)
        #self.json_dict = json_dict
        return new_json_dict

    def validate_state(self, dict_list):
        env_json = self.json
        #print(self.json)
        #check for new entities
        not_inventory_list = dict_list
        novelty_info = True
        for entity in not_inventory_list:
            # check that non-inventory entities are in the domain json
            if entity['type'] in env_json:
                pass
            else:
                print("Warning: Unknown entity type %s" % entity['type'])
                novelty_info = self.report_novelty(entity)
                print(entity)
        
        # check that the number of entities is <= expected number
        for json_entity in env_json:
            entity_list = [entity for entity in dict_list if json_entity in entity['type']]
            num_entities = len(entity_list)
            if num_entities > env_json[json_entity]['max_num_entities']:
                print("Warning: More entities of type %s than expected. Expected no more than %s. Saw %s." % (json_entity, env_json[json_entity]['max_num_entities'], num_entities))
                novelty_info = self.report_novelty(json_entity)
                #print(json_entity)
            else:
                pass
                
        

        # check that entity properties are as expected
        for json_entity in env_json: # we can recurse through all the entities in the env_json instead of all entities in the observation because we already checked that all of our observed entities exist in the env_json
            entity_list = [entity for entity in dict_list if json_entity in entity['type']]
            env_json_props = env_json[json_entity] # structure of expected properties of json_entity
            #print(env_json_props)
            for entity in entity_list:
                for prop in entity:
                    #check that all properties listed in the observed entity exist in the json list of expected properties
                    if prop not in env_json_props['prop_continuous'] and \
                        prop not in env_json_props['prop_discrete']:
                        print("Warning: Unknown property key %s in observed entity %s does not exist in the expected JSON structure" % (prop, entity))
                        novelty_info = self.report_novelty(entity)
                    #check if properties are in bounds
                    else:
                        # first check continuous properties
                        if prop in env_json_props['prop_continuous']:
                            env_json_prop = env_json_props['prop_continuous'][prop]
                            if entity[prop] < env_json_prop['min'] or entity[prop] > env_json_prop['max']:
                                print("Warning: Expected property %s to be in range: %s to %s but got property value of %s for entity %s" % (prop, env_json_prop['min'], env_json_prop['max'], entity[prop], entity))
                                novelty_info = self.report_novelty(entity)
                        # check discrete properties
                        elif prop in env_json_props['prop_discrete']:
                            env_json_prop = env_json_props['prop_discrete'][prop]
                            # if there is a mapping:
                            if 'mapping' in env_json_prop.keys():
                                # find the entity value in the mapping
                                mapped_value_list = [key for key in env_json_prop['mapping'] if env_json_prop['mapping'][key] == entity[prop]] # is a list of strings. Should have length 1.
                                if len(mapped_value_list) == 0:
                                    print("Warning: Unknown mapped value for discrete property %s in observed entity %s" % (prop, entity))
                                    novelty_info = self.report_novelty(entity)
                                else:
                                    # if we found a mapped value
                                    # TODO: handle if we found more than one mapped value?
                                    mapped_value = int(mapped_value_list[0])
                                    if 'range' in env_json_prop['min_set']:
                                        range_bound = int(env_json_prop['min_set'].split(',')[-1])
                                        acceptable_values = [x for x in range(range_bound)]
                                    else:
                                        #otherwise we are given a list of acceptable values
                                        acceptable_values = env_json_prop['min_set']

                                    if mapped_value not in acceptable_values:
                                        print("Warning: Mapped value for discrete property %s outside of acceptable range for entity %s" % (prop, entity))
                                        novelty_info = self.report_novelty(entity)
                                    else: pass
                            else: # if a range of acceptable values is given
                                print("Invalid discrete variable format")

        #print("  STATE OBSERVATION VALIDATED")
        return novelty_info

            

    def report_novelty(self, info): ##TODO for TA2: change this reporting to what TA2 expects to have
        # For TA2, we will return novelty_probability, novelty_threshold, novelty, novelty_characterization. Initialize 1s for now 
        #calls to report novelty will pass the entity that is observed to be novel to return
        return 1, 1, 1, info


#test = VizdoomSimpleNoveltyDetector()
#print(test.on_step_result())