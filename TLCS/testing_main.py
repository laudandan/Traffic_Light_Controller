from __future__ import absolute_import
from __future__ import print_function

import os
from shutil import copyfile

import traci

from testing_simulation import Simulation, simulation_step
from generator import TrafficGenerator
from model import TestModel
from visualization import Visualization
from utils import import_test_configuration, set_sumo, set_test_path


if __name__ == "__main__":

    config = import_test_configuration(config_file='testing_settings.ini')
    sumo_cmd = set_sumo(config['gui'], config['sumocfg_file_name'], config['max_steps'])
    model_path, plot_path = set_test_path(config['models_path_name'], config['model_to_test'])

    model = TestModel(
        input_dim=config['num_states'],
        model_path=model_path,
        type=0
    )
    model1 = TestModel(
        input_dim=config['num_states'],
        model_path=model_path,
        type=0
    )

    TrafficGen = TrafficGenerator(
        config['max_steps'], 
        config['n_cars_generated']
    )

    Visualization = Visualization(
        plot_path, 
        dpi=96
    )
    incoming_roads = ["E2TL", "N2TL", "W2TL", "S2TL"]
    incoming_roads1 = ["E2TL1", "N2TL1", "E2TL", "S2TL1"]
    simulation = Simulation(
        model,
        TrafficGen,
        sumo_cmd,
        config['max_steps'],
        config['green_duration'],
        config['yellow_duration'],
        config['num_states'],
        config['num_actions'],
        incoming_roads=incoming_roads,
        state_nr=0,
        ts="TL"
    )
    simulation1 = Simulation(
        model1,
        TrafficGen,
        sumo_cmd,
        config['max_steps'],
        config['green_duration'],
        config['yellow_duration'],
        config['num_states'],
        config['num_actions'],
        incoming_roads=incoming_roads1,
        state_nr=1,
        ts="TL1"
    )
    steps = 0
    simulation.start(episode=config['episode_seed'])

    while steps < config['max_steps']:
        simulation.running()
        simulation1.running()
        simulation_step()
        steps += 1
    simulation_time = simulation.close()

    print('\n----- Test episode')
    # print('Simulation time:', simulation_time, 's')

    print("----- Testing info saved at:", plot_path)

    copyfile(src='testing_settings.ini', dst=os.path.join(plot_path, 'testing_settings.ini'))

    list_reward = []
    for(item1, item2) in zip(simulation.reward_episode, simulation1.reward_episode):
        list_reward.append(item1+item2)

    list_queue = []
    for (item1, item2) in zip(simulation.queue_length_episode, simulation1.queue_length_episode):
        list_queue.append(item1 + item2)

    Visualization.save_data_and_plot(data=list_reward, filename='reward', xlabel='Action step', ylabel='Reward')
    Visualization.save_data_and_plot(data=list_queue, filename='queue', xlabel='Step', ylabel='Queue lenght (vehicles)')
