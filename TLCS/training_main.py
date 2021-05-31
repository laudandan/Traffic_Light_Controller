from __future__ import absolute_import
from __future__ import print_function

import os
import datetime
from shutil import copyfile

from training_simulation import Simulation, simulation_step
from generator import TrafficGenerator
from memory import Memory
from model import TrainModel
from visualization import Visualization
from utils import import_train_configuration, set_sumo, set_train_path


if __name__ == "__main__":

    config = import_train_configuration(config_file='training_settings.ini')
    sumo_cmd = set_sumo(config['gui'], config['sumocfg_file_name'], config['max_steps'])
    path = set_train_path(config['models_path_name'])

    model1 = TrainModel(
        config['num_layers'], 
        config['width_layers'], 
        config['batch_size'], 
        config['learning_rate'], 
        input_dim=config['num_states'], 
        output_dim=config['num_actions']
    )

    model2 = TrainModel(
        config['num_layers'],
        config['width_layers'],
        config['batch_size'],
        config['learning_rate'],
        input_dim=config['num_states'],
        output_dim=config['num_actions']
    )

    memory1 = Memory(
        config['memory_size_max'],
        config['memory_size_min']
    )

    memory2 = Memory(
        config['memory_size_max'],
        config['memory_size_min']
    )

    TrafficGen = TrafficGenerator(
        config['max_steps'], 
        config['n_cars_generated']
    )

    Visualization = Visualization(
        path, 
        dpi=96
    )

    incoming_roads = ["E2TL", "N2TL", "W2TL", "S2TL"]
    incoming_roads1 = ["E2TL1", "N2TL1", "E2TL", "S2TL1"]
    simulation = Simulation(
        model1,
        memory1,
        TrafficGen,
        sumo_cmd,
        config['gamma'],
        config['max_steps'],
        config['green_duration'],
        config['yellow_duration'],
        config['num_states'],
        config['num_actions'],
        config['training_epochs'],
        incoming_roads=incoming_roads,
        state_nr=0,
        ts="TL"
    )
    simulation1 = Simulation(
        model1,
        memory1,
        TrafficGen,
        sumo_cmd,
        config['gamma'],
        config['max_steps'],
        config['green_duration'],
        config['yellow_duration'],
        config['num_states'],
        config['num_actions'],
        config['training_epochs'],
        incoming_roads=incoming_roads1,
        state_nr=1,
        ts="TL1"
    )

    timestamp_start = datetime.datetime.now()
    steps = 0
    episode = 0
    epsilon = 0

    # Iterate episodes of simulation
    while episode < config['total_episodes']:
        simulation.before_running(episode)
        print('\n----- Episode', str(episode + 1), 'of',
              str(config['total_episodes']))
        simulation.start_sumo()
        epsilon = 1.0 - (episode / config['total_episodes'])

        # An episode of simulation
        while steps < config['max_steps']:
            simulation.running(epsilon)
            simulation1.running(epsilon)
            simulation_step()
            steps += 1
        steps = 0
        episode += 1
        simulation_time, training_time = simulation.after_running(epsilon)
        print('Simulation time:', simulation_time, 's - Training time:', training_time, 's - Total:',
              round(simulation_time + training_time, 1), 's')
        simulation.close_sumo()

    print("\n----- Start time:", timestamp_start)
    print("----- End time:", datetime.datetime.now())
    print("----- Session info saved at:", path)

    model1.save_model(path, 0)
    model2.save_model(path, 1)

    copyfile(src='training_settings.ini', dst=os.path.join(path, 'training_settings.ini'))

    Visualization.save_data_and_plot(data=simulation.reward_store, filename='reward', xlabel='Episode', ylabel='Cumulative negative reward')
    Visualization.save_data_and_plot(data=simulation.cumulative_wait_store, filename='delay', xlabel='Episode', ylabel='Cumulative delay (s)')
    Visualization.save_data_and_plot(data=simulation.avg_queue_length_store, filename='queue', xlabel='Episode', ylabel='Average queue length (vehicles)')