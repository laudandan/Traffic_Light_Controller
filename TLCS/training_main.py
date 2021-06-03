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

    memory1 = Memory(
        config['memory_size_max'],
        config['memory_size_min']
    )

    model2 = TrainModel(
        config['num_layers'],
        config['width_layers'],
        config['batch_size'],
        config['learning_rate'],
        input_dim=config['num_states'],
        output_dim=config['num_actions']
    )

    memory2 = Memory(
        config['memory_size_max'],
        config['memory_size_min']
    )

    traffic_generator = TrafficGenerator(
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
        traffic_generator,
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
        neighbor_nr=1,
        ts="TL"
    )
    simulation1 = Simulation(
        model2,
        memory2,
        traffic_generator,
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
        neighbor_nr=0,
        ts="TL1"
    )

    timestamp_start = datetime.datetime.now()
    steps = 0
    episode = 0
    epsilon = 0

    list_reward = []
    # Iterate episodes of simulation
    sum_neg_reward1 = 0
    sum_neg_reward2 = 0
    while episode < config['total_episodes']:
        simulation.before_running(episode)
        simulation1.before_running(episode)
        simulation.generate_route(episode)
        print('\n----- Episode', str(episode + 1), 'of',
              str(config['total_episodes']))
        simulation.start_sumo()
        epsilon = 1.0 - (episode / config['total_episodes'])

        # An episode of simulation
        while steps < config['max_steps']:
            simulation.running(epsilon)
            simulation1.running(epsilon)
            list_reward.append(simulation.get_reward()+simulation1.get_reward())  # +simulation1.get_reward())
            simulation_step()
            steps += 1
        steps = 0
        episode += 1
        sum_neg_reward1 += simulation.get_reward_neg()
        sum_neg_reward2 += simulation1.get_reward_neg()
        simulation_time, training_time = simulation.after_running(epsilon)
        _, _ = simulation1.after_running(epsilon)
        print("Total reward agent1:", sum_neg_reward1, "- Epsilon:", round(epsilon, 2))
        print("Total reward agent2:", simulation1.get_reward_neg(), "- Epsilon:", round(epsilon, 2))
        sum_neg_reward1 = 0
        sum_neg_reward2 = 0
        print('Simulation time:', simulation_time, 's - Training time:', training_time, 's - Total:',
              round(simulation_time + training_time, 1), 's')
        simulation.close_sumo()

    print("\n----- Start time:", timestamp_start)
    print("----- End time:", datetime.datetime.now())
    print("----- Session info saved at:", path)

    model1.save_model(path, 0)
    model2.save_model(path, 1)

    copyfile(src='training_settings.ini', dst=os.path.join(path, 'training_settings.ini'))

    Visualization.save_data_and_plot(data=list_reward, filename="reward_agents",
                                     xlabel='Episode', ylabel='Cumulative reward')
    Visualization.save_data_and_plot(data=simulation.reward_store, filename='reward',
                                     xlabel='Episode', ylabel='Cumulative negative reward')
    Visualization.save_data_and_plot(data=simulation.cumulative_wait_store, filename='delay',
                                     xlabel='Episode', ylabel='Cumulative delay (s)')
    Visualization.save_data_and_plot(data=simulation.avg_queue_length_store, filename='queue',
                                     xlabel='Episode', ylabel='Average queue length (vehicles)')