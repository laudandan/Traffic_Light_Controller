import os
import datetime
from shutil import copyfile

import traci

from training_simulation import Simulation, simulation_step
from generator import TrafficGenerator
from memory import Memory
from model import TrainModel
from visualization import Visualization
from utils import import_train_configuration, set_sumo, set_train_path


def get_incoming_lanes(ts):
    lane = []
    for links in traci.trafficlight.getControlledLanes(ts):
        data = links.split("_")
        lane.append(data[0])
    return list(dict.fromkeys(lane))


def train_main(model="1x1"):
    config = import_train_configuration(config_file='training_settings.ini')
    sumocfg_file_name = config['sumocfg_file_name']
    if model == "2x2":
        sumocfg_file_name = 'sumo_config_2x2.sumocfg'
    sumo_cmd = set_sumo(config['gui'], sumocfg_file_name, config['max_steps'])
    path = set_train_path(config['models_path_name']+model)
    isDumb = False
    if model == "1x1dumb":
        isDumb = True
    traffic_generator = TrafficGenerator(
        config['max_steps'],
        config['n_cars_generated']
    )

    visualization = Visualization(
        path,
        dpi=96
    )
    if model == "2x2":
        traffic_generator.generate_routefile2x2(0)
    else:
        traffic_generator.generate_routefile(0)
    sumo_cmd_retrive = set_sumo(False, sumocfg_file_name, config['max_steps'])
    traci.start(sumo_cmd_retrive)  # only for retrive some parameters
    listAgents = traci.trafficlight.getIDList()
    num_states = config['num_states']
    if model == "2x2":
        num_states = 240

    num_agents = len(listAgents)
    models = []
    simulations = []
    memorys = []
    for agent in listAgents:
        incoming_roads = get_incoming_lanes(agent)
        modelTrain = TrainModel(
            config['num_layers'],
            config['width_layers'],
            config['batch_size'],
            config['learning_rate'],
            input_dim=num_states,
            output_dim=config['num_actions']
        )
        memory = Memory(
            config['memory_size_max'],
            config['memory_size_min']
        )
        memorys.append(memory)
        models.append(modelTrain)
        simulation = Simulation(
            modelTrain,
            memory,
            traffic_generator,
            sumo_cmd,
            config['gamma'],
            config['max_steps'],
            config['green_duration'],
            config['yellow_duration'],
            num_states,
            config['num_actions'],
            config['training_epochs'],
            incoming_roads=incoming_roads,
            ts=agent
        )
        if agent == "TL1":
            simulation = Simulation(
                modelTrain,
                memory,
                traffic_generator,
                sumo_cmd,
                config['gamma'],
                config['max_steps'],
                config['green_duration'],
                config['yellow_duration'],
                num_states,
                config['num_actions'],
                config['training_epochs'],
                incoming_roads=incoming_roads,
                ts=agent,
                dumb=isDumb
            )
        simulation.setNeighborTs()
        simulations.append(simulation)

    traci.close()

    timestamp_start = datetime.datetime.now()
    steps = 0
    episode = 0
    epsilon = 0

    list_reward_neg = [[] for i in range(num_agents)]
    result_score_epoch = []
    list_numbers_stops = [[] for i in range(num_agents)]
    # Iterate episodes of simulation

    while episode < config['total_episodes']:
        for i in range(len(listAgents)):
            simulations[i].before_running(episode)
        print(model)
        simulations[0].generate_route(episode, model)
        print('\n----- Episode', str(episode + 1), 'of',
              str(config['total_episodes']))
        simulations[0].start_sumo()
        epsilon = 1.0 - (episode / config['total_episodes'])

        # An episode of simulation
        while steps < config['max_steps']:
            for i in range(len(listAgents)):
                simulations[i].train(epsilon)
            simulation_step()
            steps += 1
        steps = 0
        episode += 1
        simulation_time, training_time = simulations[0].after_running(epsilon)

        neg_reward = [0]*num_agents
        nr_stops = [0]*num_agents
        for i in range(len(listAgents)):
            if i:
                _, _ = simulations[i].after_running(epsilon)
            print("Total reward agent"+str(i+1)+": ", simulations[i].get_reward_neg(), "- Epsilon:", round(epsilon, 2))
            neg_reward[i] += simulations[i].get_reward_neg()
            nr_stops[i] += simulations[i].get_nr_stops()
        result_score_epoch = simulations[0].get_score_epoch()
        for j in range(num_agents):
            list_numbers_stops[j].append(nr_stops[j])
            list_reward_neg[j].append(neg_reward[j])
        print('Simulation time:', simulation_time, 's - Training time:', training_time, 's - Total:',
              round(simulation_time + training_time, 1), 's')
        simulations[0].close_sumo()

    print("\n----- Start time:", timestamp_start)
    print("----- End time:", datetime.datetime.now())
    print("----- Session info saved at:", path)

    for i, ts in enumerate(listAgents):
        models[i].save_model(path, ts)

    copyfile(src='training_settings.ini', dst=os.path.join(path, 'training_settings.ini'))
    visualization.save_data_and_plot(data=list_reward_neg, filename="reward_neg",
                                     xlabel='Episode', ylabel='Negative Reward', dumb=isDumb)
    visualization.save_data_and_plot(data=list_numbers_stops, filename="number_stops",
                                     xlabel='Episode', ylabel='Number of stops', dumb=isDumb)
    #visualization.save_data_and_plot(data=result_score_epoch, filename="score_epochs_avg_reward",
    #                                xlabel='Training Epochs', ylabel='Average score')
