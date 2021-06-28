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


def get_incoming_lanes(ts):
    lane = []
    for links in traci.trafficlight.getControlledLanes(ts):
        data = links.split("_")
        lane.append(data[0])
    return list(dict.fromkeys(lane))



def test_main(model="1x1"):
    config = import_test_configuration(config_file='testing_settings.ini')
    sumocfg_file_name = config['sumocfg_file_name']
    if model == "2x2":
        sumocfg_file_name = 'sumo_config_2x2.sumocfg'
    sumo_cmd = set_sumo(config['gui'], sumocfg_file_name, config['max_steps'])
    model_path, plot_path = set_test_path(config['models_path_name']+model, config['model_to_test'])

    isDumb = False
    if model == "1x1dumb":
        isDumb = True


    trafficgen = TrafficGenerator(
        config['max_steps'],
        config['n_cars_generated']
    )

    visualization = Visualization(
        plot_path,
        dpi=96
    )
    if model == "2x2":
        trafficgen.generate_routefile2x2(0)
    else:
        trafficgen.generate_routefile(0)
    sumo_cmd_retrive = set_sumo(False, sumocfg_file_name, config['max_steps'])
    traci.start(sumo_cmd_retrive)  # only for retrive some parameters
    listAgents = traci.trafficlight.getIDList()
    num_states = config['num_states']
    if model == "2x2":
        num_states = 240
    num_agents = len(listAgents)
    models = []
    simulations = []
    for agent in listAgents:
        incoming_roads = get_incoming_lanes(agent)
        modelTest = TestModel(
            input_dim=num_states,
            model_path=model_path,
            grade=agent
        )
        models.append(modelTest)

        simulation = Simulation(
            modelTest,
            trafficgen,
            sumo_cmd,
            config['max_steps'],
            config['green_duration'],
            config['yellow_duration'],
            num_states,
            config['num_actions'],
            incoming_roads=incoming_roads,
            ts=agent
        )
        if agent == "TL1":
            simulation = Simulation(
                modelTest,
                trafficgen,
                sumo_cmd,
                config['max_steps'],
                config['green_duration'],
                config['yellow_duration'],
                num_states,
                config['num_actions'],
                incoming_roads=incoming_roads,
                ts=agent,
                dumb=isDumb
            )
        simulation.setNeighborTs()
        simulations.append(simulation)

    traci.close()
    steps = 0
    simulations[0].start(episode=config['episode_seed'], model=model)
    reward = 0
    while steps < config['max_steps']:
        for i in range(len(listAgents)):
            simulations[i].running()
            reward += simulations[i].get_reward()
        simulation_step()
        steps += 1
    simulation_time = simulations[0].close()

    print('\n----- Test episode')
    print('Simulation time:', simulation_time, 's')
    print('Reward: ', reward)
    print("----- Testing info saved at:", plot_path)

    copyfile(src='testing_settings.ini', dst=os.path.join(plot_path, 'testing_settings.ini'))

    """"
    list_reward = []
    for i in range(len(listAgents)):
        for index in range(len(simul))
    for(item1, item2) in zip(simulation.reward_episode, simulation1.reward_episode):
        list_reward.append(item1+item2)

    list_queue = []
    for (item1, item2) in zip(simulation.queue_length_episode, simulation1.queue_length_episode):
        list_queue.append(item1 + item2)
    """
    list_queue = []
    list_reward = []
    # visualization.save_data_and_plot(data=list_reward, filename='reward', xlabel='Action step', ylabel='Reward')
    # visualization.save_data_and_plot(data=list_queue, filename='queue', xlabel='Step', ylabel='Queue lenght (vehicles)')
