import traci
import numpy as np
import random
import timeit
import os

# phase codes based on environment.net.xml
PHASE_NS_GREEN = 0  # action 0 code 00
PHASE_NS_YELLOW = 1
PHASE_NSL_GREEN = 2  # action 1 code 01
PHASE_NSL_YELLOW = 3
PHASE_EW_GREEN = 4  # action 2 code 10
PHASE_EW_YELLOW = 5
PHASE_EWL_GREEN = 6  # action 3 code 11
PHASE_EWL_YELLOW = 7


def simulation_step():
    traci.simulationStep()


class Simulation:
    def __init__(self, Model, Memory, TrafficGen, sumo_cmd, gamma, max_steps, green_duration, yellow_duration,
                 num_states, num_actions, training_epochs, incoming_roads, state_nr, neighbor_nr, ts):
        self._Model = Model
        self._Memory = Memory
        self._TrafficGen = TrafficGen
        self._gamma = gamma
        self._step = 0
        self._sumo_cmd = sumo_cmd
        self._max_steps = max_steps
        self._green_duration = green_duration
        self._yellow_duration = yellow_duration
        self._num_states = num_states
        self._num_actions = num_actions
        self._reward_store = []
        self._cumulative_wait_store = []
        self._avg_queue_length_store = []
        self._training_epochs = training_epochs

        self.incoming_roads = incoming_roads
        self.state_nr = state_nr
        self.neighbor_nr = neighbor_nr
        self.ts = ts
        self._waiting_times = {}
        self._sum_neg_reward = 0
        self._sum_queue_length = 0
        self._sum_waiting_time = 0
        self.old_state = -1
        self.old_total_wait = 0
        self.old_action = -1  # dummy init

        self.index = 0
        self.flag_observations = True
        self.flag_yellow = True
        self.flag_green = True
        self.flag_reward = True

    def running(self, epsilon):
        if self.index == 0 and self.flag_observations:
            current_state_agent1 = self._get_state(self.state_nr)
            current_state_agent2 = self._get_state(self.neighbor_nr)
            self.current_state = np.concatenate((current_state_agent1[:len(current_state_agent1)//2],
                                                 current_state_agent2[:len(current_state_agent2)//2]), axis=None)
            self.current_total_wait = self._collect_waiting_times(incoming_roads=self.incoming_roads)
            self.reward = self.old_total_wait - self.current_total_wait
            self.action = self._choose_action(self.current_state, epsilon)
            if self._step != 0:
                self._Memory.add_sample((self.old_state, self.old_action, self.reward, self.current_state))

            self.set_flags(False, True, True, False)

        if self._step != 0 and self.old_action != self.action and self.index == 0 and self.flag_yellow:
            self._set_yellow_phase(self.old_action, self.ts)
            # self._simulate(incoming_roads=self.incoming_roads)
            self.index = self._yellow_duration
            self.set_flags(False, False, True, False)

        if self.index == 0 and self.flag_green:
            self._set_green_phase(self.action, self.ts)
            # self._simulate(incoming_roads=self.incoming_roads)
            self.index = self._green_duration
            self.set_flags(False, False, False, True)

        if self.index > 0:
            self._waiting_queue(incoming_roads=self.incoming_roads)

        if self.index == 0 and self.flag_reward:
            self.old_state = self.current_state
            self.old_action = self.action
            self.old_total_wait = self.current_total_wait
            self.reward_store.append(self.reward)
            self.set_flags(True, False, False, False)

            if self.reward < 0:
                self._sum_neg_reward += self.reward
        if self.index > 0:
            self.index -= 1
        self._step += 1

    def get_reward(self):
        return self.reward

    def set_flags(self, observations, yellow, green, reward):
        self.flag_observations = observations
        self.flag_yellow = yellow
        self.flag_green = green
        self.flag_reward = reward

    def start_sumo(self):
        traci.start(self._sumo_cmd)

    def close_sumo(self):
        traci.close()

    def generate_route(self, episode):
        self._TrafficGen.generate_routefile(seed=episode)

    def before_running(self, episode):
        self._step = 0
        self._waiting_times = {}
        self._sum_neg_reward = 0
        self._sum_queue_length = 0
        self._sum_waiting_time = 0
        self._waiting_times = {}
        self.old_total_wait = 0
        self.old_action = -1

        self.start_time = timeit.default_timer()
        print("Simulating...")

    def set_reward_neg(self, sum_neg):
        self._sum_neg_reward += sum_neg

    def get_reward_neg(self):
        return self._sum_neg_reward

    def after_running(self, epsilon):
        self._save_episode_stats()
        # print("Total reward:", self._sum_neg_reward, "- Epsilon:", round(epsilon, 2))
        simulation_time = round(timeit.default_timer() - self.start_time, 1)

        print("Training...")
        start_time = timeit.default_timer()
        for _ in range(self._training_epochs):
            self._replay()
        training_time = round(timeit.default_timer() - start_time, 1)

        return simulation_time, training_time

    def _waiting_queue(self, incoming_roads):
        queue_length = self._get_queue_length(incoming_roads=incoming_roads)
        # self._queue_length_episode.append(queue_length)
        self._sum_queue_length += queue_length
        self._sum_waiting_time += queue_length

    def _collect_waiting_times(self, incoming_roads):
        """
        Retrieve the waiting time of every car in the incoming roads
        """
        car_list = traci.vehicle.getIDList()
        for car_id in car_list:
            wait_time = traci.vehicle.getAccumulatedWaitingTime(car_id)
            road_id = traci.vehicle.getRoadID(car_id)  # get the road id where the car is located
            if road_id in incoming_roads:  # consider only the waiting times of cars in incoming roads
                self._waiting_times[car_id] = wait_time
            else:
                if car_id in self._waiting_times:  # a car that was tracked has cleared the intersection
                    del self._waiting_times[car_id]
        total_waiting_time = sum(self._waiting_times.values())
        return total_waiting_time

    def _choose_action(self, state, epsilon):
        """
        Decide wheter to perform an explorative or exploitative action, according to an epsilon-greedy policy
        """
        if random.random() < epsilon:
            return random.randint(0, self._num_actions - 1) # random action
        else:
            return np.argmax(self._Model.predict_one(state)) # the best action given the current state

    def _set_yellow_phase(self, old_action, traffic_light="TL"):
        """
        Activate the correct yellow light combination in sumo
        """
        yellow_phase_code = old_action * 2 + 1  # obtain the yellow phase code, based on the old action (ref on environment.net.xml)
        traci.trafficlight.setPhase(traffic_light, yellow_phase_code)

    def _set_green_phase(self, action_number, traffic_light):
        """
        Activate the correct green light combination in sumo
        """
        traci.trafficlight.getIDList()

        if action_number == 0:
            traci.trafficlight.setPhase(traffic_light, PHASE_NS_GREEN)
        elif action_number == 1:
            traci.trafficlight.setPhase(traffic_light, PHASE_NSL_GREEN)
        elif action_number == 2:
            traci.trafficlight.setPhase(traffic_light, PHASE_EW_GREEN)
        elif action_number == 3:
            traci.trafficlight.setPhase(traffic_light, PHASE_EWL_GREEN)

    def _get_queue_length(self, incoming_roads):
        """
        Retrieve the number of cars with speed = 0 in every incoming lane
        """
        queue_length = 0
        for roads in incoming_roads:
            halt = traci.edge.getLastStepHaltingNumber(roads)
            queue_length += halt
        return queue_length

    def _get_state(self, number_traffic=0):
        """
        Retrieve the state of the intersection from sumo, in the form of cell occupancy
        """
        state = np.zeros(self._num_states)
        car_list = traci.vehicle.getIDList()

        for car_id in car_list:
            lane_pos = traci.vehicle.getLanePosition(car_id)
            lane_id = traci.vehicle.getLaneID(car_id)
            lane_pos = 750 - lane_pos  # inversion of lane pos, so if the car is close to the traffic light -> lane_pos = 0 --- 750 = max len of a road

            # distance in meters from the traffic light -> mapping into cells
            if lane_pos < 7:
                lane_cell = 0
            elif lane_pos < 14:
                lane_cell = 1
            elif lane_pos < 21:
                lane_cell = 2
            elif lane_pos < 28:
                lane_cell = 3
            elif lane_pos < 40:
                lane_cell = 4
            elif lane_pos < 60:
                lane_cell = 5
            elif lane_pos < 100:
                lane_cell = 6
            elif lane_pos < 160:
                lane_cell = 7
            elif lane_pos < 400:
                lane_cell = 8
            elif lane_pos <= 750:
                lane_cell = 9

            # finding the lane where the car is located
            # x2TL_3 are the "turn left only" lanes
            if number_traffic == 0:
                if lane_id == "W2TL_0" or lane_id == "W2TL_1" or lane_id == "W2TL_2":
                    lane_group = 0
                elif lane_id == "W2TL_3":
                    lane_group = 1
                elif lane_id == "N2TL_0" or lane_id == "N2TL_1" or lane_id == "N2TL_2":
                    lane_group = 2
                elif lane_id == "N2TL_3":
                    lane_group = 3
                elif lane_id == "E2TL_0" or lane_id == "E2TL_1" or lane_id == "E2TL_2":
                    lane_group = 4
                elif lane_id == "E2TL_3":
                    lane_group = 5
                elif lane_id == "S2TL_0" or lane_id == "S2TL_1" or lane_id == "S2TL_2":
                    lane_group = 6
                elif lane_id == "S2TL_3":
                    lane_group = 7
                else:
                    lane_group = -1
            else:
                if lane_id == "W2TL1_0" or lane_id == "W2TL1_1" or lane_id == "W2TL1_2":
                    lane_group = 0
                elif lane_id == "W2TL1_3":
                    lane_group = 1
                elif lane_id == "N2TL1_0" or lane_id == "N2TL1_1" or lane_id == "N2TL1_2":
                    lane_group = 2
                elif lane_id == "N2TL1_3":
                    lane_group = 3
                elif lane_id == "E2TL1_0" or lane_id == "E2TL1_1" or lane_id == "E2TL1_2":
                    lane_group = 4
                elif lane_id == "E2TL1_3":
                    lane_group = 5
                elif lane_id == "S2TL1_0" or lane_id == "S2TL1_1" or lane_id == "S2TL1_2":
                    lane_group = 6
                elif lane_id == "S2TL1_3":
                    lane_group = 7
                else:
                    lane_group = -1

            if lane_group >= 1 and lane_group <= 7:
                car_position = int(str(lane_group) + str(
                    lane_cell))  # composition of the two postion ID to create a number in interval 0-79
                valid_car = True
            elif lane_group == 0:
                car_position = lane_cell
                valid_car = True
            else:
                valid_car = False  # flag for not detecting cars crossing the intersection or driving away from it

            if valid_car:
                state[
                    car_position] = 1  # write the position of the car car_id in the state array in the form of "cell occupied"

        return state

    def _replay(self):
        """
        Retrieve a group of samples from the memory and for each of them update the learning equation, then train
        """
        batch = self._Memory.get_samples(self._Model.batch_size)

        if len(batch) > 0:  # if the memory is full enough
            states = np.array([val[0] for val in batch])  # extract states from the batch
            next_states = np.array([val[3] for val in batch])  # extract next states from the batch

            # prediction
            q_s_a = self._Model.predict_batch(states)  # predict Q(state), for every sample
            q_s_a_d = self._Model.predict_batch(next_states)  # predict Q(next_state), for every sample

            # setup training arrays
            x = np.zeros((len(batch), self._num_states))
            y = np.zeros((len(batch), self._num_actions))

            for i, b in enumerate(batch):
                state, action, reward, _ = b[0], b[1], b[2], b[3]  # extract data from one sample
                current_q = q_s_a[i]  # get the Q(state) predicted before
                current_q[action] = reward + self._gamma * np.amax(q_s_a_d[i])  # update Q(state, action)
                x[i] = state
                y[i] = current_q  # Q(state) that includes the updated action value

            self._Model.train_batch(x, y)  # train the NN

    def _save_episode_stats(self):
        """
        Save the stats of the episode to plot the graphs at the end of the session
        """
        self._reward_store.append(self._sum_neg_reward)  # how much negative reward in this episode
        self._cumulative_wait_store.append(self._sum_waiting_time)  # total number of seconds waited by cars in this episode
        self._avg_queue_length_store.append(self._sum_queue_length / self._max_steps)  # average number of queued cars per step, in this episode

    @property
    def reward_store(self):
        return self._reward_store

    @property
    def cumulative_wait_store(self):
        return self._cumulative_wait_store

    @property
    def avg_queue_length_store(self):
        return self._avg_queue_length_store

