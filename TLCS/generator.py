import numpy as np
import math

class TrafficGenerator:
    def __init__(self, max_steps, n_cars_generated):
        self._n_cars_generated = n_cars_generated  # how many cars per episode
        self._max_steps = max_steps

    def generate_routefile2x2(self, seed):
        np.random.seed(seed)  # make tests reproducible

        # the generation of cars is distributed according to a weibull distribution
        timings = np.random.weibull(2, self._n_cars_generated)
        timings = np.sort(timings)

        # reshape the distribution to fit the interval 0:max_steps
        car_gen_steps = []
        min_old = math.floor(timings[1])
        max_old = math.ceil(timings[-1])
        min_new = 0
        max_new = self._max_steps
        for value in timings:
            car_gen_steps = np.append(car_gen_steps,
                                      ((max_new - min_new) / (max_old - min_old)) * (value - max_old) + max_new)

        car_gen_steps = np.rint(
            car_gen_steps)  # round every value to int -> effective steps when a car will be generated

        # produce the file for cars generation, one car per line
        with open("intersection/episode_routes.rou.xml", "w") as routes:
            print("""<routes>
                    <vType accel="1.0" decel="4.5" id="standard_car" length="5.0" minGap="2.5" maxSpeed="25" sigma="0.5" />

                    <route id="W_N" edges="W2TL TL2N"/>
                    <route id="W_E" edges="W2TL TL2E TL12E"/>
                    <route id="W_S" edges="W2TL TL2S TL22W"/>
                    <route id="N_W" edges="N2TL TL2W"/>
                    <route id="N_E" edges="N2TL TL2E TL12E"/>
                    <route id="N_S" edges="N2TL TL2S TL22E TL32E"/>
                    <route id="E_W" edges="E2TL TL2W"/>
                    <route id="E_N" edges="E2TL TL2N"/>
                    <route id="E_S" edges="E2TL TL2S TL22S"/>
                    <route id="S_W" edges="S2TL TL2W"/>
                    <route id="S_N" edges="S2TL TL2N"/>
                    <route id="S_E" edges="S2TL TL2E TL12S"/>

                    <route id="W_N1" edges="TL2E TL12N"/>
                    <route id="W_E1" edges="TL2E TL12E"/>
                    <route id="W_S1" edges="TL2E TL12S E2TL2 TL22W"/>
                    <route id="N_W1" edges="N2TL1 E2TL TL2N"/>
                    <route id="N_E1" edges="N2TL1 TL12E"/>
                    <route id="N_S1" edges="N2TL1 TL12S TL32E"/>
                    <route id="E_W1" edges="E2TL1 E2TL TL2W"/>
                    <route id="E_N1" edges="E2TL1 TL12N"/>
                    <route id="E_S1" edges="E2TL1 TL12S TL32S"/>
                    <route id="S_W1" edges="S2TL1 E2TL TL2S"/>
                    <route id="S_N1" edges="S2TL1 TL12N"/>
                    <route id="S_E1" edges="S2TL1 TL12E"/>
                    
                    

                    """, file=routes)

            for car_counter, step in enumerate(car_gen_steps):
                straight_or_turn = np.random.uniform()
                if straight_or_turn < 0.50:  # choose direction: straight or turn - 75% of times the car goes straight
                    route_straight = np.random.randint(1, 6)  # choose a random source & destination
                    if route_straight == 1:
                        print(
                            '    <vehicle id="W_E_%i" type="standard_car" route="W_E" depart="%s" departLane="random" departSpeed="10" />' % (
                                car_counter, step), file=routes)
                    elif route_straight == 2:
                        print(
                            '    <vehicle id="E_W1_%i" type="standard_car" route="E_W1" depart="%s" departLane="random" departSpeed="10" />' % (
                                car_counter, step), file=routes)
                    elif route_straight == 3:
                        print(
                            '    <vehicle id="W_N_%i" type="standard_car" route="W_N" depart="%s" departLane="random" departSpeed="10" />' % (
                                car_counter, step), file=routes)

                    elif route_straight == 4:
                        print(
                            '    <vehicle id="E_S1_%i" type="standard_car" route="E_S1" depart="%s" departLane="random" departSpeed="10" />' % (
                                car_counter, step), file=routes)
                    elif route_straight == 5:
                        print(
                            '    <vehicle id="N_E_%i" type="standard_car" route="N_E" depart="%s" departLane="random" departSpeed="10" />' % (
                                car_counter, step), file=routes)
                    else:
                        print(
                            '    <vehicle id="S_W1_%i" type="standard_car" route="S_W1" depart="%s" departLane="random" departSpeed="10" />' % (
                                car_counter, step), file=routes)
                else:  # car that turn -25% of the time the car turns
                    route_turn = np.random.randint(1, 6)  # choose random source source & destination
                    if route_turn == 1:
                        print(
                            '    <vehicle id="N_S_%i" type="standard_car" route="N_S" depart="%s" departLane="random" departSpeed="10" />' % (
                                car_counter, step), file=routes)
                    elif route_turn == 2:
                        print(
                            '    <vehicle id="N_S1_%i" type="standard_car" route="N_S1" depart="%s" departLane="random" departSpeed="10" />' % (
                                car_counter, step), file=routes)
                    elif route_turn == 3:
                        print(
                            '    <vehicle id="S_N_%i" type="standard_car" route="S_N" depart="%s" departLane="random" departSpeed="10" />' % (
                                car_counter, step), file=routes)
                    elif route_turn == 4:
                        print(
                            '    <vehicle id="S_N1_%i" type="standard_car" route="S_N1" depart="%s" departLane="random" departSpeed="10" />' % (
                                car_counter, step), file=routes)
                    elif route_turn == 5:
                        print(
                            '    <vehicle id="E_N_%i" type="standard_car" route="N_W1" depart="%s" departLane="random" departSpeed="10" />' % (
                                car_counter, step), file=routes)
                    elif route_turn == 6:
                        print(
                            '    <vehicle id="S_E_%i" type="standard_car" route="S_E" depart="%s" departLane="random" departSpeed="10" />' % (
                                car_counter, step), file=routes)

            print("</routes>", file=routes)

    def generate_routefile(self, seed):
        """
        Generation of the route of every car for one episode
        """
        np.random.seed(seed)  # make tests reproducible

        # the generation of cars is distributed according to a weibull distribution
        timings = np.random.weibull(2, self._n_cars_generated)
        timings = np.sort(timings)

        # reshape the distribution to fit the interval 0:max_steps
        car_gen_steps = []
        min_old = math.floor(timings[1])
        max_old = math.ceil(timings[-1])
        min_new = 0
        max_new = self._max_steps
        for value in timings:
            car_gen_steps = np.append(car_gen_steps, ((max_new - min_new) / (max_old - min_old)) * (value - max_old) + max_new)

        car_gen_steps = np.rint(car_gen_steps)  # round every value to int -> effective steps when a car will be generated

        # produce the file for cars generation, one car per line
        with open("intersection/episode_routes.rou.xml", "w") as routes:
            print("""<routes>
            <vType accel="1.0" decel="4.5" id="standard_car" length="5.0" minGap="2.5" maxSpeed="25" sigma="0.5" />

            <route id="W_N" edges="W2TL TL2N"/>
            <route id="W_E" edges="W2TL TL2E TL12E"/>
            <route id="W_S" edges="W2TL TL2S"/>
            <route id="N_W" edges="N2TL TL2W"/>
            <route id="N_E" edges="N2TL TL2E TL12E"/>
            <route id="N_S" edges="N2TL TL2S"/>
            <route id="E_W" edges="E2TL TL2W"/>
            <route id="E_N" edges="E2TL TL2N"/>
            <route id="E_S" edges="E2TL TL2S"/>
            <route id="S_W" edges="S2TL TL2W"/>
            <route id="S_N" edges="S2TL TL2N"/>
            <route id="S_E" edges="S2TL TL2E TL12S"/>
            
            <route id="W_N1" edges="TL2E TL12N"/>
            <route id="W_E1" edges="TL2E TL12E"/>
            <route id="W_S1" edges="TL2E TL12S"/>
            <route id="N_W1" edges="N2TL1 E2TL TL2N"/>
            <route id="N_E1" edges="N2TL1 TL12E"/>
            <route id="N_S1" edges="N2TL1 TL12S"/>
            <route id="E_W1" edges="E2TL1 E2TL TL2W"/>
            <route id="E_N1" edges="E2TL1 TL12N"/>
            <route id="E_S1" edges="E2TL1 TL12S"/>
            <route id="S_W1" edges="S2TL1 E2TL TL2S"/>
            <route id="S_N1" edges="S2TL1 TL12N"/>
            <route id="S_E1" edges="S2TL1 TL12E"/>
            
            """, file=routes)

            for car_counter, step in enumerate(car_gen_steps):
                straight_or_turn = np.random.uniform()
                if straight_or_turn < 0.50:  # choose direction: straight or turn - 75% of times the car goes straight
                    route_straight = np.random.randint(1, 6)  # choose a random source & destination
                    if route_straight == 1:
                        print(
                            '    <vehicle id="W_E_%i" type="standard_car" route="W_E" depart="%s" departLane="random" departSpeed="10" />' % (
                            car_counter, step), file=routes)
                    elif route_straight == 2:
                        print(
                            '    <vehicle id="E_W1_%i" type="standard_car" route="E_W1" depart="%s" departLane="random" departSpeed="10" />' % (
                            car_counter, step), file=routes)
                    elif route_straight == 3:
                        print(
                            '    <vehicle id="W_N_%i" type="standard_car" route="W_N" depart="%s" departLane="random" departSpeed="10" />' % (
                            car_counter, step), file=routes)

                    elif route_straight == 4:
                        print(
                            '    <vehicle id="E_S1_%i" type="standard_car" route="E_S1" depart="%s" departLane="random" departSpeed="10" />' % (
                            car_counter, step), file=routes)
                    elif route_straight == 5:
                        print(
                            '    <vehicle id="N_E_%i" type="standard_car" route="N_E" depart="%s" departLane="random" departSpeed="10" />' % (
                            car_counter, step), file=routes)
                    else:
                        print(
                            '    <vehicle id="S_W1_%i" type="standard_car" route="S_W1" depart="%s" departLane="random" departSpeed="10" />' % (
                            car_counter, step), file=routes)
                else:  # car that turn -25% of the time the car turns
                    route_turn = np.random.randint(1, 6)  # choose random source source & destination
                    if route_turn == 1:
                        print(
                            '    <vehicle id="N_S_%i" type="standard_car" route="N_S" depart="%s" departLane="random" departSpeed="10" />' % (
                            car_counter, step), file=routes)
                    elif route_turn == 2:
                        print(
                            '    <vehicle id="N_S1_%i" type="standard_car" route="N_S1" depart="%s" departLane="random" departSpeed="10" />' % (
                            car_counter, step), file=routes)
                    elif route_turn == 3:
                        print(
                            '    <vehicle id="S_N_%i" type="standard_car" route="S_N" depart="%s" departLane="random" departSpeed="10" />' % (
                            car_counter, step), file=routes)
                    elif route_turn == 4:
                        print(
                            '    <vehicle id="S_N1_%i" type="standard_car" route="S_N1" depart="%s" departLane="random" departSpeed="10" />' % (
                            car_counter, step), file=routes)
                    elif route_turn == 5:
                        print(
                            '    <vehicle id="E_N_%i" type="standard_car" route="N_W1" depart="%s" departLane="random" departSpeed="10" />' % (
                            car_counter, step), file=routes)
                    elif route_turn == 6:
                        print(
                            '    <vehicle id="S_E_%i" type="standard_car" route="S_E" depart="%s" departLane="random" departSpeed="10" />' % (
                            car_counter, step), file=routes)

            print("</routes>", file=routes)
