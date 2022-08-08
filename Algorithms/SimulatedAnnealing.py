from numpy import matrix

from Patient import Patient
from Room import Room
import pandas as pd
import math
import matplotlib.pyplot as plt
# import mlrose
import numpy as np
import random
import time
import decimal
from Patient import Patient
from Room import Room
import pandas as pd
import math
import random
import numpy as np
import matplotlib.pyplot as plt
import time

random.seed(10)

######## READ DATA ###########
Patient_data = pd.read_csv('../Data/Patient50.csv')
Rooms_data = pd.read_csv('../Data/Rooms.csv')

######## Variables ##########
ROBOT_SPEED = 6
COST = 200
MAX_TIME = 10000


def init_rooms(Rooms_data):
    room_dict = {}
    for room in Rooms_data.iterrows():
        room_dict[room[1]['id']] = Room(room[1]['id'], room[1]['x'], room[1]['y'])
    return room_dict


def init_patient(Patient_data, r_data):
    patient_array = []
    for patient in Patient_data.iterrows():
        patient_array.append(
            Patient(patient[1]['id'], r_data[patient[1]['room']], patient[1]['urgency'], patient[1]['typeofdisease']))
    return patient_array


def calc_dist(room1, room2):
    """ Returns Euclidean distance between 2 points """
    return math.sqrt(((room1[0] - room2[0]) ** 2) + ((room1[1] - room2[1]) ** 2))


def distance_matrix(room_data):
    """ Returns distance matrix between all rooms """
    print(room_data)
    df = pd.DataFrame(columns=range(1, len(room_data) + 1), index=range(1, len(room_data) + 1))
    for i in room_data.keys():
        for j in room_data.keys():
            df.at[i, j] = calc_dist(room_data[i].get_loc(), room_data[j].get_loc())
    return df


# todo: we start from P0 or P1??????????????
def patient_distance_matrix(patient_data):
    """ Returns distance matrix between all patients """
    df = pd.DataFrame(columns=range(0, len(patient_data)), index=range(0, len(patient_data)))
    for i in range(0, len(patient_data)):
        for j in range(0, len(patient_data)):
            df.at[i, j] = calc_dist(patient_data[i - 1].room.get_loc(), patient_data[j - 1].room.get_loc())
    return df


def plotonimage(room_data):
    """ Create a plot of the given patients path"""
    x_arr = []
    y_arr = []
    for room in room_data.values():
        temp_room_loc = room.get_loc()
        x_arr.append(temp_room_loc[0])
        y_arr.append(temp_room_loc[1])

    plt.plot(x_arr, y_arr, linestyle='dashed', marker='s')

    # plt.scatter(x_arr,y_arr)
    plt.show()


def get_patient(patients, id):
    """ Returns patient based on id"""
    for p in patients:
        if p.id == id:
            return p


def print_solution(solution):
    l = []
    for p in solution:
        l.append(p.id)
    print(l)


def random_solution(matrix):
    """ Returns a random possible solution"""
    patients = list(range(len(matrix)))
    solution = []

    for i in range(len(matrix)):
        random_patient = patients[random.randint(0, len(patients) - 1)]
        solution.append(random_patient)
        patients.remove(random_patient)

    return solution




def get_neighbors(solution):
    """ Returns a list of possible neighbors to current solution"""
    neighbours = []
    for i in range(1, len(solution)):
        for j in range(i + 1, len(solution)):
            neighbour = solution.copy()
            neighbour[i] = solution[j]
            neighbour[j] = solution[i]
            neighbours.append(neighbour)

    return neighbours


def objective_function(tsp, solution, patients, robot_speed, cost):
    target_value = 0
    time_of = 0
    for i in range(len(solution)):
        time_of += (tsp[solution[i - 1]][solution[i]] / robot_speed) + patients[solution[i - 1]].type_of_disease
        latency = max((time_of - patients[solution[i - 1]].urgency), 0)
        target_value = (time_of + latency * cost)
    return target_value


def isDone(start_time, max_time):
    return time.time() - start_time > max_time


def simulated_annealing(tsp, patients, robot_speed, C, max_time, initial_state, initial_temp=90, final_temp=0.1,
                        alpha=0.01, time_limit=False):
    """ Find optimal solution by Simulated Annealing Heuristic"""
    print("Starting SA")

    # Set initial parameters
    initial_temp = initial_temp
    final_temp = final_temp
    alpha = alpha
    current_temp = initial_temp
    current_state = initial_state
    solution = current_state
    current_best = solution

    if time_limit:
        start_time = time.time()
        while current_temp > final_temp and not isDone(start_time, max_time):
            neighbor = random.choice(get_neighbors(solution))
            # Check if neighbor is best so far
            cost_diff = objective_function(tsp, current_state, patients, robot_speed, C) - objective_function(tsp,
                                                                                                              neighbor,
                                                                                                              patients,
                                                                                                              robot_speed,
                                                                                                              C)
            # if the new solution is better, accept it
            if cost_diff > 0:
                solution = neighbor
                current_best = solution
            # if the new solution is not better, accept it with a probability of e^(-cost/temp)
            else:
                if random.uniform(0, 1) < decimal.Decimal((-1 * cost_diff) / current_temp).exp():
                    solution = neighbor
            # decrement the temperature
            current_temp -= alpha

        if objective_function(tsp, current_best, patients, robot_speed, C) < objective_function(tsp,
                                                                                                solution,
                                                                                                patients,
                                                                                                robot_speed,
                                                                                                C):
            solution = current_best


    else:
        while current_temp > final_temp:
            neighbor = random.choice(get_neighbors(solution))

            # Check if neighbor is best so far
            cost_diff = objective_function(tsp, current_state, patients, robot_speed, C) - objective_function(tsp,
                                                                                                              neighbor,
                                                                                                              patients,
                                                                                                              robot_speed,
                                                                                                              C)
            # if the new solution is better, accept it
            if cost_diff >= 0:
                solution = neighbor
                current_best = solution
                print('found better')
            # if the new solution is not better, accept it with a probability of e^(-cost/temp)

            else:
                # print(decimal.Decimal((cost_diff) / current_temp).exp())
                print(math.exp(cost_diff / current_temp))
                # if random.uniform(0, 1) < decimal.Decimal((-1 * cost_diff) / current_temp).exp():
                if random.uniform(0, 1) < math.exp(cost_diff / current_temp):
                    solution = neighbor
            # decrement the temperature
            current_temp -= alpha

        if objective_function(tsp, current_best, patients, robot_speed, C) < objective_function(tsp,
                                                                                                solution,
                                                                                                patients,
                                                                                                robot_speed,
                                                                                                C):
            solution = current_best

    print(solution)
    print(objective_function(tsp, solution, patients, robot_speed, C))
    return solution, objective_function(tsp, solution, patients, robot_speed, C)


def patient_to_room_arr(pat_arr, patient_data):
    room_arr = []
    x_arr = []
    y_arr = []
    # room_arr.append(patient_data[pat_arr[0]].room.get_loc())
    for patient in pat_arr:
        room_arr.append(patient_data[patient].room.get_loc())
    # print(room_arr)
    for cord in room_arr:
        x_arr.append(cord[0])
        y_arr.append(cord[1])
    fig, ax = plt.subplots()
    u = np.diff(x_arr)
    v = np.diff(y_arr)
    u = np.array([2 if x == 0 else x for x in u])
    v = np.array([2 if x == 0 else x for x in v])
    pos_x = x_arr[:-1] + u / 2
    pos_y = y_arr[:-1] + v / 2
    norm = np.sqrt(abs(u * 2 + v * 2))
    ax.plot(x_arr, y_arr, marker="o")
    # ax.quiver(pos_x, pos_y, u / norm, v / norm, angles="xy", zorder=5, pivot='mid')
    plt.title("Path to Rooms")
    plt.show()
    for i in range(0, len(x_arr) - 1):
        j = i
        if j >= (len(x_arr) - 1):
            break
        while x_arr[i] == x_arr[j + 1] and y_arr[i] == y_arr[j + 1]:
            x_arr[j + 1] = x_arr[j + 1] + 3
            y_arr[j + 1] = y_arr[j + 1] + 3
            if (j + 1) < (len(x_arr) - 1):
                j = j + 1
            else:
                break
    fig, ax = plt.subplots()
    u = np.diff(x_arr)
    v = np.diff(y_arr)
    u = np.array([2 if x == 0 else x for x in u])
    v = np.array([2 if x == 0 else x for x in v])
    pos_x = x_arr[:-1] + u / 2
    pos_y = y_arr[:-1] + v / 2
    norm = np.sqrt(abs(u * 2 + v * 2))
    ax.plot(x_arr, y_arr, marker="o")
    # ax.quiver(pos_x, pos_y, u / norm, v / norm, angles="xy", zorder=5, pivot='mid')
    plt.title("Path to Patients")
    plt.show()


if __name__ == '__main__':
    room_data = init_rooms(Rooms_data)
    patient_data = init_patient(Patient_data, room_data)
    # print(patient_data)

    # print(len(patient_data))
    time_matrix = patient_distance_matrix(patient_data)

    robot_speed = ROBOT_SPEED
    c = COST
    max_time = MAX_TIME
    time_limit = False

    rand_solution = random_solution(time_matrix)
    print(rand_solution)
    print(objective_function(time_matrix, rand_solution, patient_data, robot_speed, c))

    # solution, z = simulated_annealing(time_matrix, patient_data, robot_speed, c, max_time, rand_solution,
    #                                   initial_temp=400, final_temp=0.01,
    #                                   alpha=0.001, time_limit=False)
    # print(solution)
    # print(patient_data)
    # patient_to_room_arr(solution, patient_data)

    from SA import SimulatedAnnealing

    sa = SimulatedAnnealing(tsp=time_matrix, patients=patient_data, robot_speed=robot_speed, cost=c,
                             initialTemp=90, finalTemp=0.01, tempReduction="linear",
                            iterationPerTemp=100, alpha=0.99, beta=5)

    solution, z = sa.run()
    print(solution)
    print(z)

    patient_to_room_arr(solution, patient_data)
