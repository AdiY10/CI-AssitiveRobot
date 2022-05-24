from numpy import matrix

from Patient import Patient
from Room import Room
import pandas as pd
import math
import matplotlib.pyplot as plt
# import mlrose
import numpy as np
import random

random.seed(10)

######## READ DATA ###########
Patient_data = pd.read_csv('Patient.csv')
Rooms_data = pd.read_csv('Rooms.csv')

######## Variables ##########
ROBOT_SPEED = 6
COST = 0


def init_rooms(Rooms_data):
    room_dict = {}
    for room in Rooms_data.iterrows():
        room_dict[room[1]['id']] = Room(room[1]['id'], room[1]['x'], room[1]['y'])
    return room_dict


def init_patient(Patient_data, r_data):
    patient_array = [Patient(0, Room(0, 0, 0), 0, 0)]
    patient_array = []
    for patient in Patient_data.iterrows():
        patient_array.append(
            Patient(patient[1]['id'], r_data[patient[1]['room']], patient[1]['urgency'], patient[1]['typeofdisease']))
    return patient_array


def calc_dist(room1, room2):
    return (math.sqrt(((room1[0] - room2[0]) ** 2) + ((room1[1] - room2[1]) ** 2)))


def distance_matrix(room_data):
    print(room_data)
    df = pd.DataFrame(columns=range(1, len(room_data) + 1), index=range(1, len(room_data) + 1))
    for i in room_data.keys():
        for j in room_data.keys():
            df.at[i, j] = calc_dist(room_data[i].get_loc(), room_data[j].get_loc())
    return df


def plotonimage(room_data):
    x_arr = []
    y_arr = []
    for room in room_data:
        1264


def calc_moving_value(p1, p2):
    # value formula = dist/speed + type_of_disease
    if p1 == p2:
        return 0

    return (calc_dist(p1.room.get_loc(), p2.room.get_loc()) / ROBOT_SPEED) + p2.type_of_disease


def get_patient(patients, id):
    for p in patients:
        if p.id == id:
            return p


def print_solution(solution):
    l = []
    for p in solution:
        l.append(p.id)
    print(l)

def random_solution(matrix, patients_list):
    patients = list(range(len(matrix)))
    patients.remove(0)
    solution = [get_patient(patients_list,0)]

    for i in range(1, len(matrix)):
        random_patient = patients[random.randint(0, len(patients) - 1)]
        solution.append(get_patient(patients_list,random_patient))
        patients.remove(random_patient)

    return solution


def routeLength(matrix, solution):
    routeLength = 0
    for i in range(len(solution)):
        routeLength += matrix[solution[i - 1].id][solution[i].id]

    return routeLength


def get_neighbors(solution):
    neighbours = []
    for i in range(1, len(solution)):
        for j in range(i + 1, len(solution)):
            neighbour = solution.copy()
            neighbour[i] = solution[j]
            neighbour[j] = solution[i]
            neighbours.append(neighbour)

    return neighbours


def get_cost(matrix, state):
    cost = routeLength(matrix=matrix, solution=state)
    for i in range(1, len(state)):
        current_path = state[0:i+1]
        time = routeLength(matrix=matrix, solution=current_path)
        delta_latency = max(0, time-state[i].type_of_disease)
        cost += delta_latency * COST

    return cost


def simulated_annealing(matrix, initial_state):
    # ## Set initial parameters
    initial_temp = 90
    final_temp = .1
    alpha = 0.01
    current_temp = initial_temp
    current_state = initial_state
    solution = current_state

    while current_temp > final_temp:
        neighbor = random.choice(get_neighbors(solution))

        # Check if neighbor is best so far
        cost_diff = get_cost(matrix, current_state) - get_cost(matrix, neighbor)
        print(cost_diff)

        # if the new solution is better, accept it
        if cost_diff > 0:
            solution = neighbor
        # if the new solution is not better, accept it with a probability of e^(-cost/temp)
        else:
            if random.uniform(0, 1) < math.exp(-cost_diff / current_temp):
                solution = neighbor
        # decrement the temperature
        current_temp -= alpha

    print_solution(solution)
    print(get_cost(matrix, solution))
    return solution


if __name__ == '__main__':
    room_data = init_rooms(Rooms_data)
    patient_data = init_patient(Patient_data, room_data)
    print(patient_data)
    print(len(patient_data))
    # dist = distance_matrix(room_data)

    # ## time_matrix
    df = pd.DataFrame(columns=range(0, len(patient_data)), index=range(0, len(patient_data)))
    i = 0
    for p1 in patient_data:
        j = 0
        for p2 in patient_data:
            df.at[i, j] = calc_moving_value(p1, p2)
            j += 1
        i += 1

    print(df)

    rand_solution = random_solution(df, patient_data)
    print(rand_solution)
    print_solution(rand_solution)
    print(get_cost(df, rand_solution))
    # print(getNeighbours(rand_solution))
    # print(getBestNeighbour(df, getNeighbours(rand_solution)))

    simulated_annealing(df, rand_solution)
