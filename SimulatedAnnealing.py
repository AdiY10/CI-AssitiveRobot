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

random.seed(10)

######## READ DATA ###########
Patient_data = pd.read_csv('Patient.csv')
Rooms_data = pd.read_csv('Rooms.csv')

######## Variables ##########
ROBOT_SPEED = 6
COST = 10
MAX_TIME = 10000


def init_rooms(Rooms_data):
    """ Returns dictionary of the rooms from csv file"""
    room_dict = {}
    for room in Rooms_data.iterrows():
        room_dict[room[1]['id']] = Room(room[1]['id'], room[1]['x'], room[1]['y'])
    return room_dict


def init_patient(Patient_data, r_data):
    """ Returns dictionary of the patients from csv file"""
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
                        alpha=0.01):
    """ Find optimal solution by Simulated Annealing Heuristic"""
    print("Starting SA")

    # Set initial parameters
    initial_temp = initial_temp
    final_temp = final_temp
    alpha = alpha
    current_temp = initial_temp
    current_state = initial_state
    solution = current_state

    start_time = time.time()
    while current_temp > final_temp and not isDone(start_time, max_time):
        neighbor = random.choice(get_neighbors(solution))

        # Check if neighbor is best so far
        cost_diff = objective_function(tsp, current_state, patients, robot_speed, C) - objective_function(tsp, neighbor,
                                                                                                          patients,
                                                                                                          robot_speed,
                                                                                                          C)
        # if the new solution is better, accept it
        if cost_diff > 0:
            solution = neighbor
        # if the new solution is not better, accept it with a probability of e^(-cost/temp)
        else:
            if random.uniform(0, 1) < math.exp(-cost_diff / current_temp):
                solution = neighbor
        # decrement the temperature
        current_temp -= alpha

    print(solution)
    print(objective_function(tsp, solution, patients, robot_speed, C))
    return solution


if __name__ == '__main__':
    room_data = init_rooms(Rooms_data)
    patient_data = init_patient(Patient_data, room_data)
    print(patient_data)

    print(len(patient_data))
    time_matrix = patient_distance_matrix(patient_data)

    robot_speed = ROBOT_SPEED
    c = COST
    max_time = MAX_TIME

    rand_solution = random_solution(time_matrix)
    print(rand_solution)

    print(objective_function(time_matrix, rand_solution, patient_data, robot_speed, c))

    simulated_annealing(time_matrix, patient_data, robot_speed, c, max_time, rand_solution)
