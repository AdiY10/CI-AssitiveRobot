from Patient import Patient
from Room import Room
import pandas as pd
import math
import random
import numpy as np
import matplotlib.pyplot as plt
import time
from Algorithms.SA import SimulatedAnnealing

######## READ DATA ###########
Patient_data = pd.read_csv('Data/Patient100.csv')
Rooms_data = pd.read_csv('Data/Rooms.csv')

######## Variables ##########
robot_speed = 6
cost = 200
random.seed(10)

def init_rooms(Rooms_data):
    room_dict = {}
    for room in Rooms_data.iterrows():
        room_dict[room[1]['id']] = Room(room[1]['id'],room[1]['x'],room[1]['y'])
    return room_dict


def init_patient(Patient_data,r_data):
    patient_array = []
    for patient in Patient_data.iterrows():
        patient_array.append(Patient(patient[1]['id'],r_data[patient[1]['room']],patient[1]['urgency'], patient[1]['typeofdisease']))
    return patient_array

def calc_dist(room1 , room2):
    return(math.sqrt(((room1[0] - room2[0])**2) + ((room1[1] - room2[1])**2)))



def room_distance_matrix(room_data):
    df = pd.DataFrame(columns=range(1,len(room_data)+1), index=range(1,len(room_data)+1))
    for i in room_data.keys():
        for j in room_data.keys():
            df.at[i,j] = calc_dist(room_data[i].get_loc(),room_data[j].get_loc())
    return df

def patient_distance_matrix(patient_data):
    df = pd.DataFrame(columns=range(1,len(patient_data)+1), index=range(1,len(patient_data)+1))
    for i in range(1,len(patient_data)+1):
        for j in range(1,len(patient_data)+1):
            df.at[i,j] = calc_dist(patient_data[i-1].room.get_loc(), patient_data[j-1].room.get_loc())
    return df


################# for plot ###########################
# def plotonimage(room_data):
#     x_arr = []
#     y_arr = []
#     for room in room_data.values():
#         temp_room_loc = room.get_loc()
#         x_arr.append(temp_room_loc[0])
#         y_arr.append(temp_room_loc[1])
#
#     plt.plot(x_arr , y_arr, linestyle = 'dashed', marker='s')
#
#     # plt.scatter(x_arr,y_arr)
#     plt.show()

################# for plot ###########################




def randomSolution(tsp):
    """creating a random solution for lower-bound"""
    patients = list(range(len(tsp)))
    solution = []

    for i in range(len(tsp)):
        random_patient = patients[random.randint(0, len(patients) - 1)]
        solution.append(random_patient)
        patients.remove(random_patient)

    return solution

def objective_function(tsp, solution, patients, robot_speed,cost):
    """Calculating objective function for each scenario"""
    target_value = 0
    time_of = 0
    for i in range(len(solution)):
        time_of += (tsp[solution[i - 1]][solution[i]] / robot_speed) + patients[solution[i-1]].type_of_disease
        latency = max((time_of - patients[solution[i-1]].urgency), 0)
        target_value = (time_of + latency * cost)
    return target_value

def getNeighbours(solution):
    """searching for all neighbours"""
    neighbours = []
    for i in range(len(solution)):
        for j in range(i + 1, len(solution)):
            neighbour = solution.copy()
            neighbour[i] = solution[j]
            neighbour[j] = solution[i]
            neighbours.append(neighbour)
    return neighbours

def getBestNeighbour(tsp, neighbours, patients, robot_speed,cost):
    """Looking for the best neighbour"""
    bestRouteLength = objective_function(tsp, neighbours[0], patients, robot_speed, cost)
    bestNeighbour = neighbours[0]
    for neighbour in neighbours:
        currentRouteLength = objective_function(tsp, neighbour, patients, robot_speed, cost)
        if currentRouteLength < bestRouteLength:
            bestRouteLength = currentRouteLength
            bestNeighbour = neighbour
    return bestNeighbour, bestRouteLength

def hillClimbing(tsp, patients, robot_speed, cost, max_time, time_limit):
    """Hill Climb algorithm"""
    currentSolution = randomSolution(tsp)
    currentRouteLength = objective_function(tsp, currentSolution, patients, robot_speed, cost)
    neighbours = getNeighbours(currentSolution)
    bestNeighbour, bestNeighbourRouteLength = getBestNeighbour(tsp, neighbours,patients, robot_speed, cost)
    start_time = time.time()
    if time_limit:
        """If the stop criterion is time"""
        while not isDone(start_time, max_time):  ## checks if the algorithm has more time to run
            neighbours = getNeighbours(currentSolution)
            bestNeighbour, bestNeighbourRouteLength = getBestNeighbour(tsp, neighbours, patients, robot_speed, cost)
            if bestNeighbourRouteLength < currentRouteLength:
                currentRouteLength = bestNeighbourRouteLength
                currentSolution = bestNeighbour
    else:
        """If the stop criterion is improvement"""
        while bestNeighbourRouteLength < currentRouteLength:
            currentSolution = bestNeighbour
            currentRouteLength = bestNeighbourRouteLength
            neighbours = getNeighbours(currentSolution)
            bestNeighbour, bestNeighbourRouteLength = getBestNeighbour(tsp, neighbours, patients, robot_speed, cost)
    return currentSolution, currentRouteLength

def isDone(start_time, max_time):
    """Checking if time limit as achieved"""
    return time.time() - start_time > max_time

def hill_climb_algorithm(matrix, patients, robot_speed, cost, max_time, time_limit):
    """Running Hill climb Algorithm"""
    tsp = matrix.values.tolist()
    return(hillClimbing(tsp, patients, robot_speed, cost, max_time, time_limit))

def patient_to_room_arr(pat_arr,patient_data):
    """creating a plot with the robot path"""
    room_arr = []
    x_arr = []
    y_arr =[]
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
    for i in range(0,len(x_arr)-1):
        j = i
        if j >= (len(x_arr)-1):
            break
        while x_arr[i] == x_arr[j+1] and y_arr[i] == y_arr[j+1]:
            x_arr[j + 1] = x_arr[j + 1] + 3
            y_arr[j + 1] = y_arr[j + 1] + 3
            if (j+1) < (len(x_arr) - 1):
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

def sa_tune():
    """Tune SA Hyper-parameters"""
    initial_temp = [10, 30, 50]
    alpha_list = [0.1, 0.99]
    temp_reduction = ['geometric', 'linear']
    for it in initial_temp:
        for alph in alpha_list:
            for tr in temp_reduction:
                sa = SimulatedAnnealing(tsp=patients_dist_matrix, patients=patient_data, robot_speed=robot_speed,
                                        cost=cost,
                                        initialTemp=it, finalTemp=0.01, tempReduction=tr,
                                        iterationPerTemp=100, alpha=alph, beta=5)

                sa_result = sa.run()
                print(f'intialTemp: {it}, alpha: {alph}, tempRduction: {tr} | z = {sa_result[1]} \n --path: {sa_result[0]}')


if __name__ == '__main__':
    room_data = init_rooms(Rooms_data)
    patient_data = init_patient(Patient_data, room_data)
    rooms_dist_matrix = room_distance_matrix(room_data)
    patients_dist_matrix = patient_distance_matrix(patient_data)
    time_limit = False
    # plotonimage(room_data)
    # for robot_speed in range(2,10):
    #     for cost in range(0,500,50):
    for max_time in range(5,9,5):
        hill_climb_result = hill_climb_algorithm(patients_dist_matrix, patient_data, robot_speed, cost, max_time, time_limit)
        print(hill_climb_result)

    sa = SimulatedAnnealing(tsp=patients_dist_matrix, patients=patient_data, robot_speed=robot_speed,
                            cost=cost,
                            initialTemp=30, finalTemp=0.01, tempReduction='geometric',
                            iterationPerTemp=100, alpha=0.99, beta=5)

    sa_result = sa.run(200, False)
    print(sa_result['solution'], "\n", sa_result['z'])
    # sa.plot_ZtoTEMP_sa(sa_result['z_list'], sa_result['temp_list'])
    # patient_to_room_arr(sa_result[0],patient_data)

    # hill_climb_result = hill_climb_algorithm(patients_dist_matrix, patient_data, robot_speed, cost, 10, False)
    # print(hill_climb_result)