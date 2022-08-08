from numpy import matrix
from Patient import Patient
from Room import Room
import random
import pandas as pd
import math
import matplotlib.pyplot as plt
import numpy as np
from numpy.random import randint
import time


######## READ DATA ###########
Patient_data = pd.read_csv('../Data/Patient50.csv')
Rooms_data = pd.read_csv('../Data/Rooms.csv')
random.seed(10)

######## Variables ##########
robot_speed = 6
cost = 200


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
    """Calculating distance between two rooms"""
    return (math.sqrt(((room1[0] - room2[0]) ** 2) + ((room1[1] - room2[1]) ** 2)))


def room_distance_matrix(room_data):
    """creating distance matrix"""
    df = pd.DataFrame(columns=range(1, len(room_data) + 1), index=range(1, len(room_data) + 1))
    for i in room_data.keys():
        for j in room_data.keys():
            df.at[i, j] = calc_dist(room_data[i].get_loc(), room_data[j].get_loc())
    return df


def patient_distance_matrix(patient_data):
    """creating distance matrix"""
    df = pd.DataFrame(columns=range(1, len(patient_data) + 1), index=range(1, len(patient_data) + 1))
    for i in range(1, len(patient_data) + 1):
        for j in range(1, len(patient_data) + 1):
            df.at[i, j] = calc_dist(patient_data[i - 1].room.get_loc(), patient_data[j - 1].room.get_loc())
    return df


def compute_patient_distance_coordinates(a, b):
    """Calculating distance between two patients"""
    return ((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2) ** 0.5


def objective_function(patient_a, patient_b, patients_dict,cost):
    """Objective function calculation"""
    target_value = 0
    distance = compute_patient_distance_coordinates(patients_dict[patient_a], patients_dict[patient_b])
    for pat in patient_data:
        if str(pat.id) == patient_a:
            time_of = (distance / robot_speed) + pat.type_of_disease
            latency = max(int(time_of - pat.urgency), 0)
            target_value = (time_of + latency * cost)

    return target_value


def genesis(patient_list, n_population):
    """First step: Create the first population set"""
    population_set = []
    for i in range(n_population):
        # Randomly generating a new solution
        sol_i = patient_list[np.random.choice(list(range(n_patients)), n_patients, replace=False)]
        population_set.append(sol_i)
    return np.array(population_set)


def fitnesss_eval(patient_list, patients_dict,cost):
    """Calculate individual objective function"""
    total = 0
    for i in range(n_patients - 1):
        a = patient_list[i]
        b = patient_list[i + 1]
        total += (objective_function(a, b, patients_dict,cost)*25)
    return total


def get_all_fitness(population_set, patients_dict,cost):
    """All solutions objective functions"""
    fitness_list = np.zeros(n_population)

    # Looping over all solutions computing the fitness for each solution
    for i in range(n_population):
        fitness_list[i] = fitnesss_eval(population_set[i], patients_dict,cost)

    return fitness_list


def progenitor_selection(population_set, fitness_list):
    """Selecting the progenitors"""
    total_fit = fitness_list.sum()
    prob_list = fitness_list / total_fit

    # Notice there is the chance that a progenitor. mates with oneself
    progenitor_list_a = np.random.choice(list(range(len(population_set))), len(population_set), p=prob_list,
                                         replace=True)
    progenitor_list_b = np.random.choice(list(range(len(population_set))), len(population_set), p=prob_list,
                                         replace=True)

    progenitor_list_a = population_set[progenitor_list_a]
    progenitor_list_b = population_set[progenitor_list_b]

    return np.array([progenitor_list_a, progenitor_list_b])


def mate_progenitors(prog_a, prog_b):
    """Pairs crossover"""
    offspring = prog_a[0:5]

    for patient in prog_b:

        if not patient in offspring:
            offspring = np.concatenate((offspring, [patient]))

    return offspring


def mate_population(progenitor_list):
    """Finding pairs of mates"""
    new_population_set = []
    for i in range(progenitor_list.shape[1]):
        prog_a, prog_b = progenitor_list[0][i], progenitor_list[1][i]
        offspring = mate_progenitors(prog_a, prog_b)
        new_population_set.append(offspring)

    return new_population_set


def mutate_offspring(offspring):
    """Offspring production"""
    for q in range(int(n_patients * mutation_rate)):
        a = np.random.randint(0, n_patients)
        b = np.random.randint(0, n_patients)

        offspring[a], offspring[b] = offspring[b], offspring[a]

    return offspring


def mutate_population(new_population_set):
    """New populaiton generation"""
    mutated_pop = []
    for offspring in new_population_set:
        mutated_pop.append(mutate_offspring(offspring))
    return mutated_pop


if __name__ == '__main__':
    """Prepare Data for GA algorithm, and determine parameters values"""
    room_data = init_rooms(Rooms_data)
    patient_data = init_patient(Patient_data, room_data)
    rooms_dist_matrix = room_distance_matrix(room_data)
    patients_dist_matrix = patient_distance_matrix(patient_data)
    fitness_list = patients_dist_matrix.values.tolist()[0]
    patient_coordinates_list = []
    names_list = []
    for i in patient_data:
        patient_coordinates_list.append([i.room.x, i.room.y])
    for i in patient_data:
        names_list.append(str(i.id))
    names_list = np.asarray(names_list)
    n_patients = 51
    n_population = 50
    mutation_rate = 0.1
    robot_speed = 6
    cost = 200
    generations = 500

    patients_dict = {x: y for x, y in zip(names_list, patient_coordinates_list)}
    population_set = genesis(names_list, n_population)
    fitness_list = get_all_fitness(population_set, patients_dict,cost)
    progenitor_list = progenitor_selection(population_set, fitness_list)
    new_population_set = mate_population(progenitor_list)
    mutated_pop = mutate_population(new_population_set)

    best_solution = [-1, np.inf, np.array([])]
    generations_list = [0]
    min_val_list = [0]
    mean_val_list = [0]
    start_time = time.time()

    for i in range(generations): ##stop criterion - Generations
        # if i % 100 == 0: print(best_solution, fitness_list.min()) ##print results every 100 generations
        fitness_list = get_all_fitness(mutated_pop, patients_dict,cost) ## calculate objective function value
        # Saving the best solution
        if fitness_list.min() < best_solution[1]: ## check if there is an improvement in the results
            best_solution[0] = i
            best_solution[1] = fitness_list.min()
            best_solution[2] = np.array(mutated_pop)[fitness_list.min() == fitness_list]
            print(best_solution, fitness_list.min())
        ## save data for the graphs
        generations_list.append(i)
        min_val_list.append(fitness_list.min())
        mean_val_list.append(fitness_list.mean())
        progenitor_list = progenitor_selection(population_set, fitness_list)
        new_population_set = mate_population(progenitor_list)

        mutated_pop = mutate_population(new_population_set)
        print(best_solution, fitness_list.min())
    print("for",generations," iterations, the time is: ",time.time() - start_time)
    print("for",generations," the best solution is: ",best_solution[1])


"""printing GA graphs"""
# fig = plt.figure()
# ax = fig.add_subplot()
# plt.plot(generations_list, min_val_list)
# plt.yticks( range(0,550,100) )
# plt.title('Genetic Best Values')
# plt.xlabel('generation')
# plt.ylabel('objective function')

# fig = plt.figure()
# ax = fig.add_subplot()
# plt.plot(generations_list, mean_val_list)
# plt.yticks( range(0,550,100) )
# plt.title('Genetic Mean Values')
# plt.xlabel('generation')
# plt.ylabel('objective function')
