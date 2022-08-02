from numpy import matrix

from Patient import Patient
from Room import Room
import random
import pandas as pd
import math
import matplotlib.pyplot as plt
# import mlrose
import numpy as np
from numpy.random import randint
from numpy.random import rand
from datetime import datetime


######## READ DATA ###########
Patient_data = pd.read_csv('Patient.csv')
Rooms_data = pd.read_csv('Rooms.csv')
random.seed(10)

######## Variables ##########
ROBOT_SPEED = 6
COST = 0


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
    return (math.sqrt(((room1[0] - room2[0]) ** 2) + ((room1[1] - room2[1]) ** 2)))


def room_distance_matrix(room_data):
    df = pd.DataFrame(columns=range(1, len(room_data) + 1), index=range(1, len(room_data) + 1))
    for i in room_data.keys():
        for j in room_data.keys():
            df.at[i, j] = calc_dist(room_data[i].get_loc(), room_data[j].get_loc())
    return df


def patient_distance_matrix(patient_data):
    df = pd.DataFrame(columns=range(1, len(patient_data) + 1), index=range(1, len(patient_data) + 1))
    for i in range(1, len(patient_data) + 1):
        for j in range(1, len(patient_data) + 1):
            df.at[i, j] = calc_dist(patient_data[i - 1].room.get_loc(), patient_data[j - 1].room.get_loc())
    return df


# objective functions
def compute_patient_distance_coordinates(a, b):
    return ((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2) ** 0.5


def objective_function(patient_a, patient_b, patients_dict):
    target_value = 0
    distance = compute_patient_distance_coordinates(patients_dict[patient_a], patients_dict[patient_b])
    for pat in patient_data:
        if str(pat.id) == patient_a:
            time_of = (distance / robot_speed) + pat.type_of_disease
            latency = max((time_of - pat.urgency), 0)
            target_value = (time_of + latency * cost)

    return target_value


# First step: Create the first population set
def genesis(patient_list, n_population):
    population_set = []
    for i in range(n_population):
        # Randomly generating a new solution
        sol_i = patient_list[np.random.choice(list(range(n_patients)), n_patients, replace=False)]
        population_set.append(sol_i)
    return np.array(population_set)


# 2. Evaluation of the fitnesss

# individual solution
def fitnesss_eval(patient_list, patients_dict):
    total = 0
    for i in range(n_patients - 1):
        a = patient_list[i]
        b = patient_list[i + 1]
        total += objective_function(a, b, patients_dict)
    return total


# All solutions
def get_all_fitness(population_set, patients_dict):
    fitness_list = np.zeros(n_population)

    # Looping over all solutions computing the fitness for each solution
    for i in range(n_population):
        fitness_list[i] = fitnesss_eval(population_set[i], patients_dict)

    return fitness_list


# 3. Selecting the progenitors
def progenitor_selection(population_set, fitness_list):
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


# Pairs crossover
def mate_progenitors(prog_a, prog_b):
    offspring = prog_a[0:5]

    for patient in prog_b:

        if not patient in offspring:
            offspring = np.concatenate((offspring, [patient]))

    return offspring


# Finding pairs of mates
def mate_population(progenitor_list):
    new_population_set = []
    for i in range(progenitor_list.shape[1]):
        prog_a, prog_b = progenitor_list[0][i], progenitor_list[1][i]
        offspring = mate_progenitors(prog_a, prog_b)
        new_population_set.append(offspring)

    return new_population_set


# Offspring production
def mutate_offspring(offspring):
    for q in range(int(n_patients * mutation_rate)):
        a = np.random.randint(0, n_patients)
        b = np.random.randint(0, n_patients)

        offspring[a], offspring[b] = offspring[b], offspring[a]

    return offspring


# New populaiton generation
def mutate_population(new_population_set):
    mutated_pop = []
    for offspring in new_population_set:
        mutated_pop.append(mutate_offspring(offspring))
    return mutated_pop


if __name__ == '__main__':
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

    n_patients = 101
    n_population = 20
    mutation_rate = 0.1
    robot_speed = 6
    cost = 200
    generations = 100

    patients_dict = {x: y for x, y in zip(names_list, patient_coordinates_list)}
    population_set = genesis(names_list, n_population)
    fitness_list = get_all_fitness(population_set, patients_dict)
    progenitor_list = progenitor_selection(population_set, fitness_list)
    new_population_set = mate_population(progenitor_list)
    mutated_pop = mutate_population(new_population_set)

    # Everything put together
    best_solution = [-1, np.inf, np.array([])]
    generations_list = []
    min_val_list = []
    for i in range(generations):
        # if i % 100 == 0: print(best_solution, fitness_list.min(), datetime.now().strftime("%d/%m/%y %H:%M"))
        fitness_list = get_all_fitness(mutated_pop, patients_dict)
        # Saving the best solution
        if fitness_list.min() < best_solution[1]:
            best_solution[0] = i
            best_solution[1] = fitness_list.min()
            best_solution[2] = np.array(mutated_pop)[fitness_list.min() == fitness_list]
            print(best_solution, fitness_list.min())
        generations_list.append(i)
        min_val_list.append(fitness_list.min())
        progenitor_list = progenitor_selection(population_set, fitness_list)
        new_population_set = mate_population(progenitor_list)

        mutated_pop = mutate_population(new_population_set)

print(len(generations_list),generations_list)
print(len(min_val_list),min_val_list)