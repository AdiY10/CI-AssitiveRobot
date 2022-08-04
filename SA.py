"""
Simulated Annealing Class
"""
import random
import math
import time

random.seed(10)

def randomSolution(tsp):
    patients = list(range(len(tsp)))
    solution = []

    for i in range(len(tsp)):
        random_patient = patients[random.randint(0, len(patients) - 1)]
        solution.append(random_patient)
        patients.remove(random_patient)

    return solution


class SimulatedAnnealing:
    def __init__(self, tsp, patients, robot_speed, cost, initialTemp, finalTemp, tempReduction,
                 iterationPerTemp=100, alpha=10, beta=5):
        self.tsp = tsp.values.tolist()
        self.solution = randomSolution(self.tsp)
        self.currTemp = initialTemp
        self.finalTemp = finalTemp
        self.iterationPerTemp = iterationPerTemp
        self.alpha = alpha
        self.beta = beta

        self.patients = patients
        self.robot_speed = robot_speed
        self.cost = cost

        if tempReduction == "linear":
            self.decrementRule = self.linearTempReduction
        elif tempReduction == "geometric":
            self.decrementRule = self.geometricTempReduction
        elif tempReduction == "slowDecrease":
            self.decrementRule = self.slowDecreaseTempReduction
        else:
            self.decrementRule = tempReduction

    def linearTempReduction(self):
        self.currTemp -= self.alpha

    def geometricTempReduction(self):
        self.currTemp *= self.alpha

    def slowDecreaseTempReduction(self):
        self.currTemp = self.currTemp / (1 + self.beta * self.currTemp)

    def isTerminationCriteriaMet(self, max_time, time_limit, start_time):
        if time_limit:
            return self.currTemp <= self.finalTemp or self.get_neighbors(self.solution) == 0 or self.isDone(start_time,
                                                                                                            max_time)
        else:
            return self.currTemp <= self.finalTemp or self.get_neighbors(self.solution) == 0

    def isDone(self, start_time, max_time):
        return time.time() - start_time > max_time

    def get_neighbors(self, solution):
        """ Returns a list of possible neighbors to current solution"""
        neighbours = []
        for i in range(len(solution)):
            for j in range(i + 1, len(solution)):
                neighbour = solution.copy()
                neighbour[i] = solution[j]
                neighbour[j] = solution[i]
                neighbours.append(neighbour)
        return neighbours

    def objective_function(self, solution):
        target_value = 0
        time_of = 0
        for i in range(len(solution)):
            time_of += (self.tsp[solution[i - 1]][solution[i]] / self.robot_speed) + self.patients[
                solution[i - 1]].type_of_disease
            latency = max((time_of - self.patients[solution[i - 1]].urgency), 0)
            target_value = (time_of + latency * self.cost)
        return target_value

    def run(self, max_time=10000, time_limit=False):
        start_time = time.time()
        while not self.isTerminationCriteriaMet(max_time, time_limit, start_time):
            # iterate that number of times
            for i in range(self.iterationPerTemp):
                # get all of the neighbors
                neighbors = self.get_neighbors(self.solution)
                # pick a random neighbor
                newSolution = random.choice(neighbors)
                # get the cost between the two solutions
                cost = self.objective_function(self.solution) - self.objective_function(newSolution)
                # if the new solution is better, accept it
                if cost >= 0:
                    self.solution = newSolution
                    # print('found better')
                # if the new solution is not better, accept it with a probability of e^(-cost/temp)
                else:
                    # print(math.exp(cost / self.currTemp))
                    if random.uniform(0, 1) < math.exp(cost / self.currTemp):
                        print(math.exp(cost / self.currTemp))
                        self.solution = newSolution
            # decrement the temperature
            self.decrementRule()
            print(self.objective_function(self.solution), ", temp: ", self.currTemp )
        print(f'run time: {time.time()-start_time}')
        return [self.solution, self.objective_function(self.solution)]
