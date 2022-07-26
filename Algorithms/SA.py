"""
Simulated Annealing Class
"""
import random
import math
import time
import matplotlib.pyplot as plt

random.seed(10)


def randomSolution(tsp):
    """
    Returns a random initial solution
    :param tsp: the distance matrix
    :return: list of patients order.
    """
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

    def isTerminationCriteriaMet(self, max_iter, iter_limit, current_iter):
        """
        Returns a boolean, whether the algorithm should finish the run or not.
        :param max_iter: Number of iterations
        :param iter_limit: Boolean -> whether to use iterations constarint or not.
        :param current_iter: int -> current iteration number
        :return: True or False
        """
        if iter_limit:    # if iterations constraint is applied.
            return self.currTemp <= self.finalTemp or self.get_neighbors(self.solution) == 0 or self.isDone(current_iter,
                                                                                                            max_iter)
        else:
            return self.currTemp <= self.finalTemp or self.get_neighbors(self.solution) == 0

    def isDone(self, curr_iter, max_iter):
        return curr_iter > max_iter

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
        """
        Returns the value of our objective function
        :param solution: list of patients
        :return: float.
        """
        target_value = 0
        time_of = 0
        for i in range(len(solution)):
            time_of += (self.tsp[solution[i - 1]][solution[i]] / self.robot_speed) + self.patients[
                solution[i - 1]].type_of_disease
            latency = max((time_of - self.patients[solution[i - 1]].urgency), 0)
            target_value = (time_of + latency * self.cost)
        return target_value

    def run(self, max_iter=200, iter_limit=False):
        """
        Execute the SA algorithm
        :param max_iter: max iterations
        :param iter_limit: Apply iterations constraint (bool)
        :return: solution
        """
        start_time = time.time()
        z_list = []
        temp_list = []
        curr_iter = 0
        bestSolution = self.solution
        while not self.isTerminationCriteriaMet(max_iter, iter_limit, curr_iter):
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

                if self.objective_function(self.solution) < self.objective_function(bestSolution):
                    bestSolution = self.solution

            z_list.append(self.objective_function(self.solution))
            temp_list.append(curr_iter)

            # decrement the temperature
            self.decrementRule()

            curr_iter = curr_iter + 1

            print(self.objective_function(self.solution), ", temp: ", self.currTemp)
        print(f'run time: {time.time() - start_time}')
        self.plot_ZtoTEMP_sa(z_list, temp_list)
        return {'solution': bestSolution, 'z': self.objective_function(bestSolution), 'z_list': z_list,
                'temp_list': temp_list}


    def plot_ZtoTEMP_sa(self, z_l, temp_l):
        # fig = plt.figure()
        # ax = fig.add_subplot()
        plt.plot(temp_l, z_l)
        plt.title('Simulated Annealing with Best Parameters')
        plt.xlabel('Iteration')
        plt.ylabel('Objective Function')
        plt.ticklabel_format(useOffset=False)
        plt.show()
