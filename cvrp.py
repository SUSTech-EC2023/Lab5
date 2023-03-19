import random
import numpy as np

def generate_cvrp_instance(num_customers, max_demand, max_capacity):
    customers = np.random.randint(1, max_demand, size=num_customers)
    coordinates = np.random.randint(0, 100, size=(num_customers, 2))
    return customers, coordinates, max_capacity


def euclidean_distance(a, b):
    return np.sqrt((a[0] - b[0])**2 + (a[1] - b[1])**2)


# TODO: implement the fitness evaluation function fitness()
def fitness(solution, customers, coordinates, max_capacity):
    '''
    @describe: calculate the fitness (total distance) of the solution

    @param: solution: a list of nodes, e.g., [3, 2, 4, 1]
    @param: customers: demand of each customer, e.g., [9, 7, 5, 7]
    @param: coordinates: coordinates of each customer, e.g., [array([85, 82]), array([80, 51]), array([95, 24]), array([49, 70]), array([ 5, 87])]
    @param: max_capacity: the capacity of each vehicle, e.g., 50

    @return: total distance of the solution
    '''
    return


def generate_initial_population(num_individuals, num_customers):
    population = []
    for _ in range(num_individuals):
        individual = list(range(1, num_customers))
        random.shuffle(individual)
        population.append(individual)
    return population


# TODO: implement your mutation operator
def mutate(individual):
    '''
    @describe: the mutation operator
    @param: individual: a list of nodes, e.g., [3, 2, 4, 1]
    @return: individual after mutation
    '''
    return


#TODO: implement your crossover operator
def crossover(parent1, parent2):
    '''
    @describe: the crossover operator
    @param parent1: a list of nodes, e.g., [3, 2, 4, 1]
    @param parent2: a list of nodes, e.g., [4, 3, 2, 1]
    @return child1, child2: two offspring obtained by crossover
    '''
    return


# TODO: implement your tournament selection method
def tournament_selection():
    '''
    @describe: the tournament selection method, select k individuals from polulation, the best one is selected as one parent
    @param: Determined by yourself
    @return parent
    '''
    return


def evolutionary_algorithm(num_generations, num_individuals, mutation_rate, num_customers, max_demand, max_capacity):
    customers, coordinates, capacity = generate_cvrp_instance(num_customers, max_demand, max_capacity)
    population = generate_initial_population(num_individuals, num_customers)
    best_solution = min(population, key=lambda x: fitness(x, customers, coordinates, capacity))
    best_fitness = fitness(best_solution, customers, coordinates, capacity)
    print("Best solution of current generation:", best_solution)
    print("Best fitness of current generation:", best_fitness)

    for generation in range(num_generations):
        new_population = []
        for _ in range(num_individuals // 2):
            # Selection
            parent1 = tournament_selection()
            parent2 = tournament_selection()

            # Crossover
            child1, child2 = crossover(parent1, parent2)

            # Mutation
            if random.random() < mutation_rate:
                mutate(child1)
            if random.random() < mutation_rate:
                mutate(child2)

            new_population.extend([child1, child2])

        population = new_population
        # Find the best solution in the current population
        best_solution = min(population, key=lambda x: fitness(x, customers, coordinates, capacity))
        best_fitness = fitness(best_solution, customers, coordinates, capacity)
        print("Best solution of current generation:", best_solution)
        print("Best fitness of current generation:", best_fitness)

    # Find the best solution in the final population
    best_solution = min(population, key=lambda x: fitness(x, customers, coordinates, capacity))
    best_fitness = fitness(best_solution, customers, coordinates, capacity)
    return best_solution, best_fitness




if __name__ == "__main__":
    num_generations = 100
    num_individuals = 50
    mutation_rate = 0.2
    num_customers = 20
    max_demand = 10
    max_capacity = 50

    best_solution, best_fitness = evolutionary_algorithm(num_generations, num_individuals, mutation_rate, num_customers, max_demand, max_capacity)

    print("Best solution:", best_solution)
    print("Best fitness:", best_fitness)
