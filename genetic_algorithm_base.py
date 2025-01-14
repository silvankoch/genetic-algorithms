import numpy as np

class GeneticAlgorithm:
    def __init__(self, initial_population, fitness_function, crossover_function, mutation_function):
        """A general framework for optimizing the fitness of a population with a genetic algorithm.
        
        Args:
            initial_population (list): A list of individuals in the population
            fitness_function (function): The fitness function to evaluate the individuals. fitness_function(individual) -> fitness
            crossover_function (function): The function to perform crossover between two individuals. crossover(parent1, parent2) -> child
            mutation_function (function): The function to mutate an individual. mutation(individual) -> mutated_individual
        """
        
        self.population = initial_population
        self.fitness_function = fitness_function
        self.crossover_function = crossover_function
        self.mutation_function = mutation_function

        self.population_fitness = None
        self._population_size = len(initial_population)

    def evaluate_population_fitness(self):
        """Calculate the fitness of each individual in the population"""
        self.population_fitness = np.array([self.fitness_function(individual) for individual in self.population])

    def select(self, n_individuals):
        """Select the fittest individuals from the population"""
        return [self.population[i] for i in np.argsort(self.population_fitness)[-n_individuals:]]

    def produce_offspring(self, parent_candidates, number_of_offspring):
        """Generate offspring from selected parents"""
        offspring = []
        n = len(parent_candidates)  

        for _ in range(number_of_offspring):
            parent1 = parent_candidates[np.random.randint(n)]
            parent2 = parent_candidates[np.random.randint(n)]
            child = self.crossover_function(parent1, parent2)
            child = self.mutation_function(child)
            offspring.append(child)

        return offspring

    def _next_generation(self, number_of_parents, number_of_offspring):
        """Generate the next generation from the current population"""
        # Evaluate fitness
        self.evaluate_population_fitness()

        # Pick parents & produce offspring
        parent_candidates = self.select(number_of_parents)
        offspring = self.produce_offspring(parent_candidates, number_of_offspring)

        # Evaluate offspring fitness
        offspring_fitnesses = np.array([self.fitness_function(individual) for individual in offspring])

        # Add offspring to the population
        self.population = np.concatenate((self.population, offspring))
        self.population_fitness = np.concatenate((self.population_fitness, offspring_fitnesses))
        
        # Select the fittest individuals
        self.population = self.select(self._population_size)
        # note: at this point, the population fitness size does not match the population size anymore
        # this gets fixed the next time evaluate_population_fitness is called

    def evolve(self, number_of_generations, number_of_parents, number_of_offspring):
        """Evolve the population over a number of generations"""
        
        generational_fitness = []

        for _ in range(number_of_generations):
            self._next_generation(number_of_parents, number_of_offspring)
            generational_fitness.append(max(self.population_fitness))

        return self.population, generational_fitness