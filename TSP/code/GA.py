import numpy as np
import matplotlib.pyplot as plt
import time
import random


SMALL_CITIES = np.array([
    [0, 0], [10, 15], [20, 5], [15, 10], [10, 10],
    [5, 20], [25, 15], [10, 25], [15, 5], [5, 5]
])

MEDIUM_CITIES = np.array([
    [0, 0], [10, 15], [20, 5], [15, 10], [10, 10], [5, 20], [25, 15],
    [10, 25], [15, 5], [5, 5], [30, 10], [35, 20], [25, 25], [20, 20],
    [15, 15]
])

LARGE_CITIES = np.array([
    [0, 0], [10, 15], [20, 5], [15, 10], [10, 10], [5, 20], [25, 15],
    [10, 25], [15, 5], [5, 5], [30, 10], [35, 20], [25, 25], [20, 20],
    [15, 15], [40, 10], [50, 20], [45, 25], [35, 30], [30, 25], [25, 35],
    [20, 30], [15, 40], [10, 35], [5, 30]
])


def calculate_distance(route, coordinates):
    distance = 0
    for i in range(len(route) - 1):
        distance += np.linalg.norm(coordinates[route[i]] - coordinates[route[i + 1]])
    distance += np.linalg.norm(coordinates[route[-1]] - coordinates[route[0]])  # Return to start
    return distance


def initialize_population(size, num_cities):
    """Initialize a random population of routes."""
    return [np.random.permutation(num_cities) for _ in range(size)]

def fitness_function(population, coordinates):
    """Evaluate fitness of each route."""
    return [1 / calculate_distance(route, coordinates) for route in population]

def selection(population, fitness):
    """Tournament selection."""
    selected = []
    for _ in range(len(population)):
        tournament = np.random.choice(len(population), 3, replace=False)
        winner = tournament[np.argmax([fitness[i] for i in tournament])]
        selected.append(population[winner])
    return selected

def crossover(parent1, parent2):
    """Order Crossover (OX)."""
    size = len(parent1)
    start, end = sorted(random.sample(range(size), 2))
    child = [-1] * size
    child[start:end] = parent1[start:end]

    pointer = 0
    for i in range(size):
        if child[i] == -1:
            while parent2[pointer] in child:
                pointer += 1
            child[i] = parent2[pointer]
    return child

def mutate(route, mutation_rate=0.1):
    """Swap mutation."""
    if random.random() < mutation_rate:
        i, j = random.sample(range(len(route)), 2)
        route[i], route[j] = route[j], route[i]
    return route

def genetic_algorithm(coordinates, population_size=100, generations=200, mutation_rate=0.1):
    """Run Genetic Algorithm."""
    num_cities = len(coordinates)
    population = initialize_population(population_size, num_cities)
    best_route, best_distance = None, float('inf')
    distances = []

    for generation in range(generations):
        fitness = fitness_function(population, coordinates)
        population = selection(population, fitness)
        new_population = []
        for i in range(0, len(population), 2):
            parent1, parent2 = population[i], population[(i + 1) % len(population)]
            child1 = mutate(crossover(parent1, parent2), mutation_rate)
            child2 = mutate(crossover(parent2, parent1), mutation_rate)
            new_population.extend([child1, child2])
        population = new_population
        # Track the best solution
        best_idx = np.argmax(fitness)
        current_best_distance = 1 / fitness[best_idx]
        distances.append(current_best_distance)
        if current_best_distance < best_distance:
            best_distance = current_best_distance
            best_route = population[best_idx]
    return best_route, best_distance, distances, population[0]


def calculate_route_diversity(routes):
    """Measure average route diversity (number of differing positions)."""
    num_routes = len(routes)
    total_differences = 0
    comparisons = 0

    for i in range(num_routes):
        for j in range(i + 1, num_routes):
            differences = np.sum(routes[i] != routes[j])
            total_differences += differences
            comparisons += 1

    return total_differences / comparisons


def main():
    # Dataset selection
    print("Select a dataset:")
    print("1. Small Cities")
    print("2. Medium Cities")
    print("3. Large Cities")
    choice = input("Enter the number corresponding to your choice: ")

    if choice == "1":
        coordinates = SMALL_CITIES
    elif choice == "2":
        coordinates = MEDIUM_CITIES
    elif choice == "3":
        coordinates = LARGE_CITIES
    else:
        print("Invalid choice. Exiting.")
        return


    runs = 10
    initial_distances, optimized_distances, computation_times, improvements, all_routes = [], [], [], [], []
    convergence_plots = []
    last_unoptimized_route = None
    last_optimized_route = None
    last_coordinates = None
    best_route_overall = None
    best_distance_overall = float("inf")

    print(f"Running Genetic Algorithm for {runs} iterations...")

    for run in range(runs):
        print(f"Run {run + 1}/{runs}...")
        start_time = time.time()
        best_route, best_distance, distances, unoptimized_route = genetic_algorithm(coordinates)
        end_time = time.time()


        initial_distance = calculate_distance(unoptimized_route, coordinates)
        computation_time = end_time - start_time
        improvement = ((initial_distance - best_distance) / initial_distance) * 100

        initial_distances.append(initial_distance)
        optimized_distances.append(best_distance)
        computation_times.append(computation_time)
        improvements.append(improvement)
        all_routes.append(best_route)
        convergence_plots.append(distances)


        if run == runs - 1:
            last_unoptimized_route = unoptimized_route
            last_optimized_route = best_route
            last_coordinates = coordinates

        # Track best overall route
        if best_distance < best_distance_overall:
            best_distance_overall = best_distance
            best_route_overall = best_route


    avg_initial = np.mean(initial_distances)
    avg_optimized = np.mean(optimized_distances)
    avg_time = np.mean(computation_times)
    avg_improvement = np.mean(improvements)
    best_distance = np.min(optimized_distances)
    worst_distance = np.max(optimized_distances)
    median_distance = np.median(optimized_distances)
    std_distance = np.std(optimized_distances)
    efficiency_ratio = avg_improvement / avg_time
    route_diversity = calculate_route_diversity(all_routes)

    print("\nResults:")
    print(f"Best Distance: {best_distance:.2f}")
    print(f"Worst Distance: {worst_distance:.2f}")
    print(f"Median Distance: {median_distance:.2f}")
    print(f"Standard Deviation of Distance: {std_distance:.2f}")
    print(f"Average Initial Distance: {avg_initial:.2f}")
    print(f"Average Optimized Distance: {avg_optimized:.2f}")
    print(f"Average Improvement: {avg_improvement:.2f}%")
    print(f"Average Computation Time: {avg_time:.2f}s")
    print(f"Average Algorithm Efficiency Ratio: {efficiency_ratio:.2f}")
    print(f"Average Route Diversity: {route_diversity:.2f} differing positions")


    plt.figure(figsize=(10, 6))
    for i, distances in enumerate(convergence_plots):
        plt.plot(distances, label=f"Run {i + 1}")
    plt.title("Convergence Plot")
    plt.xlabel("Generation")
    plt.ylabel("Distance")
    plt.legend()
    plt.grid()
    plt.show()


    plt.figure(figsize=(10, 6))
    plt.scatter(computation_times, optimized_distances, color='blue', label='Runs')
    plt.title("Pareto Graph")
    plt.xlabel("Computation Time (s)")
    plt.ylabel("Optimized Distance")
    plt.grid()
    plt.legend()
    plt.show()


    unoptimized_coords = [last_coordinates[i] for i in last_unoptimized_route]
    unoptimized_coords.append(last_coordinates[last_unoptimized_route[0]])  # Close the loop
    x, y = zip(*unoptimized_coords)

    plt.figure(figsize=(10, 6))
    plt.plot(x, y, marker="o", label="Unoptimized Route", color="red")
    plt.title(f"Unoptimized Route - Distance: {initial_distances[-1]:.2f}")
    plt.xlabel("X Coordinate")
    plt.ylabel("Y Coordinate")
    plt.legend()
    plt.grid()
    plt.show()


    optimized_coords = [last_coordinates[i] for i in last_optimized_route]
    optimized_coords.append(last_coordinates[last_optimized_route[0]])  # Close the loop
    x, y = zip(*optimized_coords)

    plt.figure(figsize=(10, 6))
    plt.plot(x, y, marker="o", label="Optimized Route", color="green")
    plt.title(f"Optimized Route - Distance: {optimized_distances[-1]:.2f}")
    plt.xlabel("X Coordinate")
    plt.ylabel("Y Coordinate")
    plt.legend()
    plt.grid()
    plt.show()


    best_coords = [last_coordinates[i] for i in best_route_overall]
    best_coords.append(last_coordinates[best_route_overall[0]])  # Close the loop
    x, y = zip(*best_coords)

    plt.figure(figsize=(10, 6))
    plt.plot(x, y, marker="o", label="Best Overall Route", color="blue")
    plt.title(f"Best Overall Route - Distance: {best_distance_overall:.2f}")
    plt.xlabel("X Coordinate")
    plt.ylabel("Y Coordinate")
    plt.legend()
    plt.grid()
    plt.show()


if __name__ == "__main__":
    main()

