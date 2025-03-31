import numpy as np
import matplotlib.pyplot as plt
import time
import random
from collections import deque


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


def tabu_search(coordinates, tabu_size=10, max_iterations=1000, max_no_improve=100):
    num_cities = len(coordinates)
    current_route = list(range(num_cities))
    random.shuffle(current_route)
    current_distance = calculate_distance(current_route, coordinates)

    best_route = current_route[:]
    best_distance = current_distance

    tabu_list = deque(maxlen=tabu_size)
    no_improve_counter = 0
    convergence = []

    for iteration in range(max_iterations):
        neighborhood = []
        distances = []


        for i in range(num_cities):
            for j in range(i + 1, num_cities):
                new_route = current_route[:]
                new_route[i], new_route[j] = new_route[j], new_route[i]
                if (i, j) not in tabu_list:
                    neighborhood.append((new_route, (i, j)))
                    distances.append(calculate_distance(new_route, coordinates))

        if not distances:
            break


        best_candidate_idx = np.argmin(distances)
        best_candidate, swap = neighborhood[best_candidate_idx]
        best_candidate_distance = distances[best_candidate_idx]


        current_route = best_candidate
        current_distance = best_candidate_distance
        tabu_list.append(swap)


        if current_distance < best_distance:
            best_route = current_route[:]
            best_distance = current_distance
            no_improve_counter = 0
        else:
            no_improve_counter += 1

        convergence.append(best_distance)


        if no_improve_counter >= max_no_improve:
            break

    return best_route, best_distance, convergence


def main():
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
    initial_distances, optimized_distances, computation_times, improvements, convergence_plots = [], [], [], [], []
    best_distance_overall = float('inf')
    best_route_overall = None
    last_coordinates = coordinates

    print(f"Running Tabu Search for {runs} iterations...")

    for run in range(runs):
        print(f"Run {run + 1}/{runs}...")
        start_time = time.time()
        best_route, best_distance, convergence = tabu_search(coordinates)
        end_time = time.time()

        initial_distance = calculate_distance(list(range(len(coordinates))), coordinates)
        computation_time = end_time - start_time
        improvement = ((initial_distance - best_distance) / initial_distance) * 100

        initial_distances.append(initial_distance)
        optimized_distances.append(best_distance)
        computation_times.append(computation_time)
        improvements.append(improvement)
        convergence_plots.append(convergence)

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


    plt.figure(figsize=(10, 6))
    for i, distances in enumerate(convergence_plots):
        plt.plot(distances, label=f"Run {i + 1}")
    plt.title("Convergence Plot")
    plt.xlabel("Iteration")
    plt.ylabel("Best Distance")
    plt.legend()
    plt.grid()
    plt.show()


    plt.figure(figsize=(10, 6))
    plt.scatter(computation_times, optimized_distances, color='blue', label='Runs')
    plt.title("Pareto Plot")
    plt.xlabel("Computation Time (s)")
    plt.ylabel("Optimized Distance")
    plt.grid()
    plt.legend()
    plt.show()


    unoptimized_coords = [last_coordinates[i] for i in range(len(last_coordinates))]
    unoptimized_coords.append(last_coordinates[0])  # Close the loop
    x, y = zip(*unoptimized_coords)

    plt.figure(figsize=(10, 6))
    plt.plot(x, y, marker="o", label="Unoptimized Route", color="red")
    plt.title(f"Unoptimized Route - Distance: {initial_distances[-1]:.2f}")
    plt.xlabel("X Coordinate")
    plt.ylabel("Y Coordinate")
    plt.legend()
    plt.grid()
    plt.show()


    optimized_coords = [last_coordinates[i] for i in best_route_overall]
    optimized_coords.append(last_coordinates[best_route_overall[0]])  # Close the loop
    x, y = zip(*optimized_coords)

    plt.figure(figsize=(10, 6))
    plt.plot(x, y, marker="o", label="Optimized Route", color="green")
    plt.title(f"Optimized Route - Distance: {best_distance_overall:.2f}")
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
