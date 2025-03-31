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



def initialize_particles(num_particles, num_cities):
    """Initialize particles as random routes."""
    return [random.sample(range(num_cities), num_cities) for _ in range(num_particles)]


def update_velocity_and_position(particles, velocities, p_best, g_best, w, c1, c2, num_cities):
    """Update the velocity and position of each particle."""
    for i in range(len(particles)):
        for j in range(num_cities):
            velocities[i][j] = (
                    w * velocities[i][j]
                    + c1 * random.random() * (p_best[i][j] - particles[i][j])
                    + c2 * random.random() * (g_best[j] - particles[i][j])
            )
        # Update particle position by sorting indices based on velocity
        particles[i] = sorted(range(num_cities), key=lambda k: velocities[i][k])
    return particles, velocities


def particle_swarm_optimization(coordinates, num_particles=50, iterations=100, w=0.5, c1=1.5, c2=1.5):
    """Run PSO to solve the TSP."""
    num_cities = len(coordinates)
    particles = initialize_particles(num_particles, num_cities)
    velocities = [random.sample(range(num_cities), num_cities) for _ in range(num_particles)]
    p_best = particles.copy()
    p_best_scores = [calculate_distance(route, coordinates) for route in particles]
    g_best = particles[np.argmin(p_best_scores)]
    g_best_score = min(p_best_scores)

    convergence = []

    for _ in range(iterations):
        distances = [calculate_distance(route, coordinates) for route in particles]
        for i in range(num_particles):
            if distances[i] < p_best_scores[i]:
                p_best[i] = particles[i]
                p_best_scores[i] = distances[i]
        g_best = p_best[np.argmin(p_best_scores)]
        g_best_score = min(p_best_scores)

        particles, velocities = update_velocity_and_position(
            particles, velocities, p_best, g_best, w, c1, c2, num_cities
        )
        convergence.append(g_best_score)

    return g_best, g_best_score, convergence, particles[0]



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
    best_distance_overall = float('inf')

    print(f"Running PSO for {runs} iterations...")

    for run in range(runs):
        print(f"Run {run + 1}/{runs}...")
        start_time = time.time()
        best_route, best_distance, convergence, unoptimized_route = particle_swarm_optimization(coordinates)
        end_time = time.time()

        # Collect metrics
        initial_distance = calculate_distance(unoptimized_route, coordinates)
        computation_time = end_time - start_time
        improvement = ((initial_distance - best_distance) / initial_distance) * 100

        initial_distances.append(initial_distance)
        optimized_distances.append(best_distance)
        computation_times.append(computation_time)
        improvements.append(improvement)
        all_routes.append(best_route)
        convergence_plots.append(convergence)


        if run == runs - 1:
            last_unoptimized_route = unoptimized_route
            last_optimized_route = best_route
            last_coordinates = coordinates

        # Track the best overall route
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
