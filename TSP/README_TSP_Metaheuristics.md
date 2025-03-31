# Metaheuristic Algorithms for the Traveling Salesman Problem (TSP)

This project provides Python implementations of four powerful metaheuristic algorithms to solve the classic **Traveling Salesman Problem (TSP)** — a combinatorial optimization problem where the goal is to find the shortest possible route visiting all cities exactly once and returning to the start.

---

## Algorithms Implemented

1. **Genetic Algorithm (GA)**
2. **Particle Swarm Optimization (PSO)**
3. **Simulated Annealing (SA)**
4. **Tabu Search (TS)**

Each algorithm is tailored to handle different aspects of the optimization challenge and supports comparison through consistent evaluation metrics and visualizations.

---

## About Metaheuristics
Metaheuristics are high-level problem-independent algorithmic frameworks that provide a set of guidelines or strategies to develop heuristic optimization algorithms. These methods are particularly useful for:
- Large search spaces
- Incomplete or imperfect information
- Avoiding local optima

All algorithms in this repo are tested on the TSP and evaluated based on distance optimization, convergence, runtime, and solution quality.



## Datasets
Three predefined city datasets are included in each script:
- **Small Cities** (10 nodes)
- **Medium Cities** (15 nodes)
- **Large Cities** (25 nodes)

---

##  How to Run

1. Clone the repository:
```bash
git clone https://github.com/Derrick-M99/tsp-metaheuristics.git
cd tsp-metaheuristics
```

2. Run any script:
```bash
python tsp_ga.py    # Genetic Algorithm
python tsp_pso.py   # Particle Swarm Optimization
python tsp_sa.py    # Simulated Annealing
python tsp_ts.py    # Tabu Search
```

3. Follow the on-screen prompts to select a dataset.

Each run will:
- Optimize routes for 10 iterations
- Output route statistics
- Display visualizations (convergence, route map, Pareto plot)

---

## Output Metrics
- Best, worst, median distances
- Average computation time
- Improvement percentage
- Convergence curves
- Route visualizations

---

##  Requirements
Install dependencies:
```bash
pip install numpy matplotlib
```
Tested on Python 3.8+

---
