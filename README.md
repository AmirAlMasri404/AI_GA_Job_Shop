Job Shop Scheduling Using a Genetic Algorithm

This project solves the Job Shop Scheduling Problem using a Genetic Algorithm (GA). The goal is to find an efficient order for processing job operations across multiple machines while minimizing the total completion time, known as the makespan.

The program reads job and operation data from an input file, then generates and evolves possible schedules using selection, crossover, mutation, and elitism. Each generated schedule respects job-operation order and machine availability constraints.

After the algorithm finishes, the system displays the best makespan and the full operation schedule. It also creates a Gantt chart showing when each job runs on every machine, highlights machine idle time, calculates machine utilization percentages, and produces a convergence chart showing how the GA improves across generations.

The project is implemented in Python and uses Matplotlib for visualization.
