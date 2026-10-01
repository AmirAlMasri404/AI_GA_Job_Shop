from core.ga import GeneticAlgorithm
from visualization.plots import ScheduleVisualizer
from io_handler.parser import parser
from io_handler.reporter import exportReport



def collect_ga_params() -> dict:
    print("\n--- GA Parameters (press Enter to use default) ---")

    return {
        "populationSize": int(input("  Population size [100]: ").strip() or 100),
        "generations": int(input("  Generations [200]: ").strip() or 200),
        "crossoverRate": float(input("  Crossover rate [0.85]: ").strip() or 0.85),
        "mutationRate": float(input("  Mutation rate [0.15]: ").strip() or 0.15),
        "tournamentSize": int(input("  Tournament size [3]: ").strip() or 3),
        "elitismCount": int(input("  Elitism count [2]: ").strip() or 2),
    }



def main():
    print("=== Job Shop Scheduling — Genetic Algorithm ===\n")

    filename = input("Enter job definition file path [data/input.txt]: ").strip() or "data/input.txt"
    problem = parser(filename)

    if problem is None:
        print("Failed to parse file or no jobs found. Exiting.")
        return

    print(f"\nParsed {problem.num_jobs} jobs, {problem.num_machines} machines.")

    params = collect_ga_params()

    device = input("  Compute device [cuda]: ").strip().lower() or "cuda"

    print("\nRunning Genetic Algorithm...\n")
    ga = GeneticAlgorithm(problem=problem, verbose=True, device=device, **params)
    print(f"Using: {ga.device.upper()}")
    best = ga.run()

    output_file = "data/output.txt"
    exportReport(
        filepath=output_file,
        problem=problem,
        schedule=best.schedule,
        makespan=best.makespan,
        solver=ga
    )
    print(f"Optimization finished! Schedule and report saved to: {output_file}")

    viz = ScheduleVisualizer(
        best.schedule,
        problem.num_machines,
        problem.num_jobs,
    )

    gantt_path = "data/gantt.png"
    conv_path = "data/convergence.png"

    viz.gantt_plot(
        save_path=gantt_path,
        show=False,
        fig_size=(30, 10),
    )
    ScheduleVisualizer.plot_convergence(
        ga.history_best,
        ga.history_avg,
        save_path=conv_path,
        show=False,
    )

    print("\nOptimization finished!")
    print(f"  * Schedule report:    {output_file}")
    print(f"  * Gantt chart:        {gantt_path}")
    print(f"  * Convergence curve:  {conv_path}")



if __name__ == "__main__":
    main()
