from core.models import JobShopProblem, ScheduledOperation
from core.ga import GeneticAlgorithm



def calculateMachineUtilization(problem: JobShopProblem, schedule: list[ScheduledOperation], makespan: int) -> dict[int, float]:
    if makespan <= 0:
        return {m: 0.0 for m in range(1, problem.num_machines + 1)}

    busy_time = {m: 0 for m in range(1, problem.num_machines + 1)}
    for op in schedule:
        busy_time[op.machine_id] += (op.end - op.start)

    return {m: round((busy_time[m] / makespan) * 100, 2) for m in range(1, problem.num_machines + 1)}



def exportReport(
    filepath: str,
    problem: JobShopProblem,
    schedule: list[ScheduledOperation],
    makespan: int,
    solver: GeneticAlgorithm,
    elapsed_time: float = 0.0
) -> None:
    utilization = calculateMachineUtilization(problem, schedule, makespan)

    with open(filepath, "w", encoding="utf-8") as f:
        f.write("=" * 70 + "\n")
        f.write("  MANUFACTURING PLANT JOB SHOP OPTIMIZATION REPORT\n")
        f.write("=" * 70 + "\n")
        f.write(f"  * Total Jobs:            {problem.num_jobs}\n")
        f.write(f"  * Total Machines:        {problem.num_machines}\n")
        f.write(f"  * Optimal Makespan:      {makespan} time units\n")

        if solver.history_best:
            initial_best = solver.history_best[0]
            reduction = initial_best - makespan
            pct = (reduction / initial_best) * 100 if initial_best > 0 else 0
            f.write(f"  * Initial Makespan:      {initial_best} time units\n")
            f.write(f"  * Makespan Reduction:    -{reduction} units ({pct:.2f}%)\n")

        if elapsed_time > 0:
            f.write(f"  * Execution Time:        {elapsed_time:.4f} seconds\n")

        f.write(f"  * Population Size:       {solver.populationSize}\n")
        f.write(f"  * Total Generations:     {solver.generations}\n")
        f.write(f"  * Crossover Rate:        {solver.crossoverRate}\n")
        f.write(f"  * Mutation Rate:         {solver.mutationRate}\n")
        f.write("=" * 70 + "\n\n")

        f.write("--- MACHINE UTILIZATION BREAKDOWN ---\n")
        for m_id, util in utilization.items():
            f.write(f"  Machine {m_id:2d}: {util:6.2f}% active\n")
        f.write("\n")

        f.write("--- DETAILED OPERATION SCHEDULE ---\n")
        f.write(f"  {'Machine':<12} {'Job':<10} {'Operation':<14} {'Start':<10} {'End':<10} {'Duration':<10}\n")
        f.write("  " + "-" * 66 + "\n")

        for op in schedule:
            dur = op.end - op.start
            f.write(
                f"  M{op.machine_id:<11} Job_{op.job_id:<6} Op {op.operation_id:<11} {op.start:<10} {op.end:<10} {dur:<10}\n"
            )

        f.write("=" * 70 + "\n")
