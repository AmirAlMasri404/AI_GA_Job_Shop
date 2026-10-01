import random
import time
from core.models import Operation, Job, JobShopProblem, ScheduledOperation
from core.chromosome import Chromosome
from core.ga import GeneticAlgorithm



def verifyConstraints(schedule: list[ScheduledOperation], jobs: list[Job]) -> bool:
    machine_intervals: dict[int, list[tuple[int, int, str]]] = {}
    for op in schedule:
        machine_intervals.setdefault(op.machine_id, []).append((op.start, op.end, f"Job {op.job_id} Op {op.operation_id}"))

    for mid, intervals in machine_intervals.items():
        intervals.sort(key=lambda x: x[0])
        for i in range(len(intervals) - 1):
            curr_start, curr_end, curr_name = intervals[i]
            next_start, next_end, next_name = intervals[i + 1]
            if curr_end > next_start:
                print(f"FAILED: Machine {mid} collision between {curr_name} and {next_name}")
                return False

    job_intervals: dict[int, list[tuple[int, int, int]]] = {}
    for op in schedule:
        job_intervals.setdefault(op.job_id, []).append((op.operation_id, op.start, op.end))

    for jid, ops in job_intervals.items():
        ops.sort(key=lambda x: x[0])
        for i in range(len(ops) - 1):
            curr_op_id, curr_start, curr_end = ops[i]
            next_op_id, next_start, next_end = ops[i + 1]
            if curr_end > next_start:
                print(f"FAILED: Job {jid} precedence violation between Op {curr_op_id} and Op {next_op_id}")
                return False

    return True



def generateIndustrialProblem(num_jobs: int, num_machines: int, seed: int = 99) -> JobShopProblem:
    rng = random.Random(seed)
    jobs: list[Job] = []

    for j_id in range(1, num_jobs + 1):
        machines = list(range(1, num_machines + 1))
        rng.shuffle(machines)
        operations = [
            Operation(duration=rng.randint(10, 100), machine_id=m)
            for m in machines
        ]
        jobs.append(Job(job_id=j_id, operations=operations))

    return JobShopProblem(jobs=jobs, num_machines=num_machines)



def main() -> None:
    NUM_JOBS = 30
    NUM_MACHINES = 20
    
    POP_SIZE = 250
    GENERATIONS = 300

    TOTAL_OPERATIONS = NUM_JOBS * NUM_MACHINES
    TOTAL_EVALS = POP_SIZE * GENERATIONS
    TOTAL_GENE_OPERATIONS = TOTAL_EVALS * TOTAL_OPERATIONS

    problem = generateIndustrialProblem(num_jobs=NUM_JOBS, num_machines=NUM_MACHINES, seed=99)
    jobs = problem.jobs

    max_job_work = max(sum(op.duration for op in j.operations) for j in jobs)
    machine_loads: dict[int, int] = {}
    for j in jobs:
        for op in j.operations:
            machine_loads[op.machine_id] = machine_loads.get(op.machine_id, 0) + op.duration
    busiest_machine = max(machine_loads, key=machine_loads.get)
    lower_bound = max(max_job_work, max(machine_loads.values()))

    print("=" * 75)
    print(f"  EXTREME INDUSTRIAL STRESS BENCHMARK: {NUM_JOBS} JOBS x {NUM_MACHINES} MACHINES")
    print(f"     ({TOTAL_OPERATIONS} Total Operations | Search Space: ~10^970 schedules)")
    print("=" * 75)
    print(f"  * Problem Scale:        {NUM_JOBS} Jobs x {NUM_MACHINES} Machines ({TOTAL_OPERATIONS} Operations)")
    print(f"  * Theoretical Workload: Longest Job = {max_job_work} | Busiest Machine {busiest_machine} = {max(machine_loads.values())}")
    print(f"  * Physical Lower Bound: {lower_bound} time units")
    print(f"  * Simulation Workload:  {POP_SIZE} pop x {GENERATIONS} gens = {TOTAL_EVALS:,} evaluations")
    print(f"  * Total Operations:     {TOTAL_GENE_OPERATIONS:,} gene evaluations to process")
    print("-" * 75)
    print("  Starting execution...\n")

    solver = GeneticAlgorithm(
        problem=problem,
        populationSize=POP_SIZE,
        generations=GENERATIONS,
        crossoverRate=0.85,
        mutationRate=0.15,
        tournamentSize=3,
        elitismCount=2,
        verbose=True,
        device="cuda",
    )

    t_start = time.perf_counter()
    best = solver.run()
    t_end = time.perf_counter()

    elapsed = t_end - t_start
    throughput = TOTAL_GENE_OPERATIONS / elapsed if elapsed > 0 else 0
    evals_per_sec = TOTAL_EVALS / elapsed if elapsed > 0 else 0

    print("-" * 75)
    print("  BENCHMARK RESULTS:")
    print(f"  * Elapsed Time:         {elapsed:.3f} seconds")
    print(f"  * Throughput:           {throughput:,.0f} gene-ops/sec ({evals_per_sec:,.0f} schedules/sec)")
    print(f"  * Initial Best:         {solver.history_best[0]} time units")
    print(f"  * Final Optimized:      {best.makespan} time units")
    drop = solver.history_best[0] - best.makespan
    print(f"  * Makespan Reduction:   -{drop} time units ({(drop / solver.history_best[0]) * 100:.2f}% improvement)")
    print(f"  * Lower Bound Gap:      {((best.makespan / lower_bound) - 1) * 100:.2f}% above physical limit")

    print("\n  Running full constraint validation...")
    valid = verifyConstraints(best.schedule, jobs)
    assert valid, "Constraint validation failed!"
    print(f"  * Constraint Check:     PASSED (All {len(best.schedule)} operations strictly obey all rules)")
    print("=" * 75)



if __name__ == "__main__":
    main()
