import random
from dataclasses import dataclass
from core.models import JobShopProblem, ScheduledOperation



@dataclass(slots=True)
class Chromosome:
    genes: list[int]
    makespan: int = 0
    schedule: list[ScheduledOperation] | None = None



    @classmethod
    def create(cls, problem: JobShopProblem, calculate_makespan: bool = True) -> "Chromosome":
        genes: list[int] = []

        for job in problem.jobs:
            genes.extend([job.job_id] * len(job.operations))

        random.shuffle(genes)
        chromosome = cls(genes=genes)
        if calculate_makespan:
            chromosome.calculateMakespan(problem)
        
        return chromosome



    def evaluate(self, problem: JobShopProblem, includeSchedule: bool = False) -> tuple[int, list[ScheduledOperation]]:
        next_op = [0] * problem.num_jobs
        job_end = [0] * problem.num_jobs
        machine_end = [0] * (problem.num_machines + 1)
        makespan = 0
        schedule: list[ScheduledOperation] = []

        job_index_map = problem.job_index_map
        jobs = problem.jobs

        for job_id in self.genes:
            j_idx = job_index_map[job_id]
            op_idx = next_op[j_idx]
            op = jobs[j_idx].operations[op_idx]

            next_op[j_idx] = op_idx + 1

            job_ready = job_end[j_idx]
            machine_ready = machine_end[op.machine_id]

            start = max(job_ready, machine_ready)
            end = start + op.duration

            if end > makespan:
                makespan = end

            machine_end[op.machine_id] = end
            job_end[j_idx] = end

            if includeSchedule:
                schedule.append(
                    ScheduledOperation(
                        job_id=job_id,
                        operation_id=op_idx + 1,
                        machine_id=op.machine_id,
                        start=start,
                        end=end
                    )
                )

        if includeSchedule:
            schedule.sort(key=lambda op: (op.machine_id, op.start))

        return makespan, schedule



    def calculateMakespan(self, problem: JobShopProblem) -> int:
        self.makespan = self.evaluate(problem, False)[0]
        return self.makespan



    def buildSchedule(self, problem: JobShopProblem) -> list[ScheduledOperation]:
        self.makespan, self.schedule = self.evaluate(problem, True)
        return self.schedule



    def __lt__(self, other: "Chromosome") -> bool:
        return self.makespan < other.makespan
