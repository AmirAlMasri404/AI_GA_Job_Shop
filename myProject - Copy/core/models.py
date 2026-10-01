from dataclasses import dataclass



@dataclass(slots=True)
class Operation:
    duration: int
    machine_id: int



@dataclass(slots=True)
class Job:
    job_id: int
    operations: list[Operation]



@dataclass(slots=True)
class ScheduledOperation:
    job_id: int
    operation_id: int
    machine_id: int
    start: int
    end: int



class JobShopProblem:

    def __init__(self, jobs: list[Job], num_machines: int):
        self.jobs = jobs
        self.num_jobs = len(jobs)
        self.num_machines = num_machines
        self.job_ids = tuple(job.job_id for job in jobs)
        self.job_index_map = {job.job_id: idx for idx, job in enumerate(jobs)}