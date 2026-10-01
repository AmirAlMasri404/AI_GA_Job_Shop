from core.models import Job, Operation, JobShopProblem



def parser(filename: str) -> JobShopProblem | None:
    job = {}
    jobs = []
    max_machine = 0

    try:
        with open(filename, "r") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue

                job_segment, op_segment = line.split(":", 1)
                job_segment = job_segment.strip()
                op_segment = op_segment.strip()

                job_id = int(job_segment.split("_")[1])
                job[job_id] = []

                op_tokens = op_segment.split("->")

                for op in op_tokens:
                    op = op.strip()
                    machine_part, duration_part = op.split("[")
                    machine_id = int(machine_part.removeprefix("M"))
                    duration = int(duration_part.removesuffix("]"))
                    if machine_id > max_machine:
                        max_machine = machine_id
                    mach_op = [machine_id, duration]
                    job[job_id].append(mach_op)

        for job_id, op_list in job.items():
            operations = []
            for machine_id, duration in op_list:
                operations.append(Operation(machine_id=machine_id, duration=duration))

            jobs.append(Job(job_id=job_id, operations=operations))

    except (FileNotFoundError, ValueError) as e:
        print(f"Error parsing file: {e}")
        return None

    if not jobs:
        return None

    return JobShopProblem(jobs=jobs, num_machines=max_machine)
