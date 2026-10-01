import random
import heapq

try:
    import torch
except ImportError:  # GPU support remains optional until PyTorch is installed.
    torch = None
from core.models import JobShopProblem
from core.chromosome import Chromosome



class GeneticAlgorithm:
    def __init__(self, problem: JobShopProblem, populationSize: int = 100, generations: int = 200, 
                 crossoverRate: float = 0.85, mutationRate: float = 0.15, 
                 tournamentSize: int = 3, elitismCount: int = 2, verbose: bool = True,
                 device: str = "auto"):
        
        self.problem = problem
        self.populationSize = populationSize
        self.generations = generations
        self.crossoverRate = crossoverRate
        self.mutationRate = mutationRate
        self.tournamentSize = tournamentSize
        self.elitismCount = elitismCount
        self.verbose = verbose
        self.device = self._resolve_device(device)
        self.population: list[Chromosome] = []
        self.history_best: list[int] = []
        self.history_avg: list[float] = []
        self.best_chromosome: Chromosome | None = None


    @staticmethod
    def _resolve_device(requested_device: str) -> str:
        """Choose CUDA when available, or raise if the user explicitly requires it."""
        if requested_device not in {"auto", "cpu", "cuda"}:
            raise ValueError("device must be 'auto', 'cpu', or 'cuda'")

        cuda_available = torch is not None and torch.cuda.is_available()
        if requested_device == "cuda" and not cuda_available:
            raise RuntimeError(
                "CUDA was requested but is unavailable. Install a CUDA-enabled PyTorch build."
            )
        return "cuda" if requested_device == "cuda" or (requested_device == "auto" and cuda_available) else "cpu"



    @staticmethod
    def _createOffspring(p_keep: list[int], p_order: list[int], keep_set: set[int]) -> list[int]:
        remaining = [gene for gene in p_order if gene not in keep_set]
        rem_idx = 0
        offspring: list[int] = []

        for gene in p_keep:
            if gene in keep_set:
                offspring.append(gene)
            else:
                offspring.append(remaining[rem_idx])
                rem_idx += 1

        return offspring



    def createPopulation(self) -> list[Chromosome]:
        return [
            Chromosome.create(self.problem, calculate_makespan=self.device == "cpu")
            for _ in range(self.populationSize)
        ]


    def evaluatePopulationOnGpu(self) -> None:
        """Evaluate every chromosome simultaneously on the selected CUDA device.

        Genetic operators intentionally stay on the CPU because they are highly
        branchy.  Fitness evaluation is the dominant numerical workload and is
        batched here, so all population members run in parallel on the GPU.
        """
        if self.device != "cuda" or torch is None:
            return

        job_index = self.problem.job_index_map
        genes = torch.tensor(
            [[job_index[job_id] for job_id in chromosome.genes] for chromosome in self.population],
            dtype=torch.long,
            device=self.device,
        )
        population_count, gene_count = genes.shape
        job_count = self.problem.num_jobs
        max_operations = max(len(job.operations) for job in self.problem.jobs)

        durations = torch.zeros((job_count, max_operations), dtype=torch.long, device=self.device)
        machines = torch.zeros((job_count, max_operations), dtype=torch.long, device=self.device)
        for job_idx, job in enumerate(self.problem.jobs):
            for operation_idx, operation in enumerate(job.operations):
                durations[job_idx, operation_idx] = operation.duration
                machines[job_idx, operation_idx] = operation.machine_id

        rows = torch.arange(population_count, device=self.device)
        next_operation = torch.zeros((population_count, job_count), dtype=torch.long, device=self.device)
        job_end = torch.zeros((population_count, job_count), dtype=torch.long, device=self.device)
        machine_end = torch.zeros(
            (population_count, self.problem.num_machines + 1), dtype=torch.long, device=self.device
        )
        makespans = torch.zeros(population_count, dtype=torch.long, device=self.device)

        for gene_position in range(gene_count):
            current_jobs = genes[:, gene_position]
            operation_indices = next_operation[rows, current_jobs]
            current_machines = machines[current_jobs, operation_indices]
            end_time = torch.maximum(
                job_end[rows, current_jobs], machine_end[rows, current_machines]
            ) + durations[current_jobs, operation_indices]

            next_operation[rows, current_jobs] = operation_indices + 1
            job_end[rows, current_jobs] = end_time
            machine_end[rows, current_machines] = end_time
            makespans = torch.maximum(makespans, end_time)

        for chromosome, makespan in zip(self.population, makespans.cpu().tolist()):
            chromosome.makespan = makespan



    def crossover(self, parent1: Chromosome, parent2: Chromosome) -> tuple[Chromosome, Chromosome]:
        job_ids = self.problem.job_ids
        if len(job_ids) <= 1:
            return Chromosome(genes=parent1.genes.copy(), makespan=parent1.makespan), Chromosome(genes=parent2.genes.copy(), makespan=parent2.makespan)

        k = random.randint(1, len(job_ids) - 1)
        keep_set = set(random.sample(job_ids, k))

        child1_genes = self._createOffspring(parent1.genes, parent2.genes, keep_set)
        child2_genes = self._createOffspring(parent2.genes, parent1.genes, keep_set)

        child1 = Chromosome(genes=child1_genes)
        child2 = Chromosome(genes=child2_genes)
        if self.device == "cpu":
            child1.calculateMakespan(self.problem)
            child2.calculateMakespan(self.problem)

        return child1, child2



    def mutate(self, chromosome: Chromosome) -> Chromosome:
        mutated_genes = chromosome.genes.copy()

        if len(mutated_genes) < 2 or len(set(mutated_genes)) <= 1:
            return Chromosome(genes=mutated_genes, makespan=chromosome.makespan)

        while True:
            idx1, idx2 = random.sample(range(len(mutated_genes)), 2)
            if mutated_genes[idx1] != mutated_genes[idx2]:
                break

        mutated_genes[idx1], mutated_genes[idx2] = (mutated_genes[idx2], mutated_genes[idx1])
        mutated_chromosome = Chromosome(genes=mutated_genes)
        if self.device == "cpu":
            mutated_chromosome.calculateMakespan(self.problem)
        return mutated_chromosome



    def tournamentSelection(self) -> Chromosome:
        candidates = random.sample(self.population, self.tournamentSize)
        return min(candidates)



    def run(self) -> Chromosome:
        self.population = self.createPopulation()
        self.evaluatePopulationOnGpu()
        self.history_best = []
        self.history_avg = []

        for gen in range(self.generations):
            elites = heapq.nsmallest(min(self.elitismCount, self.populationSize), self.population)
            gen_best = elites[0]
            gen_avg = sum(c.makespan for c in self.population) / self.populationSize

            self.history_best.append(gen_best.makespan)
            self.history_avg.append(gen_avg)

            if self.verbose:
                print(f"Gen {gen + 1:3d}/{self.generations} | Best: {gen_best.makespan} | Avg: {gen_avg:.2f}")

            next_generation: list[Chromosome] = [
                Chromosome(genes=c.genes.copy(), makespan=c.makespan)
                for c in elites
            ]

            while len(next_generation) < self.populationSize:
                p1 = self.tournamentSelection()
                p2 = self.tournamentSelection()

                if random.random() < self.crossoverRate:
                    c1, c2 = self.crossover(p1, p2)
                else:
                    c1 = Chromosome(genes=p1.genes.copy(), makespan=p1.makespan)
                    c2 = Chromosome(genes=p2.genes.copy(), makespan=p2.makespan)

                if random.random() < self.mutationRate:
                    c1 = self.mutate(c1)
                if random.random() < self.mutationRate:
                    c2 = self.mutate(c2)

                next_generation.append(c1)
                if len(next_generation) < self.populationSize:
                    next_generation.append(c2)

            self.population = next_generation
            self.evaluatePopulationOnGpu()

        best = min(self.population)
        best.buildSchedule(self.problem)
        self.best_chromosome = best
        return best
