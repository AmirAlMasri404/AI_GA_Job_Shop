import matplotlib.pyplot as plt
import matplotlib.patches as m_patches



class ScheduleVisualizer:
    def __init__(self, schedule, machines_count, jobs_count):
        self.schedule = schedule
        self.machines_count = machines_count
        self.jobs_count = jobs_count
        self.job_colors = self._assign_job_colors()



    def _assign_job_colors(self):
        cmap = plt.colormaps["Set1"]
        return {job_id: cmap((job_id - 1) % cmap.N) for job_id in range(1, self.jobs_count + 1)}



    def _group_by_machine(self):
        by_machine = {
            machine_id: []
            for machine_id in range(1, self.machines_count + 1)
        }

        for op in self.schedule:
            by_machine[op.machine_id].append(op)

        for operations in by_machine.values():
            operations.sort(key=lambda op: op.start)

        return by_machine



    def gantt_plot(
        self,
        save_path=None,
        show=True,
        fig_size=(14, 7),
        dark_mode=True,
    ):
        makespan = max((op.end for op in self.schedule), default=0)

        fig = plt.figure(figsize=fig_size, dpi=100)
        ax = fig.add_subplot(111)

        if dark_mode:
            fig.patch.set_facecolor("#2b2b2b")
            ax.set_facecolor("#1e1e1e")
            text_color = "white"
            grid_color = "#444444"
        else:
            fig.patch.set_facecolor("#ffffff")
            ax.set_facecolor("#f9f9f9")
            text_color = "black"
            grid_color = "#dddddd"

        by_machine = self._group_by_machine()

        for machine_id, operations in by_machine.items():
            last_end = 0

            for op in operations:
                if op.start > last_end:
                    ax.barh(
                        machine_id,
                        op.start - last_end,
                        left=last_end,
                        height=0.62,
                        facecolor="none",
                        edgecolor="red",
                        hatch="//",
                        linewidth=0.8,
                    )

                duration = op.end - op.start

                ax.barh(
                    machine_id,
                    duration,
                    left=op.start,
                    height=0.62,
                    color=self.job_colors[op.job_id],
                    edgecolor="black",
                    linewidth=0.8,
                    alpha=0.95,
                )

                if makespan > 0 and duration / makespan > 0.02:
                    ax.text(
                        op.start + duration / 2,
                        machine_id,
                        f"J{op.job_id}",
                        ha="center",
                        va="center",
                        fontsize=13,
                        fontweight="bold",
                        color="white",
                    )

                last_end = max(last_end, op.end)

        machine_ids = list(by_machine.keys())

        ax.set_yticks(machine_ids)
        ax.set_yticklabels(
            [f"Machine {machine_id}" for machine_id in machine_ids],
            color=text_color,
            fontsize=9,
        )
        ax.set_xlabel(
            "Time (Units)",
            color=text_color,
            fontsize=10,
            fontweight="bold",
        )
        ax.set_title(
            f"Optimized Schedule Gantt Chart (Makespan: {makespan})",
            color=text_color,
            fontsize=12,
            fontweight="bold",
        )
        ax.set_xlim(0, max(makespan, 1))
        ax.invert_yaxis()
        ax.grid(axis="x", linestyle="--", color=grid_color, alpha=0.6)
        ax.tick_params(colors=text_color)

        for spine in ax.spines.values():
            spine.set_color(grid_color)

        job_handles = [
            m_patches.Patch(
                facecolor=self.job_colors[job_id],
                edgecolor="black",
                label=f"Job {job_id}",
            )
            for job_id in range(1, self.jobs_count + 1)
        ]

        idle_handle = m_patches.Patch(
            facecolor="none",
            edgecolor="red",
            hatch="//",
            label="Idle",
        )

        legend = ax.legend(
            handles=job_handles + [idle_handle],
            bbox_to_anchor=(1.01, 1),
            loc="upper left",
            fontsize=8,
            title="Jobs",
            title_fontsize=9,
            facecolor="#2b2b2b" if dark_mode else "#ffffff",
            edgecolor=grid_color,
        )

        plt.setp(legend.get_texts(), color=text_color)
        plt.setp(legend.get_title(), color=text_color)

        fig.tight_layout()

        if save_path:
            fig.savefig(save_path, dpi=200, facecolor=fig.get_facecolor())

        if show:
            plt.show()
        else:
            plt.close(fig)

        return fig



    @staticmethod
    def plot_convergence(
        history_best,
        history_avg,
        save_path=None,
        show=True,
        figsize=(10, 5),
    ):
        fig = plt.figure(figsize=figsize, dpi=100)
        ax = fig.add_subplot(111)

        generations = range(1, len(history_best) + 1)

        ax.plot(generations, history_best, label="Best Makespan", linewidth=2)
        ax.plot(
            generations,
            history_avg,
            label="Average Makespan",
            linestyle="--",
        )

        ax.set_xlabel("Generation")
        ax.set_ylabel("Makespan")
        ax.set_title("GA Convergence")
        ax.legend()

        fig.tight_layout()

        if save_path:
            fig.savefig(save_path, dpi=200)

        if show:
            plt.show()
        else:
            plt.close(fig)

        return fig



    def machine_utilization(self, makespan):
        if makespan <= 0:
            return {
                machine_id: 0.0
                for machine_id in range(1, self.machines_count + 1)
            }

        busy_time = {
            machine_id: 0
            for machine_id in range(1, self.machines_count + 1)
        }

        for op in self.schedule:
            busy_time[op.machine_id] += op.end - op.start

        return {
            machine_id: round(100 * time / makespan, 2)
            for machine_id, time in busy_time.items()
        }
