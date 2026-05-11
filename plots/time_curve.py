from benchopt import BasePlot
import numpy as np


class Plot(BasePlot):
    name = "Run Curves"
    type = "scatter"
    options = {
        "objective": ...,
        "dataset": ...,
    }

    def plot(self, df, objective, dataset):
        df = df.query(f"objective_name == '{objective}' and dataset_name == '{dataset}'")
        objective_column = "objective_run_time"
        plot_data = []

        total_solvers = df['solver_name'].unique()
        solver_names = [x.split('[')[0] for x in total_solvers]
        solver_names = list(set(solver_names))

        for solver_name in solver_names:
            solvers = [s for s in total_solvers if s.split('[')[0] == solver_name]
            x_vals = []
            y_vals = []
            q1_vals = []
            q9_vals = []

            for s in solvers:
                df_filtered = df[df['solver_name'] == s]
                if objective_column not in df_filtered.columns:
                    continue

                y = df_filtered[objective_column].dropna().values.tolist()
                if len(y) == 0:
                    continue

                y_val = np.median(y) if len(y) > 1 else y[0]
                x_nodes = int(s.split('slurm_nodes=')[1].split(']')[0])

                x_vals.append(x_nodes)
                y_vals.append(y_val)
                q1_vals.append(np.percentile(y, 25))
                q9_vals.append(np.percentile(y, 75))

            # Sort by x values (number of nodes)
            x_vals, y_vals, q1_vals, q9_vals = zip(*sorted(zip(x_vals, y_vals, q1_vals, q9_vals)))

            plot_data.append({
                "x": x_vals,
                "y": y_vals,
                "y_low": q1_vals,
                "y_high": q9_vals,
                "label": solver_name,
                **self.get_style(solver_name),
            })

        return plot_data

    def get_metadata(self, df, objective, dataset):
        return {
            "title": f"Run Time Curves\n{objective}\nData: {dataset}",
            "xlabel": "Number of nodes",
            "ylabel": "Time (s)",
            "scale": "semilog-x"
        }