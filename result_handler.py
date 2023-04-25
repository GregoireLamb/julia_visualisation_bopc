import glob
import pandas as pd
import matplotlib.pyplot as plt

if __name__ == "__main__":
    files = glob.glob("results/*.dat")
    for file in files:
        df = pd.read_csv(file, sep=";", header=None, names=["size","patch", "nprocs", "time"])
        df = df.groupby(["size", "patch", "nprocs"]).mean()
        ref_dict = df.loc[df.index.get_level_values("nprocs") == 1, "time"].to_dict()
        ref_dict = {key[0]: value for key, value in ref_dict.items()}
        df["speedup"] = df.index.get_level_values("size").map(ref_dict) / df["time"]
        df["efficiency"] = df["speedup"] / df.index.get_level_values("nprocs")

        fig, axs = plt.subplots(2, 2,figsize=(9, 5))
        fig.subplots_adjust(hspace=0.5, wspace=0.5)
        fig.suptitle('Output analysis')
        nprocs = df.index.get_level_values("nprocs").unique()
        metrics = ["time", "speedup", "efficiency"]
        titles = ["absolute running time", "speedup", "efficiency"]
        for i in [0, 1]:
            for j in [0, 1]:
                if i+j < 2:
                    for size in df.index.get_level_values("size").unique():
                        axs[i, j].plot(nprocs, df.loc[size][metrics[i+2*j]], label=f"size {size}", marker="o")
                        axs[i, j].set_title(titles[i+2*j])
                        axs[i, j].set_xlabel("number of processes")
                        axs[i, j].set_ylabel(metrics[i+2*j])

        handles, labels = axs[0, 0].get_legend_handles_labels()
        fig.legend(handles, labels, loc="upper right", ncol=2)
        plt.show()
        plt.close()


