import glob
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages


if __name__ == "__main__":
    files = glob.glob("../results/*exp23*.dat")
    for file in files:
        df = pd.read_csv(file, sep=";", header=None, names=["size","patch", "nprocs", "time"])
        df = df.groupby(["size", "nprocs", "patch"]).mean()

        fig2 = plt.plot(df.index.get_level_values("patch"), df["time"], label="time", marker="o")
        # add title and label
        plt.title(f"Raw data analysis for {file[11:-4]}, c=cs")
        plt.xlabel("patch size")
        plt.ylabel("mean runtime (s)")
        plt.savefig(f"{file[:-4]}.png")
        plt.close()

        df.reset_index(inplace=True)
        fig, ax = plt.subplots(figsize=(12, 4))
        ax.axis('tight')
        ax.axis('off')
        the_table = ax.table(cellText=df.values, colLabels=df.columns, loc='center')
        ax.set_title(f"Raw data analysis for {file[11:-4]}, c=cs")
        pp = PdfPages(f"{file[:-4]}.pdf")
        pp.savefig(fig, bbox_inches='tight')
        pp.close()

