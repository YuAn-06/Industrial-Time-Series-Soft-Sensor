import argparse
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import rcParams
from sklearn.manifold import TSNE
from statsmodels.graphics.tsaplots import plot_acf
from statsmodels.tsa.stattools import adfuller

rcParams["font.sans-serif"] = ["SimHei"]
rcParams["axes.unicode_minus"] = False
rcParams["font.family"] = "Times New Roman"


DATASET_PATHS = {
    "DC": "data/DC/debutanizer_column.csv",
    "SRU": "data/SRU/SRU_data.csv",
    "Ironmaking": "data/Ironmaking/Ironmaking.csv",
    "PPGAS": "data/PPGAS/gt_2012.csv",
    "MP": "data/MP/MP_data.csv",
}


def plot_time_series(data, columns):
    _, d = data.shape
    fig, axes = plt.subplots(math.ceil(d / 3), 3, figsize=(24, 18), squeeze=False)
    axes = axes.flatten()
    for i in range(d):
        axes[i].plot(data[:, i], label=columns[i])
        axes[i].legend()

    for ax in axes[d:]:
        ax.set_visible(False)
    plt.tight_layout()
    return fig


def plot_acf_lag(data, columns, lags=100):
    _, d = data.shape
    fig, axes = plt.subplots(math.ceil(d / 3), 3, figsize=(18, 12), squeeze=False)
    axes = axes.flatten()
    for i in range(d):
        max_lag = min(lags, max(data.shape[0] - 2, 1))
        plot_acf(data[:, i], ax=axes[i], lags=max_lag, alpha=0.05, label=columns[i])
        axes[i].legend(fontsize=8)
        axes[i].set_xlabel("Lag", fontsize=4)
        axes[i].set_ylabel("Autocorrelation", fontsize=4)

    for ax in axes[d:]:
        ax.set_visible(False)
    plt.tight_layout()
    return fig


def plot_spearmanr(data, columns, corr):
    fig = plt.figure(figsize=(12, 10))
    sns.heatmap(corr, cmap="coolwarm", annot=True, fmt=".2f", xticklabels=columns, yticklabels=columns)
    plt.tight_layout()
    return fig


def plot_data_2d(train_data, test_data):
    tsne = TSNE(n_components=2, random_state=42, init="pca", learning_rate="auto")
    all_data = np.vstack([train_data, test_data])
    all_data_tsne = tsne.fit_transform(all_data)
    train_data_tsne = all_data_tsne[: len(train_data)]
    test_data_tsne = all_data_tsne[len(train_data) :]
    df = pd.DataFrame(
        {
            "x": np.concatenate((train_data_tsne[:, 0], test_data_tsne[:, 0])),
            "y": np.concatenate((train_data_tsne[:, 1], test_data_tsne[:, 1])),
            "label": ["training"] * len(train_data_tsne) + ["testing"] * len(test_data_tsne),
        }
    )

    g = sns.jointplot(
        x="x",
        y="y",
        data=df,
        kind="scatter",
        hue="label",
        palette={"training": "#3E4F94", "testing": "#B02425"},
        joint_kws={"alpha": 0.5, "s": 80, "edgecolors": "black", "linewidths": 0.3},
    )
    g.ax_joint.tick_params(labelsize=12)
    g.ax_joint.set_xlabel(" ")
    g.ax_joint.set_ylabel(" ")
    g.ax_joint.grid(True)
    return g.fig


def ADF_test(data, columns):
    results = []
    for i in range(data.shape[1]):
        result = adfuller(data[:, i])
        stationary = result[1] < 0.05
        message = (
            f"{columns[i]}: ADF Statistic: {result[0]}, p-value: {result[1]}, "
            f"the series is {'stationary' if stationary else 'non-stationary'}"
        )
        print(message)
        results.append(
            {
                "column": columns[i],
                "adf_statistic": float(result[0]),
                "p_value": float(result[1]),
                "stationary": stationary,
                "message": message,
            }
        )
    return results


def detect_outliers(data, columns):
    n_samples = data.shape[0]
    results = []

    for i in range(data.shape[1]):
        q1 = np.percentile(data[:, i], 25)
        q3 = np.percentile(data[:, i], 75)
        iqr = q3 - q1
        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr

        iqr_outliers = (data[:, i] < lower_bound) | (data[:, i] > upper_bound)
        iqr_percentage = (iqr_outliers.sum() / n_samples) * 100
        message = f"{columns[i]}: IQR Outliers Percentage: {iqr_percentage:.5f}%"
        print(message)
        results.append(
            {
                "column": columns[i],
                "count": int(iqr_outliers.sum()),
                "percentage": float(iqr_percentage),
                "message": message,
            }
        )
    return results


def parse_args():
    parser = argparse.ArgumentParser(description="Visualize dataset characteristics and save a report.")
    parser.add_argument("--data-name", "--dataset", default="MP", dest="data_name")
    parser.add_argument("--csv", default=None, help="Optional CSV path. Defaults to the known path for data-name.")
    parser.add_argument("--target", default=None, help="Optional target column for report correlations.")
    parser.add_argument("--output-dir", default=None, help="Optional output directory.")
    parser.add_argument("--plot-format", default="png", choices=["png", "pdf", "svg"])
    parser.add_argument("--acf-lags", type=int, default=100)
    parser.add_argument("--tsne-sample-rows", type=int, default=3000)
    parser.add_argument("--show", action="store_true", help="Show figures interactively after saving.")
    parser.add_argument("--no-plots", action="store_true", help="Only save the Markdown report.")
    return parser.parse_args()


def load_dataset(data_name, csv_path):
    path = csv_path or DATASET_PATHS.get(data_name)
    if path is None:
        raise ValueError(f"Unknown data_name '{data_name}'. Pass --csv explicitly.")

    data_df = pd.read_csv(path)
    numeric_df = data_df.drop(columns=["date"], errors="ignore")
    numeric_df = numeric_df.select_dtypes(include=[np.number])
    return Path(path), data_df, numeric_df


def save_report(report_path, data_name, csv_path, data_df, numeric_df, adf_results, outlier_results, plot_paths, target):
    lines = [
        f"# Characteristics Report: {data_name}",
        "",
        f"- CSV: `{csv_path}`",
        f"- Rows: `{len(data_df)}`",
        f"- Columns: `{data_df.shape[1]}`",
        f"- Numeric columns used by `data/data_visualization.py`: `{numeric_df.shape[1]}`",
        f"- Target: `{target or 'not specified'}`",
        "",
        "## ADF Test",
        "",
    ]
    for item in adf_results:
        label = "stationary" if item["stationary"] else "non-stationary"
        lines.append(f"- `{item['column']}`: p={item['p_value']:.6g}, {label}")

    lines.extend(["", "## IQR Outliers", ""])
    for item in sorted(outlier_results, key=lambda value: value["percentage"], reverse=True):
        lines.append(f"- `{item['column']}`: {item['count']} ({item['percentage']:.5f}%)")

    if target and target in numeric_df.columns:
        lines.extend(["", "## Target Spearman Correlations", ""])
        corr = numeric_df.corr(method="spearman")[target].drop(labels=[target], errors="ignore")
        corr = corr.reindex(corr.abs().sort_values(ascending=False).index)
        for column, value in corr.head(10).items():
            lines.append(f"- `{column}`: Spearman={value:.4f}")

    if plot_paths:
        lines.extend(["", "## Plots", ""])
        for title, path in plot_paths.items():
            lines.extend(
                [
                    f"### {title.replace('_', ' ').title()}",
                    "",
                    f"`{path}`",
                    "",
                    f"![{title}]({Path(path).name})",
                    "",
                ]
            )

    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    args = parse_args()
    csv_path, data_df, numeric_df = load_dataset(args.data_name, args.csv)
    output_dir = (
        Path(args.output_dir)
        if args.output_dir
        else Path("data") / args.data_name / "characteristics report"
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    data = numeric_df.values
    columns = numeric_df.columns.tolist()
    if data.shape[1] == 0:
        raise ValueError("No numeric columns available for visualization.")

    adf_results = ADF_test(data, columns)
    outlier_results = detect_outliers(data, columns)

    plot_paths = {}
    if not args.no_plots:
        fig = plot_time_series(data, columns)
        path = output_dir / f"{args.data_name}_time_series.{args.plot_format}"
        fig.savefig(path, dpi=300)
        plot_paths["time_series"] = str(path)
        plt.close(fig)

        fig = plot_acf_lag(data, columns, lags=args.acf_lags)
        path = output_dir / f"{args.data_name}_acf.{args.plot_format}"
        fig.savefig(path, dpi=300)
        plot_paths["acf"] = str(path)
        plt.close(fig)

        fig = plot_spearmanr(data, columns, corr=numeric_df.corr(method="spearman"))
        path = output_dir / f"{args.data_name}_spearman.{args.plot_format}"
        fig.savefig(path, dpi=300)
        plot_paths["spearman"] = str(path)
        plt.close(fig)

        tsne_data = data
        if len(tsne_data) > args.tsne_sample_rows:
            sample_idx = np.linspace(0, len(tsne_data) - 1, args.tsne_sample_rows).astype(int)
            tsne_data = tsne_data[sample_idx]
        data_train = tsne_data[: int(0.7 * len(tsne_data))]
        data_test = tsne_data[int(0.8 * len(tsne_data)) :]
        if len(data_train) > 1 and len(data_test) > 1:
            fig = plot_data_2d(data_train, data_test)
            path = output_dir / f"{args.data_name}_tsne.{args.plot_format}"
            fig.savefig(path, dpi=300)
            plot_paths["tsne"] = str(path)
            plt.close(fig)

    report_path = output_dir / f"{args.data_name}_characteristics.md"
    save_report(
        report_path=report_path,
        data_name=args.data_name,
        csv_path=csv_path,
        data_df=data_df,
        numeric_df=numeric_df,
        adf_results=adf_results,
        outlier_results=outlier_results,
        plot_paths=plot_paths,
        target=args.target,
    )

    print(f"Saved report: {report_path}")
    for name, path in plot_paths.items():
        print(f"Saved {name}: {path}")

    if args.show:
        plt.show()


if __name__ == "__main__":
    main()
