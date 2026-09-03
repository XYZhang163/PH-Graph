import argparse
import time
from pathlib import Path

import numpy as np
import scipy.io as scio
from sklearn import cluster
from sklearn.preprocessing import MinMaxScaler

from PH_Graph import PHGraph
from utils import compute_score, norm


def load_dataset(path):
    if path.suffix.lower() == ".mat":
        data = scio.loadmat(path)
        x = data["data"].astype(float)
        y = data["labels"].reshape(-1).astype(int)
        x = MinMaxScaler().fit_transform(x)
        return x, y

    if path.suffix.lower() == ".txt":
        data = np.loadtxt(path)
        x = data[:, :-1].astype(float)
        y = data[:, -1].astype(int)
        x = norm(x)
        return x, y

    raise ValueError(f"Unsupported dataset format: {path.name}")


def run_once(x, y, tau, max_value, max_dimension, random_state):
    start = time.perf_counter()

    model = PHGraph(rate=tau, max_value=max_value, max_dimension=max_dimension)
    affinity = model.fit_transform(x)

    spectral = cluster.SpectralClustering(
        n_clusters=len(np.unique(y)),
        eigen_solver="arpack",
        affinity="precomputed",
        random_state=random_state,
    ).fit(affinity)

    ari, nmi, acc = compute_score(y, spectral.labels_.astype(int))
    runtime = time.perf_counter() - start
    return ari, nmi, acc, runtime


def print_result(prefix, ari, nmi, acc, runtime):
    print(
        f"{prefix} | "
        f"ARI={ari:.2f} | "
        f"NMI={nmi:.2f} | "
        f"ACC={acc:.2f} | "
        f"Runtime={runtime:.4f}s",
        flush=True,
    )


def run_dataset(path, args):
    x, y = load_dataset(path)
    print(
        f"\nDataset: {path.stem} | Samples={x.shape[0]} | "
        f"Features={x.shape[1]} | Classes={len(np.unique(y))}",
        flush=True,
    )

    scores = []
    for trial in range(1, args.trials + 1):
        ari, nmi, acc, runtime = run_once(
            x,
            y,
            args.tau,
            args.max_value,
            args.max_dimension,
            args.random_state + trial - 1,
        )
        scores.append((ari, nmi, acc, runtime))
        label = "Single run" if trial == 1 else f"Run {trial}"
        print_result(label, ari, nmi, acc, runtime)

    mean_scores = np.mean(np.array(scores), axis=0)
    print_result(f"Average of {args.trials} runs", *mean_scores)


def list_datasets(data_dir):
    return sorted(
        [
            path
            for path in data_dir.iterdir()
            if path.suffix.lower() in {".mat", ".txt"} and path.is_file()
        ]
    )


def main():
    parser = argparse.ArgumentParser(description="Run PH-Graph demo on datasets in data/.")
    parser.add_argument("--data-dir", default="data", help="Directory containing .mat or .txt datasets.")
    parser.add_argument("--datasets", nargs="*", help="Optional dataset names, e.g. DS1 DC1.")
    parser.add_argument("--trials", type=int, default=10, help="Number of repeated runs per dataset.")
    parser.add_argument("--tau", type=float, default=0.90)
    parser.add_argument("--max-value", type=float, default=1.1)
    parser.add_argument("--max-dimension", type=int, default=1)
    parser.add_argument("--random-state", type=int, default=0)
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    datasets = list_datasets(data_dir)

    if args.datasets:
        wanted = {name.lower() for name in args.datasets}
        datasets = [path for path in datasets if path.stem.lower() in wanted]

    if not datasets:
        raise FileNotFoundError("No .mat or .txt datasets found in the data directory.")

    for dataset_path in datasets:
        run_dataset(dataset_path, args)


if __name__ == "__main__":
    main()
