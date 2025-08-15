# monte_carlo.py
import matplotlib

matplotlib.use("Agg")
import argparse
import json
import os
from math import erf, exp, log, sqrt
from time import perf_counter
from typing import Dict, Iterable, List, Tuple

import matplotlib.pyplot as plt
import numpy as np


# ----------------------------
# Closed-form (Black–Scholes)
# ----------------------------
def norm_cdf(x: float) -> float:
    """Standard normal CDF via erf (keeps dependencies light)."""
    return 0.5 * (1.0 + erf(x / sqrt(2.0)))


def bs_call_price_delta(
    S0: float, K: float, r: float, sigma: float, T: float
) -> Tuple[float, float]:
    """Black–Scholes price and Delta for a European call."""
    if any(v <= 0 for v in (S0, K, sigma, T)):
        raise ValueError("S0, K, sigma, T must be positive")
    srt = sigma * sqrt(T)
    d1 = (log(S0 / K) + (r + 0.5 * sigma * sigma) * T) / srt
    d2 = d1 - srt
    price = S0 * norm_cdf(d1) - K * exp(-r * T) * norm_cdf(d2)
    delta = norm_cdf(d1)
    return price, delta


# ----------------------------
# Monte Carlo engines
# ----------------------------
def _simulate_terminal(
    S0: float,
    r: float,
    sigma: float,
    T: float,
    n: int,
    seed: int,
    antithetic: bool = False,
) -> Tuple[np.ndarray, np.ndarray]:
    """Simulate terminal price ST under GBM; optionally antithetic pairs."""
    rng = np.random.default_rng(seed)
    if antithetic:
        m = n // 2
        Z = rng.standard_normal(m)
        Z = np.concatenate([Z, -Z])
        if len(Z) < n:  # pad if n is odd
            Z = np.append(Z, rng.standard_normal(1))
    else:
        Z = rng.standard_normal(n)

    drift = (r - 0.5 * sigma * sigma) * T
    vol = sigma * sqrt(T)
    ST = S0 * np.exp(drift + vol * Z)
    return ST, Z


def _ci95(se: float) -> float:
    """Half-width of a 95% normal-approx confidence interval."""
    return 1.96 * se


def euro_call_mc(
    S0: float = 100.0,
    K: float = 100.0,
    r: float = 0.01,
    sigma: float = 0.2,
    T: float = 1.0,
    n: int = 100_000,
    seed: int = 0,
    method: str = "naive",  # "naive" | "antithetic" | "cv"
) -> Dict[str, float]:
    """
    Monte Carlo estimator for a European call with variance-reduction options.

    Pathwise Delta (valid here): Δ = e^{-rT} E[ 1_{ST>K} * ST/S0 ]
    Pathwise Vega (for GBM):    ∂ST/∂σ = ST * (-σT + sqrt(T) Z)
    """
    if any(v <= 0 for v in (S0, K, sigma, T)):
        raise ValueError("S0, K, sigma, T must be positive")
    if n <= 0:
        raise ValueError("n must be positive")

    antithetic = (method == "antithetic") or (method == "cv")
    ST, Z = _simulate_terminal(S0, r, sigma, T, n, seed, antithetic=antithetic)
    disc = exp(-r * T)

    # Payoff and standard (naive) estimator
    payoff = np.maximum(ST - K, 0.0)

    if method == "cv":
        # Control variate: X = ST with known E[X] = S0 * e^{rT}
        X = ST
        EX = S0 * exp(r * T)
        # Work with *undiscounted* payoff for covariance, discount afterward
        cov = np.cov(payoff, X, ddof=1)[0, 1]
        varX = np.var(X, ddof=1)
        b_opt = 0.0 if varX == 0 else cov / varX
        payoff_adj = payoff - b_opt * (X - EX)
        price = disc * payoff_adj.mean()
        se = disc * payoff_adj.std(ddof=1) / sqrt(len(payoff_adj))
    elif method == "antithetic":
        price = disc * payoff.mean()  # antithetics handled in simulation pairing
        se = disc * payoff.std(ddof=1) / sqrt(len(payoff))
    else:
        price = disc * payoff.mean()
        se = disc * payoff.std(ddof=1) / sqrt(len(payoff))

    # Pathwise Greeks
    delta_pw = disc * np.mean((ST / S0) * (ST > K))
    vega_pw = disc * np.mean((ST > K) * ST * (-sigma * T + sqrt(T) * Z))

    return {
        "price": float(price),
        "se": float(se),
        "ci95": float(_ci95(se)),
        "delta_pw": float(delta_pw),
        "vega_pw": float(vega_pw),
        "paths": int(n),
        "seed": int(seed),
        "method": method,
    }


def delta_fd(
    S0: float,
    K: float,
    r: float,
    sigma: float,
    T: float,
    n: int = 100_000,
    seed: int = 0,
    eps: float = 1e-2,
) -> float:
    """Finite-difference Delta (central) using the MC price as a function of S0."""
    up = euro_call_mc(S0 + eps, K, r, sigma, T, n=n, seed=seed, method="antithetic")["price"]
    dn = euro_call_mc(S0 - eps, K, r, sigma, T, n=n, seed=seed, method="antithetic")["price"]
    return float((up - dn) / (2.0 * eps))


# ----------------------------
# Convergence plotting
# ----------------------------
def convergence_plot_multi(
    S0: float,
    K: float,
    r: float,
    sigma: float,
    T: float,
    seed: int,
    methods: Iterable[str],
    path_grid: Iterable[int],
    out_png: str,
) -> Dict[str, object]:
    """Plot MC convergence for multiple methods on one figure."""
    bs_price, _ = bs_call_price_delta(S0, K, r, sigma, T)

    plt.figure()
    legends = []
    all_points = {}

    for method in methods:
        ns: List[int] = []
        est: List[float] = []
        ci: List[float] = []
        for n in path_grid:
            res = euro_call_mc(S0, K, r, sigma, T, n=n, seed=seed, method=method)
            ns.append(n)
            est.append(res["price"])
            ci.append(res["ci95"])
        plt.plot(ns, est, marker="o")
        # Draw CI bands as thin error bars (optional)
        upper = [m + c for m, c in zip(est, ci)]
        lower = [m - c for m, c in zip(est, ci)]
        plt.fill_between(ns, lower, upper, alpha=0.08)
        legends.append(f"MC ({method})")
        all_points[method] = list(zip(ns, est, ci))

    plt.axhline(bs_price, color="black", linestyle="--", label="Black-Scholes")
    plt.xscale("log")
    plt.xlabel("Paths (log scale)")
    plt.ylabel("Price")
    plt.title("European Call — MC Convergence (multi-method)")
    plt.legend(legends + ["Black-Scholes"])
    plt.tight_layout()

    out_dir = os.path.dirname(out_png)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    plt.savefig(out_png, dpi=150)
    plt.close()

    return {"bs_price": bs_price, "points": all_points, "out_png": out_png}


def convergence_plot(S0, K, r, sigma, T, seed, method, path_grid, out_png):
    return convergence_plot_multi(S0, K, r, sigma, T, seed, [method], path_grid, out_png)


# ----------------------------
# CLI
# ----------------------------
def main() -> None:
    p = argparse.ArgumentParser(description="European call pricing via Monte Carlo")
    p.add_argument("--S0", type=float, default=100.0)
    p.add_argument("--K", type=float, default=100.0)
    p.add_argument("--r", type=float, default=0.01)
    p.add_argument("--sigma", type=float, default=0.2)
    p.add_argument("--T", type=float, default=1.0)
    p.add_argument("--paths", type=int, default=100_000)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--method", choices=["naive", "antithetic", "cv"], default="naive")
    p.add_argument("--repeat", type=int, default=1, help="repeat timing N times and average")
    p.add_argument("--delta-fd", action="store_true", help="also compute finite-difference Delta")
    p.add_argument("--plot", type=str, default="", help="save a convergence plot PNG")
    p.add_argument(
        "--plot-methods",
        type=str,
        default="naive,antithetic,cv",
        help="comma-separated list for convergence plot (e.g., naive,cv)",
    )
    args = p.parse_args()

    # Timed runs
    latencies = []
    last_res = None
    for _ in range(max(1, args.repeat)):
        t0 = perf_counter()
        last_res = euro_call_mc(
            S0=args.S0,
            K=args.K,
            r=args.r,
            sigma=args.sigma,
            T=args.T,
            n=args.paths,
            seed=args.seed,
            method=args.method,
        )
        latencies.append(perf_counter() - t0)

    assert last_res is not None
    latency_mean = float(np.mean(latencies))
    latency_std = float(np.std(latencies, ddof=1)) if len(latencies) > 1 else 0.0
    last_res["latency_sec"] = round(latency_mean, 6)
    last_res["latency_std_sec"] = round(latency_std, 6)
    last_res["throughput_paths_per_sec"] = (
        int(args.paths / latency_mean) if latency_mean > 0 else None
    )

    # Reference BS and absolute error
    bs_price, _ = bs_call_price_delta(args.S0, args.K, args.r, args.sigma, args.T)
    last_res["bs_price"] = round(bs_price, 6)
    last_res["abs_err"] = round(abs(last_res["price"] - bs_price), 6)

    # Optional finite-difference Delta
    if args.delta_fd:
        last_res["delta_fd"] = float(
            delta_fd(args.S0, args.K, args.r, args.sigma, args.T, n=args.paths, seed=args.seed)
        )

    # Optional multi-method convergence plot
    if args.plot:
        grid = [1_000, 5_000, 10_000, 20_000, 50_000, max(100_000, args.paths)]
        methods = [m.strip() for m in args.plot_methods.split(",") if m.strip()]
        conv = convergence_plot_multi(
            args.S0, args.K, args.r, args.sigma, args.T, args.seed, methods, grid, args.plot
        )
        last_res["convergence_png"] = conv["out_png"]

    # Stable single JSON object for easy parsing/CI
    print(json.dumps(last_res, indent=2))


if __name__ == "__main__":
    main()
