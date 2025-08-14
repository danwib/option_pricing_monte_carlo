import argparse
import json
from math import exp, log, sqrt, erf
import numpy as np
from time import perf_counter
import matplotlib.pyplot as plt


def norm_cdf(x: float) -> float:
    # Standard normal CDF via erf (no SciPy needed)
    return 0.5 * (1.0 + erf(x / sqrt(2.0)))


def bs_call_price_delta(S0: float, K: float, r: float, sigma: float, T: float):
    if sigma <= 0 or T <= 0 or S0 <= 0 or K <= 0:
        raise ValueError("S0,K,sigma,T must be positive")
    srt = sigma * sqrt(T)
    d1 = (log(S0 / K) + (r + 0.5 * sigma * sigma) * T) / srt
    d2 = d1 - srt
    price = S0 * norm_cdf(d1) - K * exp(-r * T) * norm_cdf(d2)
    delta = norm_cdf(d1)
    return price, delta


def _simulate_terminal(S0, r, sigma, T, n, seed, antithetic=False):
    rng = np.random.default_rng(seed)
    if antithetic:
        m = n // 2
        Z = rng.standard_normal(m)
        Z = np.concatenate([Z, -Z])
        if len(Z) < n:
            Z = np.append(Z, rng.standard_normal(1))
    else:
        Z = rng.standard_normal(n)
    drift = (r - 0.5 * sigma * sigma) * T
    vol = sigma * sqrt(T)
    ST = S0 * np.exp(drift + vol * Z)
    return ST, Z


def euro_call_mc(
    S0=100.0,
    K=100.0,
    r=0.01,
    sigma=0.2,
    T=1.0,
    n=100_000,
    seed=0,
    method="naive",  # "naive" | "antithetic" | "cv"
):
    antithetic = method == "antithetic" or method == "cv"
    ST, Z = _simulate_terminal(S0, r, sigma, T, n, seed, antithetic=antithetic)
    payoff = np.maximum(ST - K, 0.0)
    df = exp(-r * T)

    # Control variate (use ST with known E[ST] = S0*e^{rT})
    if method == "cv":
        Y = ST
        Y_mean_known = S0 * exp(r * T)
        cov = np.cov(payoff, Y, ddof=1)[0, 1]
        varY = np.var(Y, ddof=1)
        c_star = 0.0 if varY == 0 else cov / varY
        payoff_adj = payoff - c_star * (Y - Y_mean_known)
        price_est = df * payoff_adj.mean()
        se = df * payoff_adj.std(ddof=1) / sqrt(len(payoff_adj))
    else:
        price_est = df * payoff.mean()
        se = df * payoff.std(ddof=1) / sqrt(len(payoff))

    # Pathwise Greeks (valid for smooth part; indicator for call is fine)
    # Delta: E[e^{-rT} 1_{ST>K} * (ST/S0)]
    delta_pw = df * np.mean((ST / S0) * (ST > K))
    # Vega: ∂/∂σ E[e^{-rT} max(ST-K,0)]  with ∂ST/∂σ = ST * (-σT + sqrt(T) Z)
    vega_pw = df * np.mean((ST > K) * ST * (-sigma * T + sqrt(T) * Z))

    return {
        "price": float(price_est),
        "se": float(se),
        "delta_pw": float(delta_pw),
        "vega_pw": float(vega_pw),
        "n": int(n),
        "method": method,
    }


def convergence_plot(S0, K, r, sigma, T, seed, method, path_grid, out_png):
    estimates = []
    for n in path_grid:
        res = euro_call_mc(S0, K, r, sigma, T, n=n, seed=seed, method=method)
        estimates.append((n, res["price"], res["se"]))
    bs_price, _ = bs_call_price_delta(S0, K, r, sigma, T)

    ns = [x[0] for x in estimates]
    prices = [x[1] for x in estimates]
    ses = [x[2] for x in estimates]

    plt.figure()
    plt.plot(ns, prices, marker="o", label=f"MC ({method})")
    plt.axhline(bs_price, color="black", linestyle="--", label="Black–Scholes")
    # ±1.96*SE bands around MC estimates
    upper = [p + 1.96 * s for p, s in zip(prices, ses)]
    lower = [p - 1.96 * s for p, s in zip(prices, ses)]
    plt.fill_between(ns, lower, upper, color="gray", alpha=0.2, label="95% CI (MC)")
    plt.xscale("log")
    plt.xlabel("Paths (log scale)")
    plt.ylabel("Price")
    plt.title("European Call — Monte Carlo Convergence")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_png, dpi=150)
    plt.close()
    return {"bs_price": bs_price, "points": estimates, "out_png": out_png}


def main():
    p = argparse.ArgumentParser(description="European call pricing via Monte Carlo")
    p.add_argument("--S0", type=float, default=100.0)
    p.add_argument("--K", type=float, default=100.0)
    p.add_argument("--r", type=float, default=0.01)
    p.add_argument("--sigma", type=float, default=0.2)
    p.add_argument("--T", type=float, default=1.0)
    p.add_argument("--paths", type=int, default=100000)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--method", choices=["naive", "antithetic", "cv"], default="naive")
    p.add_argument("--plot", type=str, default="")
    args = p.parse_args()

    t0 = perf_counter()
    res = euro_call_mc(
        S0=args.S0,
        K=args.K,
        r=args.r,
        sigma=args.sigma,
        T=args.T,
        n=args.paths,
        seed=args.seed,
        method=args.method,
    )
    dt = perf_counter() - t0
    res["latency_sec"] = round(dt, 4)
    res["throughput_paths_per_sec"] = int(args.paths / dt) if dt > 0 else None

    if args.plot:
        grid = [1_000, 5_000, 10_000, 20_000, 50_000, max(100_000, args.paths)]
        conv = convergence_plot(args.S0, args.K, args.r, args.sigma, args.T, args.seed, args.method, grid, args.plot)
        res["bs_price"] = round(conv["bs_price"], 6)
        res["convergence_png"] = conv["out_png"]

    print(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()

