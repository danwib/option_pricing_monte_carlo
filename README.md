# Option Pricing via Monte Carlo

**European call option pricing** implemented in pure Python/NumPy with a focus on **benchmarking, variance reduction, and reproducibility**. Includes a Black–Scholes baseline, **antithetic sampling**, pathwise **Greeks**, a CLI, and an optional **convergence plot**.

> Educational project — **not** investment advice.

---

## Why

- Show how plain-vanilla **Monte Carlo** converges to Black–Scholes.
- Demonstrate simple **variance reduction** (antithetic sampling).
- Report **throughput, latency, and sampling error** in a reproducible way.
- Provide a clean, interview-friendly codebase with tests & CI hooks.

---

## What’s inside

- `src/monte_carlo.py` – MC pricer with CLI, Black–Scholes closed form, pathwise **Delta**/**Vega**, convergence plotting.
- `tests/` – sanity tests (pricing monotonicity, proximity to Black–Scholes, variance reduction).
- `figures/` – saved convergence plots (created on demand).
- `requirements.txt`, `.github/workflows/ci.yml` – quick setup and CI.

---

## Quick start

```bash
# create & activate a virtual env
python -m venv .venv && source .venv/bin/activate

# install deps
pip install -r requirements.txt

# run a 50k-path antithetic MC with a convergence plot
python -m src.monte_carlo --paths 50000 --method antithetic --plot figures/convergence.png
```

The script prints a JSON summary (price, standard error, Greeks, runtime):

```json
{
  "price": 8.411713566704494,
  "se": 0.06006077177398317,
  "delta_pw": 0.5604301106837697,
  "vega_pw": 39.28911561823092,
  "n": 50000,
  "method": "antithetic",
  "latency_sec": 0.007,
  "throughput_paths_per_sec": 7112973,
  "bs_price": 8.433319,
  "convergence_png": "figures/convergence.png"
}
```

> Tip: if `figures/` doesn’t exist, the script will create it (or `mkdir -p figures` first).

---

## CLI

```bash
python -m src.monte_carlo --help
```

Key flags:

- `--S0` (float, default 100.0) – spot  
- `--K` (float, default 100.0) – strike  
- `--r` (float, default 0.01) – risk-free rate  
- `--sigma` (float, default 0.2) – volatility  
- `--T` (float, default 1.0) – maturity in years  
- `--paths` (int, default 100000) – number of MC paths  
- `--seed` (int, default 0) – RNG seed for reproducibility  
- `--method {naive,antithetic}` – variance-reduction choice  
- `--plot <png path>` – save a **convergence** plot (MC vs. Black–Scholes)

Examples:

```bash
# Baseline Monte Carlo
python -m src.monte_carlo --paths 100000 --method naive

# Antithetic sampling (usually lower SE for same paths)
python -m src.monte_carlo --paths 50000 --method antithetic

# Convergence study (log-scale paths vs. price) + PNG
python -m src.monte_carlo --paths 20000 --method antithetic --plot figures/convergence.png
```

---

## Results (reproducible example)

`S0=100, K=100, r=0.01, sigma=0.2, T=1.0, paths=50,000, method=antithetic, seed=0`

| Metric                     | Value      | Notes                     |
|---------------------------|------------|---------------------------|
| **MC Price**              | 8.4117     | Antithetic sampling       |
| **Standard Error (SE)**   | 0.0601     | 95% CI ≈ ±1.96×SE = ±0.118 |
| **Black–Scholes (closed)**| 8.4333     | Reference                 |
| **Delta (pathwise)**      | 0.5604     |                           |
| **Vega (pathwise)**       | 39.2891    |                           |
| **Latency (seconds)**     | 0.007      | End-to-end                |
| **Throughput (paths/s)**  | 7,112,973  | CPU, NumPy vectorised     |

Convergence plot: see `figures/convergence.png`.

---

## How it works (high level)

- **Black–Scholes baseline**: Closed-form call price `C = S0 N(d1) − K e^{-rT} N(d2)` used as ground truth.  
- **MC simulation**: Sample terminal price `S_T = S0 exp((r − 0.5σ²)T + σ√T·Z)`, compute discounted payoff `e^{-rT} max(S_T − K, 0)`, average over paths.  
- **Variance reduction**: **Antithetic sampling** reuses `Z` and `−Z` to reduce variance with minimal extra cost.  
- **Greeks (pathwise)**:  
  - **Delta**: `E[e^{-rT} 1_{S_T>K}·(S_T/S0)]`  
  - **Vega**: uses `∂S_T/∂σ` inside the expectation (pathwise derivative).

---

## Testing

```bash
pytest
```

Tests cover:

- **Proximity to Black–Scholes** within a loose tolerance at modest path counts.
- **Monotonicity** checks (price increases with `S0` and with `σ`).
- **Variance reduction**: antithetic SE ≤ naive SE (allowing small fluctuations).

---

## Performance notes

- Vectorised NumPy operations keep Python overhead low; throughput scales with path count.  
- SE ≈ `O(1/√N)`: doubling paths reduces SE by ~√2.  
- For fair comparisons, keep **seed** fixed and report **paths, method, and latency**.

---

## Roadmap (nice to have)

- **Control variates** (e.g., use `S_T` as a CV — already set up in code if you enable it).  
- **Barrier options** (path dependent).  
- **Quasi-MC** (Sobol sequences).  
- **Batch benchmarks** (table of SE vs. paths vs. method).

---

## Reproducibility

- All commands above are deterministic given `--seed`.  
- Environment: Python 3.12, NumPy ≥ 2.x, Matplotlib for plotting.  
- Printout is JSON for easy logging/CI comparisons.

---

## License

MIT — see `LICENSE`.

---

## Acknowledgements

- Classic Black–Scholes and Monte Carlo methods; pathwise Greeks as commonly taught in computational finance courses.

