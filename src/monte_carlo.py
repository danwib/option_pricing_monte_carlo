import numpy as np
from time import perf_counter

def euro_call_mc(S0=100.0, K=100.0, r=0.01, vol=0.2, T=1.0, n=100_000, seed=0):
    rng = np.random.default_rng(seed)
    Z = rng.standard_normal(n)
    ST = S0 * np.exp((r - 0.5*vol**2)*T + vol*np.sqrt(T)*Z)
    payoff = np.maximum(ST - K, 0.0)
    price = np.exp(-r*T) * payoff.mean()
    # simple pathwise delta for GBM
    delta = np.exp(-r*T) * (ST > K).mean()
    return price, delta

def main():
    t0 = perf_counter()
    price, delta = euro_call_mc()
    dt = perf_counter() - t0
    print({"price": round(price, 4), "delta": round(delta, 4), "latency_sec": round(dt, 4)})

if __name__ == "__main__":
    main()
