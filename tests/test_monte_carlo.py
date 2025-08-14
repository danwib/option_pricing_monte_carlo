from src.monte_carlo import euro_call_mc, bs_call_price_delta, convergence_plot
import os

def test_price_sanity():
    res = euro_call_mc(n=10_000, seed=1, method="antithetic")
    assert res["price"] > 0 and 0 <= res["delta_pw"] <= 1

def test_close_to_bs():
    S0=100; K=100; r=0.01; sigma=0.2; T=1.0
    bs, _ = bs_call_price_delta(S0,K,r,sigma,T)
    res = euro_call_mc(S0,K,r,sigma,T, n=50_000, seed=42, method="antithetic")
    assert abs(res["price"] - bs) < 0.5

def test_bs_close_enough():
    S0,K,r,sigma,T = 100.0,100.0,0.01,0.2,1.0
    bs, _ = bs_call_price_delta(S0,K,r,sigma,T)
    # Keep paths modest for CI; check loose tolerance
    res = euro_call_mc(S0,K,r,sigma,T, n=50_000, seed=42, method="antithetic")
    assert abs(res["price"] - bs) < 0.5

def test_monotone_S0_sigma():
    base = euro_call_mc(100,100,0.01,0.2,1.0, n=30_000, seed=1)
    higher_S0 = euro_call_mc(110,100,0.01,0.2,1.0, n=30_000, seed=1)
    higher_sigma = euro_call_mc(100,100,0.01,0.3,1.0, n=30_000, seed=1)
    assert higher_S0["price"] > base["price"]
    assert higher_sigma["price"] > base["price"]

def test_antithetic_reduces_se():
    # Compare SE (should be smaller or equal with antithetic)
    naive = euro_call_mc(100,100,0.01,0.2,1.0, n=40_000, seed=7, method="naive")
    anti  = euro_call_mc(100,100,0.01,0.2,1.0, n=40_000, seed=7, method="antithetic")
    assert anti["se"] <= naive["se"] * 1.05  # allow small fluctuation

def test_convergence_plot_tmp(tmp_path):
    out = tmp_path / "subdir" / "conv.png"
    res = convergence_plot(100,100,0.01,0.2,1.0, seed=0, method="antithetic",
                           path_grid=[1000, 2000], out_png=str(out))
    assert os.path.exists(res["out_png"])
