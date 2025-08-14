from src.monte_carlo import euro_call_mc, bs_call_price_delta

def test_price_sanity():
    res = euro_call_mc(n=10_000, seed=1, method="antithetic")
    assert res["price"] > 0 and 0 <= res["delta_pw"] <= 1

def test_close_to_bs():
    S0=100; K=100; r=0.01; sigma=0.2; T=1.0
    bs, _ = bs_call_price_delta(S0,K,r,sigma,T)
    res = euro_call_mc(S0,K,r,sigma,T, n=50_000, seed=42, method="antithetic")
    assert abs(res["price"] - bs) < 0.5

