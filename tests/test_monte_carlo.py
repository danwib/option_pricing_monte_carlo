from src.monte_carlo import euro_call_mc

def test_price_sanity():
    p, d = euro_call_mc(n=10000, seed=1)
    assert p > 0 and 0 <= d <= 1
