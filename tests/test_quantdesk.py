import numpy as np
import pytest
from quantdesk import *
from quantdesk.pricing import d1_d2
from quantdesk.stochastic import expected_avg_iv
from quantdesk.vol import svi_w
from quantdesk.strategies import payoff_at_expiry, breakevens, value
from quantdesk.backtest import synthetic_market, run_backtest, summarize, run_universe

S, K, T, r, q, sig = 100.0, 105.0, 0.5, 0.04, 0.01, 0.3


def test_put_call_parity():
    c = bs_price(S, K, T, r, sig, q, "call")
    p = bs_price(S, K, T, r, sig, q, "put")
    assert c - p == pytest.approx(S * np.exp(-q * T) - K * np.exp(-r * T), abs=1e-10)


def test_known_value():
    # classic textbook case: S=K=100, T=1, r=5%, sigma=20% -> call 10.4506
    assert bs_price(100, 100, 1, 0.05, 0.2) == pytest.approx(10.4506, abs=1e-3)


def test_iv_roundtrip_and_arbitrage_bounds():
    for kind in ("call", "put"):
        px = bs_price(S, K, T, r, 0.37, q, kind)
        assert implied_vol(px, S, K, T, r, q, kind) == pytest.approx(0.37, abs=1e-8)
    assert np.isnan(implied_vol(0.0001 + max(S - K, 0) - 50, S, K, T, r))  # below intrinsic


@pytest.mark.parametrize("kind", ["call", "put"])
def test_greeks_vs_finite_difference(kind):
    g = greeks(S, K, T, r, sig, q, kind)
    h = 1e-4 * S
    f = lambda s=S, v=sig, t=T, rr=r: bs_price(s, K, t, rr, v, q, kind)
    assert g["delta"] == pytest.approx((f(S + h) - f(S - h)) / (2 * h), abs=1e-6)
    assert g["gamma"] == pytest.approx((f(S + h) - 2 * f() + f(S - h)) / h**2, abs=1e-5)
    assert g["vega"] == pytest.approx((f(v=sig + 1e-5) - f(v=sig - 1e-5)) / 2e-5 / 100, abs=1e-6)
    assert g["theta"] == pytest.approx(-(f(t=T + 1e-5) - f(t=T - 1e-5)) / 2e-5 / 365, abs=1e-6)
    assert g["rho"] == pytest.approx((f(rr=r + 1e-5) - f(rr=r - 1e-5)) / 2e-5 / 100, abs=1e-6)
    # vanna = d(delta)/d(vol)
    d_up = greeks(S, K, T, r, sig + 1e-5, q, kind)["delta"]
    d_dn = greeks(S, K, T, r, sig - 1e-5, q, kind)["delta"]
    assert g["vanna"] == pytest.approx((d_up - d_dn) / 2e-5 / 100, abs=1e-6)
    # volga = d(vega)/d(vol)
    v_up = greeks(S, K, T, r, sig + 1e-5, q, kind)["vega"]
    v_dn = greeks(S, K, T, r, sig - 1e-5, q, kind)["vega"]
    assert g["volga"] == pytest.approx((v_up - v_dn) / 2e-5 / 100, abs=1e-6)


def test_american_vs_european():
    eu_p = bs_price(S, K, T, r, sig, 0.0, "put")
    am_p = american_crr(S, K, T, r, sig, 0.0, "put", steps=600)
    assert am_p >= eu_p - 1e-9 and am_p - eu_p > 0.01            # early-exercise premium exists
    eu_c = bs_price(S, K, T, r, sig, 0.0, "call")
    assert american_crr(S, K, T, r, sig, 0.0, "call", steps=600) == pytest.approx(eu_c, abs=5e-3)
    assert american_crr(S, K, T, r, sig, 0.0, "put", steps=600) == pytest.approx(
        american_crr(S, K, T, r, sig, 0.0, "put", steps=1200), abs=5e-3)   # convergence


def test_svi_recovers_curve_and_is_arb_free():
    true = (0.02, 0.15, -0.5, 0.0, 0.2)
    k = np.linspace(-0.5, 0.5, 41)
    p, rmse = svi_fit(k, svi_w(k, *true))
    assert rmse < 1e-5
    assert svi_butterfly_ok(true)[0]
    assert not svi_butterfly_ok((0.0, 2.0, 0.0, 0.0, 0.001))[0]       # spiky smile violates g(k)>=0


def test_garch_recovers_parameters():
    rng = np.random.default_rng(0)
    n, om, a, b = 20000, 2e-6, 0.08, 0.90
    h, r_ = om / (1 - a - b), np.empty(n)
    for t in range(n):
        r_[t] = np.sqrt(h) * rng.standard_normal()
        h = om + a * r_[t] ** 2 + b * h
    g = garch_fit(r_)
    assert g.alpha == pytest.approx(a, abs=0.02)
    assert g.beta == pytest.approx(b, abs=0.03)
    assert g.persistence == pytest.approx(a + b, abs=0.015)


def test_gbm_is_martingale_under_riskneutral_drift():
    ST = gbm_terminal(100, 0.04, 0.3, 1.0, n=400_000, seed=3)
    assert ST.mean() == pytest.approx(100 * np.exp(0.04), rel=3e-3)


def test_heston_degenerates_to_bs_when_volvol_zero():
    p = Heston(0.09, 2.0, 0.09, 1e-8, -0.5)
    ST, vT = heston_terminal(100, 0.04, 0.5, p, n=200_000, seed=4)
    mc = np.exp(-0.04 * 0.5) * np.maximum(ST - 100, 0).mean()
    assert mc == pytest.approx(bs_price(100, 100, 0.5, 0.04, 0.3), rel=0.015)
    assert expected_avg_iv(vT, 0.25, p).mean() == pytest.approx(0.3, abs=1e-3)


def test_heston_negative_rho_creates_negative_skew_in_returns():
    p = Heston(0.09, 2.0, 0.09, 0.8, -0.8)
    ST, _ = heston_terminal(100, 0.0, 1.0, p, n=100_000, seed=5)
    lr = np.log(ST / 100)
    skew = ((lr - lr.mean()) ** 3).mean() / lr.std() ** 3
    assert skew < -0.1


def test_spread_has_less_vega_and_less_cost_than_naked_call():
    naked = [Leg("call", 100, 1)]
    spread = bull_call_spread(100, 110)
    gn = net_greeks(naked, 100, 0.2, 0.04, 0.3)
    gs = net_greeks(spread, 100, 0.2, 0.04, 0.3)
    assert abs(gs["vega"]) < abs(gn["vega"])
    assert value(spread, 100, 0.2, 0.04, 0.3) < value(naked, 100, 0.2, 0.04, 0.3)


def test_payoff_and_breakeven():
    legs = bull_call_spread(100, 110)
    assert payoff_at_expiry(legs, [90, 105, 120]).tolist() == [0.0, 500.0, 1000.0]
    be = breakevens(legs, 100, 0.25, 0.04, 0.3)
    assert len(be) == 1 and 100 < be[0] < 110


def test_costs_reduce_pnl():
    legs = [Leg("call", 100, 1)]
    ST = np.array([110.0])
    free, *_ = position_pnl(legs, 100, 0.2, 0.1, 0.04, ST, 0.3, 0.3, 0, CostModel(0, 0, 0))
    paid, *_ = position_pnl(legs, 100, 0.2, 0.1, 0.04, ST, 0.3, 0.3, 0, CostModel())
    assert paid[0] < free[0]


def test_kelly_zero_for_negative_edge_and_positive_for_positive_edge():
    rng = np.random.default_rng(1)
    bad = rng.choice([-100.0, 90.0], 20000, p=[0.6, 0.4])      # EV = -24
    good = rng.choice([-100.0, 150.0], 20000, p=[0.5, 0.5])    # EV = +25
    assert kelly_fraction(bad, 100.0)[0] == 0.0
    f, _ = kelly_fraction(good, 100.0)
    # analytic Kelly for win +1.5x / lose -1x at p = 0.5: f* = (p*b - q)/b = 1/6
    assert f == pytest.approx(1 / 6, abs=0.02)


def test_gates_block_violations():
    pnl = np.random.default_rng(0).normal(30, 100, 5000)
    v = run_gates(pnl, 300.0, 300.0, 1000.0, 0.3, 0.3, dte=30, hold_days=20,
                  days_to_earnings=10, open_positions=2)
    failed = {g.name for g in v.gates if g.hard and not g.passed}
    assert {"5 DTE minimum", "6 Earnings: exit day before", "7 Size cap", "8 Open positions"} <= failed
    assert not v.allowed and v.recommended_premium == 0.0


def test_full_analysis_runs_both_models():
    prices = 100 * np.exp(np.cumsum(np.random.default_rng(2).normal(0.0004, 0.017, 500)))
    for model in ("heston", "gbm"):
        a = full_analysis(prices, K=100, dte=60, hold_days=30, iv=0.3, model=model,
                          n_paths=8000, days_to_earnings=75)
        assert "VERDICT" in format_report(a)


def test_backtest_detects_pump_and_rejects_no_pump():
    with_pump = summarize(run_universe(8, pump=0.35, n_calib=0))
    no_pump = summarize(run_universe(8, pump=0.0, n_calib=0))
    assert with_pump["mean_R"] > no_pump["mean_R"] + 0.05
    assert no_pump["mean_R"] < 0.0          # costs + theta: no edge means a loss
    assert with_pump["mean_R_ci95"][0] > no_pump["mean_R_ci95"][0]
