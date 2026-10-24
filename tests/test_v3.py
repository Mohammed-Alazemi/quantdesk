"""Tests for the v3 modules. Every test checks the code against an exact or published result."""
import os
import sys

# Let this file run on its own (e.g. "Run file" in Spyder) as well as under pytest.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pytest
from scipy.stats import lognorm

from quantdesk import bs_price, kelly_fraction, implied_vol
from quantdesk.stochastic import Heston, heston_terminal
from quantdesk.heston import heston_price, heston_smile, calibrate_heston
from quantdesk.hedging import (delta_hedge, locked_in_pnl, hedging_error_std,
                               gamma_theta_breakeven_move)
from quantdesk.density import (bl_density, svi_density, density_moments, variance_swap_strike,
                               event_vol, implied_event_move, iv_after_event)
from quantdesk.vol import svi_w
from quantdesk.stats import (probabilistic_sharpe, deflated_sharpe, min_track_record,
                             walk_forward_splits)
from quantdesk.marketmaking import ASParams, optimal_quotes, simulate

S, r, q = 100.0, 0.03, 0.01
P = Heston(0.05, 1.2, 0.06, 0.9, -0.75)


# ------------------------------------------------------------------ pricing fixes
def test_kelly_refined_to_analytic_value():
    rng = np.random.default_rng(1)
    good = rng.choice([-100.0, 150.0], 200_000, p=[0.5, 0.5])
    # win +1.5x / lose -1x at p = 0.5: f* = (p b - q) / b = 1/6 (sample noise ~ 0.003)
    assert kelly_fraction(good, 100.0, max_n=200_000)[0] == pytest.approx(1 / 6, abs=0.006)


def test_implied_vol_vectorised_over_a_chain():
    K = np.array([80.0, 100.0, 120.0])
    vols = np.array([0.35, 0.25, 0.22])
    px = bs_price(S, K, 0.5, r, vols, q)
    assert np.allclose(implied_vol(px, S, K, 0.5, r, q), vols, atol=1e-8)


def test_bad_option_kind_is_rejected():
    with pytest.raises(ValueError):
        bs_price(S, 100, 1, r, 0.2, kind="cal")


# ------------------------------------------------------------------ Heston
@pytest.mark.parametrize("T", [0.05, 1.0, 5.0])
def test_heston_reduces_to_black_scholes(T):
    """xi -> 0 with v0 = theta: constant variance, so Heston must equal Black-Scholes."""
    p = Heston(0.04, 1.5, 0.04, 1e-4, 0.0)
    K = np.array([60.0, 80.0, 100.0, 120.0, 150.0])
    assert np.allclose(heston_price(S, K, T, r, p, q), bs_price(S, K, T, r, 0.2, q), atol=1e-6)


def test_heston_matches_adaptive_quadrature():
    from scipy.integrate import quad
    from quantdesk.heston import char_func
    T, K = 0.25, np.array([80.0, 100.0, 115.0])
    F = S * np.exp((r - q) * T)

    def ref(k):
        x = np.log(F / k)
        f = lambda u: np.real(np.exp(1j * u * x) * char_func(u - 0.5j, T, P)) / (u * u + 0.25)
        I = quad(f, 0, np.inf, limit=2000, epsabs=1e-12, epsrel=1e-12)[0]
        return S * np.exp(-q * T) - np.sqrt(F * k) * np.exp(-r * T) / np.pi * I

    assert np.allclose(heston_price(S, K, T, r, P, q), [ref(k) for k in K], atol=1e-8)


def test_heston_matches_monte_carlo():
    ST, _ = heston_terminal(S, r - q, 1.0, P, n=300_000, seed=1, steps_per_year=400)
    K = np.array([80.0, 100.0, 120.0])
    pay = np.maximum(ST[:, None] - K, 0) * np.exp(-r)
    se = pay.std(0) / np.sqrt(len(ST))
    assert np.all(np.abs(pay.mean(0) - heston_price(S, K, 1.0, r, P, q)) < 4 * se + 0.01)


def test_heston_put_call_parity_and_negative_skew():
    K = np.array([80.0, 100.0, 120.0])
    c = heston_price(S, K, 0.5, r, P, q, "call")
    p = heston_price(S, K, 0.5, r, P, q, "put")
    assert np.allclose(c - p, S * np.exp(-q * 0.5) - K * np.exp(-r * 0.5), atol=1e-10)
    iv = heston_smile(S, K, 0.5, r, P, q)
    assert iv[0] > iv[1] > iv[2]                 # rho < 0: downside strikes trade at higher IV


def _surface(true):
    quotes = []
    for T in (0.1, 0.25, 0.5, 1.0, 2.0):
        K = S * np.exp(np.linspace(-0.4, 0.3, 9) * np.sqrt(T))
        quotes += [(T, k, v) for k, v in zip(K, heston_smile(S, K, T, r, true, q))]
    return quotes


def test_calibration_recovers_parameters_exactly():
    true = Heston(0.045, 2.5, 0.06, 0.7, -0.7)
    fit = calibrate_heston(S, r, _surface(true), q)
    got = np.array([fit.params.v0, fit.params.kappa, fit.params.theta, fit.params.xi, fit.params.rho])
    assert np.allclose(got, [0.045, 2.5, 0.06, 0.7, -0.7], rtol=1e-4)
    assert fit.rmse_iv < 1e-6


def test_calibration_with_noisy_quotes_fits_to_the_noise_level():
    true = Heston(0.045, 2.5, 0.06, 0.7, -0.7)
    rng = np.random.default_rng(0)
    noisy = [(T, k, v + rng.normal(0, 0.002)) for T, k, v in _surface(true)]
    fit = calibrate_heston(S, r, noisy, q)
    assert fit.rmse_iv < 0.0025                  # cannot beat 0.002 noise by much, must not miss it
    assert fit.params.rho == pytest.approx(-0.7, abs=0.05)
    assert fit.params.v0 == pytest.approx(0.045, abs=0.003)


# ------------------------------------------------------------------ hedging
def test_hedging_at_realised_vol_locks_in_the_vol_edge():
    target = locked_in_pnl(S, 100, 0.5, r, 0.2, 0.3)
    coarse = delta_hedge(S, 100, 0.5, r, 0.2, 0.3, n_steps=25, sigma_hedge=0.3, n_paths=20_000)
    fine = delta_hedge(S, 100, 0.5, r, 0.2, 0.3, n_steps=400, sigma_hedge=0.3, n_paths=20_000)
    assert fine.mean == pytest.approx(target, abs=0.02)
    assert fine.std == pytest.approx(coarse.std / 4, rel=0.1)     # 16x steps -> 1/4 the noise


def test_hedging_error_follows_derman_kamal():
    for n in (25, 100, 400):
        h = delta_hedge(S, 100, 0.5, 0.0, 0.2, 0.2, n_steps=n, n_paths=20_000)
        assert abs(h.mean) < 0.03                                 # fair price: no edge
        assert h.std == pytest.approx(hedging_error_std(S, 100, 0.5, 0.0, 0.2, n), rel=0.06)


def test_hedged_pnl_does_not_depend_on_drift():
    means = [delta_hedge(S, 100, 0.5, r, 0.2, 0.3, n_steps=400, sigma_hedge=0.3, mu=mu,
                         n_paths=20_000).mean for mu in (-0.3, 0.0, 0.4)]
    assert np.ptp(means) < 0.03


def test_transaction_costs_reduce_hedged_pnl():
    free = delta_hedge(S, 100, 0.5, r, 0.2, 0.2, n_steps=100, n_paths=4000)
    paid = delta_hedge(S, 100, 0.5, r, 0.2, 0.2, n_steps=100, n_paths=4000, cost_per_share_pct=5e-4)
    assert paid.mean < free.mean - 0.05


def test_gamma_theta_breakeven_is_one_daily_standard_deviation():
    assert gamma_theta_breakeven_move(S, 100, 0.5, 0.0, 0.2) == pytest.approx(
        S * 0.2 / np.sqrt(365), rel=1e-9)


# ------------------------------------------------------------------ densities and event vol
def test_breeden_litzenberger_recovers_lognormal():
    T = 0.5
    K = np.linspace(20, 300, 2801)
    Ki, d = bl_density(K, bs_price(S, K, T, r, 0.25), r, T)
    s, mu = 0.25 * np.sqrt(T), np.log(S) + (r - 0.5 * 0.25**2) * T
    assert np.max(np.abs(d - lognorm.pdf(Ki, s, scale=np.exp(mu)))) < 1e-6
    m = density_moments(Ki, d)
    assert m["mass"] == pytest.approx(1, abs=1e-6) and m["mean"] == pytest.approx(S * np.exp(r * T), rel=1e-6)


def test_svi_density_is_proper_and_left_skewed():
    params = (0.01, 0.1, -0.6, 0.0, 0.15)
    K, d = svi_density(S, 0.5, r, params)
    m = density_moments(K, d)
    assert m["mass"] == pytest.approx(1, abs=1e-4)
    assert m["mean"] == pytest.approx(S * np.exp(r * 0.5), rel=1e-4)
    assert m["skew"] < 0 and d.min() > -1e-10


def test_variance_swap_strike():
    K = np.linspace(5, 600, 4000)
    assert variance_swap_strike(S, 0.5, r, K, np.full_like(K, 0.25)) == pytest.approx(0.0625, rel=1e-4)
    params = (0.01, 0.1, -0.6, 0.0, 0.15)
    F = S * np.exp(r * 0.5)
    iv = np.sqrt(svi_w(np.log(K / F), *params) / 0.5)
    atm_var = svi_w(0.0, *params) / 0.5
    assert variance_swap_strike(S, 0.5, r, K, iv) > atm_var    # skew makes variance dearer than ATM


def test_event_vol_matches_worked_example_and_round_trips():
    e = event_vol(0.36, 40, 0.24)                  # desk-reference example: 171%, move ~8.6%
    assert e == pytest.approx(1.714, abs=1e-3)
    assert implied_event_move(e) == pytest.approx(0.0861, abs=1e-3)
    assert iv_after_event(0.36, 40, e) == pytest.approx(0.24, abs=1e-12)


# ------------------------------------------------------------------ statistics
def test_deflated_sharpe_rejects_best_of_many_random_strategies():
    rng = np.random.default_rng(0)
    false_naive = false_dsr = 0
    for _ in range(100):
        R = rng.normal(0, 0.01, (100, 250))        # 100 skill-less strategies
        srs = R.mean(1) / R.std(1, ddof=1)
        best = R[np.argmax(srs)]
        false_naive += probabilistic_sharpe(best) > 0.95
        false_dsr += deflated_sharpe(best, 100, srs) > 0.95
    assert false_naive > 90                         # naive test is fooled almost every time
    assert false_dsr <= 5                           # deflated test is not


def test_deflated_sharpe_keeps_real_skill_and_falls_with_trials():
    r_ = np.random.default_rng(1).normal(0.003, 0.01, 500)      # true Sharpe 0.3 per period
    assert probabilistic_sharpe(r_, 0.3) == pytest.approx(0.5, abs=0.4)  # true SR is a coin flip
    assert deflated_sharpe(r_, 10) > deflated_sharpe(r_, 1000)
    assert deflated_sharpe(r_, 100) > 0.95
    assert 1 < min_track_record(r_) < 500


def test_walk_forward_never_tests_on_the_past():
    for tr, te in walk_forward_splits(100, 30, 10, embargo=2):
        assert te.min() == tr.max() + 3 and len(tr) == 30 and len(te) == 10


# ------------------------------------------------------------------ market making
def test_reservation_price_skews_against_inventory():
    p = ASParams()
    bid0, ask0, r0 = optimal_quotes(100, 0, 0.5, p)
    bid1, ask1, r1 = optimal_quotes(100, 5, 0.5, p)
    assert r0 == 100 and r1 < 100 and bid1 < bid0 and ask1 < ask0
    assert ask1 - bid1 == pytest.approx(ask0 - bid0)


def test_avellaneda_stoikov_reproduces_published_comparison():
    """Paper (gamma = 0.1): inventory strategy P&L std ~6, symmetric ~13; spread ~1.49."""
    inv = simulate(strategy="inventory", n_paths=1000).summary()
    sym = simulate(strategy="symmetric", n_paths=1000).summary()
    assert inv["spread"] == pytest.approx(1.49, abs=0.01)
    assert inv["std_pnl"] < 0.6 * sym["std_pnl"]
    assert inv["std_q"] < 0.5 * sym["std_q"]
    assert 55 < inv["mean_pnl"] < sym["mean_pnl"] < 75


# ------------------------------------------------------------------ data loading
def test_load_closes_reads_yahoo_format_not_volume(tmp_path):
    from quantdesk.data import load_closes
    import pandas as pd
    dates = pd.bdate_range("2025-01-01", periods=100)
    close = np.linspace(100, 120, 100)
    df = pd.DataFrame({"Date": dates[::-1].strftime("%Y-%m-%d"), "Open": close[::-1], "High": close[::-1],
                       "Low": close[::-1], "Close": close[::-1], "Adj Close": close[::-1] * 0.99,
                       "Volume": np.full(100, 5_000_000)})          # newest first, like some exports
    f = tmp_path / "yahoo.csv"
    df.to_csv(f, index=False)
    out = load_closes(f)
    assert np.allclose(out, close * 0.99)                           # adjusted, oldest first, not volume


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-q"]))
