"""Is the backtest lying? Statistics that correct a Sharpe ratio for luck and for trying many ideas.

* Probabilistic Sharpe Ratio (Bailey & Lopez de Prado 2012): the probability that the true Sharpe
  exceeds a benchmark, given the sample length, skewness and fat tails of the returns.
* Deflated Sharpe Ratio (Bailey & Lopez de Prado 2014): the PSR measured against the Sharpe you
  would expect from the BEST of N strategies with no skill. Test 50 variants and keep the winner,
  and its Sharpe is inflated; the DSR removes that selection bias.
* Walk-forward splits with an embargo: fit only on the past, test on the next block, never tune
  on the test window.

All Sharpe ratios here are per period (per trade or per day), not annualised.
"""
from __future__ import annotations
import numpy as np
from scipy.stats import norm, skew, kurtosis

EULER_GAMMA = 0.5772156649015329


def sharpe(returns):
    r = np.asarray(returns, float)
    return float(r.mean() / r.std(ddof=1))


def probabilistic_sharpe(returns, sr_benchmark=0.0):
    """P(true Sharpe > sr_benchmark). Uses the Mertens (2002) standard error of the Sharpe ratio,
    which grows with negative skew and fat tails."""
    r = np.asarray(returns, float)
    n = len(r)
    sr = sharpe(r)
    g3 = skew(r)
    g4 = kurtosis(r, fisher=False)              # raw kurtosis (3 for a normal)
    se = np.sqrt((1 - g3 * sr + (g4 - 1) / 4 * sr**2) / (n - 1))
    return float(norm.cdf((sr - sr_benchmark) / se))


def expected_max_sharpe(n_trials, sr_std):
    """Expected maximum Sharpe among n_trials skill-less strategies whose Sharpe estimates have
    standard deviation sr_std (extreme-value approximation)."""
    if n_trials <= 1:
        return 0.0
    return float(sr_std * ((1 - EULER_GAMMA) * norm.ppf(1 - 1 / n_trials)
                           + EULER_GAMMA * norm.ppf(1 - 1 / (n_trials * np.e))))


def deflated_sharpe(returns, n_trials, trial_sharpes=None):
    """Probability that the selected strategy's true Sharpe beats the best-of-N luck benchmark.

    trial_sharpes: Sharpe ratios of every variant tried (their spread estimates sr_std). If not
    given, the spread under the null, 1/sqrt(n-1), is used.
    """
    r = np.asarray(returns, float)
    sr_std = (float(np.std(trial_sharpes, ddof=1)) if trial_sharpes is not None
              else 1 / np.sqrt(len(r) - 1))
    return probabilistic_sharpe(r, expected_max_sharpe(n_trials, sr_std))


def min_track_record(returns, sr_benchmark=0.0, confidence=0.95):
    """Number of observations needed before the observed Sharpe beats sr_benchmark at the given
    confidence (Bailey & Lopez de Prado). Returns inf if the Sharpe does not beat the benchmark."""
    r = np.asarray(returns, float)
    sr = sharpe(r)
    if sr <= sr_benchmark:
        return float("inf")
    g3, g4 = skew(r), kurtosis(r, fisher=False)
    z = norm.ppf(confidence)
    return float(1 + (1 - g3 * sr + (g4 - 1) / 4 * sr**2) * (z / (sr - sr_benchmark)) ** 2)


def walk_forward_splits(n, train, test, step=None, embargo=0):
    """Yield (train_idx, test_idx) index arrays for a rolling walk-forward test.

    The test block always lies strictly after its training block, separated by `embargo`
    observations so that overlapping trades cannot leak information across the boundary.
    """
    step = test if step is None else step
    start = 0
    while start + train + embargo + test <= n:
        tr = np.arange(start, start + train)
        te = np.arange(start + train + embargo, start + train + embargo + test)
        yield tr, te
        start += step
