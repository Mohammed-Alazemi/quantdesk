"""Figures and numbers for the quantdesk paper. Everything is computed from the package itself."""
import json, os, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import lognorm

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
os.chdir(HERE)
from quantdesk import bs_price, greeks, implied_vol, american_crr
from quantdesk.stochastic import Heston
from quantdesk.heston import heston_smile, calibrate_heston, heston_price
from quantdesk.hedging import delta_hedge, locked_in_pnl, hedging_error_std
from quantdesk.density import svi_density, density_moments, variance_swap_strike, event_vol, implied_event_move
from quantdesk.vol import svi_w, garch_fit
from quantdesk.stats import probabilistic_sharpe, deflated_sharpe, expected_max_sharpe
from quantdesk.marketmaking import simulate
from quantdesk.backtest import run_universe, summarize
from quantdesk.risk import kelly_fraction

plt.rcParams.update({
    "font.family": "serif", "font.serif": ["DejaVu Serif"], "mathtext.fontset": "dejavuserif",
    "font.size": 9.5, "axes.labelsize": 9.5, "legend.fontsize": 8, "axes.spines.top": False,
    "axes.spines.right": False, "lines.linewidth": 1.4, "savefig.bbox": "tight",
})
INK, NUM, ACC, GREY = "#1b1f24", "#c0392b", "#1f5fa8", "#8a8f98"
os.makedirs("figures", exist_ok=True)
out = {}
S, r = 100.0, 0.04


def save(fig, name):
    fig.savefig(f"figures/{name}.pdf"); fig.savefig(f"figures/{name}.png", dpi=200); plt.close(fig)


# ---------------- validation numbers for pricing
c, p = bs_price(100, 105, 0.5, 0.04, 0.3, 0.01, "call"), bs_price(100, 105, 0.5, 0.04, 0.3, 0.01, "put")
out["parity_err"] = abs(c - p - (100 * np.exp(-0.005) - 105 * np.exp(-0.02)))
out["textbook"] = bs_price(100, 100, 1, 0.05, 0.2)
out["iv_rt"] = abs(implied_vol(bs_price(100, 105, 0.5, 0.04, 0.37, 0.01), 100, 105, 0.5, 0.04, 0.01) - 0.37)
out["am_put"] = american_crr(100, 105, 0.5, 0.04, 0.3, 0, "put", 600)
out["eu_put"] = bs_price(100, 105, 0.5, 0.04, 0.3, 0, "put")

# ---------------- Figure 1: Heston calibration
true = Heston(0.045, 2.5, 0.06, 0.7, -0.7)
rng = np.random.default_rng(0)
quotes = []
Ts = (0.1, 0.25, 0.5, 1.0, 2.0)
for T in Ts:
    K = S * np.exp(np.linspace(-0.4, 0.3, 9) * np.sqrt(T))
    quotes += [(T, k, v + rng.normal(0, 0.002)) for k, v in zip(K, heston_smile(S, K, T, r, true))]
fit = calibrate_heston(S, r, quotes)
out["heston_true"] = [true.v0, true.kappa, true.theta, true.xi, true.rho]
out["heston_fit"] = [fit.params.v0, fit.params.kappa, fit.params.theta, fit.params.xi, fit.params.rho]
out["heston_rmse"] = fit.rmse_iv
fig, ax = plt.subplots(figsize=(4.8, 3.0))
cols = plt.cm.viridis(np.linspace(0.05, 0.85, len(Ts)))
Q = np.array(quotes)
for T, col in zip(Ts, cols):
    m = Q[:, 0] == T
    k = np.log(Q[m, 1] / (S * np.exp(r * T)))
    ax.plot(k, Q[m, 2] * 100, "o", ms=3.5, mfc="none", color=col)
    kk = np.linspace(k.min(), k.max(), 80)
    ax.plot(kk, heston_smile(S, S * np.exp(r * T) * np.exp(kk), T, r, fit.params) * 100, color=col,
            label=fr"$T={T:g}$")
ax.set_xlabel(r"log-moneyness $\ln(K/F)$"); ax.set_ylabel("implied vol (%)")
ax.legend(frameon=False, ncol=2)
save(fig, "fig1_heston")

# ---------------- Figure 2: hedging
ns = np.array([6, 12, 25, 50, 100, 200, 400])
std_real, std_imp, std_none, mean_real = [], [], [], []
for n in ns:
    a = delta_hedge(S, 100, 0.5, r, 0.2, 0.3, n_steps=n, sigma_hedge=0.3, n_paths=20_000)
    b = delta_hedge(S, 100, 0.5, r, 0.2, 0.3, n_steps=n, n_paths=20_000)
    c0 = delta_hedge(S, 100, 0.5, 0.0, 0.2, 0.2, n_steps=n, n_paths=20_000)
    std_real.append(a.std); std_imp.append(b.std); std_none.append(c0.std); mean_real.append(a.mean)
dk = [hedging_error_std(S, 100, 0.5, 0.0, 0.2, n) for n in ns]
out["locked"] = locked_in_pnl(S, 100, 0.5, r, 0.2, 0.3)
out["hedge_rows"] = [dict(n=int(n), mr=m, sr=s1, si=s2, s0=s3, dk=d)
                     for n, m, s1, s2, s3, d in zip(ns, mean_real, std_real, std_imp, std_none, dk)]
a200 = delta_hedge(S, 100, 0.5, r, 0.2, 0.3, n_steps=200, sigma_hedge=0.3, n_paths=20_000)
b200 = delta_hedge(S, 100, 0.5, r, 0.2, 0.3, n_steps=200, n_paths=20_000)
fig, ax = plt.subplots(1, 2, figsize=(7.2, 2.8), gridspec_kw=dict(width_ratios=[1.15, 1]))
bins = np.linspace(-3, 9, 90)
ax[0].hist(b200.pnl, bins=bins, color=GREY, alpha=0.55, label=r"hedge at $\sigma_i$")
ax[0].hist(a200.pnl, bins=bins, color=ACC, alpha=0.75, label=r"hedge at $\sigma_r$")
ax[0].axvline(out["locked"], color=NUM, ls="--", lw=1.1, label=r"$[V(\sigma_r)-V(\sigma_i)]e^{rT}$")
ax[0].set_xlabel("hedged P&L at expiry ($)"); ax[0].set_ylabel("paths"); ax[0].legend(frameon=False, loc="upper right")
ax[0].set_xlim(-1.5, 9.5)
ax[1].loglog(ns, std_none, "o", color=NUM, ms=4, mfc="none", label=r"no edge, simulated")
ax[1].loglog(ns, dk, "-", color=INK, lw=1, label=r"$\sqrt{\pi/4}\,\nu\sigma/\sqrt{n}$")
ax[1].loglog(ns, std_real, "s", color=ACC, ms=3.5, label=r"hedge at $\sigma_r$")
ax[1].loglog(ns, std_imp, "^", color=GREY, ms=3.5, label=r"hedge at $\sigma_i$")
ax[1].set_xlabel("rebalances $n$"); ax[1].set_ylabel("std of P&L ($)"); ax[1].legend(frameon=False, fontsize=7.5)
from matplotlib.ticker import FixedLocator, FixedFormatter, NullLocator
ax[1].yaxis.set_major_locator(FixedLocator([0.25, 0.5, 1, 2, 3])); ax[1].yaxis.set_major_formatter(FixedFormatter(["0.25", "0.5", "1", "2", "3"]))
ax[1].yaxis.set_minor_locator(NullLocator())
save(fig, "fig2_hedging")

# ---------------- Figure 3: risk-neutral density
params, T = (0.01, 0.1, -0.6, 0.0, 0.15), 0.5
K, d = svi_density(S, T, r, params)
m = density_moments(K, d)
atm_vol = np.sqrt(svi_w(0, *params) / T)
s_ln = atm_vol * np.sqrt(T); mu = np.log(S) + (r - 0.5 * atm_vol**2) * T
Ks = np.linspace(5, 600, 4000)
iv = np.sqrt(svi_w(np.log(Ks / (S * np.exp(r * T))), *params) / T)
out["density"] = m; out["atm_vol"] = atm_vol
out["varswap_vol"] = np.sqrt(variance_swap_strike(S, T, r, Ks, iv))
e = event_vol(0.36, 40, 0.24); out["event_vol"] = e; out["event_move"] = implied_event_move(e)
fig, ax = plt.subplots(1, 2, figsize=(7.2, 2.7))
kk = np.linspace(-0.6, 0.45, 200)
ax[0].plot(kk, np.sqrt(svi_w(kk, *params) / T) * 100, color=INK)
ax[0].axhline(atm_vol * 100, color=GREY, ls=":", lw=1)
ax[0].set_xlabel(r"$\ln(K/F)$"); ax[0].set_ylabel("implied vol (%)"); ax[0].set_title("SVI smile", fontsize=9)
sel = (K > 40) & (K < 170)
ax[1].plot(K[sel], d[sel], color=NUM, label="implied by the smile")
ax[1].plot(K[sel], lognorm.pdf(K[sel], s_ln, scale=np.exp(mu)), color=GREY, ls="--", label="lognormal at ATM vol")
ax[1].set_xlabel(r"$S_T$"); ax[1].set_ylabel("risk-neutral density"); ax[1].legend(frameon=False, loc="upper left", fontsize=7.5); ax[1].set_ylim(0, 0.044)
ax[1].set_title("Breeden–Litzenberger", fontsize=9)
save(fig, "fig3_density")

# ---------------- Figure 4: overfitting
rng = np.random.default_rng(3)
best_sr, naive, defl = [], [], []
for _ in range(300):
    R = rng.normal(0, 0.01, (100, 250))
    srs = R.mean(1) / R.std(1, ddof=1)
    b = R[np.argmax(srs)]
    best_sr.append(srs.max() * np.sqrt(252)); naive.append(probabilistic_sharpe(b))
    defl.append(deflated_sharpe(b, 100, srs))
out["dsr"] = dict(best_sr_median=float(np.median(best_sr)), naive_fp=float(np.mean(np.array(naive) > 0.95)),
                  dsr_fp=float(np.mean(np.array(defl) > 0.95)),
                  emax=expected_max_sharpe(100, 1 / np.sqrt(249)) * np.sqrt(252))
fig, ax = plt.subplots(1, 2, figsize=(7.2, 2.6))
ax[0].hist(best_sr, bins=30, color=ACC, alpha=0.8)
ax[0].axvline(out["dsr"]["emax"], color=NUM, ls="--", lw=1.1, label="expected max (theory)")
ax[0].set_xlabel("best annualised Sharpe of 100 skill-less strategies"); ax[0].set_ylabel("trials")
ax[0].legend(frameon=False)
ax[1].hist(naive, bins=np.linspace(0, 1, 26), color=GREY, alpha=0.6, label="naive PSR")
ax[1].hist(defl, bins=np.linspace(0, 1, 26), color=NUM, alpha=0.7, label="deflated SR")
ax[1].axvline(0.95, color=INK, ls=":", lw=1)
ax[1].set_xlabel("reported probability of skill"); ax[1].legend(frameon=False, loc="upper center")
save(fig, "fig4_overfitting")

# ---------------- backtest numbers
bt = {}
for pump in (0.0, 0.35):
    s = summarize(run_universe(20, pump=pump, n_calib=0))
    bt[str(pump)] = dict(n=s["n"], mean=s["mean_R"], lo=s["mean_R_ci95"][0], hi=s["mean_R_ci95"][1],
                         t=s["t_stat"], psr=s["psr"], win=s["win_rate"])
exp_rows = {}
for b in (14, 45, 90, 120):
    s = summarize(run_universe(20, pump=0.35, n_calib=0, dte_buffer_cal=b))
    exp_rows[b] = dict(mean=s["mean_R"], sharpe=s["sharpe_trade"])
out["backtest"] = bt; out["expiry"] = exp_rows
rng = np.random.default_rng(1)
good = rng.choice([-100.0, 150.0], 200_000, p=[0.5, 0.5])
out["kelly_check"] = kelly_fraction(good, 100.0, max_n=200_000)[0]

# ---------------- Figure 5: market making
inv = simulate(strategy="inventory", n_paths=1000)
sym = simulate(strategy="symmetric", n_paths=1000)
out["mm"] = dict(inv=inv.summary(), sym=sym.summary())
fig, ax = plt.subplots(1, 2, figsize=(7.2, 2.6))
bins = np.linspace(20, 110, 46)
ax[0].hist(sym.pnl, bins=bins, color=GREY, alpha=0.6, label="symmetric")
ax[0].hist(inv.pnl, bins=bins, color=ACC, alpha=0.75, label="inventory-aware")
ax[0].set_xlabel("terminal P&L"); ax[0].set_ylabel("paths"); ax[0].legend(frameon=False, loc="upper left")
ax[0].set_ylim(0, ax[0].get_ylim()[1] * 1.3)
qb = np.arange(-30, 31, 2)
ax[1].hist(sym.inventory, bins=qb, color=GREY, alpha=0.6, label="symmetric")
ax[1].hist(inv.inventory, bins=qb, color=ACC, alpha=0.75, label="inventory-aware")
ax[1].set_xlabel("terminal inventory"); ax[1].legend(frameon=False, loc="upper left")
ax[1].set_ylim(0, ax[1].get_ylim()[1] * 1.3)
save(fig, "fig5_mm")

json.dump(out, open("numbers.json", "w"), indent=1, default=float)
print(json.dumps(out, indent=1, default=float)[:4000])
