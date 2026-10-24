"""quantdesk: an options research desk in pure numpy/scipy/pandas."""
from .pricing import bs_price, greeks, implied_vol, american_crr
from .vol import hist_vol, garch_fit, garch_forecast_vol, svi_fit, svi_w, svi_iv, svi_butterfly_ok
from .stochastic import Heston, gbm_terminal, heston_terminal
from .heston import heston_price, heston_smile, calibrate_heston
from .hedging import delta_hedge, locked_in_pnl, hedging_error_std, gamma_theta_breakeven_move
from .density import (bl_density, svi_density, density_moments, variance_swap_strike, event_vol,
                      implied_event_move, iv_after_event)
from .strategies import Leg, CostModel, bull_call_spread, position_pnl, net_greeks
from .risk import kelly_fraction, kelly_bootstrap, kelly_vs_drift, var_cvar, stress_grid
from .stats import (probabilistic_sharpe, deflated_sharpe, min_track_record, walk_forward_splits)
from .marketmaking import ASParams, optimal_quotes
from .checklist import Rules, run_gates
from .analysis import full_analysis, format_report

__version__ = "3.0"
