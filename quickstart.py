"""See the desk in action with one click (no pytest needed).

Runs the research experiments (Heston calibration, delta hedging, risk-neutral density,
deflated Sharpe, market making), then a full verdict-first trade report and backtest.
Unzip the project first, then run this file from its folder.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from quantdesk.cli import main  # noqa: E402

if __name__ == "__main__":
    main(["research"])
    print()
    main(["demo"])
