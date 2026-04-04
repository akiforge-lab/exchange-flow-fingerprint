# exchange-flow-fingerprint

A local monitoring tool for cross-exchange funding rate behaviour.
Identifies medium-horizon trading opportunities (500–1000 min holding) by detecting when "smart flow" on leading exchanges has not yet been reflected in your execution venues.

## How it works

- Collects funding rate snapshots every 60 s from [loris.tools](https://api.loris.tools/funding)
- Builds a panel of per-exchange, per-symbol rates over time
- Scores opportunities using: rate z-score vs history, leader-consensus trend persistence, and leader→exec gap
- Classifies exchanges as **leaders** or **followers** based on cross-symbol average lead scores
- Surfaces results in a Streamlit dashboard with auto-mode (auto-collect, auto-rebuild, no babysitting)

## Setup

```bash
pip install -r requirements.txt
```

Python 3.11+ recommended.  All data fetching uses the stdlib (`urllib`) — no API key required.

## Usage

### Auto mode (recommended)

Start the dashboard — it handles everything automatically:

```bash
streamlit run dashboard.py
```

Turn on **Auto mode** in the sidebar. The dashboard will:
1. Start the collector (funding snapshots every 60 s)
2. Rebuild the panel automatically after every 5 new snapshots
3. Recompute the opportunity table in the background
4. Update the Opportunities tab live via `@st.fragment` (no full page reload)

### Manual workflow

```bash
# 1. Collect snapshots (runs until interrupted)
py fetch.py --loop 60

# 2. Build panel from snapshots
py build_panel.py

# 3. Launch dashboard
streamlit run dashboard.py

# Optional: fetch price data for the Detail tab
py fetch_prices.py --symbols ETH SOL PEPE SUI ENA
```

## Files

| File | Purpose |
|---|---|
| `fetch.py` | Polls loris.tools API, saves raw snapshots to `data/snapshots/` |
| `build_panel.py` | Assembles snapshots into `data/panel.csv` |
| `dashboard.py` | Streamlit dashboard (main entry point) |
| `fetch_prices.py` | Fetches 1-minute OHLCV from Binance for price context |
| `analyze.py` | CLI analysis of exchange lead/lag dynamics |
| `analyze_price.py` | CLI analysis of funding–price alignment |

## Data directory

`data/` is excluded from version control (snapshots and panel can be large).
The directory is created automatically on first run.

## Deployment note

Designed to run locally or on a headless server.  `.streamlit/config.toml` disables the email prompt and runs non-interactively out of the box.
