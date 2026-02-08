# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

CMU MSBA capstone project for **stablecoin depeg prediction**. Collects multi-source data (on-chain, exchange, DeFi, macro) for stablecoins (USDT, USDC, DAI, FRAX) and trains ML classifiers to predict depeg events (price deviations from $1.00 peg).

## Commands

```bash
# Install dependencies
pip install -r requirements.txt

# Collect all data for a stablecoin (or "all")
python src/data/collect_all.py usdt --start-date 2020-01-01
python src/data/collect_all.py all --no-onchain

# Collect individual sources
python src/data/collect_prices.py      # CoinGecko prices
python src/data/collect_binance.py     # Binance OHLCV
python src/data/collect_kraken.py     # Kraken fiat pairs (USDTUSD, USDCUSD, DAIUSD)
python src/data/collect_defillama.py   # DefiLlama supply metrics
python src/data/collect_onchain.py     # Etherscan transfers (needs ETHERSCAN_API_KEY)
python src/data/collect_market.py      # BTC/ETH prices + Fear & Greed
python src/data/collect_fred.py        # FRED macro data (needs FRED_API_KEY)

# Merge raw sources into processed files
python src/data/merge_sources.py

# Train models
python src/models/baseline.py         # Single-coin (USDT)
python src/models/multi_coin.py       # Multi-coin (USDT + USDC)

# Test feature engineering standalone
python src/features/engineering.py

# Run tests
pytest tests/
```

## Architecture

### Data Pipeline

`collect_all.py` orchestrates 7 collectors that each write to `data/raw/`:

1. **CoinGeckoCollector** (`collect_prices.py`) - stablecoin USD prices, market cap, volume
2. **BinanceCollector** (`collect_binance.py`) - hourly OHLCV for trading pairs (e.g., BTCUSDT), buy pressure, spread
3. **KrakenCollector** (`collect_kraken.py`) - hourly OHLCV for fiat pairs (USDTUSD, USDCUSD, DAIUSD), VWAP, spread
4. **DefiLlamaCollector** (`collect_defillama.py`) - circulating supply, chain distribution, implied price
5. **EtherscanCollector** (`collect_onchain.py`) - ERC-20 token transfers, whale transactions, gas metrics
6. **MarketDataCollector** (`collect_market.py`) - BTC/ETH prices, Fear & Greed Index, stablecoin market share
7. **FREDCollector** (`collect_fred.py`) - DXY, VIX, Treasury rates, Fed Funds rate

`merge_sources.py` joins all raw sources on date into `data/processed/{coin}_merged_daily.csv` and `combined_stablecoins_daily.csv`. Binance and Kraken data are aggregated from hourly to daily. FRED, Kraken, and Fear & Greed are left-joined (optional).

### Feature Engineering & Labeling

- `src/features/engineering.py` - creates ~25 features from merged data: BTC returns/volatility/drawdown, volume ratios, spread z-scores, supply momentum, buy pressure, stress indicators, flight-to-safety interaction terms. `prepare_modeling_data()` is the main entry point returning `(X, y, feature_names, df)`.
- `src/features/labeling.py` - labels contiguous depeg events using a deviation threshold (default 1% from `config/settings.py:DEPEG_THRESHOLD`). Creates forward-looking prediction targets.

### Models

- `src/models/baseline.py` - trains Logistic Regression, Random Forest, and Gradient Boosting on single-coin data with `TimeSeriesSplit` cross-validation and balanced class weights.
- `src/models/multi_coin.py` - same model suite on combined USDT+USDC data, adds `is_usdc` dummy feature.
- Saved artifacts in `models/`: `best_model.joblib`, `scaler.joblib`, `features.txt`, `config.txt`.

### Notebooks (in `notebooks/`)

- `data_collection.ipynb` - interactive data collection walkthrough
- `data_visualization.ipynb` - 10-section EDA (price deviations, depeg distributions, correlations, key market events)
- `descriptive_analytics.ipynb` - statistical analysis
- `model_training.ipynb` - model training and evaluation with visualizations

## Key Configuration

All stablecoin configs (CoinGecko IDs, DefiLlama IDs, Ethereum contracts, Binance pairs, Kraken pairs) are in `config/settings.py`. The depeg threshold defaults to 1% (`DEPEG_THRESHOLD = 0.01`), but models also use 0.5% for more positive training samples.

## API Keys

Set via environment variables (or `.env` file):
- `COINGECKO_API_KEY` - optional, increases rate limits
- `ETHERSCAN_API_KEY` - required for on-chain data
- `FRED_API_KEY` - required for macro data

## File Naming Convention

Raw data files follow: `{coin}_{source}_{detail}.csv` (e.g., `usdt_binance_btcusdt.csv`, `usdt_kraken_usdtusd.csv`, `usdc_defillama_metrics.csv`). Market-wide files: `market_{type}.csv`. All raw data is gitignored.

## Important Patterns

- All source modules use `sys.path.insert(0, ...)` for imports from project root. Run scripts from the project root directory.
- Collectors include rate limiting (`time.sleep()`) to respect API quotas.
- The prediction target is forward-looking: "will abs deviation exceed threshold in the next N days?"
- Class imbalance is significant (depeg events are rare). Models use balanced class weights and `TimeSeriesSplit` (not random CV) to respect temporal ordering.
