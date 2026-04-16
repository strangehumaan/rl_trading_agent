# RL Trading Agent (PPO) - Single Asset

A compact Reinforcement Learning project that trains a **PPO (Proximal Policy Optimization)** agent to trade a single asset using historical OHLCV data and technical indicators.

This repository includes:
- A complete notebook pipeline: data download -> feature engineering -> custom Gymnasium environment -> PPO training -> evaluation
- Saved model artifact: `ppo_trading_agent.zip`
- Exported result plots for quick analysis

## Project Goal

Build and evaluate a single-asset RL trading system that learns **Buy / Hold / Sell** actions from market features, then compare it against:
- **Random Agent** (sanity baseline)
- **Buy & Hold** (market baseline)

## Method (Brief)

### 1) Data
- Source: `yfinance`
- Default ticker: `^NSEI` (NIFTY 50)
- Default period: `2019-01-01` to `2024-01-01`
- Train/test split: 80% / 20%

### 2) Features
- `Close`, `MA20`, `MA50`, `RSI(14)`, `MACD`

### 3) Environment
- Custom `TradingEnv` built with `gymnasium`
- **State**: sliding window of normalized features
- **Actions**: `0=Hold`, `1=Buy`, `2=Sell`
- **Reward**: step-wise percentage change in portfolio value

### 4) Agent
- Algorithm: `stable-baselines3` PPO (`MlpPolicy`)
- Timesteps: `75,000` (default notebook config)
- Initial cash: `100,000`

### 5) Evaluation Metrics
- Total Return (%)
- Sharpe Ratio (annualized)
- Max Drawdown (%)
- Number of trades

## Results Artifacts

The repository already contains generated outputs:
- `reward_curve.png` - training reward progression
- `cumulative_returns.png` - PPO vs Random vs Buy & Hold (test set)
- `buy_sell_markers.png` - test price chart with PPO buy/sell markers
- `train_vs_test.png` - train vs unseen test comparison
- `metrics_bar.png` - side-by-side metric comparison

> Note: Metrics are based on backtesting and should not be treated as live-trading performance.

## Repository Structure

- `RL.ipynb` - main end-to-end notebook
- `requirements.txt` - Python dependencies
- `ppo_trading_agent.zip` - trained PPO model checkpoint
- `*.png` - evaluation and visualization outputs

## Quick Start

### 1) Create and activate a virtual environment (Windows PowerShell)
```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

### 2) Install dependencies
```powershell
pip install -r requirements.txt
```

### 3) Run the project
```powershell
jupyter notebook RL.ipynb
```

Run notebook cells top-to-bottom to reproduce training, evaluation, and plots.

## Configuration

Edit the `CONFIG` dictionary in `RL.ipynb` to change:
- ticker symbol
- date range
- train/test split
- training timesteps
- window size
- initial capital
- random seed

## Limitations

- Backtest-only evaluation (no live deployment)
- No transaction costs, slippage, or brokerage fees in environment logic
- Risk of overfitting to a single historical period/asset
- Daily OHLCV features may not capture all market microstructure signals

## Future Improvements

- Walk-forward validation across rolling windows
- Add realistic market frictions (fees, slippage, position sizing constraints)
- Better reward shaping (risk-adjusted objective)
- Multi-asset portfolio environment
- Hyperparameter tuning and experiment tracking

## Disclaimer

This project is for **educational/research purposes only** and is **not financial advice**.

