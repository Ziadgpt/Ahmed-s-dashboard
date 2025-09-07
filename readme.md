Ahmed's Crypto Holdings
Overview
Ahmed's Crypto Holdings is an AI-powered backtest analyzer designed for institutional crypto traders to evaluate TradingView trade export data. It delivers precise performance metrics, advanced visualizations, and actionable insights for portfolio optimization, providing a professional dashboard for strategic decision-making.
Features

Performance Metrics: Comprehensive portfolio metrics, including net profit, max drawdown, Sharpe ratio (daily, annualized), Sortino ratio (daily, annualized), buy & hold return, open P&L, profit factor, total closed trades, winning/losing trades, and more.
Visualizations: Equity curve, underwater plot (drawdown), PnL per trade, trade duration distribution, MFE vs. MAE scatter, monthly returns heatmap, and rolling Sharpe ratio.
Advanced Analytics:
Anomaly detection for identifying outlier trades (e.g., lucky wins or poor exits).
Monte Carlo simulations (Student's t or historical methods).
Portfolio optimization (Sharpe-optimized and hierarchical risk parity allocations).


Exports: PDF report with metrics and AI insights; CSV export of trade data.
User Interface: Modern, responsive Streamlit dashboard with tooltips, date filters, and customizable inputs (initial equity, risk-free rate, timeframe).

Setup Instructions

Prerequisites:
Python 3.8 or higher
pip package manager


Clone or Download:
Download the project files: app.py, requirements.txt, and your trade export Excel file.


Install Dependencies:pip install -r requirements.txt

Dependencies:
streamlit
pandas
numpy
plotly
scikit-learn
scipy
reportlab


Run the Dashboard:streamlit run app.py


Access:
Open the provided URL (e.g., http://localhost:8501) in a web browser.



Input Requirements

File Format: Excel file (.xlsx) with a "List of trades" sheet.
Required Columns:
Trade #: Unique trade identifier
Type: "Entry long" or "Exit long"
Date/Time: Trade timestamp (Excel serial or datetime format)
Price USDT: Trade price
Position size (qty): Quantity traded
Position size (value): Position value in USDT
Net P&L USDT: Profit/loss in USDT
Net P&L %: Profit/loss percentage
Run-up USDT: Maximum favorable excursion
Drawdown USDT: Maximum adverse excursion
Cumulative P&L USDT: Cumulative profit/loss


Optional Column: Symbol (defaults to "PYRUSDT" if missing).

Usage

Launch: Run streamlit run app.py and access the dashboard in a web browser.
Upload: Upload your TradingView trade export Excel file.
Configure:
Set Initial Equity to your starting account balance.
Set Risk-Free Rate (e.g., USDT staking rate; affects Sharpe/Sortino ratios).
Set Chart Timeframe (hours per bar, for trade duration calculations).
Filter trades by date range or use presets (Last Day, Week, Month, Year, Max).


Explore Tabs:
Overview: View key metrics with tooltips explaining each metric.
Trades: Inspect detailed trade data (trade number, symbol, dates, P&L, etc.).
Performance: Analyze PnL and trade duration distributions.
Risk: Review Value at Risk (VaR), Conditional VaR (CVaR), and correlation matrix.
Visuals: Explore equity curves, drawdown plots, and other visualizations.
Advanced Analytics: Access anomaly detection, Monte Carlo simulations, and portfolio allocation insights.


Export: Download a PDF report or CSV trades from the Advanced Analytics tab.

Notes

Metric Methodology: Metrics like Sharpe and Sortino ratios are calculated using daily returns, which may differ from TradingView’s per-trade calculations. Daily-based metrics are standard for portfolio analysis.
Open Trades: Open trades are detected and reported separately in metrics like open P&L and total open trades.
CSS Design: Metric cards are styled for responsive display, preventing text overflow for large values.
Support: For issues or customization, contact support@ahmedscryptoholdings.com.

Troubleshooting

Excel Errors: Ensure the "List of trades" sheet exists and columns match the required format.
Display Issues: Use a compatible browser (Chrome, Firefox recommended) and check screen resolution for optimal rendering.
Performance: Large datasets may require caching optimization; contact support for assistance.

Contact
For questions or support, email: support@ahmedscryptoholdings.com.# Ahmed-s-dashboard
