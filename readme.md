Ahmed's Crypto Holdings Dashboard
Overview
Ahmed's Crypto Holdings, crafted by Ziad Abdelaziz (zozquant@gmail.com), is a powerful, AI-driven Streamlit dashboard designed for crypto traders to analyze TradingView backtest data with precision. Tailored for professional traders, it transforms raw trade data from Excel files into actionable insights through advanced metrics, interactive visualizations, and intelligent analytics. With robust features like anomaly detection, Monte Carlo simulations, and portfolio optimization, the dashboard delivers exceptional value for optimizing trading strategies. Export reports in PDF, CSV, and Excel formats to share comprehensive analyses with ease, ensuring a seamless and professional experience.
Key Features

Data Processing: Seamlessly ingests TradingView Excel files ("List of trades" sheet), supporting serial dates and multi-symbol trades (e.g., PYRUSDT).
Performance Metrics: Computes precise metrics, including:
Net Profit (e.g., $195,738.44 for 1310 trades)
Max Drawdown (e.g., 9.65%)
Sharpe Ratio (e.g., 3.29, annualized daily)
Sortino Ratio, Calmar Ratio, VaR, CVaR, and more


Interactive Visualizations: Engaging Plotly charts, including:
Equity Curve
Underwater Plot (Drawdown)
PnL per Trade
Trade Duration Distribution
Correlation Matrix (multi-symbol)
Monthly Returns Heatmap
MFE vs. MAE Scatter Plot


Advanced Analytics:
Anomaly Detection: Identifies "Lucky Wins" and "Poor Exits" using Isolation Forest.
Monte Carlo Simulations: Forecasts equity paths with Student's t or historical bootstrap methods.
Portfolio Optimization: Provides Sharpe-optimized and Hierarchical Risk Parity (HRP) allocations.


Exports:
PDF Report: Professional text-based report with AI insights and metrics table.
CSV Export: Detailed trade data for external analysis.
Excel Report: Multi-sheet report with Metrics, Trades, Anomalies, Allocations, and AI Summary.


User Interface: Futuristic, responsive design with custom CSS, intuitive tabs, and interactive controls for a premium experience.
Reliability: Robust error handling ensures smooth operation, even with complex datasets.

Prerequisites

Python: Version 3.8 or higher.

Dependencies: Install required packages via:
pip install streamlit pandas numpy reportlab plotly scikit-learn scipy openpyxl


TradingView Export: An Excel file (.xlsx) with a "List of trades" sheet containing:

Required columns: Trade #, Type, Date/Time, Price USDT, Position size (qty), Position size (value), Net P&L USDT, Net P&L %, Run-up USDT, Drawdown USDT, Cumulative P&L USDT.
Optional: Symbol (defaults to PYRUSDT if missing).



Installation

Save the Code: Place the dashboard code in app.py.

Create requirements.txt:
streamlit>=1.46.1
pandas>=2.0.0
numpy>=1.24.0
reportlab>=4.0.0
plotly>=5.0.0
scikit-learn>=1.2.0
scipy>=1.10.0
openpyxl>=3.1.0


Install Dependencies:
pip install -r requirements.txt


Run the Application:
streamlit run app.py

Access the dashboard at http://localhost:8501 in your web browser.


Usage

Upload Trade Data:

In the sidebar’s "Control Panel," upload your TradingView Excel file (.xlsx).
Ensure the "List of trades" sheet includes required columns (e.g., as tested with trades.xlsx containing 1310 closed trades for PYRUSDT).


Configure Settings:

Initial Equity: Set starting balance (default: $10,000).
Risk-Free Rate: Input annualized rate (e.g., 2% for USDT staking, used in Sharpe/Sortino ratios).
Chart Timeframe: Specify hours per bar (default: 0.88 for ~1-minute bars).
Default Symbol: Set fallback symbol (default: PYRUSDT).
Anomaly Sensitivity: Adjust detection threshold (default: 0.02).


Filter Trades:

Use "Time Filter" to select date ranges or presets (Last Day, Week, Month, Year, Max).
Reset settings with the "Reset to Default" button.


Explore Tabs:

Overview: View key metrics (e.g., Net Profit: $195,738.44, Max Drawdown: 9.65%, Sharpe Ratio: 3.29) in hoverable cards.
Trades: Inspect trade details (trade number, symbol, dates, PnL, etc.).
Performance: Analyze PnL per Trade and Trade Duration Distribution.
Risk: Review VaR, CVaR, and Correlation Matrix (for multi-symbol data).
Visuals: Explore Equity Curve, Underwater Plot, Rolling Sharpe Ratio, Monthly Returns Heatmap, and MFE vs. MAE.
Advanced Analytics:
Access AI insights (e.g., anomaly warnings, risk alerts).
Run Monte Carlo simulations for equity forecasting.
View anomalies (Lucky Wins, Poor Exits).
Check Sharpe-optimized and HRP allocations.
Export reports (PDF, CSV, Excel).




Export Reports:

In "Advanced Analytics," download:
PDF Report: Text-based with AI insights and metrics (e.g., Sharpe Ratio, Max Drawdown).
CSV Export: Filtered trades for external use.
Excel Report: Multi-sheet with Metrics, Trades, Anomalies, Allocations, and AI Summary.





File Structure

app.py: Core dashboard script.
requirements.txt: Dependency list (as shown above).
README.md: This documentation.

Example Excel Format
Use a trades.xlsx file with a "List of trades" sheet. Example:



Trade #
Type
Date/Time
Symbol
Price USDT
Position size (qty)
Position size (value)
Net P&L USDT
Net P&L %
Run-up USDT
Drawdown USDT
Cumulative P&L USDT



1
Entry
2025-01-01 10:00:00
PYRUSDT
1.50
1000
1500.0
0.0
0.0
0.0
0.0
0.0


1
Exit
2025-01-01 12:00:00
PYRUSDT
1.60
1000
1600.0
100.0
6.67
150.0
-50.0
100.0


2
Entry
2025-01-02 09:00:00
PYRUSDT
1.55
500
775.0
0.0
0.0
0.0
0.0
100.0


PDF Report
The PDF report delivers a professional summary, including:

Title: "Ahmed's Crypto Holdings Report."
AI Insights: Actionable analysis (e.g., warnings for high drawdowns or anomalies).
Metrics Table: Comprehensive metrics (e.g., Net Profit, Sharpe Ratio, Max Drawdown).
Note: Visuals are available in the dashboard’s "Visuals" tab for interactive analysis.

Technical Details

Dependencies:
streamlit: Web interface.
pandas, numpy: Data processing.
plotly: Interactive visualizations.
scikit-learn: Anomaly detection.
scipy: Monte Carlo and optimization.
reportlab: Text-based PDF generation.
openpyxl: Excel exports.


Caching: @st.cache_data ensures fast performance with large datasets.
Error Handling:
Validates Excel columns and dates.
Skips invalid trades with clear warnings.
Handles edge cases (e.g., single trade, missing symbols).


Analytics:
Metrics computed daily for accuracy (252 trading days for annualization).
Anomaly detection flags unusual trades (e.g., high MAE/PnL ratios).
Monte Carlo supports Student's t and historical methods.
Allocations optimize return/risk (Sharpe) or minimize risk (HRP).



Known Limitations

Open Trade PnL: Unrealized PnL is approximated as 0 (no live price data). Contact the author for API integration (e.g., ccxt).
PDF Report: Text-only; use the dashboard for visualizations.
Large Datasets: May slow with >10,000 trades; caching mitigates this.

Troubleshooting

Invalid Dates: Ensure Date/Time column uses valid formats (e.g., 2025-01-01 10:00:00 or Excel serial dates).
Missing Columns: Verify column names match the required format.
Chart/Report Errors: Confirm dependencies (pip list) and memory availability.
High Skip Rate: Check warnings for skipped trades; ensure entries/exits are complete.
Contact: zozquant@gmail.com for prompt support.

Why Choose This Dashboard?

Proven Accuracy: Validated with real data (e.g., $195,738.44 net profit, 9.65% max drawdown for 1310 PYRUSDT trades).
Professional Design: Futuristic UI with tight, client-ready metric cards.
Actionable Insights: AI-driven recommendations enhance trading decisions.
Reliable Exports: PDF, CSV, and Excel reports for seamless sharing.
Support: Dedicated assistance from Ziad Abdelaziz to ensure success.

Author

Name: Ziad Abdelaziz
Email: zozquant@gmail.com

