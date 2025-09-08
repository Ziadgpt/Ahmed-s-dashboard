import streamlit as st
import pandas as pd
import numpy as np
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet
import io
from datetime import datetime, timedelta
import plotly.express as px
import plotly.graph_objects as go
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from scipy.optimize import minimize
from scipy.stats import t, norm
import scipy.cluster.hierarchy as sch

# Page Config
st.set_page_config(page_title="Ahmed's Crypto Holdings", layout="wide", initial_sidebar_state="expanded")
st.markdown("""
    <style>
    .main {background-color: #0A0F1A; color: #E0E7FF; padding: 20px; font-family: 'Orbitron', sans-serif;}
    .stTabs [data-baseweb="tab-list"] {gap: 20px; border-bottom: none;}
    .stTabs [data-baseweb="tab"] {height: 50px; font-size: 16px; font-weight: 500; background: linear-gradient(90deg, #1A237E, #283593); border-radius: 10px 10px 0 0; padding: 12px 24px; color: #BBDEFB; transition: all 0.3s ease; box-shadow: 0 2px 4px rgba(26, 35, 126, 0.5);}
    .stTabs [data-baseweb="tab"]:hover {transform: translateY(-3px); box-shadow: 0 4px 8px rgba(26, 35, 126, 0.7); color: #FFFFFF;}
    .stTabs [aria-selected="true"] {background: linear-gradient(90deg, #303F9F, #3F51B5); color: #FFFFFF; border-bottom: 3px solid #536DFE;}
    .metric-card {
        background: linear-gradient(135deg, #1A237E 0%, #303F9F 100%);
        padding: 20px;
        border-radius: 12px;
        margin: 5px;
        border: 2px solid #536DFE;
        box-sizing: border-box;
        box-shadow: 0 6px 12px rgba(26, 35, 126, 0.6);
        transition: transform 0.3s ease;
        min-width: 200px;
        max-width: 33.33%;
        display: inline-block;
        min-height: 120px;
        overflow: hidden;
        text-overflow: ellipsis;
        white-space: nowrap;
    }
    .metric-card:hover {transform: scale(1.03); box-shadow: 0 8px 16px rgba(83, 109, 254, 0.7);}
    .metric-title {
        color: #BBDEFB;
        font-weight: 600;
        font-size: 1.0em;
        margin-bottom: 10px;
        text-transform: uppercase;
        letter-spacing: 1px;
        overflow-wrap: break-word;
    }
    .metric-value {
        color: #FFFFFF;
        font-size: 1.4em;
        font-weight: bold;
        overflow-wrap: break-word;
        white-space: normal;
    }
    .stPlotlyChart {background-color: #0A0F1A; border-radius: 12px; padding: 15px; margin-bottom: 20px; box-shadow: 0 4px 8px rgba(26, 35, 126, 0.5);}
    .stButton > button {background: linear-gradient(90deg, #536DFE 0%, #3F51B5 100%); color: #FFFFFF; border: none; border-radius: 8px; padding: 10px 20px; font-weight: 600;}
    .stButton > button:hover {background: linear-gradient(90deg, #3F51B5 0%, #536DFE 100%); box-shadow: 0 4px 8px rgba(83, 109, 254, 0.5);}
    .heatmap-cell {cursor: pointer;}
    @import url('https://fonts.googleapis.com/css2?family=Orbitron:wght@400;500;600;700&display=swap');
    </style>
""", unsafe_allow_html=True)

st.title("Ahmed's Crypto Holdings")
st.markdown(
    '<div style="padding-bottom: 20px; color: #BBDEFB;">AI-Powered Backtest Analyzer for Crypto Traders. Precise Metrics, Advanced Insights, Optimal Strategies.</div>',
    unsafe_allow_html=True)

# Sidebar Controls
with st.sidebar:
    st.header("Control Panel")
    with st.expander("Help"):
        st.markdown("""
        **Ahmed's Crypto Holdings**
        - Upload TradingView Excel ('List of trades' sheet).
        - Set Initial Equity, Risk-Free Rate (e.g., USDT staking rate), Chart Timeframe (hours per bar).
        - Filter by date range or use presets.
        - Explore tabs: Overview (metrics), Trades (details), Performance (PnL bars), Risk (VaR), Visuals (equity curve), Advanced Analytics (anomalies, Monte Carlo).
        - Click 'Generate' in Monte Carlo to refresh simulations.
        - Export PDF/CSV/Excel from Advanced Analytics.
        Contact: zozquant@gmail.com
        """)
    uploaded_file = st.file_uploader("Upload TradingView Excel", type=["xlsx"])
    initial_equity = st.number_input("Initial Equity", value=10000.0, step=1000.0, min_value=1.0, key="initial_equity")
    risk_free_rate = st.number_input("Risk-Free Rate (%)", value=2.0, step=0.1, min_value=0.0, key="risk_free_rate") / 100
    timeframe_hours = st.number_input("Chart Timeframe (hours per bar)", value=0.88, step=0.1, min_value=0.01, key="timeframe_hours")
    default_symbol = st.text_input("Default Symbol (if missing)", value="PYRUSDT", key="default_symbol")
    anomaly_contamination = st.slider("Anomaly Detection Sensitivity", 0.01, 0.1, 0.02, step=0.01, key="anomaly_contamination")

    if uploaded_file:
        try:
            df = pd.read_excel(uploaded_file, sheet_name="List of trades")
            min_date = pd.to_datetime(df['Date/Time'], errors='coerce').min()
            max_date = pd.to_datetime(df['Date/Time'], errors='coerce').max()
            if pd.isna(min_date) or pd.isna(max_date):
                st.error("Invalid dates in Excel. Ensure 'Date/Time' column has valid dates.")
                st.stop()
            st.session_state['min_date'] = min_date.date()
            st.session_state['max_date'] = max_date.date()
            st.subheader("Time Filter")
            presets = ["Last Day", "Last Week", "Last Month", "Last Year", "Max"]
            selected_preset = st.selectbox("Quick Presets", presets, key="presets")
            if selected_preset == "Last Day":
                start_date = max_date - timedelta(days=1)
            elif selected_preset == "Last Week":
                start_date = max_date - timedelta(days=7)
            elif selected_preset == "Last Month":
                start_date = max_date - timedelta(days=30)
            elif selected_preset == "Last Year":
                start_date = max_date - timedelta(days=365)
            else:
                start_date = min_date
            end_date = max_date
            start_date = st.date_input("Start Date", start_date.date(), min_value=min_date.date(),
                                       max_value=max_date.date(), key="start_date")
            end_date = st.date_input("End Date", end_date.date(), min_value=min_date.date(), max_value=max_date.date(),
                                     key="end_date")
            if st.button("Reset to Default"):
                st.session_state['presets'] = "Max"
                st.session_state['risk_free_rate'] = 2.0
                st.session_state['default_symbol'] = "PYRUSDT"
                st.session_state['timeframe_hours'] = 0.88
                st.session_state['anomaly_contamination'] = 0.02
                st.session_state['start_date'] = min_date.date()
                st.session_state['end_date'] = max_date.date()
        except Exception as e:
            st.error(f"Failed to load Excel: {str(e)}. Ensure 'List of trades' sheet exists.")
            st.stop()

# Modular Functions
@st.cache_data
def excel_serial_to_datetime(value):
    if pd.isna(value):
        return pd.NaT
    if isinstance(value, datetime):
        return pd.Timestamp(value)
    try:
        serial = float(value)
        if serial < 0 or serial > 100000:
            raise ValueError("Invalid serial date")
        base_date = datetime(1899, 12, 30)
        delta = timedelta(days=serial)
        return pd.Timestamp(base_date + delta)
    except Exception:
        return pd.NaT

@st.cache_data
def process_trades(df, default_symbol, timeframe_hours):
    if pd.api.types.is_numeric_dtype(df['Date/Time']):
        df['datetime'] = df['Date/Time'].apply(excel_serial_to_datetime)
    else:
        df['datetime'] = pd.to_datetime(df['Date/Time'], errors='coerce')
    df = df.sort_values(['Trade #', 'datetime'])
    required_cols = ['Trade #', 'Type', 'Date/Time', 'Price USDT', 'Position size (qty)', 'Position size (value)',
                     'Net P&L USDT', 'Net P&L %', 'Run-up USDT', 'Drawdown USDT', 'Cumulative P&L USDT']
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        st.error(f"Missing columns: {', '.join(missing_cols)}. Ensure Excel has 'List of trades' sheet with required format.")
        st.info("Tip: Check for alternate names like 'Trade ID' or 'Time'. Contact zozquant@gmail.com.")
        return pd.DataFrame()
    if 'Symbol' not in df.columns:
        st.info(f"No Symbol column found. Using default: {default_symbol}. For multi-asset analysis, include Symbol.")
        df['Symbol'] = default_symbol
    trades = []
    skipped_trades = []
    for trade_num, group in df.groupby('Trade #'):
        try:
            entry_rows = group[group['Type'].str.lower().str.contains('entry', na=False)]
            exit_rows = group[group['Type'].str.lower().str.contains('exit', na=False)]
            if entry_rows.empty:
                raise ValueError("Missing entry row")
            entry = entry_rows.iloc[-1]
            symbol = entry['Symbol']
            entry_date = entry['datetime']
            entry_price = float(str(entry['Price USDT']).replace(',', ''))
            quantity = float(str(entry['Position size (qty)']).replace(',', ''))
            position_size = float(str(entry['Position size (value)']).replace(',', ''))
            mfe = float(str(entry.get('Run-up USDT', 0)).replace(',', ''))
            mae = abs(float(str(entry.get('Drawdown USDT', 0)).replace(',', '')))
            cumulative_pnl = float(str(group['Cumulative P&L USDT'].iloc[-1]).replace(',', ''))
            if pd.isna(entry_date):
                raise ValueError("Invalid entry date")
            if not exit_rows.empty:
                exit = exit_rows.iloc[0]
                exit_date = exit['datetime']
                if pd.isna(exit_date):
                    raise ValueError("Invalid exit date")
                exit_price = float(str(exit['Price USDT']).replace(',', ''))
                pnl = float(str(exit['Net P&L USDT']).replace(',', ''))
                return_pct = float(str(exit['Net P&L %']).replace(',', ''))
                duration_hours = (exit_date - entry_date).total_seconds() / 3600
                bars = duration_hours / timeframe_hours if timeframe_hours > 0 else 0
                trades.append({
                    'trade_num': trade_num, 'symbol': symbol, 'entry_date': entry_date, 'exit_date': exit_date,
                    'entry_price': entry_price, 'exit_price': exit_price, 'quantity': quantity,
                    'position_size': position_size, 'pnl': pnl, 'return_pct': return_pct, 'duration': duration_hours,
                    'bars': bars, 'is_open': False, 'mfe': mfe, 'mae': mae, 'cumulative_pnl': cumulative_pnl
                })
            else:
                trades.append({
                    'trade_num': trade_num, 'symbol': symbol, 'entry_date': entry_date,
                    'entry_price': entry_price, 'quantity': quantity, 'position_size': position_size,
                    'pnl': float(str(group['Net P&L USDT'].iloc[-1]).replace(',', '')), 'return_pct': 0.0,
                    'duration': 0.0, 'bars': 0.0, 'is_open': True, 'mfe': mfe, 'mae': mae,
                    'cumulative_pnl': cumulative_pnl
                })
        except Exception as e:
            skipped_trades.append((trade_num, str(e)))
    trades_df = pd.DataFrame(trades)
    if skipped_trades:
        skip_ratio = len(skipped_trades) / (len(skipped_trades) + len(trades)) if trades else 1.0
        st.warning(f"Skipped {len(skipped_trades)} trades: {skipped_trades[:5]}{'...' if len(skipped_trades) > 5 else ''}")
        if skip_ratio > 0.1:
            st.warning(f"High skip rate ({skip_ratio:.1%}). Check Excel for invalid dates or missing entries.")
    return trades_df

@st.cache_data
def compute_metrics(closed_df, initial_equity, risk_free_rate=0.0):
    """
    Compute portfolio performance metrics. Fixed open_pnl and drawdown calculations.
    """
    closed_trades = closed_df[~closed_df['is_open']].copy()
    if closed_trades.empty or len(closed_trades) < 2:
        return {key: "N/A" for key in ['net_profit', 'max_drawdown', 'sharpe_ratio', 'sortino_ratio', 'calmar_ratio',
                                       'var_95', 'cvar_95', 'max_runup', 'gross_profit', 'gross_loss', 'profit_factor',
                                       'total_trades', 'winning_trades', 'losing_trades', 'percent_profitable', 'avg_pnl',
                                       'avg_win', 'avg_loss', 'win_loss_ratio', 'avg_bars', 'avg_bars_win', 'avg_bars_loss',
                                       'largest_win', 'largest_loss', 'largest_win_pct', 'largest_loss_pct',
                                       'margin_calls', 'open_pnl', 'total_open_trades',
                                       'longest_drawdown', 'time_to_recovery', 'returns_skew', 'returns_kurtosis']}, pd.Series([initial_equity]), pd.Series(), pd.Series()

    closed_trades = closed_trades.dropna(
        subset=['pnl', 'return_pct', 'entry_price', 'exit_price', 'position_size', 'exit_date', 'entry_date', 'cumulative_pnl'])

    equity_curve = initial_equity + closed_trades['cumulative_pnl']
    equity_curve.index = closed_trades['exit_date']

    daily_equity = equity_curve.resample("D").last().ffill()
    daily_returns = daily_equity.pct_change().dropna()

    per_trade_returns = closed_trades['return_pct'] / 100.0
    log_returns = np.log1p(per_trade_returns)

    total_days = (closed_trades['exit_date'].max() - closed_trades['entry_date'].min()).days
    trades_per_year = len(closed_trades) / (total_days / 365.25) if total_days > 0 else 1
    annualization_factor_trades = np.sqrt(trades_per_year)

    mean_return = daily_returns.mean()
    std_return = daily_returns.std(ddof=0) or 1e-9
    downside_std = daily_returns[daily_returns < 0].std(ddof=0) or 1e-9
    sharpe_ratio = ((mean_return - risk_free_rate/252) / std_return) * np.sqrt(252) if std_return else "N/A"
    sortino_ratio = ((mean_return - risk_free_rate/252) / downside_std) * np.sqrt(252) if downside_std else "N/A"

    net_profit = closed_trades['pnl'].sum()
    gross_profit = closed_trades.loc[closed_trades['pnl'] > 0, 'pnl'].sum()
    gross_loss = abs(closed_trades.loc[closed_trades['pnl'] < 0, 'pnl'].sum())
    profit_factor = gross_profit / gross_loss if gross_loss > 0 else np.nan
    percent_profitable = (closed_trades['pnl'] > 0).mean() * 100
    avg_pnl = closed_trades['pnl'].mean()
    avg_win = closed_trades.loc[closed_trades['pnl'] > 0, 'pnl'].mean()
    avg_loss = abs(closed_trades.loc[closed_trades['pnl'] < 0, 'pnl'].mean())
    win_loss_ratio = avg_win / avg_loss if avg_loss > 0 else np.nan

    high_water = daily_equity.cummax()
    drawdowns = (high_water - daily_equity) / high_water
    max_drawdown = drawdowns.max() * 100
    drawdown_starts = (drawdowns > 0) & (drawdowns.shift() <= 0)
    drawdown_ends = (drawdowns <= 0) & (drawdowns.shift() > 0)
    longest_drawdown = 0
    time_to_recovery = 0
    if len(daily_equity) > 1:
        in_drawdown = False
        current_dd_days = 0
        current_recovery_days = 0
        for i in range(1, len(daily_equity)):
            if drawdowns.iloc[i] > 0:
                if not in_drawdown:
                    in_drawdown = True
                    current_dd_days = 1
                else:
                    current_dd_days += 1
                current_recovery_days = 0
            else:
                if in_drawdown:
                    in_drawdown = False
                    time_to_recovery = max(time_to_recovery, current_recovery_days)
                    current_recovery_days = 1
                else:
                    current_recovery_days += 1
            longest_drawdown = max(longest_drawdown, current_dd_days)
    max_runup = (daily_equity - daily_equity.cummin()).max()

    returns_skew = per_trade_returns.skew()
    returns_kurtosis = per_trade_returns.kurtosis()

    var_95 = norm.ppf(0.05, mean_return, std_return) * np.sqrt(252) * 100 if std_return else "N/A"
    cvar_95 = (mean_return - std_return * norm.pdf(norm.ppf(0.05)) / 0.05) * np.sqrt(252) * 100 if std_return else "N/A"

    annual_return = (equity_curve.iloc[-1] / initial_equity) ** (365.25/total_days) - 1 if total_days > 0 else np.nan
    calmar = annual_return / (max_drawdown/100) if max_drawdown > 0 else np.nan

    margin_calls = (daily_equity < 0.1 * initial_equity).sum() if len(daily_equity) > 1 else 0
    open_trades = closed_df[closed_df['is_open']]
    open_pnl = open_trades['pnl'].sum() if not open_trades.empty else 0.0
    total_open_trades = len(open_trades)

    metrics = {
        'net_profit': float(net_profit) if not np.isnan(net_profit) else "N/A",
        'max_drawdown': float(max_drawdown) if not np.isnan(max_drawdown) else "N/A",
        'sharpe_ratio': float(sharpe_ratio) if sharpe_ratio != "N/A" else "N/A",
        'sortino_ratio': float(sortino_ratio) if sortino_ratio != "N/A" else "N/A",
        'calmar_ratio': float(calmar) if not np.isnan(calmar) else "N/A",
        'var_95': float(var_95) if var_95 != "N/A" else "N/A",
        'cvar_95': float(cvar_95) if cvar_95 != "N/A" else "N/A",
        'max_runup': float(max_runup) if not np.isnan(max_runup) else "N/A",
        'gross_profit': float(gross_profit) if not np.isnan(gross_profit) else "N/A",
        'gross_loss': float(gross_loss) if not np.isnan(gross_loss) else "N/A",
        'profit_factor': float(profit_factor) if not np.isnan(profit_factor) else "N/A",
        'total_trades': int(len(closed_trades)),
        'winning_trades': int((closed_trades['pnl'] > 0).sum()),
        'losing_trades': int((closed_trades['pnl'] < 0).sum()),
        'percent_profitable': float(percent_profitable) if not np.isnan(percent_profitable) else "N/A",
        'avg_pnl': float(avg_pnl) if not np.isnan(avg_pnl) else "N/A",
        'avg_win': float(avg_win) if not np.isnan(avg_win) else "N/A",
        'avg_loss': float(avg_loss) if not np.isnan(avg_loss) else "N/A",
        'win_loss_ratio': float(win_loss_ratio) if not np.isnan(win_loss_ratio) else "N/A",
        'avg_bars': float(closed_trades['bars'].mean()) if not closed_trades['bars'].empty else "N/A",
        'avg_bars_win': float(closed_trades.loc[closed_trades['pnl'] > 0, 'bars'].mean()) if not closed_trades[closed_trades['pnl'] > 0].empty else "N/A",
        'avg_bars_loss': float(closed_trades.loc[closed_trades['pnl'] < 0, 'bars'].mean()) if not closed_trades[closed_trades['pnl'] < 0].empty else "N/A",
        'largest_win': float(closed_trades['pnl'].max()) if not closed_trades['pnl'].empty else "N/A",
        'largest_loss': float(abs(closed_trades['pnl'].min())) if not closed_trades['pnl'].empty else "N/A",
        'largest_win_pct': float(closed_trades['return_pct'].max()) if not closed_trades.empty else "N/A",
        'largest_loss_pct': float(abs(closed_trades['return_pct'].min())) if not closed_trades.empty else "N/A",
        'margin_calls': int(margin_calls),
        'open_pnl': float(open_pnl) if not np.isnan(open_pnl) else "N/A",
        'total_open_trades': int(total_open_trades),
        'longest_drawdown': int(longest_drawdown) if not np.isnan(longest_drawdown) else 0,
        'time_to_recovery': int(time_to_recovery) if not np.isnan(time_to_recovery) else 0,
        'returns_skew': float(returns_skew) if not np.isnan(returns_skew) else "N/A",
        'returns_kurtosis': float(returns_kurtosis) if not np.isnan(returns_kurtosis) else "N/A"
    }

    return metrics, equity_curve, log_returns, daily_returns

def monte_carlo_simulation(daily_returns, initial_equity, start_date, sim_type='t', n_simulations=100, horizon=252, df=5):
    """
    Simulate Monte Carlo equity paths with Student's t or historical bootstrap.
    Note: No caching to ensure fresh random simulations each time.
    """
    if daily_returns.empty or daily_returns.std() == 0:
        return np.array([initial_equity] * horizon * n_simulations).reshape(horizon, n_simulations)
    if sim_type == 't':
        mean = daily_returns.mean()
        std = daily_returns.std()
        simulations = t.rvs(df, loc=mean, scale=std, size=(horizon, n_simulations))
    else:  # Historical bootstrap
        simulations = np.random.choice(daily_returns, size=(horizon, n_simulations), replace=True)
    paths = initial_equity * np.exp(np.cumsum(simulations, axis=0))
    return paths

@st.cache_data
def detect_anomalies(closed_df, contamination=0.02):
    closed_only = closed_df[~closed_df['is_open']].copy()
    if len(closed_only) < 10:
        return pd.DataFrame()
    features = closed_only[['pnl', 'bars', 'position_size', 'mfe', 'mae']].dropna()
    if len(features) < 10:
        return pd.DataFrame()
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features)
    iso = IsolationForest(contamination=contamination, random_state=42)
    anomalies = iso.fit_predict(features_scaled)
    result = closed_only.iloc[features.index][anomalies == -1].copy()
    result['anomaly_type'] = result.apply(
        lambda x: 'Lucky Win' if x['pnl'] > 0 and x['mae'] > 0.8 * abs(x['pnl']) else 'Poor Exit' if x['pnl'] > 0 and x['mfe'] > 1.5 * x['pnl'] else 'Outlier', axis=1)
    return result

def generate_ai_summary(anomalies, metrics):
    summary = []
    if not anomalies.empty:
        lucky_wins = len(anomalies[anomalies['anomaly_type'] == 'Lucky Win'])
        poor_exits = len(anomalies[anomalies['anomaly_type'] == 'Poor Exit'])
        if lucky_wins > 0:
            summary.append(f"Warning: {lucky_wins} trades flagged as 'Lucky Wins' with MAE > 80% of PnL. Refine entry timing.")
        if poor_exits > 0:
            summary.append(f"Caution: {poor_exits} trades flagged as 'Poor Exits' with MFE > 1.5x PnL. Optimize exits.")
    if isinstance(metrics.get('max_drawdown'), (int, float)) and metrics['max_drawdown'] > 30:
        summary.append(f"High Risk: Max drawdown ({metrics['max_drawdown']:.2f}%) exceeds 30%. Reduce position sizes.")
    if isinstance(metrics.get('returns_skew'), (int, float)) and metrics['returns_skew'] < -0.5:
        summary.append("Negative Skew: Frequent large losses. Review risk management.")
    if isinstance(metrics.get('returns_kurtosis'), (int, float)) and metrics['returns_kurtosis'] > 3:
        summary.append(f"Fat Tails: High kurtosis ({metrics['returns_kurtosis']:.2f}). Use robust risk models.")
    return "\n".join(summary) if summary else "No critical issues detected."

@st.cache_data
def compute_correlation(filtered_df):
    if len(filtered_df['symbol'].unique()) < 2:
        return None
    pivot = filtered_df[~filtered_df['is_open']].pivot(index='exit_date', columns='symbol', values='pnl').dropna()
    if pivot.shape[0] < 2:
        return None
    corr = pivot.corr()
    return corr

@st.cache_data
def optimize_allocation(returns_dict):
    symbols = list(returns_dict.keys())
    if len(symbols) < 2:
        return {symbols[0]: 1.0} if symbols else {}
    valid_returns = {sym: returns for sym, returns in returns_dict.items() if len(returns.dropna()) >= 2}
    if len(valid_returns) < 2:
        return {sym: 1 / len(symbols) for sym in symbols}
    symbols = list(valid_returns.keys())
    returns_df = pd.DataFrame({sym: valid_returns[sym] for sym in symbols}).T.dropna()
    if returns_df.shape[0] < 2:
        return {sym: 1 / len(symbols) for sym in symbols}
    cov_matrix = returns_df.cov()

    def objective(weights):
        port_ret = np.sum([weights[i] * valid_returns[sym].mean() for i, sym in enumerate(symbols)])
        port_std = np.sqrt(np.dot(weights.T, np.dot(cov_matrix * 252, weights)))
        return -port_ret / port_std if port_std else np.inf

    constraints = {'type': 'eq', 'fun': lambda w: np.sum(w) - 1}
    bounds = [(0, 1) for _ in symbols]
    initial_weights = np.array([1 / len(symbols)] * len(symbols))
    result = minimize(objective, initial_weights, method='SLSQP', bounds=bounds, constraints=constraints)
    if not result.success:
        st.warning("Portfolio optimization failed. Using equal weights.")
        return {sym: 1 / len(symbols) for sym in symbols}
    return dict(zip(symbols, result.x))

@st.cache_data
def hierarchical_risk_parity(daily_returns, symbols):
    if len(symbols) < 2 or (isinstance(daily_returns, pd.Series) and daily_returns.shape[0] < 2) or (isinstance(daily_returns, pd.DataFrame) and daily_returns.shape[1] < 2):
        return {sym: 1.0 / len(symbols) for sym in symbols}
    if isinstance(daily_returns, pd.Series):
        daily_returns = pd.DataFrame({symbols[0]: daily_returns})
    cov = daily_returns.cov() * 252
    corr = daily_returns.corr()
    distance = np.sqrt((1 - corr) / 2)
    linkage = sch.linkage(distance, method='ward')

    def get_rec_bipart(cov, sort_idx):
        if len(sort_idx) <= 1:
            return np.ones(1) / len(sort_idx)
        weights = np.zeros(len(sort_idx))
        left_cluster = sort_idx[:len(sort_idx) // 2]
        right_cluster = sort_idx[len(sort_idx) // 2:]
        left_var = np.sqrt(np.dot(np.ones(len(left_cluster)), np.dot(cov.iloc[left_cluster, left_cluster], np.ones(len(left_cluster)))))
        right_var = np.sqrt(np.dot(np.ones(len(right_cluster)), np.dot(cov.iloc[right_cluster, right_cluster], np.ones(len(right_cluster)))))
        alloc_factor = 1 - left_var / (left_var + right_var) if (left_var + right_var) else 0.5
        weights[left_cluster] = alloc_factor * get_rec_bipart(cov, left_cluster)
        weights[right_cluster] = (1 - alloc_factor) * get_rec_bipart(cov, right_cluster)
        return weights

    sorted_idx = sch.leaves_list(linkage)
    weights = get_rec_bipart(cov, sorted_idx)
    return dict(zip(daily_returns.columns, weights))

# Main Logic
if uploaded_file:
    processed_df = process_trades(df, default_symbol, timeframe_hours)
    st.session_state['processed_df'] = processed_df
    st.session_state['filter_date'] = None
    if not processed_df.empty:
        closed = processed_df[~processed_df['is_open']]
        opens = processed_df[processed_df['is_open'] & (processed_df['entry_date'].dt.date >= start_date)]
        filtered_df = pd.concat([closed[(closed['entry_date'].dt.date >= start_date) & (closed['exit_date'].dt.date <= end_date)], opens])
        if filtered_df.empty:
            st.warning("No trades in selected date range. Try expanding dates or uploading a larger dataset.")
        else:
            symbols = filtered_df['symbol'].unique()
            symbol_returns = {sym: filtered_df[filtered_df['symbol'] == sym]['pnl'] / filtered_df[filtered_df['symbol'] == sym]['position_size'] for sym in symbols if not filtered_df[filtered_df['symbol'] == sym].empty}
            portfolio_metrics, equity, log_returns, daily_returns = compute_metrics(filtered_df, initial_equity, risk_free_rate)
            corr_matrix = compute_correlation(filtered_df)
            sharpe_alloc = optimize_allocation(symbol_returns)
            hrp_alloc = hierarchical_risk_parity(daily_returns, symbols)

            # Tabs
            tab_overview, tab_trades, tab_performance, tab_risk, tab_visuals, tab_ai = st.tabs(
                ["Overview", "Trades", "Performance", "Risk", "Visuals", "Advanced Analytics"])

            with tab_overview:
                st.header("Portfolio Snapshot")
                st.markdown("**Key Metrics** (hover for details)")
                tooltips = {
                    'net_profit': 'Total profit/loss after commissions',
                    'max_drawdown': 'Largest peak-to-trough decline in equity (%)',
                    'sharpe_ratio': 'Risk-adjusted return (daily, annualized)',
                    'sortino_ratio': 'Risk-adjusted return (downside risk only, daily)',
                    'calmar_ratio': 'Annualized return divided by max drawdown',
                    'var_95': 'Value at Risk at 95% confidence (daily, annualized %)',
                    'cvar_95': 'Conditional VaR at 95% confidence (daily, annualized %)',
                    'max_runup': 'Largest peak-to-valley increase in equity',
                    'gross_profit': 'Total profit from winning trades',
                    'gross_loss': 'Total loss from losing trades',
                    'profit_factor': 'Gross profit divided by gross loss',
                    'total_trades': 'Total number of closed trades',
                    'winning_trades': 'Number of profitable trades',
                    'losing_trades': 'Number of losing trades',
                    'percent_profitable': 'Percentage of profitable trades (%)',
                    'avg_pnl': 'Average profit/loss per trade',
                    'avg_win': 'Average profit of winning trades',
                    'avg_loss': 'Average loss of losing trades',
                    'win_loss_ratio': 'Average win divided by average loss',
                    'avg_bars': 'Average number of bars per trade',
                    'avg_bars_win': 'Average bars in winning trades',
                    'avg_bars_loss': 'Average bars in losing trades',
                    'largest_win': 'Largest profit from a single trade',
                    'largest_loss': 'Largest loss from a single trade',
                    'largest_win_pct': 'Largest profit as a percentage',
                    'largest_loss_pct': 'Largest loss as a percentage',
                    'margin_calls': 'Times equity fell below 10% of initial equity',
                    'open_pnl': 'Unrealized profit/loss of open trades',
                    'total_open_trades': 'Number of open trades',
                    'longest_drawdown': 'Longest drawdown period (days)',
                    'time_to_recovery': 'Longest recovery time (days)',
                    'returns_skew': 'Skewness of trade returns',
                    'returns_kurtosis': 'Kurtosis of trade returns'
                }
                metrics_list = list(portfolio_metrics.items())
                for i in range(0, len(metrics_list), 3):
                    cols = st.columns([1, 1, 1], gap="small")
                    for j, (key, value) in enumerate(metrics_list[i:i + 3]):
                        with cols[j]:
                            if isinstance(value, (int, float)):
                                if key in ['var_95', 'cvar_95', 'max_drawdown', 'percent_profitable', 'largest_win_pct', 'largest_loss_pct', 'returns_skew', 'returns_kurtosis']:
                                    display_value = f"{value:.2f}%"
                                elif key in ['net_profit', 'max_runup', 'gross_profit', 'gross_loss', 'avg_pnl', 'avg_win', 'avg_loss', 'largest_win', 'largest_loss', 'open_pnl']:
                                    display_value = f"{value:,.2f}"
                                else:
                                    display_value = f"{value:.2f}"
                            else:
                                display_value = str(value)
                            st.markdown(
                                f'<div class="metric-card" title="{tooltips.get(key, "")}"><div class="metric-title">{key.replace("_", " ").title()}</div><div class="metric-value">{display_value}</div></div>',
                                unsafe_allow_html=True)

            with tab_trades:
                st.dataframe(filtered_df[['trade_num', 'symbol', 'entry_date', 'exit_date', 'pnl', 'bars', 'mfe', 'mae', 'is_open']])

            with tab_performance:
                try:
                    fig_pnl = px.bar(filtered_df, x='trade_num', y='pnl', color='pnl', title='PnL per Trade',
                                     color_continuous_scale='RdYlGn')
                    fig_pnl.update_layout(plot_bgcolor='#0A0F1A', paper_bgcolor='#0A0F1A', font=dict(color='#E0E7FF'))
                    st.plotly_chart(fig_pnl, use_container_width=True)
                    fig_bars = px.histogram(filtered_df[~filtered_df['is_open']], x='bars', title='Trade Duration Distribution (Bars)', nbins=50)
                    fig_bars.update_layout(plot_bgcolor='#0A0F1A', paper_bgcolor='#0A0F1A', font=dict(color='#E0E7FF'))
                    st.plotly_chart(fig_bars, use_container_width=True)
                except Exception as e:
                    st.warning(f"Failed to render performance charts: {str(e)}")

            with tab_risk:
                try:
                    if corr_matrix is not None and not corr_matrix.empty:
                        fig_corr = go.Figure(
                            data=go.Heatmap(z=corr_matrix.values, x=corr_matrix.columns, y=corr_matrix.index,
                                            colorscale='Plasma', text=corr_matrix.values.round(2), texttemplate="%{text}"))
                        fig_corr.update_layout(title='Correlation Matrix', plot_bgcolor='#0A0F1A', paper_bgcolor='#0A0F1A',
                                               font=dict(color='#E0E7FF'))
                        st.plotly_chart(fig_corr, use_container_width=True)
                    col1, col2 = st.columns(2)
                    with col1:
                        st.markdown(
                            f'<div class="metric-card" title="Value at Risk at 95% confidence (daily, annualized %)"><div class="metric-title">Value at Risk (95%)</div><div class="metric-value">{portfolio_metrics.get("var_95", "N/A"):.2f}%</div></div>',
                            unsafe_allow_html=True)
                    with col2:
                        st.markdown(
                            f'<div class="metric-card" title="Conditional VaR at 95% confidence (daily, annualized %)"><div class="metric-title">Conditional VaR (95%)</div><div class="metric-value">{portfolio_metrics.get("cvar_95", "N/A"):.2f}%</div></div>',
                            unsafe_allow_html=True)
                except Exception as e:
                    st.warning(f"Failed to render risk charts: {str(e)}")

            with tab_visuals:
                try:
                    fig_equity = go.Figure(
                        go.Scatter(x=equity.index, y=equity, mode='lines', line=dict(color='#536DFE'), name='Equity Curve'))
                    fig_equity.update_layout(title='Equity Curve', xaxis_title='Date', yaxis_title='Equity ($)',
                                             plot_bgcolor='#0A0F1A', paper_bgcolor='#0A0F1A', font=dict(color='#E0E7FF'))
                    st.plotly_chart(fig_equity, use_container_width=True)

                    underwater = (equity.cummax() - equity) / equity.cummax() * 100
                    fig_underwater = go.Figure(
                        go.Scatter(x=equity.index, y=-underwater, mode='lines', fill='tozeroy', line=dict(color='#FF6B6B')))
                    fig_underwater.update_layout(title='Underwater Plot (Drawdown)', xaxis_title='Date',
                                                 yaxis_title='Drawdown (%)', plot_bgcolor='#0A0F1A',
                                                 paper_bgcolor='#0A0F1A', font=dict(color='#E0E7FF'))
                    st.plotly_chart(fig_underwater, use_container_width=True)

                    rolling_window = 30
                    if not daily_returns.empty:
                        rolling_returns = daily_returns.rolling(window=rolling_window).mean() * 252
                        rolling_std = daily_returns.rolling(window=rolling_window).std() * np.sqrt(252)
                        rolling_sharpe = (rolling_returns - risk_free_rate) / rolling_std
                        fig_rolling = go.Figure(
                            go.Scatter(x=rolling_sharpe.index, y=rolling_sharpe, mode='lines',
                                       line=dict(color='#536DFE')))
                        fig_rolling.update_layout(title=f'Rolling {rolling_window}-Day Sharpe Ratio', xaxis_title='Date',
                                                  yaxis_title='Sharpe Ratio', plot_bgcolor='#0A0F1A',
                                                  paper_bgcolor='#0A0F1A', font=dict(color='#E0E7FF'))
                        st.plotly_chart(fig_rolling, use_container_width=True)

                        if len(symbols) > 1:
                            monthly_returns = daily_returns.resample('ME').apply(lambda x: x.mean() * 21)
                            fig_heatmap = go.Figure(data=go.Heatmap(z=monthly_returns.values.T, x=monthly_returns.index.strftime('%Y-%m'), y=monthly_returns.columns,
                                                                    colorscale='RdYlGn', text=[[f"{val:.2f}%" for val in row] for row in monthly_returns.values.T],
                                                                    texttemplate="%{text}", textfont={"size":10}))
                        else:
                            monthly_returns = daily_returns.resample('ME').sum()
                            monthly_returns.index = monthly_returns.index.strftime('%Y-%m')
                            fig_heatmap = go.Figure(
                                data=go.Heatmap(z=monthly_returns.values.reshape(-1, 1), x=[symbols[0]], y=monthly_returns.index,
                                                colorscale='RdYlGn', text=monthly_returns.values.round(2).reshape(-1, 1),
                                                texttemplate="%{text}%"))
                        fig_heatmap.update_layout(title='Monthly Returns Heatmap', xaxis_title='Year-Month',
                                                  yaxis_title='Symbol', plot_bgcolor='#0A0F1A', paper_bgcolor='#0A0F1A',
                                                  font=dict(color='#E0E7FF'))
                        st.plotly_chart(fig_heatmap, use_container_width=True)

                    fig_mfe_mae = px.scatter(filtered_df, x='mfe', y='mae', color='pnl', size='position_size',
                                             hover_data=['trade_num'], title='MFE vs. MAE by Trade')
                    fig_mfe_mae.update_layout(plot_bgcolor='#0A0F1A', paper_bgcolor='#0A0F1A', font=dict(color='#E0E7FF'))
                    st.plotly_chart(fig_mfe_mae, use_container_width=True)
                except Exception as e:
                    st.warning(f"Failed to render visuals: {str(e)}")

            with tab_ai:
                st.header("Advanced Analytics")
                anomalies = detect_anomalies(filtered_df, anomaly_contamination)
                ai_summary = generate_ai_summary(anomalies, portfolio_metrics)
                st.markdown(f"**AI Insights Summary**\n\n{ai_summary}", unsafe_allow_html=True)

                with st.expander("Monte Carlo Simulations"):
                    sim_type = st.selectbox("Simulation Type", ["Student's t", "Historical"], key="sim_type")
                    n_simulations = st.slider("Number of Simulations", 50, 1000, 100, step=50, key="n_simulations")
                    if 'monte_carlo_seed' not in st.session_state:
                        st.session_state['monte_carlo_seed'] = np.random.randint(0, 1000000)
                    if st.button("Generate New Simulation"):
                        st.session_state['monte_carlo_seed'] = np.random.randint(0, 1000000)
                    plot_placeholder = st.empty()
                    try:
                        np.random.seed(st.session_state['monte_carlo_seed'])
                        paths = monte_carlo_simulation(daily_returns, equity.iloc[-1] if len(equity) > 0 else initial_equity, filtered_df['exit_date'].max(),
                                                       sim_type.lower().replace("'s t", "t"), n_simulations=n_simulations)
                        fig_mc = go.Figure()
                        dates = pd.date_range(start=filtered_df['exit_date'].max(), periods=252, freq='B')
                        # Plot up to n_simulations paths (default 100 for performance)
                        for path in paths.T[:n_simulations]:
                            fig_mc.add_trace(go.Scatter(x=dates, y=path, mode='lines',
                                                        line=dict(width=0.5, color='rgba(83, 109, 254, 0.3)')))
                        median_path = np.median(paths, axis=1)
                        fig_mc.add_trace(go.Scatter(x=dates, y=median_path, mode='lines', name='Median',
                                                    line=dict(width=2, color='#FF6B6B')))
                        fig_mc.update_layout(title=f'Monte Carlo Equity Paths ({sim_type}, 252 Days)', xaxis_title='Date',
                                             yaxis_title='Equity ($)', plot_bgcolor='#0A0F1A', paper_bgcolor='#0A0F1A',
                                             font=dict(color='#E0E7FF'), height=600, autosize=True)
                        with plot_placeholder:
                            st.plotly_chart(fig_mc, use_container_width=True, config={'responsive': True})
                    except Exception as e:
                        st.warning(f"Failed to render Monte Carlo simulation: {str(e)}")

                with st.expander("Anomaly Detection"):
                    if not anomalies.empty:
                        st.dataframe(anomalies[['trade_num', 'symbol', 'pnl', 'bars', 'mfe', 'mae', 'anomaly_type']])
                    else:
                        st.info("No anomalies detected with current sensitivity.")

                with st.expander("Portfolio Allocation"):
                    st.write("**Sharpe-Optimized Allocation:**", sharpe_alloc)
                    st.write("**Hierarchical Risk Parity Allocation:**", hrp_alloc)

                st.subheader("Export Reports")
                try:
                    buffer = io.BytesIO()
                    doc = SimpleDocTemplate(buffer, pagesize=letter)
                    styles = getSampleStyleSheet()
                    elements = [Paragraph("Ahmed's Crypto Holdings Report", styles['Title']),
                               Paragraph("AI Insights Summary", styles['Heading2']),
                               Paragraph(ai_summary.replace('\n', '<br />'), styles['Normal']),
                               Paragraph("<br />", styles['Normal'])]
                    data = [['Metric', 'Value']] + [[key.replace('_', ' ').title(),
                                                    f"{value:.2f}%" if isinstance(value, (int, float)) and key in ['var_95', 'cvar_95', 'max_drawdown',
                                                                                                                 'percent_profitable', 'largest_win_pct',
                                                                                                                 'largest_loss_pct', 'returns_skew', 'returns_kurtosis']
                                                    else f"{value:,.2f}" if isinstance(value, (int, float)) and key in ['net_profit', 'max_runup', 'gross_profit', 'gross_loss', 'avg_pnl', 'avg_win', 'avg_loss', 'largest_win', 'largest_loss', 'open_pnl']
                                                    else str(value)]
                                                   for key, value in portfolio_metrics.items()]
                    table = Table(data)
                    table.setStyle(TableStyle([
                        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#1A237E')),
                        ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
                        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                        ('FONTSIZE', (0, 0), (-1, 0), 14),
                        ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
                        ('BACKGROUND', (0, 1), (-1, -1), colors.HexColor('#303F9F')),
                        ('TEXTCOLOR', (0, 1), (-1, -1), colors.white),
                        ('FONTNAME', (0, 1), (-1, -1), 'Helvetica'),
                        ('FONTSIZE', (0, 1), (-1, -1), 12),
                        ('GRID', (0, 0), (-1, -1), 1, colors.white)
                    ]))
                    elements.append(table)
                    doc.build(elements)
                    st.download_button("Download PDF Report", buffer.getvalue(), "portfolio_report.pdf",
                                      mime="application/pdf")
                    st.download_button("Export Trades CSV", filtered_df.to_csv(index=False).encode('utf-8'), "trades.csv", "text/csv")

                    buffer = io.BytesIO()
                    with pd.ExcelWriter(buffer, engine='openpyxl') as writer:
                        metrics_df = pd.DataFrame(list(portfolio_metrics.items()), columns=['Metric', 'Value'])
                        metrics_df.to_excel(writer, sheet_name='Metrics', index=False)
                        filtered_df.to_excel(writer, sheet_name='Trades', index=False)
                        if not anomalies.empty:
                            anomalies.to_excel(writer, sheet_name='Anomalies', index=False)
                        alloc_df = pd.DataFrame({
                            'Type': ['Sharpe-Optimized', 'Hierarchical Risk Parity'],
                            'Allocation': [str(sharpe_alloc), str(hrp_alloc)]
                        })
                        alloc_df.to_excel(writer, sheet_name='Allocations', index=False)
                        summary_df = pd.DataFrame({'AI Insights': [ai_summary]})
                        summary_df.to_excel(writer, sheet_name='AI_Summary', index=False)
                    st.download_button("Download Excel Report", buffer.getvalue(), "portfolio_report.xlsx",
                                      mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
                except Exception as e:
                    st.warning(f"Failed to generate report: {str(e)}")
else:
    st.info("Upload a TradingView Excel file to activate the dashboard. Sample format: trades.xlsx with 'List of trades' sheet.")
