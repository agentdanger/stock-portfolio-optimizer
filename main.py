from flask import Flask, request, jsonify, make_response
from google.cloud import storage

import requests
from io import BytesIO

import yfinance as yf

import pandas as pd
import numpy as np

import scipy.optimize as sco
import scipy.interpolate as sci
from sklearn.covariance import LedoitWolf

import json

from datetime import datetime, date
import math

app = Flask(__name__)

def normalize_number(value):
    if value is None:
        return None
    if isinstance(value, (np.floating, np.integer)):
        value = value.item()
    if isinstance(value, (float, int)):
        return value if math.isfinite(value) else None
    return value


def sanitize_for_json(payload):
    if isinstance(payload, dict):
        return {key: sanitize_for_json(value) for key, value in payload.items()}
    if isinstance(payload, list):
        return [sanitize_for_json(value) for value in payload]
    return normalize_number(payload)


def safe_dividend_yield(info):
    try:
        dy = info["dividendYield"]
        if dy is None:
            return None
        return float(dy)
    except KeyError:
        return None
    except (TypeError, ValueError):
        return None


def filter_universe_by_dividend_yield(tickers, max_dividend_yield=0.5, keep_if_missing=True):
    info_map = {}
    dividend_yield_map = {}
    kept, removed = [], []

    for ticker in tickers:
        try:
            info = yf.Ticker(ticker).info or {}
        except Exception:
            info = {}

        info_map[ticker] = info
        dy = safe_dividend_yield(info)
        dividend_yield_map[ticker] = dy

        if dy is None:
            (kept if keep_if_missing else removed).append(ticker)
        elif dy <= max_dividend_yield:
            kept.append(ticker)
        else:
            removed.append(ticker)

    return kept, removed, dividend_yield_map, info_map


def load_universe_from_gcs(bucket_name, blob_name, fallback):
    try:
        storage_client = storage.Client()
        blob = storage_client.bucket(bucket_name).blob(blob_name)
        raw = blob.download_as_string()
        data = json.loads(raw)
        return data if isinstance(data, list) and len(data) > 0 else fallback
    except Exception:
        return fallback


def calculate_max_drawdown(values):
    """Calculate maximum drawdown from a series of portfolio values."""
    if len(values) < 2:
        return 0.0
    values = np.array(values)
    peak = np.maximum.accumulate(values)
    drawdown = (values - peak) / peak
    return float(np.min(drawdown))


def calculate_sharpe_ratio(returns, risk_free_rate=0.0):
    """Calculate annualized Sharpe ratio from daily returns."""
    if len(returns) < 2:
        return 0.0
    returns = np.array(returns)
    excess_returns = returns - risk_free_rate / 252
    if np.std(excess_returns) == 0:
        return 0.0
    return float(np.mean(excess_returns) / np.std(excess_returns) * np.sqrt(252))


def get_quarterly_rebalance_dates(start_date, end_date):
    """Generate list of quarter-end rebalance dates between start and end."""
    rebalance_dates = []
    current = pd.Timestamp(start_date)
    end = pd.Timestamp(end_date)

    # Move to first quarter end after start
    quarter_ends = {3: 31, 6: 30, 9: 30, 12: 31}

    while current <= end:
        month = current.month
        # Find next quarter end
        if month <= 3:
            next_qe = pd.Timestamp(year=current.year, month=3, day=31)
        elif month <= 6:
            next_qe = pd.Timestamp(year=current.year, month=6, day=30)
        elif month <= 9:
            next_qe = pd.Timestamp(year=current.year, month=9, day=30)
        else:
            next_qe = pd.Timestamp(year=current.year, month=12, day=31)

        if next_qe >= current and next_qe <= end:
            rebalance_dates.append(next_qe)

        # Move to next quarter
        if next_qe.month == 12:
            current = pd.Timestamp(year=next_qe.year + 1, month=1, day=1)
        else:
            current = pd.Timestamp(year=next_qe.year, month=next_qe.month + 1, day=1)

    return rebalance_dates


def run_backtest(tickers, price_data, benchmark_data, start_date, end_date, lookback_days=252):
    """
    Run walk-forward backtest simulation.

    Args:
        tickers: List of stock tickers to include
        price_data: DataFrame with price data for all tickers
        benchmark_data: Series with benchmark (S&P 500) prices
        start_date: Backtest start date
        end_date: Backtest end date
        lookback_days: Days of history for each optimization

    Returns:
        Dict with portfolio values, benchmark values, and metrics
    """
    initial_value = 100000

    # Align price data and benchmark to common dates
    common_dates = price_data.index.intersection(benchmark_data.index)
    price_data = price_data.loc[common_dates]
    benchmark_data = benchmark_data.loc[common_dates]

    # Filter to backtest period
    mask = (price_data.index >= pd.Timestamp(start_date)) & (price_data.index <= pd.Timestamp(end_date))
    bt_prices = price_data.loc[mask]
    bt_benchmark = benchmark_data.loc[mask]

    if len(bt_prices) < lookback_days + 20:
        return None  # Not enough data

    # Get rebalance dates
    rebalance_dates = get_quarterly_rebalance_dates(start_date, end_date)

    # Filter rebalance dates to those present in our data (or nearest prior date)
    valid_rebalance_dates = []
    for rd in rebalance_dates:
        # Find nearest date in data that is <= rd
        available = bt_prices.index[bt_prices.index <= rd]
        if len(available) > 0:
            valid_rebalance_dates.append(available[-1])

    # Remove duplicates and sort
    valid_rebalance_dates = sorted(list(set(valid_rebalance_dates)))

    if len(valid_rebalance_dates) < 2:
        return None  # Not enough rebalance periods

    # Initialize tracking
    portfolio_values = []
    benchmark_values = []
    quarterly_values = []

    current_weights = np.array([1.0 / len(tickers)] * len(tickers))  # Start equal weight
    portfolio_value = initial_value
    benchmark_value = initial_value

    # Track benchmark starting price
    benchmark_start_price = bt_benchmark.iloc[0]

    # Walk through each day
    all_dates = bt_prices.index.tolist()
    prev_prices = None
    rebalance_idx = 0

    for i, current_date in enumerate(all_dates):
        current_prices = bt_prices.loc[current_date].values
        current_benchmark_price = bt_benchmark.loc[current_date]

        # Check if we need to rebalance
        if rebalance_idx < len(valid_rebalance_dates) and current_date >= valid_rebalance_dates[rebalance_idx]:
            # Get training data (lookback_days before this date)
            train_end_idx = i
            train_start_idx = max(0, train_end_idx - lookback_days)

            if train_end_idx - train_start_idx >= 60:  # Need at least 60 days of data
                train_prices = bt_prices.iloc[train_start_idx:train_end_idx]
                train_returns = train_prices.pct_change().dropna()

                if len(train_returns) >= 30:
                    # Run optimization on training data
                    try:
                        mean_returns = train_returns.mean() * 252
                        lw_bt = LedoitWolf().fit(train_returns.values)
                        cov_matrix = lw_bt.covariance_ * 252

                        def neg_sharpe(w):
                            port_ret = np.sum(mean_returns * w)
                            port_vol = np.sqrt(w.T @ cov_matrix @ w)
                            if port_vol == 0:
                                return 0
                            return -port_ret / port_vol

                        constraints = {'type': 'eq', 'fun': lambda x: np.sum(x) - 1}
                        bounds = tuple((0, 1) for _ in range(len(tickers)))
                        init_weights = np.array([1.0 / len(tickers)] * len(tickers))

                        result = sco.minimize(
                            neg_sharpe,
                            init_weights,
                            method='SLSQP',
                            bounds=bounds,
                            constraints=constraints
                        )

                        if result.success:
                            current_weights = result.x
                    except Exception:
                        pass  # Keep previous weights on optimization failure

            rebalance_idx += 1

        # Calculate daily returns and update portfolio value
        if prev_prices is not None and i > 0:
            daily_returns = (current_prices - prev_prices) / prev_prices
            daily_returns = np.nan_to_num(daily_returns, nan=0.0)
            portfolio_return = np.sum(current_weights * daily_returns)
            portfolio_value = portfolio_value * (1 + portfolio_return)

        # Update benchmark value (buy and hold)
        benchmark_value = initial_value * (current_benchmark_price / benchmark_start_price)

        portfolio_values.append(portfolio_value)
        benchmark_values.append(benchmark_value)

        # Record quarterly values
        if current_date in valid_rebalance_dates:
            quarter = (current_date.month - 1) // 3 + 1
            quarterly_values.append({
                "date": f"{current_date.year}-Q{quarter}",
                "portfolio": round(portfolio_value, 2),
                "benchmark": round(benchmark_value, 2)
            })

        prev_prices = current_prices

    # Calculate metrics
    portfolio_returns_daily = np.diff(portfolio_values) / portfolio_values[:-1]
    benchmark_returns_daily = np.diff(benchmark_values) / benchmark_values[:-1]

    portfolio_total_return = (portfolio_values[-1] - initial_value) / initial_value
    benchmark_total_return = (benchmark_values[-1] - initial_value) / initial_value

    # Calculate annualized returns
    num_years = len(all_dates) / 252  # Trading days per year
    if num_years > 0:
        portfolio_annualized_return = (1 + portfolio_total_return) ** (1 / num_years) - 1
        benchmark_annualized_return = (1 + benchmark_total_return) ** (1 / num_years) - 1
    else:
        portfolio_annualized_return = 0.0
        benchmark_annualized_return = 0.0

    return {
        "period": f"{start_date} to {end_date}",
        "portfolio_annualized_return": round(portfolio_annualized_return, 4),
        "benchmark_annualized_return": round(benchmark_annualized_return, 4),
        "outperformance": round(portfolio_annualized_return - benchmark_annualized_return, 4),
        "portfolio_sharpe": round(calculate_sharpe_ratio(portfolio_returns_daily), 2),
        "benchmark_sharpe": round(calculate_sharpe_ratio(benchmark_returns_daily), 2),
        "portfolio_max_drawdown": round(calculate_max_drawdown(portfolio_values), 4),
        "benchmark_max_drawdown": round(calculate_max_drawdown(benchmark_values), 4),
        "quarterly_values": quarterly_values
    }


@app.route('/optimize', methods=['GET'])
def optimize():
    # define stock universe and earliest date to start from
    stock_universe = [
        'AAPL', 'ABBV', 'ABT', 'ACN', 'ADI', 'ADP', 'AEE', 'AEP', 'AFL', 'ALL', 'AMD', 'AME', 'AMT', 'AMZN', 'APH', 'ATO', 'AVGO',
        'AWK', 'AXP', 'BA', 'BAC', 'BCE', 'BDX', 'BLK', 'BP', 'BRK-B', 'C', 'CAE', 'CARR', 'CB', 'CHD', 'CI', 'CL', 'CMCSA', 'CMI', 'CNP', 'COP', 'COST', 
        'CP', 'CRM', 'CSCO', 'CSX', 'CTVA', 'CVX', 'DCI', 'DE', 'DG', 'DHR', 'DIS', 'DLR', 'DTE', 'DUK', 'ECL', 'EL', 'ELV', 'EMR', 'ENB', 'EQR', 'EVRG', 'EXC', 
        'F', 'FDX', 'FISV', 'FMC', 'FTNT', 'GD', 'GIB', 'GIS', 'GM', 'GOOGL', 'GS', 'HD', 'HLN', 'HON', 'HPQ', 'IBM', 
        'INTC', 'INTU', 'J', 'JNJ', 'JPM', 'KEYS', 'KMB', 'KO', 'LIN', 'LLY', 'LMT', 'LNT', 'LOW', 'LUV', 'MA', 'MCO',
        'MDT', 'MDU', 'META', 'MFC', 'MMM', 'MRK', 'MSCI', 'MSFT', 'NDAQ', 'NEE', 'NI', 'NKE', 'NOW', 'NTR', 'NVDA', 'O', 'OKE', 'ORCL', 'ORLY', 'PEP', 'PFE', 'PH',
        'PLD', 'PSA', 'QCOM', 'RF', 'ROP', 'ROST', 'RTX', 'SBUX', 'SHEL', 'SHW', 'SO', 'SPGI', 'STT', 'SU', 'SWK', 'SWX', 'SYK',
        'T', 'TFC', 'TGT', 'TJX', 'TMO', 'TRMB', 'TRP', 'TRV', 'TTE', 'ULTA', 'UNH', 'UNP', 'UPS', 'V', 'VFC', 'VZ', 
        'WMT', 'WTRG', 'WWD', 'YUM', 'ZBH', 'ZTS',"ADBE", "ANET", "ABNB", "AZO", "CMG", "CRWD", "ISRG", "LULU",
        "NFLX", "PANW", "SNPS", "TSLA", "UBER", "VRTX", "REGN", "BKNG", "PYPL", "HUBS", "WDAY", "ADSK", "VEEV", "CDNS", "TTD", "CSGP", "SHOP", "PLTR", "DDOG", "NET", "TEAM", "MDB", "OKTA", "ZS", "SNOW",
        "MELI","SPOT","ROKU","DASH","DLTR","DECK","CELH","MNST", "BIIB","MRNA","ILMN","IDXX","INCY","PODD",
        "CPRT","AXON","MTD","TTWO","SMCI", "COIN", "XYZ", "ARM"
        ]

    stock_universe = list(set(stock_universe))

    earliest_date = datetime(2016, 1, 1)

    raw_threshold = request.args.get("max_dividend_yield", 0.5)
    keep_if_missing = request.args.get("keep_if_missing", "true").lower() == "true"
    auto_adjust = request.args.get("auto_adjust", "false").lower() == "true"
    max_div_yield = float(raw_threshold)

    stock_universe, removed, dy_map, info_map = filter_universe_by_dividend_yield(
        stock_universe,
        max_dividend_yield=max_div_yield,
        keep_if_missing=keep_if_missing
    )

    if len(stock_universe) < 5:
        return jsonify({
            "error": "Dividend filter left too few tickers to optimize reliably.",
            "kept": stock_universe,
            "removed": removed,
            "threshold_used": max_div_yield
        }), 400

    # Helper functions for pulling data
    def get_current_ticker_price_yf(ticker):
        try:
            stock = yf.Ticker(ticker)
            price = stock.history(period='1d')['Close'].values[0]
            return price.item()
        except Exception as e:
            print(f"Error fetching current price for {ticker}: {e}")
            return None

    def get_historical_data_yf(ticker):
        try:
            stock = yf.download(ticker, start=earliest_date, auto_adjust=auto_adjust)
            first_date = stock.index.min()
            return stock, first_date
        except Exception as e:
            print(f"Error fetching historical data for {ticker}: {e}")
            return None, earliest_date

    stocks = {}
    price_series = {}
    first_dates = {}
    last_dates = {}
    data_removed = []

    for ticker in stock_universe:
        stocks[ticker] = {}
        stocks[ticker]['current_price'] = get_current_ticker_price_yf(ticker)
        stocks[ticker]['info'] = info_map.get(ticker, {})
        historical_df, _ = get_historical_data_yf(ticker)
        if historical_df is None or historical_df.empty or 'Close' not in historical_df:
            data_removed.append(ticker)
            continue
        series = historical_df['Close'].dropna()
        if series.empty:
            data_removed.append(ticker)
            continue
        price_series[ticker] = series
        first_dates[ticker] = series.index.min()
        last_dates[ticker] = series.index.max()

    stock_universe = [ticker for ticker in stock_universe if ticker in price_series]

    if len(stock_universe) < 5:
        return jsonify({
            "error": "Too few tickers with historical data after filtering.",
            "kept": stock_universe,
            "removed": removed,
            "data_removed": data_removed
        }), 400

    # Guard the common window: a single ticker with a short or stale history
    # (e.g. a broken Yahoo listing returning a few weeks of prices) would
    # otherwise collapse common_start/common_end for the whole universe.
    MIN_HISTORY_TRADING_DAYS = 504   # ~2 years: enough for the 252-day backtest lookback plus rebalances
    MAX_STALE_DAYS = 7
    latest_available = max(last_dates.values())
    data_removed_reasons = {}
    for ticker in list(price_series.keys()):
        series = price_series[ticker]
        reason = None
        if (latest_available - last_dates[ticker]).days > MAX_STALE_DAYS:
            reason = f"stale: last price {last_dates[ticker].date()} vs {latest_available.date()}"
        elif len(series) < MIN_HISTORY_TRADING_DAYS:
            reason = f"short history: {len(series)} trading days from {first_dates[ticker].date()}"
        if reason:
            print(f"Dropping {ticker}: {reason}")
            data_removed.append(ticker)
            data_removed_reasons[ticker] = reason
            del price_series[ticker]
            del first_dates[ticker]
            del last_dates[ticker]

    stock_universe = [ticker for ticker in stock_universe if ticker in price_series]

    if len(stock_universe) < 5:
        return jsonify({
            "error": "Too few tickers with sufficient price history after filtering.",
            "kept": stock_universe,
            "removed": removed,
            "data_removed": data_removed,
            "data_removed_reasons": data_removed_reasons
        }), 400

    common_start = max(first_dates.values())
    common_end = min(last_dates.values())

    if common_start >= common_end:
        return jsonify({
            "error": "No overlapping price history across tickers.",
            "kept": stock_universe,
            "removed": removed,
            "data_removed": data_removed
        }), 400

    sliced_series = []
    overlap_removed = []
    for ticker, series in price_series.items():
        sliced = series.loc[common_start:common_end]
        if isinstance(sliced, pd.DataFrame):
            if sliced.empty:
                overlap_removed.append(ticker)
                continue
            if sliced.shape[1] == 1:
                sliced = sliced.iloc[:, 0]
            else:
                overlap_removed.append(ticker)
                continue
        if isinstance(sliced, pd.Series):
            sliced = sliced.dropna()
            if sliced.empty:
                overlap_removed.append(ticker)
                continue
            sliced_series.append(sliced.rename(ticker))
        else:
            if pd.isna(sliced):
                overlap_removed.append(ticker)
                continue
            sliced_series.append(pd.Series([sliced], index=[common_start], name=ticker))

    if overlap_removed:
        data_removed.extend(overlap_removed)
        stock_universe = [ticker for ticker in stock_universe if ticker not in overlap_removed]

    if not sliced_series:
        return jsonify({
            "error": "No usable overlapping price data across tickers.",
            "kept": stock_universe,
            "removed": removed,
            "data_removed": data_removed
        }), 400

    stocks_df = pd.concat(sliced_series, axis=1).dropna(how='any')

    daily_returns = stocks_df.pct_change().dropna()
    if daily_returns.empty:
        return jsonify({
            "error": "Not enough return data to optimize.",
            "kept": stock_universe,
            "removed": removed,
            "data_removed": data_removed
        }), 400

    # Compute Ledoit-Wolf shrinkage covariance matrix (more stable than sample covariance)
    lw = LedoitWolf().fit(daily_returns.values)
    cov_matrix_annual = lw.covariance_ * 253

    # Portfolio return function using log returns (for optimization stability)
    def portfolio_returns(weights):
        daily_return = np.sum(daily_returns.mean() * weights)
        return np.log(1 + daily_return) * 253

    # Convert log return to annualized simple return (for display)
    def annualized_return(weights):
        log_return = portfolio_returns(weights)
        return np.exp(log_return) - 1

    # Portfolio standard deviation function
    def portfolio_sd(weights):
        return np.sqrt(np.transpose(weights) @ cov_matrix_annual @ weights)

    # Sharpe function
    def sharpe_fun(weights):
        return - (portfolio_returns(weights) / portfolio_sd(weights))

    # Constraints for the optimizer (weights sum to 1)
    constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})

    # Bounds for the weights (between 0 and 1)
    bounds = tuple((0, 1) for _ in range(len(stock_universe)))

    # Initial guess (equal weighting)
    equal_weights = np.array([1 / len(stock_universe)] * len(stock_universe))

    # Minimize negative Sharpe ratio to maximize the actual Sharpe ratio
    max_sharpe_results = sco.minimize(
        fun=sharpe_fun,
        x0=equal_weights,
        method='SLSQP',
        bounds=bounds,
        constraints=constraints
    )

    # Calculate expected return, standard deviation, and Sharpe ratio
    max_sharpe_port_return = annualized_return(max_sharpe_results["x"])
    max_sharpe_port_sd = portfolio_sd(max_sharpe_results["x"])
    if not np.isfinite(max_sharpe_port_sd) or max_sharpe_port_sd == 0:
        max_sharpe_port_sharpe = None
    else:
        max_sharpe_port_sharpe = max_sharpe_port_return / max_sharpe_port_sd

    # Initialize an array of target returns for efficient frontier calculation
    target_returns = np.linspace(start=0.15, stop=0.50, num=15)

    # Instantiate an empty container for storing the results
    obj_sd = []
    frontier = []

    # Loop to minimize standard deviation for each target return
    for target in target_returns:
        def portfolio_return_constraint(weights):
            return portfolio_returns(weights) - target

        constraints = [
            {'type': 'eq', 'fun': portfolio_return_constraint},
            {'type': 'eq', 'fun': lambda x: np.sum(x) - 1}
        ]

        min_result_object = sco.minimize(
            fun=portfolio_sd,
            x0=equal_weights,
            method='SLSQP',
            bounds=bounds,
            constraints=constraints
        )

        frontier.append(min_result_object)
        obj_sd.append(min_result_object['fun'])

    # Store the final results
    final_results = {}

    today = date.today()

    formatted_date = today.strftime("%Y-%m-%d")

    final_results['latest_run_date'] = formatted_date
    final_results["dividend_filter"] = {
        "max_dividend_yield_used": max_div_yield,
        "keep_if_missing": keep_if_missing,
        "removed": removed,
        "dividend_yield": {k: (None if v is None else float(v)) for k, v in dy_map.items()}
    }
    final_results["price_series"] = {
        "auto_adjust": auto_adjust,
        "data_removed": data_removed,
        "data_removed_reasons": data_removed_reasons,
        "data_window": {
            "start": common_start.strftime("%Y-%m-%d"),
            "end": common_end.strftime("%Y-%m-%d")
        }
    }

    # Results for max Sharpe portfolio
    final_results['max_sharpe'] = {
        'return': normalize_number(max_sharpe_port_return),
        'sd': normalize_number(max_sharpe_port_sd),
        'sharpe': normalize_number(max_sharpe_port_sharpe),
        'weights': [
            {
                'ticker': stock_universe[i],
                'weight': round(max_sharpe_results["x"][i], 4).item(),
                'price': stocks[stock_universe[i]]['current_price'],
                'info': stocks[stock_universe[i]]['info']
            } for i in range(len(stock_universe))
        ]
    }

    # Results for each target return
    for i in range(len(target_returns)):
        result = frontier[i]
        final_results[f'target_{i}'] = {
            'return': normalize_number(np.exp(target_returns[i]) - 1),
            'sd': normalize_number(obj_sd[i]),
            'weights': [
                {
                    'ticker': stock_universe[j],
                    'weight': round(result["x"][j], 4).item(),
                    'price': stocks[stock_universe[j]]['current_price']
                } for j in range(len(stock_universe))
            ]
        }

    # Run backtest validation
    try:
        # Fetch S&P 500 benchmark data
        benchmark_ticker = "^GSPC"
        benchmark_df = yf.download(benchmark_ticker, start=earliest_date, auto_adjust=auto_adjust)

        if benchmark_df is not None and not benchmark_df.empty and 'Close' in benchmark_df:
            benchmark_series = benchmark_df['Close'].dropna()
            if isinstance(benchmark_series, pd.DataFrame):
                benchmark_series = benchmark_series.iloc[:, 0]

            # Determine backtest period (5 years back from common_end or available data)
            backtest_end = common_end
            backtest_start = backtest_end - pd.DateOffset(years=5)

            # Ensure backtest_start is not before our data starts
            if backtest_start < common_start:
                backtest_start = common_start

            backtest_start_str = backtest_start.strftime("%Y-%m-%d")
            backtest_end_str = backtest_end.strftime("%Y-%m-%d")

            # Run the backtest
            backtest_result = run_backtest(
                tickers=stock_universe,
                price_data=stocks_df,
                benchmark_data=benchmark_series,
                start_date=backtest_start_str,
                end_date=backtest_end_str,
                lookback_days=252
            )

            # backtest is either a full result or null; the reason lives in backtest_error
            # so the frontend never has to distinguish an error object from real metrics.
            if backtest_result is not None:
                final_results['backtest'] = backtest_result
                final_results['backtest_error'] = None
            else:
                final_results['backtest'] = None
                final_results['backtest_error'] = "Insufficient data for backtest"
        else:
            final_results['backtest'] = None
            final_results['backtest_error'] = "Could not fetch benchmark data"
    except Exception as e:
        final_results['backtest'] = None
        final_results['backtest_error'] = f"Backtest failed: {str(e)}"

    final_results = sanitize_for_json(final_results)

    # Save final results to Google Cloud Storage
    storage_client = storage.Client()
    bucket = storage_client.bucket('portfolio-optimizer-35')
    blob = bucket.blob('portfolio-results.json')
    blob.upload_from_string(json.dumps(final_results, allow_nan=False))

    # Return final results as JSON response
    response = jsonify(final_results)
    return response

@app.route('/results', methods=['GET'])

def results():
    try:
        # Initialize the Google Cloud Storage client
        storage_client = storage.Client()

        # Get the bucket where the JSON file is stored
        bucket = storage_client.bucket('portfolio-optimizer-35')

        # Get the blob (file) from the bucket
        blob = bucket.blob('portfolio-results.json')

        # Download the JSON content as a string
        results = blob.download_as_string()

        # Convert the JSON string into a Python dictionary
        results_dict = json.loads(results)

        # Return the results as pretty JSON with an indent of 4
        # Use jsonify to ensure correct headers and formatting
        response = jsonify(results_dict)
        
        # Set CORS headers manually
        response.headers.set('Access-Control-Allow-Origin', '*')
        response.headers.set('Access-Control-Allow-Methods', 'GET, OPTIONS')
        response.headers.set('Access-Control-Allow-Headers', 'Content-Type')

        return response

    except Exception as e:
        # Log the error and return a 500 Internal Server Error response
        app.logger.error(f"Failed to retrieve results: {e}")
        return jsonify({"error": "Failed to retrieve results"}), 500

@app.route("/")
def home():
    # return basic html page "hello world"
    return "<h1>Portfolio Optimizer Works!</h1>"

if __name__ == "__main__":
    app.run(host='127.0.0.1', port=8080)
