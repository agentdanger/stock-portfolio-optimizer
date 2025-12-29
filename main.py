from flask import Flask, request, jsonify, make_response
from google.cloud import storage

import requests
from io import BytesIO

import yfinance as yf

import pandas as pd
import numpy as np

import scipy.optimize as sco
import scipy.interpolate as sci

import json

from datetime import datetime, date

app = Flask(__name__)


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



@app.route('/optimize', methods=['GET'])
def optimize():
    # define stock universe and earliest date to start from
    stock_universe = [
        'AAPL', 'ABBV', 'ABT', 'ACN', 'ADI', 'ADP', 'AEE', 'AEP', 'AFL', 'ALL', 'AMD', 'AME', 'AMT', 'AMZN', 'APH', 'ATO', 'AVGO',
        'AWK', 'AXP', 'BA', 'BAC', 'BCE', 'BDX', 'BLK', 'BP', 'BRK-B', 'C', 'CAE', 'CARR', 'CB', 'CHD', 'CI', 'CL', 'CMCSA', 'CMI', 'CNP', 'COP', 'COST', 
        'CP', 'CRM', 'CSCO', 'CSX', 'CTVA', 'CVX', 'DCI', 'DE', 'DG', 'DHR', 'DIS', 'DLR', 'DTE', 'DUK', 'ECL', 'EL', 'ELV', 'EMR', 'ENB', 'EQR', 'EVRG', 'EXC', 
        'F', 'FDX', 'FI', 'FMC', 'FTNT', 'GD', 'GIB', 'GIS', 'GM', 'GOOGL', 'GS', 'HD', 'HLN', 'HON', 'HPQ', 'IBM', 
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

    historical_data, fd = get_historical_data_yf(stock_universe[0])
    if historical_data is None or historical_data.empty:
        return jsonify({
            "error": "Failed to download historical data for optimization.",
            "kept": stock_universe
        }), 400
    
    # Create dataframe with dates from the first ticker's historical data.
    stocks_df = pd.DataFrame(index=historical_data.index)

    stocks = {}

    for ticker in stock_universe:
        stocks[ticker] = {}
        stocks[ticker]['current_price'] = get_current_ticker_price_yf(ticker)
        stocks[ticker]['info'] = info_map.get(ticker, {})
        historical_df, first_date = get_historical_data_yf(ticker)
        if first_date > earliest_date:
            print(f'{ticker} has no data before {first_date}')
            earliest_date = first_date
        if historical_df is not None:
            stocks_df = stocks_df.join(historical_df['Close']).rename(columns={'Close': ticker})

    daily_returns = stocks_df.pct_change().dropna()

    # Portfolio return function
    def portfolio_returns(weights):
        return (np.sum(daily_returns.mean() * weights)) * 253

    # Portfolio standard deviation function
    def portfolio_sd(weights):
        return np.sqrt(np.transpose(weights) @ (daily_returns.cov() * 253) @ weights)

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
    max_sharpe_port_return = portfolio_returns(max_sharpe_results["x"])
    max_sharpe_port_sd = portfolio_sd(max_sharpe_results["x"])
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
        "auto_adjust": auto_adjust
    }

    # Results for max Sharpe portfolio
    final_results['max_sharpe'] = {
        'return': max_sharpe_port_return.item(),
        'sd': max_sharpe_port_sd.item(),
        'sharpe': max_sharpe_port_sharpe.item(),
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
            'return': target_returns[i].item(),
            'sd': obj_sd[i].item(),
            'weights': [
                {
                    'ticker': stock_universe[j],
                    'weight': round(result["x"][j], 4).item(),
                    'price': stocks[stock_universe[j]]['current_price']
                } for j in range(len(stock_universe))
            ]
        }

    # Save final results to Google Cloud Storage
    storage_client = storage.Client()
    bucket = storage_client.bucket('portfolio-optimizer-35')
    blob = bucket.blob('portfolio-results.json')
    blob.upload_from_string(json.dumps(final_results))

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
