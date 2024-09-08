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

from datetime import datetime

app = Flask(__name__)



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
        'WMT', 'WTRG', 'WWD', 'YUM', 'ZBH', 'ZTS'
        ]

    earliest_date = datetime(2014, 1, 1)

    # helper functions for pulling data

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
            stock = yf.download(ticker, start=earliest_date)
            first_date = stock.index[stock['Close'].notna()][0]
            return stock, first_date
        except Exception as e:
            print(f"Error fetching historical data for {ticker}: {e}")
            return None, earliest_date

    historical_data, fd = get_historical_data_yf('AAPL')
    
    # Create dataframe with dates from AAPL historical data.
    stocks_df = pd.DataFrame(index=historical_data.index)

    stocks = {}

    for ticker in stock_universe:
        stocks[ticker] = {}
        stocks[ticker]['current_price'] = get_current_ticker_price_yf(ticker)
        historical_df, first_date = get_historical_data_yf(ticker)
        if first_date > earliest_date:
            print(f'{ticker} has no data before {first_date}')
            earliest_date = first_date
        if historical_df is not None:
            stocks_df = stocks_df.join(historical_df['Adj Close']).rename(columns={'Adj Close': ticker})

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

        obj_sd.append(min_result_object['fun'])

    # Store the final results
    final_results = {}

    # Results for max Sharpe portfolio
    final_results['max_sharpe'] = {
        'return': max_sharpe_port_return.item(),
        'sd': max_sharpe_port_sd.item(),
        'sharpe': max_sharpe_port_sharpe.item(),
        'weights': [
            {
                'ticker': stock_universe[i],
                'weight': round(max_sharpe_results["x"][i], 4).item(),
                'price': stocks[stock_universe[i]]['current_price']
            } for i in range(len(stock_universe))
        ]
    }

    # Results for each target return
    for i in range(len(target_returns)):
        final_results[f'target_{i}'] = {
            'return': target_returns[i].item(),
            'sd': obj_sd[i].item(),
            'weights': [
                {
                    'ticker': stock_universe[j],
                    'weight': round(min_result_object["x"][j], 4).item(),
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