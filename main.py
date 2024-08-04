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
        'AAPL', 'ABBV', 'ABT', 'ACN', 'ADP', 'AEP', 'AFL', 'AMD', 'AME', 'AMZN', 
        'AWK', 'BA', 'BLK', 'BP', 'BRK-B', 'C', 'CAE', 'CB', 'CHD', 'CI', 'COST', 
        'CRM', 'CSCO', 'DE', 'DIS', 'DUK', 'ECL', 'EL', 'EMR', 'EQR', 'EXC', 
        'F', 'FDX', 'FTNT', 'GD', 'GIB', 'GM', 'GOOGL', 'GS', 'HD', 'HPQ', 'IBM', 
        'INTC', 'J', 'JNJ', 'JPM', 'KMB', 'KO', 'LLY', 'LMT', 'LOW', 'MA', 
        'MDT', 'MMM', 'MRK', 'MSFT', 'NEE', 'NVDA', 'OKE', 'ORLY', 'PEP', 'PFE', 
        'PLD', 'PSA', 'ROST', 'RTX', 'SBUX', 'SHEL', 'SHW', 'SO', 'SU', 'SWK', 
        'T', 'TFC', 'TGT', 'TJX', 'TRP', 'TRV', 'UNP', 'UPS', 'V', 'VZ', 
        'WMT', 'WTRG', 'YUM'
        ]

    earliest_date = datetime(2014, 1, 1)

    # helper functions for pulling data

    def get_current_ticker_price_yf(ticker):
        stock = yf.Ticker(ticker)
        price = stock.history(period='1d')['Close'].values[0]
        price_number = price.item()
        return price_number

    def get_historical_data_yf(ticker):
        stock = yf.download(ticker)
        # print earliest non-na date in stock
        first_date = stock.index[stock['Close'].notna()][0]
        return stock, first_date

    historical_data, fd = get_historical_data_yf('AAPL')

    # create dataframe with dates from AAPL historical data.
    stocks_df = pd.DataFrame(index=historical_data.index)

    stocks = {}

    for ticker in stock_universe:
        stocks[ticker] = {}
        stocks[ticker]['current_price'] = get_current_ticker_price_yf(ticker)
        historical_df, first_date = get_historical_data_yf(ticker)
        if first_date > earliest_date:
            print(f'{ticker} has no data before {first_date}')
            earliest_date = first_date
        stocks_df = stocks_df.join(historical_df['Adj Close']).rename(columns={'Adj Close': ticker})

    daily_returns = stocks_df.pct_change().dropna(
                # Drop the first row since we have NaN's
                # The first date 2011-09-13 does not have a value since it is our cut-off date
                axis = 0,
                how = 'any',
                inplace = False
                )

    # Function for computing portfolio return
    def portfolio_returns(weights):
        return (np.sum(daily_returns.mean() * weights)) * 253

    # Function for computing standard deviation of portfolio returns
    def portfolio_sd(weights):
        return np.sqrt(np.transpose(weights) @ (daily_returns.cov() * 253) @ weights)

    def sharpe_fun(weights):
        return - (portfolio_returns(weights) / portfolio_sd(weights))

    constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})

    bounds = tuple((0, 1) for x in range(len(stock_universe)))

    equal_weights = np.array(
    [1 / len(stock_universe)] * len(stock_universe)
    )

    # Minimization results
    max_sharpe_results = sco.minimize(
    # Objective function
    fun = sharpe_fun, 

    # Initial guess, equal weights
    x0 = equal_weights, 
    method = 'SLSQP',
    bounds = bounds, 
    constraints = constraints
    )

    # Expected return
    max_sharpe_port_return = portfolio_returns(max_sharpe_results["x"])

    # Standard deviation
    max_sharpe_port_sd = portfolio_sd(max_sharpe_results["x"])

    # Sharpe ratio
    max_sharpe_port_sharpe = max_sharpe_port_return / max_sharpe_port_sd

    # Initialize an array of target returns
    target = np.linspace(
    start = 0.15, 
    stop = 0.35,
    num = 10
    )

    # We use anonymous lambda functions
    # The argument x will be the weights
    constraints = (
    {'type': 'eq', 'fun': lambda x: portfolio_returns(x) - target},
    {'type': 'eq', 'fun': lambda x: np.sum(x) - 1}
    )
    # instantiate empty container for the objective values to be minimized
    obj_sd = []
    # For loop to minimize objective function
    for target in target:
        min_result_object = sco.minimize(
            # Objective function
            fun = portfolio_sd, 
            
            # Initial guess, equal weights
            x0 = equal_weights, 
            method = 'SLSQP',
            bounds = bounds, 
            constraints = constraints
        )
        obj_sd.append(min_result_object['fun'])

    obj_sd = np.array(obj_sd)

    # Reinstatiate the list of target returns
    target = np.linspace(
    start = 0.15, 
    stop = 0.35,
    num = 10
    )

    # add results to stocks dictionary
    final_results = {}

    final_results['max_sharpe'] = {}
    
    final_results['max_sharpe']['return'] = max_sharpe_port_return.item()
    final_results['max_sharpe']['sd'] = max_sharpe_port_sd.item()
    final_results['max_sharpe']['sharpe'] = max_sharpe_port_sharpe.item()
    
    final_results['max_sharpe']['weights'] = {}
    
    for i in range(len(stock_universe)):
        final_results['max_sharpe']['weights'][stock_universe[i]] = {}
        final_results['max_sharpe']['weights'][stock_universe[i]]['weight'] = round(max_sharpe_results["x"][i], 4).item()
        final_results['max_sharpe']['weights'][stock_universe[i]]['price'] = stocks[stock_universe[i]]['current_price']

    for i in range(len(target)):
        final_results[f'target_{i}'] = {}
        final_results[f'target_{i}']['return'] = target[i].item()
        final_results[f'target_{i}']['sd'] = obj_sd[i].item()
        final_results[f'target_{i}']['weights'] = {}
        for j in range(len(stock_universe)):
            final_results[f'target_{i}']['weights'][stock_universe[j]] = {}
            final_results[f'target_{i}']['weights'][stock_universe[j]]['weight'] = round(min_result_object["x"][j], 4).item()
            final_results[f'target_{i}']['weights'][stock_universe[j]]['price'] = stocks[stock_universe[j]]['current_price']

    # save final results to json in Google Cloud Storage

    storage_client = storage.Client()

    bucket = storage_client.bucket('portfolio-optimizer-35')

    blob = bucket.blob('portfolio-results.json')

    blob.upload_from_string(json.dumps(final_results))

    # return final results as pretty json format indent 4
    return json.dumps(final_results, indent=4)

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
        response.headers.add('Content-Type', 'application/json; charset=utf-8')
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