from flask import Flask, render_template, request, redirect, url_for
import yfinance as yf
import plotly.graph_objs as go
import plotly.io as pio
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np

NLPInterFace = Flask(__name__)
tracked_tickers = []

@NLPInterFace.route('/')
def home():
    stock_data = {}
    for ticker in tracked_tickers:
        stock = yf.Ticker(ticker)
        stock_history = stock.history(interval="1m", period="1d")
        stock_data[ticker] = stock_history

    combined_graph_div, individual_graph_divs = create_stock_graph(stock_data)

    return render_template('index.html',
                           combined_graph_div=combined_graph_div,
                           individual_graph_divs=individual_graph_divs)

@NLPInterFace.route('/add', methods=['POST'])
def add_ticker():
    ticker = request.form['ticker']
    if ticker not in tracked_tickers:
        tracked_tickers.append(ticker.upper())
    return redirect(url_for('home'))


def create_stock_graph(stock_data):
    combined_fig = make_subplots(rows=1, cols=1)

    individual_graph_divs = {}

    for ticker, data in stock_data.items():
        if not data.empty:
            combined_fig.add_trace(
                go.Scatter(x=data.index, y=data['Close'], mode='lines', name=ticker)
            )

            individual_fig = go.Figure()
            individual_fig.add_trace(
                go.Scatter(x=data.index, y=data['Close'], mode='lines', name=ticker)
            )
            individual_fig.update_layout(
                title=f"{ticker} Stock Price",
                xaxis_title="Time",
                yaxis_title="Price",
                template="plotly_dark",
                height=400,
                width=600
            )
          
            individual_graph_divs[ticker] = pio.to_html(individual_fig, full_html=False)

    combined_fig.update_layout(
        title="Combined Stock Prices",
        xaxis_title="Time",
        yaxis_title="Price",
        template="plotly_dark",
        height=600,
        width=1000
    )

    combined_graph_div = pio.to_html(combined_fig, full_html=False)

    return combined_graph_div, individual_graph_divs

if __name__ == "__main__":
    NLPInterFace.run(debug=True)
