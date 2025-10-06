from flask import Flask, jsonify
from flask_cors import CORS
import yfinance as yf
import pandas as pd

app = Flask(__name__)
CORS(app)

@app.route('/')
def home():
    return jsonify({'message': 'Trading AI API is running ✅'})

@app.route('/price/<symbol>')
def get_price(symbol):
    data = yf.download(symbol, period="1d", interval="1h")
    if data.empty:
        return jsonify({'error': 'Invalid symbol or no data'}), 404
    price = data['Close'].iloc[-1]
    return jsonify({'symbol': symbol, 'price': round(price, 2)})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
