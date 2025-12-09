# Zerodha Options Scanner (RSI + Supertrend + OI/Price Δ)

A lightweight Streamlit web app that connects to Zerodha Kite Connect and scans **NIFTY / BANKNIFTY / FINNIFTY / SENSEX** option strikes for **momentum + trade ideas** using a simple two-layer interpretation:

1) **Momentum Label**
2) **Trade Label** + **Action Confidence**

The app is designed for **fast, transparent scanning** and avoids writing sensitive tokens to disk.

---

## Features

- Secure daily login flow using **request_token**
- **Access token stored only in memory** for the current Streamlit session
- Auto-expiry dropdown
- **ATM ± X strike range filter**
- RSI + Supertrend computed on option candles
- Price Δ + OI Δ interpretation
- Two-layer model for **PE and CE**
- Confidence filtering
- Diagnostic mode when no trade ideas appear
- Export results to CSV

---

## Project Structure

zerodha_option_scanner/
│
├── app.py
├── scanner.py
├── indicators.py
├── test_indicators.py
├── requirements.txt
├── .gitignore
└── .env (not committed)

---

## Requirements

- Python 3.10+ recommended
- Zerodha Kite Connect API access

Install dependencies:

```bash
pip install -r requirements.txt

Environment Setup

Create a .env file in the project root:

KITE_API_KEY=your_api_key
KITE_API_SECRET=your_api_secret

Important:
.env must not be committed. Ensure .gitignore includes:

.env
.venv/

Run the App
python -m streamlit run app.py

Daily Login Flow

Click Login to Zerodha inside the app

Complete login

Copy the request_token from the redirect URL

Paste into the app and click Generate session

Notes on Signals

This scanner provides idea generation, not guaranteed trades.

The two-layer model helps avoid confusion between:

Momentum interpretation

Trade bias (mean reversion vs momentum continuation)

Always confirm:

broader market trend

IV regime

risk limits

liquidity

Testing
python test_indicators.py

Disclaimer

This tool is for educational and research use only.
Options trading carries significant risk. Use proper risk management.


Save.

---

## 2) Make sure your .gitignore is correct
Open:

```powershell
notepad .gitignore


Ensure it includes at least:

__pycache__/
*.pyc
.venv/
.env
.streamlit/
*.csv
*.xlsx
Thumbs.db
.DS_Store
