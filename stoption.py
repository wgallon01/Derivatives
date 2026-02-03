import yfinance as yf
import pandas as pd
import numpy as np
from scipy.stats import norm
import matplotlib.pyplot as plt

r = 0.03
q = 0.0

def bs_price(S, K, T, sigma, opt_type, r):
    if T <= 0 or sigma <= 0:
        return max(0, (S-K) if opt_type=="call" else (K-S))
    d1 = (np.log(S/K) + (r-q + 0.5*sigma**2)*T)/(sigma*np.sqrt(T))
    d2 = d1 - sigma*np.sqrt(T)
    df = np.exp(-r*T)
    return df*(S*np.exp((r-q)*T)*norm.cdf(d1) - K*norm.cdf(d2)) if opt_type=="call" else df*(K*norm.cdf(-d2) - S*np.exp((r-q)*T)*norm.cdf(-d1))

def greeks(S, K, T, sigma, opt_type, r, q):
    if T <= 0 or sigma <= 0:
        return pd.Series([np.nan]*3,index=["delta","gamma","vega"])
    d1 = (np.log(S/K) + (r-q+0.5*sigma**2)*T)/(sigma*np.sqrt(T))
    pdf = norm.pdf(d1)
    call = opt_type=="call"
    df_q = np.exp(-q*T)
    delta = df_q*(norm.cdf(d1) if call else norm.cdf(d1)-1)
    gamma = df_q*pdf/(S*sigma*np.sqrt(T))
    vega  = S*df_q*pdf*np.sqrt(T)
    return pd.Series([delta,gamma,vega],index=["delta","gamma","vega"])

# Paramètres du backtest
ticker = "AAPL"
data = yf.download(ticker, start="2020-01-01", end="2026-01-01")["Close"].copy()
data = data.astype(float)

rolling_days = 21  #Rolling
sigma = 0.20
spread_width = 10
results = []

for start_idx in range(0, len(data), rolling_days):
    start_date = data.index[start_idx]
    if start_idx + rolling_days >= len(data):
        break
    end_date = data.index[start_idx + rolling_days]

    S0 = float(data.loc[start_date])
    ST = float(data.loc[end_date])

    K1 = S0
    K2 = S0 + spread_width

    call_long = bs_price(S0, K1, rolling_days/252, sigma, "call", r)
    call_short = bs_price(S0, K2, rolling_days/252, sigma, "call", r)
    net_premium = call_long - call_short

    payoff_long = max(ST-K1,0)
    payoff_short = max(ST-K2,0)
    pnl = (payoff_long - payoff_short) - net_premium

    g_long = greeks(S0, K1, rolling_days/252, sigma, "call", r, q)
    g_short = greeks(S0, K2, rolling_days/252, sigma, "call", r, q)

    results.append({
        "start": start_date,
        "end": end_date,
        "S0": S0,
        "ST": ST,
        "K1": K1,
        "K2": K2,
        "premium": net_premium,
        "PnL": pnl,
        "Delta": g_long["delta"] - g_short["delta"],
        "Gamma": g_long["gamma"] - g_short["gamma"],
        "Vega": g_long["vega"] - g_short["vega"]
    })

df_results = pd.DataFrame(results)

if not df_results.empty:
    df_results["PnL_cum"] = df_results["PnL"].cumsum()
    print(df_results)

    plt.figure(figsize=(12,6))
    plt.plot(df_results["end"], df_results["PnL_cum"], marker='o')
    plt.xlabel("Expiration")
    plt.ylabel("Cumulative PnL")
    plt.title("Rolling ATM Call Spread Backtest (2022)")
    plt.grid(True)
    plt.show()

    plt.figure(figsize=(12,6))
    plt.plot(df_results["end"], df_results["Delta"], label="Delta")
    plt.xlabel("Expiration")
    plt.ylabel("Value")
    plt.title("Delta Evolution for Rolling ATM Call Spread (2022)")
    plt.legend()
    plt.grid(True)
    plt.show()
    plt.figure(figsize=(12,6))
    plt.plot(df_results["end"], df_results["Gamma"], label="Gamma")
    plt.xlabel("Expiration")
    plt.ylabel("Value")
    plt.title("Gamma Evolution for Rolling ATM Call Spread (2022)")
    plt.legend()
    plt.grid(True)
    plt.show()

    plt.figure(figsize=(12,6))
    plt.plot(df_results["end"], df_results["Vega"], label="Vega")
    plt.xlabel("Expiration")
    plt.ylabel("Value")
    plt.title("Vega Evolution for Rolling ATM Call Spread (2022)")
    plt.legend()
    plt.grid(True)
    plt.show()
else:
    print("Aucun trade généré.")
