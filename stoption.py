import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime
from scipy.stats import norm
from scipy.optimize import brentq
import plotly.express as px
import plotly.graph_objects as go

st.set_page_config(layout="wide", page_title="Options Dashboard")
st.title("Options Dashboard, Implied Volatility and Greeks")

def bs_forward_price(F, K, T, r, sigma, option_type):
    d1 = (np.log(F / K) + 0.5 * sigma**2 * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    df = np.exp(-r * T)
    if option_type == "call":
        return df * (F * norm.cdf(d1) - K * norm.cdf(d2))
    else:
        return df * (K * norm.cdf(-d2) - F * norm.cdf(-d1))

def implied_vol_forward(price, F, K, T, r, option_type):
    try:
        f = lambda sigma: bs_forward_price(F, K, T, r, sigma, option_type) - price
        return brentq(f, 1e-6, 5)
    except:
        return np.nan

def greeks_forward(F, S, K, T, r, q, sigma, option_type):
    if T <= 0 or sigma <= 0:
        return [np.nan]*9
    d1 = (np.log(F / K) + 0.5*sigma**2*T)/(sigma*np.sqrt(T))
    d2 = d1 - sigma*np.sqrt(T)
    df_r = np.exp(-r*T)
    df_q = np.exp(-q*T)
    pdf = norm.pdf(d1)
    delta = df_q*(norm.cdf(d1) if option_type=="call" else -norm.cdf(-d1))
    gamma = df_q*pdf/(S*sigma*np.sqrt(T))
    vega  = df_q*S*pdf*np.sqrt(T)
    theta = (-df_q*S*pdf*sigma/(2*np.sqrt(T))
             - r*K*df_r*norm.cdf(d2 if option_type=="call" else -d2)
             + q*S*norm.cdf(d1 if option_type=="call" else -d1))
    rho   = K*T*df_r*norm.cdf(d2 if option_type=="call" else -d2)
    vanna = df_q*pdf*(1 - d1/(sigma*np.sqrt(T)))
    charm = -df_q*pdf*(2*(r-q)*T - d2*sigma*np.sqrt(T))/(2*T*sigma*np.sqrt(T))
    return delta, gamma, vega, theta, rho, vanna, charm, d1, d2

@st.cache_data
def load_market_data(ticker):
    tk = yf.Ticker(ticker)
    hist = tk.history(period="1d")
    S = hist["Close"].iloc[-1] if not hist.empty else 0.0
    chains = []
    try:
        for exp in tk.options:
            opt = tk.option_chain(exp)
            for df_opt, t in [(opt.calls, "call"), (opt.puts, "put")]:
                d = df_opt.copy()
                d["type"] = t
                d["expiration"] = pd.to_datetime(exp)
                chains.append(d)
        df = pd.concat(chains, ignore_index=True) if chains else pd.DataFrame()
    except:
        df = pd.DataFrame()
    return S, df

@st.cache_data
def compute_dividend_yield(ticker, S):
    try:
        tk = yf.Ticker(ticker)
        q_yahoo = tk.info.get("dividendYield")
        if q_yahoo is not None:
            return float(q_yahoo / 100)
        hist = tk.dividends
        if hist.empty or S <= 0:
            return 0.0
        one_year_ago = pd.Timestamp.utcnow() - pd.DateOffset(years=1)
        recent_divs = hist[hist.index >= one_year_ago]
        annual_div = recent_divs.sum()
        return float(annual_div / S)
    except:
        return 0.0

@st.cache_data
def realized_volatility(ticker, window_days=63):
    tk = yf.Ticker(ticker)
    hist = tk.history(period="1y")
    if hist.empty:
        return pd.DataFrame(), pd.Series()
    hist["log_ret"] = np.log(hist["Close"] / hist["Close"].shift(1))
    rv = hist["log_ret"].rolling(window_days).std() * np.sqrt(252)
    return hist[["Close"]], rv

st.sidebar.header("Market Settings")
ticker = st.sidebar.text_input("Ticker", "AAPL")
r = st.sidebar.number_input("Risk-free rate (decimal, e.g., 0.04)", 0.0, 0.1, 0.04)
st.sidebar.header("Filters")
min_oi = st.sidebar.number_input("Min Open Interest", 0, 100000, 100)
min_vol = st.sidebar.number_input("Min Volume", 0, 100000, 10)
moneyness_filter = st.sidebar.selectbox("Moneyness", ["All", "OTM only", "ATM ±5%", "ITM only"])
st.sidebar.header("Realized Volatility")
rv_window = st.sidebar.selectbox("RV Window", [21, 63, 126], format_func=lambda x: f"{x} trading days")

if ticker:
    S, df = load_market_data(ticker)
    q = compute_dividend_yield(ticker, S)
    today = datetime.utcnow()
    if not df.empty and S > 0:
        df["T"] = (df["expiration"] - today).dt.days / 365
        df = df[df["T"] > 0]
        df["mid"] = (df["bid"] + df["ask"]) / 2
        df = df[df["mid"] > 0]
        df = df[(df["openInterest"] >= min_oi) & (df["volume"] >= min_vol)]
        df["F"] = S * np.exp((r - q) * df["T"])
        df["moneyness"] = df["strike"] / S
        if moneyness_filter == "OTM only":
            df = df[((df["type"]=="call") & (df["strike"]>S)) | ((df["type"]=="put") & (df["strike"]<S))]
        elif moneyness_filter == "ATM ±5%":
            df = df[(df["moneyness"]>0.95) & (df["moneyness"]<1.05)]
        elif moneyness_filter == "ITM only":
            df = df[((df["type"]=="call") & (df["strike"]<S)) | ((df["type"]=="put") & (df["strike"]>S))]
        with st.spinner("Calibrating implied vols..."):
            df["iv"] = df.apply(lambda x: implied_vol_forward(x["mid"], x["F"], x["strike"], x["T"], r, x["type"]), axis=1)
        df[["delta","gamma","vega","theta","rho","vanna","charm","d1","d2"]] = df.apply(
            lambda x: greeks_forward(x["F"], S, x["strike"], x["T"], r, q, x["iv"], x["type"])
            if not np.isnan(x["iv"]) else [np.nan]*9, axis=1, result_type="expand"
        )
        contract_size = 100
        df["GEX"] = df["gamma"]*S**2*df["openInterest"]*contract_size * np.where(df["type"]=="call",1,-1)
        df["VEX"] = df["vega"]*df["openInterest"]*contract_size
        df["DEX"] = df["delta"]*df["openInterest"]*contract_size
        hist_price, rv = realized_volatility(ticker, rv_window)
        current_rv = rv.iloc[-1] if not rv.empty else 0.0
        df["RV"] = current_rv
        df["vol_gap"] = df["iv"] - df["RV"]
        st.subheader(f"{ticker} | Spot {S:.2f} | Dividend Yield {q:.2%} | Risk-Free Rate {r:.2%} | Realized Vol ({rv_window}d) {current_rv:.2%}")

        tab1, tab2, tab3, tab4 = st.tabs(["Option Chain","Smile / Skew","Greeks & GEX","Vol Surface"])
        with tab1:
            st.dataframe(df[["type","strike","expiration","mid","iv","RV","vol_gap",
                             "delta","gamma","vega","theta","rho",
                             "vanna","charm","GEX","VEX","DEX","openInterest","volume"]].sort_values(["expiration","strike"]))
        with tab2:
            exp = st.selectbox("Expiration", sorted(df["expiration"].unique()))
            d = df[df["expiration"] == exp]
            st.subheader("Volatility Smile")
            st.plotly_chart(px.scatter(d, x="strike", y="iv", color="type", title="Volatility Smile"),
                            use_container_width=True, key="smile_strike")
            st.plotly_chart(px.scatter(d, x="delta", y="iv", color="type", title="Delta-normalized Smile"),
                            use_container_width=True, key="smile_delta")
            st.subheader("Greeks vs Strike")
            greeks = ["delta", "gamma", "vega", "theta", "rho", "vanna", "charm"]
            for g in greeks:
                st.plotly_chart(px.scatter(d, x="strike", y=g, color="type", title=f"{g.capitalize()} vs Strike"),
                                use_container_width=True, key=f"{g}_strike_{exp}")
        with tab3:
            exp = st.selectbox("Expiration for Greeks", sorted(df["expiration"].unique()), key="greeks_exp")
            d = df[df["expiration"]==exp]
            st.plotly_chart(px.scatter(d, x="strike", y="delta", color="type", title="Delta vs Strike"),
                            use_container_width=True, key="delta_greeks")
            st.plotly_chart(px.scatter(d, x="strike", y="gamma", color="type", title="Gamma vs Strike"),
                            use_container_width=True, key="gamma_greeks")
            gex_strike = d.groupby("strike")["GEX"].sum().reset_index()
            st.plotly_chart(px.bar(gex_strike, x="strike", y="GEX", title="Gamma Exposure by Strike"),
                            use_container_width=True, key="gex_bar")
        with tab4:
            surf = df.groupby(["strike","T"])["iv"].mean().reset_index().pivot_table(index="strike", columns="T", values="iv", aggfunc='mean')
            if not surf.empty:
                fig = go.Figure(data=[go.Surface(z=surf.values, x=surf.columns, y=surf.index, colorscale="Viridis")])
                fig.update_layout(title="Implied Volatility Surface",
                                  scene=dict(xaxis_title="Time to Maturity",yaxis_title="Strike",zaxis_title="Implied Vol"))
                st.plotly_chart(fig,use_container_width=True)
