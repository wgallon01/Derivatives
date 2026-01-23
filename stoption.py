import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime
from scipy.stats import norm
from scipy.optimize import brentq
from scipy.interpolate import griddata
import plotly.express as px
import plotly.graph_objects as go

st.set_page_config(layout="wide", page_title="Options Analytics Dashboard")
st.title("Options Analytics Dashboard")

def bs_forward_price(F, K, T, r, sigma, opt_type):
    d1 = (np.log(F / K) + 0.5 * sigma**2 * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    df = np.exp(-r * T)
    if opt_type == "call":
        return df * (F * norm.cdf(d1) - K * norm.cdf(d2))
    return df * (K * norm.cdf(-d2) - F * norm.cdf(-d1))

def implied_vol(price, F, K, T, r, opt_type):
    if price <= 0 or T <= 0:
        return np.nan
    try:
        f = lambda s: bs_forward_price(F, K, T, r, s, opt_type) - price
        return brentq(f, 1e-6, 5.0)
    except:
        return np.nan

def greeks(F, S, K, T, r, q, sigma, opt_type):
    if T <= 0 or sigma <= 0:
        return pd.Series([np.nan]*5)
    d1 = (np.log(F / K) + 0.5*sigma**2*T)/(sigma*np.sqrt(T))
    d2 = d1 - sigma*np.sqrt(T)
    df_r = np.exp(-r*T)
    df_q = np.exp(-q*T)
    pdf = norm.pdf(d1)
    call = opt_type == "call"
    delta = df_q*(norm.cdf(d1) if call else -norm.cdf(-d1))
    gamma = df_q*pdf/(S*sigma*np.sqrt(T))
    vega  = df_q*S*pdf*np.sqrt(T)
    theta = -df_q*S*pdf*sigma/(2*np.sqrt(T)) - r*K*df_r*norm.cdf(d2 if call else -d2) + q*S*norm.cdf(d1 if call else -1)
    rho   = K*T*df_r*norm.cdf(d2 if call else -d2)
    return pd.Series([delta, gamma, vega, theta, rho])

def callspread_price(F, K1, K2, T, r, sigma):
    c1 = bs_forward_price(F, K1, T, r, sigma, "call")
    c2 = bs_forward_price(F, K2, T, r, sigma, "call")
    return c1 - c2

def putspread(F, K1, K2, T, r, sigma):
    p1 = bs_forward_price(F, K1, T, r, sigma, "put")
    p2 = bs_forward_price(F, K2, T, r, sigma, "put")
    return p1 - p2

@st.cache_data
def load_option_chain(ticker):
    tk = yf.Ticker(ticker)
    spot = tk.history(period="1d")["Close"].iloc[-1]
    chains = []
    for exp in tk.options:
        oc = tk.option_chain(exp)
        for df, t in [(oc.calls, "call"), (oc.puts, "put")]:
            d = df.copy()
            d["type"] = t
            d["expiration"] = pd.to_datetime(exp)
            chains.append(d)
    return spot, pd.concat(chains, ignore_index=True) if chains else pd.DataFrame()

@st.cache_data
def dividend_yield(ticker, S):
    tk = yf.Ticker(ticker)
    q = tk.info.get("dividendYield")
    if q is not None:
        return q / 100  # yfinance gives it in percentage
    divs = tk.dividends
    if divs.empty or S <= 0:
        return 0.0
    one_year_ago = pd.Timestamp.utcnow() - pd.DateOffset(years=1)
    return divs[divs.index >= one_year_ago].sum() / S

st.sidebar.header("Market")
ticker = st.sidebar.text_input("Ticker", "AAPL")
r = st.sidebar.number_input("Risk-free rate", 0.0, 0.1, 0.04)

st.sidebar.header("Filters")
min_oi = st.sidebar.number_input("Min Open Interest", 0, 100000, 1)
min_vol = st.sidebar.number_input("Min Volume", 0, 100000, 1)

if ticker:
    S, df = load_option_chain(ticker)
    if df.empty:
        st.warning("No options data found.")
    else:
        q = dividend_yield(ticker, S)
        today = datetime.utcnow()
        df["T"] = (df["expiration"] - today).dt.days / 365
        df = df[(df["T"] > 0) & (df["openInterest"] >= min_oi) & (df["volume"] >= min_vol)]
        df["mid"] = (df["bid"] + df["ask"]) / 2
        df = df[df["mid"] > 0]
        df["F"] = S * np.exp((r - q) * df["T"])
        df["iv"] = df.apply(lambda x: implied_vol(x["mid"], x["F"], x["strike"], x["T"], r, x["type"]), axis=1)
        greek_cols = ["delta","gamma","vega","theta","rho"]
        df[greek_cols] = df.apply(lambda x: greeks(x["F"], S, x["strike"], x["T"], r, q, x["iv"], x["type"]), axis=1)
        df = df.dropna(subset=["iv"])

        st.subheader(f"{ticker} | Spot {S:.2f} | q {q:.2%} | r {r:.2%}")

        tab1, tab2, tab3, tab4 = st.tabs(["Option Chain", "Greeks", "Volatility Surface", "Trade analysis"])

        with tab1:
            st.dataframe(
                df[["type","strike","expiration","mid","iv"] + greek_cols + ["openInterest","volume"]]
                .sort_values(["expiration","strike"])
            )

        with tab2:
            for g in greek_cols:
                fig = px.scatter(
                    df,
                    x="strike",
                    y=g,
                    color="type",              # Call/Put
                    symbol="expiration",        # Different symbols for each expiration
                    labels={"strike":"Strike", g:g.capitalize(), "type":"Option Type", "expiration":"Expiration"},
                    title=f"{g.capitalize()} vs Strike for All Expirations",
                    hover_data=["expiration","T","iv","mid"]
                )
                st.plotly_chart(fig, use_container_width=True)


        with tab3:
            st.subheader("Implied Volatility Surface")
            surface_df = df[["strike","T","iv"]]
            strikes = np.linspace(surface_df["strike"].min(), surface_df["strike"].max(), 50)
            maturities = np.linspace(surface_df["T"].min(), surface_df["T"].max(), 50)
            grid_K, grid_T = np.meshgrid(strikes, maturities)
            grid_iv = griddata(points=(surface_df["strike"], surface_df["T"]),
                               values=surface_df["iv"],
                               xi=(grid_K, grid_T),
                               method="cubic")
            fig = go.Figure(
                data=[go.Surface(x=grid_T, y=grid_K, z=grid_iv, colorscale="Viridis")]
            )
            fig.update_layout(scene=dict(xaxis_title="Time to Maturity",
                                         yaxis_title="Strike",
                                         zaxis_title="Implied Volatility"))
            st.plotly_chart(fig, use_container_width=True)
        with tab4:
            st.subheader("Trade Analysis")
            col1, col2 = st.columns(2)
            with col1:
                opt_type = st.selectbox("Option Type", ["call spread", "put spread"])
                K1 = st.number_input("Strike 1", value=float(S))
                K2 = st.number_input("Strike 2", value=float(S + 10))
                exp_trade = st.date_input("Expiration Date", value=today + pd.DateOffset(days=30))
                T_trade = (pd.to_datetime(exp_trade) - today).days / 365
                sigma_trade = st.number_input("Implied Volatility", min_value=0.01, max_value=3.0, value=0.2, step=0.01)
            with col2:
                if opt_type == "call spread":
                    price = callspread_price(S * np.exp((r - q) * T_trade), K1, K2, T_trade, r, sigma_trade)
                else:
                    price = putspread(S * np.exp((r - q) * T_trade), K1, K2, T_trade, r, sigma_trade)
                st.markdown(f"### Spread Price: {price:.2f}")
                greeks_trade = greeks(S * np.exp((r - q) * T_trade), S, (K1 + K2) / 2, T_trade, r, q, sigma_trade, opt_type)
                greek_names = ["Delta", "Gamma", "Vega", "Theta", "Rho"]
                for name, val in zip(greek_names, greeks_trade):
                    st.markdown(f"**{name}:** {val:.4f}")




