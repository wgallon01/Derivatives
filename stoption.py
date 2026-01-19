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

# ======================================================
# Pricing & Greeks
# ======================================================

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
        return pd.Series([np.nan]*7)

    d1 = (np.log(F / K) + 0.5*sigma**2*T)/(sigma*np.sqrt(T))
    d2 = d1 - sigma*np.sqrt(T)

    df_r = np.exp(-r*T)
    df_q = np.exp(-q*T)
    pdf = norm.pdf(d1)
    call = opt_type == "call"

    delta = df_q*(norm.cdf(d1) if call else -norm.cdf(-d1))
    gamma = df_q*pdf/(S*sigma*np.sqrt(T))
    vega  = df_q*S*pdf*np.sqrt(T)
    theta = (
        -df_q*S*pdf*sigma/(2*np.sqrt(T))
        - r*K*df_r*norm.cdf(d2 if call else -d2)
        + q*S*norm.cdf(d1 if call else -d1)
    )
    rho = K*T*df_r*norm.cdf(d2 if call else -d2)

    return pd.Series([delta, gamma, vega, theta, rho, d1, d2])


# ======================================================
# Market Data
# ======================================================

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

    return spot, pd.concat(chains, ignore_index=True)


@st.cache_data
def dividend_yield(ticker, S):
    tk = yf.Ticker(ticker)
    q = tk.info.get("dividendYield")
    if q is not None:
        return q / 100
    divs = tk.dividends
    if divs.empty:
        return 0.0
    return divs[divs.index >= pd.Timestamp.utcnow() - pd.DateOffset(years=1)].sum() / S


# ======================================================
# Sidebar
# ======================================================

st.sidebar.header("Market")
ticker = st.sidebar.text_input("Ticker", "AAPL")
r = st.sidebar.number_input("Risk-free rate", 0.0, 0.1, 0.04)

st.sidebar.header("Filters")
min_oi = st.sidebar.number_input("Min Open Interest", 0, 100000, 100)
min_vol = st.sidebar.number_input("Min Volume", 0, 100000, 10)

# ======================================================
# Main
# ======================================================

if ticker:
    S, df = load_option_chain(ticker)
    q = dividend_yield(ticker, S)
    today = datetime.utcnow()

    df["T"] = (df["expiration"] - today).dt.days / 365
    df = df[(df["T"] > 0) & (df["openInterest"] >= min_oi) & (df["volume"] >= min_vol)]

    df["mid"] = (df["bid"] + df["ask"]) / 2
    df = df[df["mid"] > 0]

    df["F"] = S * np.exp((r - q) * df["T"])

    df["iv"] = df.apply(
        lambda x: implied_vol(x["mid"], x["F"], x["strike"], x["T"], r, x["type"]),
        axis=1
    )

    greek_cols = ["delta","gamma","vega","theta","rho","d1","d2"]
    df[greek_cols] = df.apply(
        lambda x: greeks(x["F"], S, x["strike"], x["T"], r, q, x["iv"], x["type"]),
        axis=1
    )

    df = df.dropna(subset=["iv"])

    st.subheader(f"{ticker} | Spot {S:.2f} | q {q:.2%} | r {r:.2%}")

    tab1, tab2, tab3 = st.tabs(["Option Chain", "Greeks", "Volatility Surface"])

    # ======================================================
    # Option Chain
    # ======================================================

    with tab1:
        st.dataframe(
            df[["type","strike","expiration","mid","iv","delta","gamma","vega","theta","rho",
                "openInterest","volume"]]
            .sort_values(["expiration","strike"])
        )

    # ======================================================
    # Greeks
    # ======================================================

    with tab2:
        exp = st.selectbox("Expiration", sorted(df["expiration"].unique()))
        greek = st.selectbox("Greek", ["delta","gamma","vega","theta","rho"])

        d = df[df["expiration"] == exp]

        st.plotly_chart(
            px.scatter(
                d,
                x="strike",
                y=greek,
                color="type",
                title=f"{greek.capitalize()} vs Strike"
            ),
            use_container_width=True
        )

    # ======================================================
    # Volatility Surface (Interpolated)
    # ======================================================

    with tab3:
        surface_df = df[["strike","T","iv"]]

        strikes = np.linspace(surface_df["strike"].min(), surface_df["strike"].max(), 50)
        maturities = np.linspace(surface_df["T"].min(), surface_df["T"].max(), 50)

        grid_K, grid_T = np.meshgrid(strikes, maturities)

        grid_iv = griddata(
            points=(surface_df["strike"], surface_df["T"]),
            values=surface_df["iv"],
            xi=(grid_K, grid_T),
            method="cubic"
        )

        fig = go.Figure(
            data=[
                go.Surface(
                    x=grid_T,
                    y=grid_K,
                    z=grid_iv,
                    colorscale="Viridis"
                )
            ]
        )

        fig.update_layout(
            title="Interpolated Implied Volatility Surface",
            scene=dict(
                xaxis_title="Time to Maturity",
                yaxis_title="Strike",
                zaxis_title="Implied Volatility"
            )
        )

        st.plotly_chart(fig, use_container_width=True)



        

