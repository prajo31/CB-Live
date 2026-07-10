# capital_budgeting_app.py
# Streamlit Capital Budgeting App with Live Company Data + MIRR + Real Rates
# Author: Dr. Prashant Joshi (Finance, Decoded)

import math
import numpy as np
import pandas as pd
import yfinance as yf
import streamlit as st
import plotly.graph_objects as go

# -----------------------------
# Fetch real risk-free rate (10-year Treasury)
# -----------------------------
def get_real_risk_free_rate():
    try:
        tnx = yf.Ticker("^TNX").history(period="1d")
        rf = tnx["Close"].iloc[-1] / 100
        return rf
    except:
        return None

# -----------------------------
# Fetch real market return (S&P 500 annualized)
# -----------------------------
def get_real_market_return():
    try:
        sp500 = yf.Ticker("^GSPC").history(period="5y")
        daily_returns = sp500["Close"].pct_change().dropna()
        annualized = daily_returns.mean() * 252
        return annualized
    except:
        return None

# -----------------------------
# Company data
# -----------------------------
def get_company_data(ticker: str):
    try:
        stock = yf.Ticker(ticker)
        info = stock.info

        company_name = info.get("longName", ticker)
        sector = info.get("sector", "N/A")
        industry = info.get("industry", "N/A")
        beta = info.get("beta", None)

        cf = stock.cashflow
        if cf is not None and not cf.empty:
            cf = cf.T
            capex_col = None
            for c in ["Capital Expenditures", "CapitalExpenditures"]:
                if c in cf.columns:
                    capex_col = c
                    break
            capex_series = cf[capex_col] if capex_col else pd.Series(dtype="float64")
        else:
            capex_series = pd.Series(dtype="float64")

        return {
            "name": company_name,
            "sector": sector,
            "industry": industry,
            "beta": beta,
            "capex_series": capex_series,
        }
    except Exception:
        return None

# -----------------------------
# WACC
# -----------------------------
def compute_wacc(beta, rf, rm, tax_rate, debt_ratio, cost_of_debt):
    if beta is None:
        return None, None
    equity_ratio = 1 - debt_ratio
    ke = rf + beta * (rm - rf)
    wacc = debt_ratio * cost_of_debt * (1 - tax_rate) + equity_ratio * ke
    return wacc, ke

# -----------------------------
# NPV, IRR, MIRR, Payback
# -----------------------------
def npv(rate, cash_flows):
    return sum(cf / ((1 + rate) ** t) for t, cf in enumerate(cash_flows))

def irr(cash_flows):
    try:
        return np.irr(cash_flows)
    except:
        return float("nan")

def mirr(cash_flows, finance_rate, reinvest_rate):
    positive = []
    negative = []
    n = len(cash_flows)
    for t, cf in enumerate(cash_flows):
        if cf > 0:
            positive.append(cf * (1 + reinvest_rate)**(n - t - 1))
        elif cf < 0:
            negative.append(cf / ((1 + finance_rate)**t))
    if sum(negative) == 0:
        return float("nan")
    return (sum(positive) / -sum(negative))**(1/(n - 1)) - 1

def payback_period(cash_flows):
    cumulative = 0.0
    for t, cf in enumerate(cash_flows):
        cumulative += cf
        if cumulative >= 0:
            prev = cumulative - cf
            if cf == 0:
                return float(t)
            return t - 1 + (0 - prev) / cf
    return math.inf

# -----------------------------
# Streamlit UI
# -----------------------------
st.set_page_config(page_title="Capital Budgeting Lab", page_icon="💹", layout="wide")

st.title("💹 Capital Budgeting Lab with Live Company Data + MIRR + Real Rates")

st.markdown("""
This app includes:

- Real **risk-free rate** (10-year Treasury)
- Real **market return** (S&P 500 annualized)
- Live company data (Yahoo Finance)
- WACC (CAPM)
- NPV, IRR, **MIRR**, Payback, PI
- Sensitivity analysis
- Tornado chart
""")

# -----------------------------
# Sidebar Inputs
# -----------------------------
st.sidebar.header("Global Settings")

ticker = st.sidebar.text_input("Company ticker", value="MSFT")

# Auto-fetch real rates
real_rf = get_real_risk_free_rate()
real_rm = get_real_market_return()

st.sidebar.write("### Real Market Inputs")
st.sidebar.write(f"Real Risk-Free Rate (10Y Treasury): {real_rf:.2%}" if real_rf else "Could not fetch risk-free rate")
st.sidebar.write(f"Real Market Return (S&P 500): {real_rm:.2%}" if real_rm else "Could not fetch market return")

# Allow override
rf = st.sidebar.number_input("Risk-free rate (override)", 0.0, 0.20, real_rf if real_rf else 0.04)
rm = st.sidebar.number_input("Market return (override)", 0.0, 0.30, real_rm if real_rm else 0.09)

tax_rate = st.sidebar.number_input("Corporate tax rate", 0.0, 0.60, 0.21)
debt_ratio = st.sidebar.slider("Debt ratio", 0.0, 0.90, 0.30)
cost_of_debt = st.sidebar.number_input("Cost of debt", 0.0, 0.20, 0.05)
project_years = st.sidebar.slider("Project years", 3, 15, 7)

# -----------------------------
# Company Data
# -----------------------------
st.subheader("1️⃣ Company Overview")

company_data = get_company_data(ticker)
if company_data is None:
    st.error("Could not fetch data. Try another ticker.")
    st.stop()

col1, col2, col3 = st.columns(3)
with col1:
    st.metric("Company", company_data["name"])
    st.text(f"Sector: {company_data['sector']}")
    st.text(f"Industry: {company_data['industry']}")
with col2:
    st.metric("Beta", company_data["beta"])
with col3:
    if company_data["capex_series"].empty:
        st.write("No CAPEX data available.")
    else:
        st.write("CAPEX History")
        st.bar_chart(company_data["capex_series"])

# -----------------------------
# WACC
# -----------------------------
st.subheader("2️⃣ Cost of Capital (WACC)")

wacc, ke = compute_wacc(company_data["beta"], rf, rm, tax_rate, debt_ratio, cost_of_debt)

col_w1, col_w2, col_w3 = st.columns(3)
with col_w1:
    st.write("**Inputs**")
    st.write(f"rf: {rf:.2%}")
    st.write(f"rm: {rm:.2%}")
    st.write(f"Tax rate: {tax_rate:.2%}")
    st.write(f"Debt ratio: {debt_ratio:.0%}")
    st.write(f"Cost of debt: {cost_of_debt:.2%}")
with col_w2:
    st.metric("Cost of Equity (CAPM)", f"{ke:.2%}" if ke else "N/A")
with col_w3:
    st.metric("WACC", f"{wacc:.2%}" if wacc else "N/A")

# -----------------------------
# Project Inputs
# -----------------------------
st.subheader("3️⃣ Project Inputs")

col_p1, col_p2 = st.columns(2)
with col_p1:
    initial_investment = st.number_input("Initial investment (Year 0)", value=-50_000_000.0)
    salvage_value = st.number_input("Salvage value (final year)", value=5_000_000.0)
    wc_change = st.number_input("Working capital change (Year 0)", value=-2_000_000.0)
with col_p2:
    st.write("Annual operating cash flows:")
    cash_flows_years = []
    for year in range(1, project_years + 1):
        cf = st.number_input(f"Year {year} CF", value=10_000_000.0, key=f"cf_{year}")
        cash_flows_years.append(cf)

# -----------------------------
# Discount Rate
# -----------------------------
st.subheader("4️⃣ Discount Rate")

use_wacc = st.checkbox("Use WACC", value=(wacc is not None))
discount_rate = wacc if use_wacc and wacc else st.number_input(
    "Manual discount rate", 0.0, 0.30, 0.10
)
st.info(f"Discount rate used: **{discount_rate:.2%}**")

# -----------------------------
# Capital Budgeting Results
# -----------------------------
st.subheader("5️⃣ Capital Budgeting Results")

full_cf = [initial_investment + wc_change]
if project_years > 1:
    full_cf.extend(cash_flows_years[:-1])
last_cf = cash_flows_years[-1] + salvage_value - wc_change
full_cf.append(last_cf)

project_npv = npv(discount_rate, full_cf)
project_irr = irr(full_cf)
project_mirr = mirr(full_cf, discount_rate, discount_rate)
project_payback = payback_period(full_cf)
profitability_index = project_npv / abs(full_cf[0]) if full_cf[0] != 0 else float("nan")

col_r1, col_r2, col_r3, col_r4, col_r5 = st.columns(5)
with col_r1:
    st.metric("NPV", f"${project_npv:,.0f}")
with col_r2:
    st.metric("IRR", f"{project_irr:.2%}")
with col_r3:
    st.metric("MIRR", f"{project_mirr:.2%}")
with col_r4:
    st.metric("Payback", f"{project_payback:.2f}" if not math.isinf(project_payback) else "No payback")
with col_r5:
    st.metric("Profitability Index", f"{profitability_index:.2f}")

st.markdown("### Cash Flow Timeline")
cf_df = pd.DataFrame({"Year": range(project_years + 1), "Cash Flow": full_cf})
st.dataframe(cf_df)
st.bar_chart(cf_df.set_index("Year"))

# -----------------------------
# Sensitivity Analysis
# -----------------------------
st.subheader("6️⃣ Sensitivity Analysis")

dr_low = st.slider("Lower discount rate", 0.0, 0.30, max(0.0, discount_rate - 0.05))
dr_high = st.slider("Upper discount rate", 0.0, 0.30, min(0.30, discount_rate + 0.05))
cf_shock = st.slider("Cash flow shock (±%)", -0.50, 0.50, 0.0)

shocked_cf = [initial_investment + wc_change]
shocked_years = [cf * (1 + cf_shock) for cf in cash_flows_years]
if project_years > 1:
    shocked_cf.extend(shocked_years[:-1])
shocked_cf.append(shocked_years[-1] + salvage_value - wc_change)

dr_range = np.linspace(dr_low, dr_high, 11)
sens_data = [{"Discount Rate": r, "NPV": npv(r, shocked_cf)} for r in dr_range]
sens_df = pd.DataFrame(sens_data)
st.line_chart(sens_df.set_index("Discount Rate"))

# -----------------------------
# Tornado Chart
# -----------------------------
st.subheader("7️⃣ Tornado Chart (Key Variable Sensitivity)")

variables = {
    "Initial Investment": initial_investment,
    "Salvage Value": salvage_value,
    "Working Capital": wc_change,
    "Annual Cash Flow": np.mean(cash_flows_years),
    "Discount Rate": discount_rate,
}

tornado_data = []
for var, base_val in variables.items():
    low = base_val * 0.9
    high = base_val * 1.1
    temp_cf_low = full_cf.copy()
    temp_cf_high = full_cf.copy()

    if var == "Initial Investment":
        temp_cf_low[0] = low + wc_change
        temp_cf_high[0] = high + wc_change
    elif var == "Salvage Value":
        temp_cf_low[-1] = cash_flows_years[-1] + low - wc_change
        temp_cf_high[-1] = cash_flows_years[-1] + high - wc_change
    elif var == "Working Capital":
        temp_cf_low[0] = initial_investment + low
        temp_cf_high[0] = initial_investment + high
    elif var == "Annual Cash Flow":
        temp_cf_low = [initial_investment + wc_change] + \
                      [low] * (project_years - 1) + \
                      [low + salvage_value - wc_change]
        temp_cf_high = [initial_investment + wc_change] + \
                       [high] * (project_years - 1) + \
                       [high + salvage_value - wc_change]
    elif var == "Discount Rate":
        npv_low = npv(low, full_cf)
        npv_high = npv(high, full_cf)
        tornado_data.append([var, npv_low, npv_high])
        continue

    npv_low = npv(discount_rate, temp_cf_low)
    npv_high = npv(discount_rate, temp_cf_high)
    tornado_data.append([var, npv_low, npv_high])

tornado_df = pd.DataFrame(tornado_data, columns=["Variable", "Low", "High"])

fig = go.Figure()
fig.add_trace(go.Bar(
    y=tornado_df["Variable"],
    x=tornado_df["Low"],
    orientation='h',
    name="Low Case",
))
fig.add_trace(go.Bar(
    y=tornado_df["Variable"],
    x=tornado_df["High"],
    orientation='h',
    name="High Case",
))
fig.update_layout(
    title="Tornado Chart – NPV Sensitivity",
    barmode='group',
    height=500,
)
st.plotly_chart(fig, use_container_width=True)

# -----------------------------
# Download
# -----------------------------
st.subheader("8️⃣ Download Results")

csv_bytes = cf_df.to_csv(index=False).encode("utf-8")
st.download_button(
    "Download CSV",
    csv_bytes,
    file_name=f"capital_budgeting_{ticker}.csv",
    mime="text/csv",
)
