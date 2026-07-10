# capital_budgeting_app.py
# Streamlit Capital Budgeting App – Retail Store Project Model
# Live data + WACC + NPV + IRR + MIRR + Sensitivity + Tornado
# Author: Dr. Prashant Joshi (Finance, Decoded)

import math
import numpy as np
import pandas as pd
import yfinance as yf
import streamlit as st
import plotly.graph_objects as go

# -----------------------------
# Real market inputs
# -----------------------------
def get_real_risk_free_rate():
    try:
        tnx = yf.Ticker("^TNX").history(period="1d")
        rf = tnx["Close"].iloc[-1] / 100
        return rf
    except:
        return None

def get_real_market_return():
    try:
        sp500 = yf.Ticker("^GSPC").history(period="5y")
        daily_returns = sp500["Close"].pct_change().dropna()
        annualized = daily_returns.mean() * 252
        return annualized
    except:
        return None

# -----------------------------
# Company data + retail store model
# -----------------------------
def get_company_data(ticker: str):
    try:
        stock = yf.Ticker(ticker)
        info = stock.info

        company_name = info.get("longName", ticker)
        sector = info.get("sector", "N/A")
        industry = info.get("industry", "N/A")
        beta = info.get("beta", None)

        # Income statement
        fin = stock.financials
        rev = fin.loc["Total Revenue"] if "Total Revenue" in fin.index else None
        ebit = fin.loc["Ebit"] if "Ebit" in fin.index else None

        # Cash flow
        cf = stock.cashflow
        dep = cf.loc["Depreciation"] if cf is not None and "Depreciation" in cf.index else None
        capex = cf.loc["Capital Expenditures"] if cf is not None and "Capital Expenditures" in cf.index else None

        # Balance sheet for working capital
        bs = stock.balance_sheet
        if bs is not None:
            ca = bs.loc["Total Current Assets"] if "Total Current Assets" in bs.index else None
            cl = bs.loc["Total Current Liabilities"] if "Total Current Liabilities" in bs.index else None
        else:
            ca, cl = None, None

        return {
            "name": company_name,
            "sector": sector,
            "industry": industry,
            "beta": beta,
            "rev": rev,
            "ebit": ebit,
            "dep": dep,
            "capex": capex,
            "ca": ca,
            "cl": cl,
        }
    except Exception:
        return None

def estimate_retail_store_project(company_data, years, scale_factor=0.01):
    """
    Estimate project cash flows for a new retail store using company averages.
    scale_factor ~ fraction of company revenue represented by one store.
    """
    rev = company_data["rev"]
    ebit = company_data["ebit"]
    dep = company_data["dep"]
    capex = company_data["capex"]
    ca = company_data["ca"]
    cl = company_data["cl"]

    if rev is None or ebit is None:
        return None

    # Use last 3 years if available
    rev_vals = rev.dropna().values
    ebit_vals = ebit.dropna().values

    if len(rev_vals) < 2:
        return None

    # Revenue CAGR
    n = len(rev_vals) - 1
    cagr = (rev_vals[-1] / rev_vals[0])**(1 / n) - 1 if rev_vals[0] > 0 else 0.03

    # Operating margin
    op_margin = (ebit_vals[-1] / rev_vals[-1]) if rev_vals[-1] != 0 else 0.10

    # CAPEX ratio
    if capex is not None:
        capex_vals = capex.dropna().values
        if len(capex_vals) > 0:
            avg_capex = np.mean(np.abs(capex_vals))
            capex_ratio = avg_capex / rev_vals[-1]
        else:
            capex_ratio = 0.05
    else:
        capex_ratio = 0.05

    # Depreciation ratio
    if dep is not None:
        dep_vals = dep.dropna().values
        if len(dep_vals) > 0:
            avg_dep = np.mean(dep_vals)
            dep_ratio = avg_dep / rev_vals[-1]
        else:
            dep_ratio = 0.03
    else:
        dep_ratio = 0.03

    # Working capital ratio
    if ca is not None and cl is not None:
        ca_vals = ca.dropna().values
        cl_vals = cl.dropna().values
        if len(ca_vals) > 0 and len(cl_vals) > 0:
            wc = ca_vals[-1] - cl_vals[-1]
            wc_ratio = wc / rev_vals[-1]
        else:
            wc_ratio = 0.10
    else:
        wc_ratio = 0.10

    # Base project revenue (scale_factor of company revenue)
    base_rev = rev_vals[-1] * scale_factor

    years_list = list(range(1, years + 1))
    project_cf = []

    for t in years_list:
        revenue_t = base_rev * (1 + cagr)**(t - 1)
        ebit_t = revenue_t * op_margin
        dep_t = revenue_t * dep_ratio
        # NOPAT
        # tax rate will be applied later in main app
        project_cf.append({
            "year": t,
            "revenue": revenue_t,
            "ebit": ebit_t,
            "dep": dep_t,
            "wc": revenue_t * wc_ratio,
            "capex": revenue_t * capex_ratio,
        })

    return project_cf, cagr, op_margin, capex_ratio, dep_ratio, wc_ratio, base_rev

# -----------------------------
# WACC and metrics
# -----------------------------
def compute_wacc(beta, rf, rm, tax_rate, debt_ratio, cost_of_debt):
    if beta is None:
        return None, None
    equity_ratio = 1 - debt_ratio
    ke = rf + beta * (rm - rf)
    wacc = debt_ratio * cost_of_debt * (1 - tax_rate) + equity_ratio * ke
    return wacc, ke

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
st.set_page_config(page_title="Retail Store Capital Budgeting Lab", page_icon="🏬", layout="wide")

st.title("🏬 Retail Store Capital Budgeting Lab (Live Data + MIRR + Real Rates)")

st.markdown("""
This app models a **new retail store project** using **live company data**:

- Auto-estimated project cash flows from company financials
- Real risk-free rate (10Y Treasury)
- Real market return (S&P 500)
- WACC (CAPM)
- NPV, IRR, MIRR, Payback, PI
- Sensitivity analysis
- Tornado chart
""")

# -----------------------------
# Sidebar – global inputs
# -----------------------------
st.sidebar.header("Global Settings")

ticker = st.sidebar.text_input("Company ticker (e.g., COST, WMT, TGT)", value="COST")

real_rf = get_real_risk_free_rate()
real_rm = get_real_market_return()

st.sidebar.write("### Real Market Inputs")
st.sidebar.write(f"Real Risk-Free (10Y): {real_rf:.2%}" if real_rf else "Risk-free: unavailable")
st.sidebar.write(f"Real Market Return (S&P 500): {real_rm:.2%}" if real_rm else "Market return: unavailable")

rf = st.sidebar.number_input("Risk-free rate (override)", 0.0, 0.20, real_rf if real_rf else 0.04)
rm = st.sidebar.number_input("Market return (override)", 0.0, 0.30, real_rm if real_rm else 0.09)
tax_rate = st.sidebar.number_input("Corporate tax rate", 0.0, 0.60, 0.21)
debt_ratio = st.sidebar.slider("Debt ratio", 0.0, 0.90, 0.30)
cost_of_debt = st.sidebar.number_input("Cost of debt", 0.0, 0.20, 0.05)
project_years = st.sidebar.slider("Project years", 3, 15, 7)
scale_factor = st.sidebar.number_input("Store scale factor (fraction of company revenue)", 0.0, 0.10, 0.01)

# -----------------------------
# Company overview
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
    if company_data["rev"] is not None:
        st.write("Revenue (last years)")
        st.bar_chart(company_data["rev"].dropna())
    else:
        st.write("Revenue data unavailable.")

# -----------------------------
# Auto-estimated retail store project
# -----------------------------
st.subheader("2️⃣ Auto-Estimated Retail Store Project Cash Flows")

project_est = estimate_retail_store_project(company_data, project_years, scale_factor)
if project_est is None:
    st.error("Not enough data to estimate project cash flows.")
    st.stop()

project_cf, cagr, op_margin, capex_ratio, dep_ratio, wc_ratio, base_rev = project_est

st.markdown(f"""
**Estimated project based on live data:**

- Base store revenue (Year 1): ${base_rev:,.0f}
- Revenue CAGR (company-based): {cagr:.2%}
- Operating margin: {op_margin:.2%}
- CAPEX ratio (CAPEX / Revenue): {capex_ratio:.2%}
- Depreciation ratio (Dep / Revenue): {dep_ratio:.2%}
- Working capital ratio (WC / Revenue): {wc_ratio:.2%}
""")

proj_df = pd.DataFrame(project_cf)
proj_df["NOPAT"] = proj_df["ebit"] * (1 - tax_rate)
proj_df["FCF"] = proj_df["NOPAT"] + proj_df["dep"] - proj_df["capex"] - proj_df["wc"].diff().fillna(proj_df["wc"])

st.markdown("**Auto-estimated project cash flows (per store):**")
st.dataframe(proj_df[["year", "revenue", "ebit", "NOPAT", "dep", "capex", "wc", "FCF"]])

# -----------------------------
# Initial investment and manual overrides
# -----------------------------
st.subheader("3️⃣ Initial Investment and Overrides")

col_i1, col_i2 = st.columns(2)
with col_i1:
    initial_investment = st.number_input("Initial store investment (Year 0)", value=-proj_df["capex"].iloc[0])
    salvage_value = st.number_input("Salvage value at end of project", value=0.0)
with col_i2:
    use_auto_cf = st.checkbox("Use auto-estimated FCF", value=True)
    manual_cf_list = []
    if not use_auto_cf:
        st.write("Manual FCF overrides (Years 1..N):")
        for row in proj_df.itertuples():
            cf_val = st.number_input(f"Year {row.year} FCF", value=float(row.FCF), key=f"manual_fcf_{row.year}")
            manual_cf_list.append(cf_val)

# -----------------------------
# Discount rate
# -----------------------------
st.subheader("4️⃣ Discount Rate")

wacc, ke = compute_wacc(company_data["beta"], rf, rm, tax_rate, debt_ratio, cost_of_debt)
use_wacc = st.checkbox("Use WACC", value=(wacc is not None))
discount_rate = wacc if use_wacc and wacc else st.number_input("Manual discount rate", 0.0, 0.30, 0.10)

col_w1, col_w2, col_w3 = st.columns(3)
with col_w1:
    st.write(f"rf: {rf:.2%}, rm: {rm:.2%}")
with col_w2:
    st.metric("Cost of Equity (CAPM)", f"{ke:.2%}" if ke else "N/A")
with col_w3:
    st.metric("WACC", f"{wacc:.2%}" if wacc else "N/A")

st.info(f"Discount rate used: **{discount_rate:.2%}**")

# -----------------------------
# Capital budgeting results
# -----------------------------
st.subheader("5️⃣ Capital Budgeting Results")

# Build cash flow list
if use_auto_cf:
    fcf_list = proj_df["FCF"].tolist()
else:
    fcf_list = manual_cf_list

full_cf = [initial_investment] + fcf_list[:-1] + [fcf_list[-1] + salvage_value]

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

cf_df = pd.DataFrame({"Year": range(0, project_years + 1), "Cash Flow": full_cf})
st.markdown("### Cash Flow Timeline")
st.dataframe(cf_df)
st.bar_chart(cf_df.set_index("Year"))

# -----------------------------
# Sensitivity analysis
# -----------------------------
st.subheader("6️⃣ Sensitivity Analysis")

dr_low = st.slider("Lower discount rate", 0.0, 0.30, max(0.0, discount_rate - 0.05))
dr_high = st.slider("Upper discount rate", 0.0, 0.30, min(0.30, discount_rate + 0.05))
cf_shock = st.slider("Cash flow shock (±%)", -0.50, 0.50, 0.0)

shocked_cf = [initial_investment]
shocked_years = [cf * (1 + cf_shock) for cf in fcf_list]
if project_years > 1:
    shocked_cf.extend(shocked_years[:-1])
shocked_cf.append(shocked_years[-1] + salvage_value)

dr_range = np.linspace(dr_low, dr_high, 11)
sens_data = [{"Discount Rate": r, "NPV": npv(r, shocked_cf)} for r in dr_range]
sens_df = pd.DataFrame(sens_data)
st.line_chart(sens_df.set_index("Discount Rate"))

# -----------------------------
# Tornado chart
# -----------------------------
st.subheader("7️⃣ Tornado Chart (Key Variable Sensitivity)")

variables = {
    "Initial Investment": initial_investment,
    "Base Revenue (Year 1)": base_rev,
    "Operating Margin": op_margin,
    "CAPEX Ratio": capex_ratio,
    "Discount Rate": discount_rate,
}

tornado_data = []
for var, base_val in variables.items():
    low = base_val * 0.9
    high = base_val * 1.1

    temp_cf_low = full_cf.copy()
    temp_cf_high = full_cf.copy()

    if var == "Initial Investment":
        temp_cf_low[0] = low
        temp_cf_high[0] = high
    elif var == "Base Revenue (Year 1)":
        # Rebuild project CF with adjusted base revenue
        adj_proj_cf, _, _, _, _, _, _ = estimate_retail_store_project(company_data, project_years, scale_factor * (low / base_rev))
        adj_df = pd.DataFrame(adj_proj_cf)
        adj_df["NOPAT"] = adj_df["ebit"] * (1 - tax_rate)
        adj_df["FCF"] = adj_df["NOPAT"] + adj_df["dep"] - adj_df["capex"] - adj_df["wc"].diff().fillna(adj_df["wc"])
        adj_fcf = adj_df["FCF"].tolist()
        temp_cf_low = [initial_investment] + adj_fcf[:-1] + [adj_fcf[-1] + salvage_value]

        adj_proj_cf_h, _, _, _, _, _, _ = estimate_retail_store_project(company_data, project_years, scale_factor * (high / base_rev))
        adj_df_h = pd.DataFrame(adj_proj_cf_h)
        adj_df_h["NOPAT"] = adj_df_h["ebit"] * (1 - tax_rate)
        adj_df_h["FCF"] = adj_df_h["NOPAT"] + adj_df_h["dep"] - adj_df_h["capex"] - adj_df_h["wc"].diff().fillna(adj_df_h["wc"])
        adj_fcf_h = adj_df_h["FCF"].tolist()
        temp_cf_high = [initial_investment] + adj_fcf_h[:-1] + [adj_fcf_h[-1] + salvage_value]
    elif var == "Operating Margin":
        # Adjust EBIT and recompute FCF
        adj_cf_low = []
        for row in project_cf:
            ebit_t = row["revenue"] * low
            dep_t = row["dep"]
            wc_t = row["wc"]
            capex_t = row["capex"]
            nopat_t = ebit_t * (1 - tax_rate)
            adj_cf_low.append(nopat_t + dep_t - capex_t - wc_t)
        temp_cf_low = [initial_investment] + adj_cf_low[:-1] + [adj_cf_low[-1] + salvage_value]

        adj_cf_high = []
        for row in project_cf:
            ebit_t = row["revenue"] * high
            dep_t = row["dep"]
            wc_t = row["wc"]
            capex_t = row["capex"]
            nopat_t = ebit_t * (1 - tax_rate)
            adj_cf_high.append(nopat_t + dep_t - capex_t - wc_t)
        temp_cf_high = [initial_investment] + adj_cf_high[:-1] + [adj_cf_high[-1] + salvage_value]
    elif var == "CAPEX Ratio":
        adj_cf_low = []
        for row in project_cf:
            capex_t = row["revenue"] * low
            dep_t = row["dep"]
            wc_t = row["wc"]
            ebit_t = row["ebit"]
            nopat_t = ebit_t * (1 - tax_rate)
            adj_cf_low.append(nopat_t + dep_t - capex_t - wc_t)
        temp_cf_low = [initial_investment] + adj_cf_low[:-1] + [adj_cf_low[-1] + salvage_value]

        adj_cf_high = []
        for row in project_cf:
            capex_t = row["revenue"] * high
            dep_t = row["dep"]
            wc_t = row["wc"]
            ebit_t = row["ebit"]
            nopat_t = ebit_t * (1 - tax_rate)
            adj_cf_high.append(nopat_t + dep_t - capex_t - wc_t)
        temp_cf_high = [initial_investment] + adj_cf_high[:-1] + [adj_cf_high[-1] + salvage_value]
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
    file_name=f"retail_store_capital_budgeting_{ticker}.csv",
    mime="text/csv",
)
