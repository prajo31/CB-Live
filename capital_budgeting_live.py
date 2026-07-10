# capital_budgeting_app.py
# Multi-page Streamlit app – Retail Store Capital Budgeting Lab
# Live data + WACC + NPV + IRR (safe) + MIRR + Sensitivity + Tornado + Fallback modes

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
# Fallback assumption sets
# -----------------------------
def get_fallback_set(mode, custom=None):
    if mode == "Conservative":
        return {
            "cagr": 0.03,
            "op_margin": 0.05,
            "capex_ratio": 0.04,
            "dep_ratio": 0.03,
            "wc_ratio": 0.08,
        }
    elif mode == "Industry":
        return {
            "cagr": 0.05,
            "op_margin": 0.07,
            "capex_ratio": 0.05,
            "dep_ratio": 0.04,
            "wc_ratio": 0.10,
        }
    elif mode == "Aggressive":
        return {
            "cagr": 0.08,
            "op_margin": 0.10,
            "capex_ratio": 0.06,
            "dep_ratio": 0.04,
            "wc_ratio": 0.12,
        }
    elif mode == "Custom" and custom is not None:
        return custom
    else:
        return {
            "cagr": 0.05,
            "op_margin": 0.07,
            "capex_ratio": 0.05,
            "dep_ratio": 0.04,
            "wc_ratio": 0.10,
        }

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

        fin = stock.financials
        rev = fin.loc["Total Revenue"] if "Total Revenue" in fin.index else None
        ebit = fin.loc["Ebit"] if "Ebit" in fin.index else None

        cf = stock.cashflow
        dep = cf.loc["Depreciation"] if cf is not None and "Depreciation" in cf.index else None
        capex = cf.loc["Capital Expenditures"] if cf is not None and "Capital Expenditures" in cf.index else None

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

# -----------------------------
# Retail store project estimation
# -----------------------------
def estimate_retail_store_project(company_data, years, scale_factor, fallback):
    rev = company_data["rev"]
    ebit = company_data["ebit"]
    dep = company_data["dep"]
    capex = company_data["capex"]
    ca = company_data["ca"]
    cl = company_data["cl"]

    # Defaults from fallback
    cagr = fallback["cagr"]
    op_margin = fallback["op_margin"]
    capex_ratio = fallback["capex_ratio"]
    dep_ratio = fallback["dep_ratio"]
    wc_ratio = fallback["wc_ratio"]

    live_flags = {
        "cagr_live": False,
        "op_margin_live": False,
        "capex_ratio_live": False,
        "dep_ratio_live": False,
        "wc_ratio_live": False,
    }

    if rev is not None:
        rev_vals = rev.dropna().values
        if len(rev_vals) >= 2 and rev_vals[0] > 0:
            n = len(rev_vals) - 1
            cagr = (rev_vals[-1] / rev_vals[0])**(1 / n) - 1
            live_flags["cagr_live"] = True
        base_rev = rev_vals[-1] * scale_factor
    else:
        base_rev = 1_000_000 * scale_factor

    if rev is not None and ebit is not None:
        rev_vals = rev.dropna().values
        ebit_vals = ebit.dropna().values
        if len(rev_vals) > 0 and len(ebit_vals) > 0 and rev_vals[-1] != 0:
            op_margin = ebit_vals[-1] / rev_vals[-1]
            live_flags["op_margin_live"] = True

    if capex is not None and rev is not None:
        capex_vals = capex.dropna().values
        rev_vals = rev.dropna().values
        if len(capex_vals) > 0 and len(rev_vals) > 0 and rev_vals[-1] != 0:
            avg_capex = np.mean(np.abs(capex_vals))
            capex_ratio = avg_capex / rev_vals[-1]
            live_flags["capex_ratio_live"] = True

    if dep is not None and rev is not None:
        dep_vals = dep.dropna().values
        rev_vals = rev.dropna().values
        if len(dep_vals) > 0 and len(rev_vals) > 0 and rev_vals[-1] != 0:
            avg_dep = np.mean(dep_vals)
            dep_ratio = avg_dep / rev_vals[-1]
            live_flags["dep_ratio_live"] = True

    if ca is not None and cl is not None and rev is not None:
        ca_vals = ca.dropna().values
        cl_vals = cl.dropna().values
        rev_vals = rev.dropna().values
        if len(ca_vals) > 0 and len(cl_vals) > 0 and len(rev_vals) > 0 and rev_vals[-1] != 0:
            wc = ca_vals[-1] - cl_vals[-1]
            wc_ratio = wc / rev_vals[-1]
            live_flags["wc_ratio_live"] = True

    years_list = list(range(1, years + 1))
    project_cf = []

    for t in years_list:
        revenue_t = base_rev * (1 + cagr)**(t - 1)
        ebit_t = revenue_t * op_margin
        dep_t = revenue_t * dep_ratio
        wc_t = revenue_t * wc_ratio
        capex_t = revenue_t * capex_ratio
        project_cf.append({
            "year": t,
            "revenue": revenue_t,
            "ebit": ebit_t,
            "dep": dep_t,
            "wc": wc_t,
            "capex": capex_t,
        })

    return project_cf, cagr, op_margin, capex_ratio, dep_ratio, wc_ratio, base_rev, live_flags

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

def safe_irr(cf):
    # IRR requires CF0 < 0 and at least one positive CF later
    if len(cf) < 2 or cf[0] >= 0 or not any(x > 0 for x in cf[1:]):
        return float("nan")
    try:
        return np.irr(cf)
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
# App config
# -----------------------------
st.set_page_config(page_title="Retail Store Capital Budgeting Lab", page_icon="🏬", layout="wide")

st.sidebar.title("Navigation")
page = st.sidebar.radio(
    "Go to",
    ["1. Company & Market Data", "2. Project & Valuation", "3. Sensitivity & Tornado", "4. Fallback Assumptions"],
)

ticker = st.sidebar.text_input("Company ticker (e.g., COST, WMT, TGT)", value="COST")
project_years = st.sidebar.slider("Project years", 3, 15, 7)
scale_factor = st.sidebar.number_input("Store scale factor (fraction of company revenue)", 0.0, 0.10, 0.01)

real_rf = get_real_risk_free_rate()
real_rm = get_real_market_return()

rf = st.sidebar.number_input("Risk-free rate (override)", 0.0, 0.20, real_rf if real_rf else 0.04)
rm = st.sidebar.number_input("Market return (override)", 0.0, 0.30, real_rm if real_rm else 0.09)
tax_rate = st.sidebar.number_input("Corporate tax rate", 0.0, 0.60, 0.21)
debt_ratio = st.sidebar.slider("Debt ratio", 0.0, 0.90, 0.30)
cost_of_debt = st.sidebar.number_input("Cost of debt", 0.0, 0.20, 0.05)

fallback_mode = st.sidebar.selectbox(
    "Fallback assumption mode",
    ["Conservative", "Industry", "Aggressive", "Custom"],
)

custom_fallback = None
if fallback_mode == "Custom":
    st.sidebar.write("Custom fallback assumptions")
    c_cagr = st.sidebar.number_input("Revenue CAGR (e.g., 0.05)", 0.0, 0.50, 0.05)
    c_op_margin = st.sidebar.number_input("Operating margin", 0.0, 0.50, 0.07)
    c_capex_ratio = st.sidebar.number_input("CAPEX ratio", 0.0, 0.50, 0.05)
    c_dep_ratio = st.sidebar.number_input("Depreciation ratio", 0.0, 0.50, 0.04)
    c_wc_ratio = st.sidebar.number_input("Working capital ratio", 0.0, 0.50, 0.10)
    custom_fallback = {
        "cagr": c_cagr,
        "op_margin": c_op_margin,
        "capex_ratio": c_capex_ratio,
        "dep_ratio": c_dep_ratio,
        "wc_ratio": c_wc_ratio,
    }

fallback = get_fallback_set(fallback_mode, custom_fallback)

company_data = get_company_data(ticker)

if company_data is None:
    st.error("Could not fetch company data. Try another ticker.")
    st.stop()

project_est = estimate_retail_store_project(company_data, project_years, scale_factor, fallback)
project_cf, cagr, op_margin, capex_ratio, dep_ratio, wc_ratio, base_rev, live_flags = project_est

proj_df = pd.DataFrame(project_cf)
proj_df["NOPAT"] = proj_df["ebit"] * (1 - tax_rate)
proj_df["FCF"] = proj_df["NOPAT"] + proj_df["dep"] - proj_df["capex"] - proj_df["wc"].diff().fillna(proj_df["wc"])

wacc, ke = compute_wacc(company_data["beta"], rf, rm, tax_rate, debt_ratio, cost_of_debt)

# -----------------------------
# Page 1 – Company & Market Data
# -----------------------------
if page.startswith("1"):
    st.title("🏬 Company & Market Data")

    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Company", company_data["name"])
        st.text(f"Sector: {company_data['sector']}")
        st.text(f"Industry: {company_data['industry']}")
    with col2:
        st.metric("Beta", company_data["beta"])
        st.metric("Risk-free (10Y)", f"{rf:.2%}")
    with col3:
        st.metric("Market return (S&P 500)", f"{rm:.2%}")
        if company_data["rev"] is not None:
            st.write("Revenue (last years)")
            st.bar_chart(company_data["rev"].dropna())

    st.subheader("WACC (CAPM)")
    col_w1, col_w2, col_w3 = st.columns(3)
    with col_w1:
        st.write(f"rf: {rf:.2%}, rm: {rm:.2%}")
        st.write(f"Tax rate: {tax_rate:.2%}")
        st.write(f"Debt ratio: {debt_ratio:.0%}")
        st.write(f"Cost of debt: {cost_of_debt:.2%}")
    with col_w2:
        st.metric("Cost of Equity (CAPM)", f"{ke:.2%}" if ke else "N/A")
    with col_w3:
        st.metric("WACC", f"{wacc:.2%}" if wacc else "N/A")

# -----------------------------
# Page 2 – Project & Valuation
# -----------------------------
elif page.startswith("2"):
    st.title("🏬 Auto-Estimated Retail Store Project & Valuation")

    st.subheader("Auto-Estimated Project Cash Flows (Per Store)")
    st.markdown(f"""
- Base store revenue (Year 1): **${base_rev:,.0f}**
- Revenue CAGR: **{cagr:.2%}** ({'live' if live_flags['cagr_live'] else 'fallback'})
- Operating margin: **{op_margin:.2%}** ({'live' if live_flags['op_margin_live'] else 'fallback'})
- CAPEX ratio: **{capex_ratio:.2%}** ({'live' if live_flags['capex_ratio_live'] else 'fallback'})
- Depreciation ratio: **{dep_ratio:.2%}** ({'live' if live_flags['dep_ratio_live'] else 'fallback'})
- Working capital ratio: **{wc_ratio:.2%}** ({'live' if live_flags['wc_ratio_live'] else 'fallback'})
""")

    st.dataframe(proj_df[["year", "revenue", "ebit", "NOPAT", "dep", "capex", "wc", "FCF"]])

    st.subheader("Initial Investment and Overrides")
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

    st.subheader("Discount Rate")
    use_wacc = st.checkbox("Use WACC", value=(wacc is not None))
    discount_rate = wacc if use_wacc and wacc else st.number_input("Manual discount rate", 0.0, 0.30, 0.10)
    st.info(f"Discount rate used: **{discount_rate:.2%}**")

    if use_auto_cf:
        fcf_list = proj_df["FCF"].tolist()
    else:
        fcf_list = manual_cf_list

    # FULL cash-flow series including CF0
    full_cf = [initial_investment] + fcf_list[:-1] + [fcf_list[-1] + salvage_value]

    project_npv = npv(discount_rate, full_cf)
    project_irr = safe_irr(full_cf)
    project_mirr = mirr(full_cf, discount_rate, discount_rate)
    project_payback = payback_period(full_cf)
    profitability_index = project_npv / abs(full_cf[0]) if full_cf[0] != 0 else float("nan")

    col_r1, col_r2, col_r3, col_r4, col_r5 = st.columns(5)
    with col_r1:
        st.metric("NPV", f"${project_npv:,.0f}")
    with col_r2:
        st.metric("IRR", f"{project_irr:.2%}" if not math.isnan(project_irr) else "NaN")
    with col_r3:
        st.metric("MIRR", f"{project_mirr:.2%}")
    with col_r4:
        st.metric("Payback", f"{project_payback:.2f}" if not math.isinf(project_payback) else "No payback")
    with col_r5:
        st.metric("Profitability Index", f"{profitability_index:.2f}")

    if math.isnan(project_irr):
        st.warning("IRR cannot be computed because cash flows do not contain a valid sign change (CF0 must be negative and at least one future CF positive).")

    cf_df = pd.DataFrame({"Year": range(0, project_years + 1), "Cash Flow": full_cf})
    st.markdown("### Cash Flow Timeline")
    st.dataframe(cf_df)
    st.bar_chart(cf_df.set_index("Year"))

    st.session_state["full_cf"] = full_cf
    st.session_state["fcf_list"] = fcf_list
    st.session_state["discount_rate"] = discount_rate
    st.session_state["initial_investment"] = initial_investment
    st.session_state["salvage_value"] = salvage_value
    st.session_state["base_rev"] = base_rev
    st.session_state["op_margin"] = op_margin
    st.session_state["capex_ratio"] = capex_ratio
    st.session_state["project_cf"] = project_cf

# -----------------------------
# Page 3 – Sensitivity & Tornado
# -----------------------------
elif page.startswith("3"):
    st.title("📈 Sensitivity & Tornado Analysis")

    if "full_cf" not in st.session_state:
        st.warning("Please visit 'Project & Valuation' page first to compute cash flows.")
    else:
        full_cf = st.session_state["full_cf"]
        fcf_list = st.session_state["fcf_list"]
        discount_rate = st.session_state["discount_rate"]
        initial_investment = st.session_state["initial_investment"]
        salvage_value = st.session_state["salvage_value"]
        base_rev = st.session_state["base_rev"]
        op_margin = st.session_state["op_margin"]
        capex_ratio = st.session_state["capex_ratio"]
        project_cf = st.session_state["project_cf"]

        st.subheader("Discount Rate & Cash Flow Sensitivity")

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

        st.subheader("Tornado Chart – Key Variable Sensitivity")

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
                adj_proj_cf, _, _, _, _, _, _, _ = estimate_retail_store_project(company_data, project_years, scale_factor * (low / base_rev), fallback)
                adj_df = pd.DataFrame(adj_proj_cf)
                adj_df["NOPAT"] = adj_df["ebit"] * (1 - tax_rate)
                adj_df["FCF"] = adj_df["NOPAT"] + adj_df["dep"] - adj_df["capex"] - adj_df["wc"].diff().fillna(adj_df["wc"])
                adj_fcf = adj_df["FCF"].tolist()
                temp_cf_low = [initial_investment] + adj_fcf[:-1] + [adj_fcf[-1] + salvage_value]

                adj_proj_cf_h, _, _, _, _, _, _, _ = estimate_retail_store_project(company_data, project_years, scale_factor * (high / base_rev), fallback)
                adj_df_h = pd.DataFrame(adj_proj_cf_h)
                adj_df_h["NOPAT"] = adj_df_h["ebit"] * (1 - tax_rate)
                adj_df_h["FCF"] = adj_df_h["NOPAT"] + adj_df_h["dep"] - adj_df_h["capex"] - adj_df_h["wc"].diff().fillna(adj_df_h["wc"])
                adj_fcf_h = adj_df_h["FCF"].tolist()
                temp_cf_high = [initial_investment] + adj_fcf_h[:-1] + [adj_fcf_h[-1] + salvage_value]
            elif var == "Operating Margin":
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
# Page 4 – Fallback Assumptions
# -----------------------------
elif page.startswith("4"):
    st.title("⚙️ Fallback Assumptions (A/B/C/D + Custom)")

    st.markdown("""
This page controls the **fallback assumptions** used when live data is missing or incomplete.

Modes:
- **Conservative** – low growth, low margins, lower CAPEX
- **Industry** – typical big-box retail averages
- **Aggressive** – high growth, high margins, higher CAPEX
- **Custom** – you define all ratios
""")

    fb = get_fallback_set(fallback_mode, custom_fallback)

    data_rows = [
        ("Revenue CAGR", f"{cagr:.2%}", "live" if live_flags["cagr_live"] else "fallback", f"{fb['cagr']:.2%}"),
        ("Operating Margin", f"{op_margin:.2%}", "live" if live_flags["op_margin_live"] else "fallback", f"{fb['op_margin']:.2%}"),
        ("CAPEX Ratio", f"{capex_ratio:.2%}", "live" if live_flags["capex_ratio_live"] else "fallback", f"{fb['capex_ratio']:.2%}"),
        ("Depreciation Ratio", f"{dep_ratio:.2%}", "live" if live_flags["dep_ratio_live"] else "fallback", f"{fb['dep_ratio']:.2%}"),
        ("Working Capital Ratio", f"{wc_ratio:.2%}", "live" if live_flags["wc_ratio_live"] else "fallback", f"{fb['wc_ratio']:.2%}"),
    ]
    fb_df = pd.DataFrame(data_rows, columns=["Driver", "Final Value Used", "Source", "Fallback Set Value"])
    st.dataframe(fb_df)

    st.markdown("""
Use this page in class to:
- Show how missing data is handled.
- Compare conservative vs aggressive scenarios.
- Let students design their own custom assumption set.
""")

# -----------------------------
# Download
# -----------------------------
st.sidebar.markdown("---")
if "full_cf" in st.session_state:
    cf_df = pd.DataFrame({"Year": range(0, project_years + 1), "Cash Flow": st.session_state["full_cf"]})
    csv_bytes = cf_df.to_csv(index=False).encode("utf-8")
    st.sidebar.download_button(
        "Download Cash Flows CSV",
        csv_bytes,
        file_name=f"retail_store_capital_budgeting_{ticker}.csv",
        mime="text/csv",
    )
