# CB-Live
# Retail Store Capital Budgeting Lab (Streamlit + Live Data)

Interactive capital budgeting app for a **new retail store project** using **live company data**.

## Features

- Auto-estimated project cash flows from company financials (revenue, margins, CAPEX, depreciation, working capital)
- Real risk-free rate (10-year Treasury, ^TNX)
- Real market return (S&P 500, ^GSPC)
- WACC via CAPM
- NPV, IRR, MIRR, Payback period, Profitability index
- Sensitivity analysis (discount rate, cash flow shocks)
- Tornado chart for key variable sensitivity
- CSV download of cash flows and results

## Run locally

```bash
pip install -r requirements.txt
streamlit run capital_budgeting_app.py
