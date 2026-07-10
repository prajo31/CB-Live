# CB-Live
# Capital Budgeting Lab (Streamlit + Live Data + MIRR + Real Rates)

Interactive capital budgeting app for teaching and student experimentation.

## Features

- Real risk-free rate (10-year Treasury)
- Real market return (S&P 500 annualized)
- Live company data via Yahoo Finance
- WACC calculation using CAPM
- Capital budgeting model:
  - NPV
  - IRR
  - MIRR
  - Payback period
  - Profitability index
- Sensitivity analysis:
  - Discount rate range
  - Cash flow shocks
- Tornado chart for key variable sensitivity
- CSV download of cash flows and results

## How to run locally

```bash
pip install -r requirements.txt
streamlit run capital_budgeting_app.py
