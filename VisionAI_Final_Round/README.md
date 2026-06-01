# VisionAI Outlet Intelligence

Streamlit dashboard for outlet-level predictions, filtering, outlet drilldown reasoning, and Western Province budget optimization.

## Run locally

```powershell
.venv\Scripts\activate
pip install -r requirements.txt
streamlit run app.py
```

## Secrets

Put your Gemini key in `\.streamlit\secrets.toml`:

```toml
GEMINI_API_KEY = "your_key_here"
```

The app also reads `GEMINI_API_KEY` or `GOOGLE_API_KEY` from the environment.

## Pages

- Executive Overview
- Outlet Explorer
- Budget Optimizer
- Outlet Drilldown

## Data notes

The app uses `VisionAI_budget_allocations.csv` and `Web_App_Data.csv` from the project root.

## Competition scope

The app supports browsing outlet-level predictions across the full dataset, filtering by distributor and province, drilling into a specific outlet to view its predicted potential and reasoning behind the score, and reviewing the Western Province budget allocation view.