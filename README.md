## City of Fort Worth Crime Forecasting Analysis
Time-series analysis of Fort Worth (TX) crime data, uncovering trends, seasonality, and temperature relationships, and generating forecasts to understand and predict crime patterns over time.

### Key Resources
- Deployed Streamlit dashboard: `https://fortworthcrimeforecast.streamlit.app/`
- Final report: `https://drive.google.com/file/d/1h01G7i495uj7aLs-cc5WlltMPdf4DcT6/view`

### Project Scope
- analyze monthly Fort Worth crime trends over time
- identify recurring seasonal patterns
- compare crime activity with average monthly temperature
- evaluate seasonal forecasting models
- communicate results through notebooks, scripts, and an interactive dashboard


### Data
Primary cleaned datasets used in the analysis:

- `Data/Cleaned/CFW_Monthly_Crime_Count_Cleaned.csv`
- `Data/Cleaned/CFW_Avg_Tempr_Cleaned.csv`

These support the core monthly analysis, temperature alignment, and forecasting workflow.

### Analysis Assets
- `Python-Notebooks/analysis.ipynb`: crime trend, seasonality, STL decomposition, and forecasting exploration
- `Python-Notebooks/TemprVsCrime.ipynb`: temperature-versus-crime comparison and merged monthly views
- `R/analysis.R`: seasonal ARIMA, ARIMAX, and validation workflow in R
- `app.py`: Streamlit dashboard entry point
- `utils.py`: reusable Python helpers for loading data, decomposition, validation, and forecasting

### Streamlit Dashboard
The dashboard presents the analysis in a clean, portfolio-style format with sections for:

- overview and dataset summary
- crime trend analysis
- temperature versus crime
- forecasting
- diagnostics and model validation

### Run Locally
1. Create and activate a virtual environment:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Run the Streamlit dashboard from the project root:

```bash
streamlit run app.py
```

### Note
- Forecasting is implemented in Python with `statsmodels`, with seasonal ARIMA and temperature-informed ARIMAX views included in the dashboard.

