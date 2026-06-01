# VisionAI Data Pipeline

Backend data engineering, spatial modeling, and MILP optimization engine for the Western Province allocation strategy.

## Run locally

Ensure you have the raw competition data (`transactions_history_final.csv`, `outlet_coordinates.csv`, `outlet_master.csv`, `VisionAI_predictions.csv`) in the project root.

```powershell
.venv\Scripts\activate
pip install -r requirements.txt

# Execute the pipeline in this exact order:
python 01_silver_layer_pipeline.py
python 02_spatial_engine.py
python 03_budget_optimizer.py


Pipeline stages
01_silver_layer_pipeline: Out of core Polars script to clean transactions and calculate the de-seasonalized historical baseline.

02_spatial_engine: GeoPandas & OSMnx script to calculate 1km competitor density and POI distance-decay metrics.

03_budget_optimizer: PuLP knapsack solver that optimizes the LKR 5M budget based on latent volume and dynamic spatial costs.

Data notes
Outputs generated: silver_cleaned_transactions.csv, Web_App_Data.csv, and VisionAI_budget_allocations.csv.

Pass Web_App_Data.csv and VisionAI_budget_allocations.csv to the Streamlit app repository to power the frontend dashboard.
