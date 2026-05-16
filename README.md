# Data Storm 7.0: Preliminary Round
**Team:** VisionAI (Chamika Uduwaka & Ishinika Jayarathna)

## 📌 Project Overview
This repository contains Team VisionAI's submission for the Data Storm 7.0 Preliminary Round, organized by the Rotaract Club of University of Moratuwa and powered by OCTAVE. 

The objective of this project is to estimate the **Latent Demand (Maximum Monthly Purchase Potential)** for over 20,000 retail outlets for January 2026. Rather than relying purely on historically constrained sales data, our approach utilizes external spatial signals (foot-traffic drivers) and a hyper-local "Look-Alike" peer benchmarking methodology to uncap true sales potential.

## 📂 Repository Contents
* `Data_Storm_7_0_Codes.ipynb`: The main Jupyter Notebook containing the end-to-end Lakehouse pipeline (Bronze, Silver, Gold, and Modeling layers).
* `VisionAI_predictions.csv`: The final unconstrained volume predictions for all outlets.
* `Data_Storm_7_0.pdf`: Our 5-page Summary Report detailing our data forensics, spatial mapping strategy, and causal base logic.

## 🚀 How to Run the Code
1. Clone this repository.
2. Ensure you have the raw competition datasets (`transactions_history_final.csv`, `outlet_master.csv`, `outlet_coordinates.csv`,`distributor_seasonality_details.csv`,`holiday_list.csv`,`1. dataset_description.xlsx`) in the same root directory as the notebook.
3. Install the required dependencies:
   ```bash
   pip install polars pandas numpy geopandas osmnx shapely
4. Run the `Data_Storm_7_0_Codes.ipynb` notebook
