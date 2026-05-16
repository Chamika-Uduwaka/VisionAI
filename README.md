# 🌪️ Data Storm 7.0: Preliminary Round
**Team:** VisionAI (Chamika Uduwaka & Ishinika Jayarathna)

## 📌 Project Overview
This repository contains Team VisionAI's submission for the Data Storm 7.0 Preliminary Round, organized by the Rotaract Club of University of Moratuwa and powered by OCTAVE. 

The objective of this project is to estimate the **Latent Demand (Maximum Monthly Purchase Potential)** for over 20,000 retail outlets for January 2026. Rather than relying purely on historically constrained sales data, our approach utilizes external spatial signals (foot-traffic drivers) and a hyper-local "Look-Alike" peer benchmarking methodology to uncap true sales potential.

## 📂 Repository Contents
* `Data_Storm_7_0_Codes.ipynb`: The main Jupyter Notebook containing the end-to-end Lakehouse pipeline (Bronze, Silver, Gold, and Modeling layers).
* `VisionAI_predictions.csv`: The final unconstrained volume predictions for all outlets.
* `Data_Storm_7_0.pdf`: Our 5-page Summary Report detailing our data forensics, spatial mapping strategy, and causal base logic.

## ⚙️ Methodology & Architecture

Our pipeline is structured using a standard Lakehouse architecture to ensure data integrity and memory efficiency:

### 1. Data Forensics & Hygiene (The Silver Layer)
To handle the massive scale of the transaction data without exhausting memory, we utilized **Polars** for out-of-core execution.
* **System Ghost Filtering:** Isolated and removed zero-volume records representing system fees rather than physical demand.
* **Deduplication:** Dropped twin records caused by distributor upload retries.
* **Seasonality Smoothing:** Redistributed severe holiday volume spikes (e.g., Awurudu/Christmas) by aggregating data into a true `Avg_Monthly_Volume` to establish a stable baseline for our January 2026 target.

### 2. Spatial POI Extraction (The Gold Layer)
We discarded inefficient sequential API loops in favor of a bulk-extraction approach.
* **OSMnx:** Used the Overpass API to extract complete bounding boxes for the target provinces, isolating key foot-traffic drivers (Schools, Hospitals, Bus Stations, Banks).
* **GeoPandas Mapping:** Projected all raw GPS coordinates to `EPSG:32644` (Sri Lanka UTM Zone) to accurately draw 1,000-meter (1km) catchment radii around every outlet and calculate their localized POI density.

### 3. Latent Potential Modeling (The Scientist Phase)
Historical volume is left-censored (constrained by credit limits or stockouts). To uncap this, we used a peer-benchmarking strategy:
* **Hyper-Local Clustering:** Outlets were grouped into 10 distinct traffic deciles based on their mapped POI environment, not outdated master-data labels. 
* **Star Benchmarking:** Within each granular local cluster (Province + Traffic Tier), we identified the 90th percentile performer.
* **Uncapping:** The Star's monthly volume became the potential ceiling for all peers in that exact micro-environment, seamlessly executed via array evaluation (`np.maximum`).

## 🛠️ Technology Stack
* **Polars:** High-speed, memory-efficient data forensics.
* **OSMnx & Overpass API:** Bulk geospatial data extraction.
* **GeoPandas & Shapely:** Spatial joins and meter-based geometric buffering.
* **Pandas & NumPy:** Clustering, aggregations, and array mathematics.

## 🚀 How to Run the Code
1. Clone this repository.
2. Ensure you have the raw competition datasets (`transactions_history_final.csv`, `outlet_master.csv`, `outlet_coordinates.csv`) in the same root directory as the notebook.
3. Install the required dependencies:
   ```bash
   pip install polars pandas numpy geopandas osmnx shapely
