import polars as pl

print("Regenerating Silver Layer from raw transactions...")

# 1. Load the massive raw data using Polars
df_transactions = pl.scan_csv('transactions_history_final.csv')

# 2. Re-run the Round 1 cleaning logic
cleaned_transactions = (
    df_transactions
    .unique()
    .drop_nulls(subset=['Outlet_ID'])
    .filter(pl.col('Volume_Liters') > 0) # Quarantines the zero/negative system ghost adjustments
)

# 3. Calculate volume PER MONTH
monthly_sales = (
    cleaned_transactions
    .group_by(["Outlet_ID", "Year", "Month", "Distributor_ID"])
    .agg(pl.col("Volume_Liters").sum().alias("Monthly_Volume"))
)

# 4. Calculate the TRUE Average Monthly Volume
historical_sales = (
    monthly_sales
    .group_by("Outlet_ID")
    .agg([
        pl.col("Monthly_Volume").mean().alias("Avg_Monthly_Volume"),
        pl.col("Distributor_ID").first().alias("Distributor_ID")
    ])
    .collect() # Execute the Polars lazy frame
)

# 5. Save the file so our Spatial Engine can use it
historical_sales.write_csv('silver_cleaned_transactions.csv')
print("Success! 'silver_cleaned_transactions.csv' has been generated.")