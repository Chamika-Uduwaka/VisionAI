import pandas as pd
import pulp

print("Initializing MILP Budget Optimizer...")

# 1. Load the Gold Layer spatial data
df = pd.read_csv('Web_App_Data.csv')

# 2. Calculate Latent Volume (Growth Gap)
# Ensure we don't calculate negative latent volume if an outlet is over-performing
df['Latent_Volume'] = (df['Predicted_Maximum_Monthly_Liters'] - df['Avg_Monthly_Volume']).clip(lower=0)

# 3. Dynamic Cost of Acquisition Math
BASE_COST_PER_LITER = 120  

# Apply spatial modifiers (Cheaper in high foot traffic, more expensive in high competition)
df['Adjusted_Cost'] = BASE_COST_PER_LITER * ( (1 + df['Competitor_Density']) / (1 + df['Decay_POI_Score']) )

# Total investment cost to capture the entire latent volume for the outlet
df['Total_Investment_Cost'] = df['Adjusted_Cost'] * df['Latent_Volume']

# Filter out shops that have zero latent volume to optimize solver speed
df_opt = df[df['Latent_Volume'] > 0].copy()

# 4. Set up the Knapsack Problem in PuLP
print("Running MILP Knapsack Solver...")
budget_limit = 5000000  # Strict LKR 5,000,000 limit

prob = pulp.LpProblem("Maximize_Latent_Volume", pulp.LpMaximize)

# Decision variables (1 = Invest, 0 = Do Not Invest)
outlet_vars = pulp.LpVariable.dicts("Invest", df_opt['Outlet_ID'], cat='Binary')

# Objective Function: Maximize total Latent Volume captured
prob += pulp.lpSum([df_opt[df_opt['Outlet_ID'] == i]['Latent_Volume'].values[0] * outlet_vars[i] for i in df_opt['Outlet_ID']])

# Constraint: Total spend must be <= Budget Limit
prob += pulp.lpSum([df_opt[df_opt['Outlet_ID'] == i]['Total_Investment_Cost'].values[0] * outlet_vars[i] for i in df_opt['Outlet_ID']]) <= budget_limit

# Solve
prob.solve(pulp.PULP_CBC_CMD(msg=0))

# 5. Extract Winners and Format Output
winning_outlets = []
for i in df_opt['Outlet_ID']:
    if outlet_vars[i].value() == 1.0:
        spend = df_opt[df_opt['Outlet_ID'] == i]['Total_Investment_Cost'].values[0]
        winning_outlets.append({
            'Outlet_ID': i, 
            'Trade_Spend_Allocation_LKR': spend
        })

df_winners = pd.DataFrame(winning_outlets)

# Save
df_winners.to_csv('VisionAI_budget_allocations.csv', index=False)

print(f"Optimization Complete!")
print(f"Total Outlets Invested: {len(df_winners)}")
print(f"Total Budget Allocated: LKR {df_winners['Trade_Spend_Allocation_LKR'].sum():,.2f}")
print("Saved successfully to 'VisionAI_budget_allocations.csv'")