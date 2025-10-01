import pandas as pd
import numpy as np

# Load the dataset
df = pd.read_csv('car_fuel_efficiency.csv')

print("=== HOMEWORK ANSWERS ===\n")

# Q1. Pandas version
print(f"Q1. Pandas version: {pd.__version__}")

# Q2. Records count
records_count = len(df)
print(f"Q2. Records count: {records_count}")

# Q3. Fuel types
fuel_types = df['fuel_type'].nunique()
print(f"Q3. Number of fuel types: {fuel_types}")

# Q4. Missing values
missing_columns = df.isnull().sum()
columns_with_missing = (missing_columns > 0).sum()
print(f"Q4. Columns with missing values: {columns_with_missing}")
print("Missing values per column:")
for col, missing in missing_columns.items():
    if missing > 0:
        print(f"  {col}: {missing}")

# Q5. Max fuel efficiency of cars from Asia
asia_cars = df[df['origin'] == 'Asia']
max_fuel_efficiency_asia = asia_cars['fuel_efficiency_mpg'].max()
print(f"Q5. Max fuel efficiency of cars from Asia: {max_fuel_efficiency_asia}")

# Q6. Median value of horsepower
print("\nQ6. Horsepower analysis:")
original_median = df['horsepower'].median()
print(f"Original median horsepower: {original_median}")

# Find the most frequent value
most_frequent_hp = df['horsepower'].mode()[0]
print(f"Most frequent horsepower value: {most_frequent_hp}")

# Fill missing values with most frequent value
df_filled = df.copy()
df_filled['horsepower'] = df_filled['horsepower'].fillna(most_frequent_hp)

# Calculate new median
new_median = df_filled['horsepower'].median()
print(f"New median horsepower after filling: {new_median}")

if new_median > original_median:
    print("Result: Yes, it increased")
elif new_median < original_median:
    print("Result: Yes, it decreased")
else:
    print("Result: No")

# Q7. Sum of weights (Linear regression implementation)
print("\nQ7. Linear regression calculation:")

# Select all cars from Asia
asia_cars = df[df['origin'] == 'Asia']

# Select only columns vehicle_weight and model_year
selected_data = asia_cars[['vehicle_weight', 'model_year']]

# Select the first 7 values
first_7 = selected_data.head(7)
print("First 7 rows of Asia cars (vehicle_weight, model_year):")
print(first_7)

# Get the underlying NumPy array
X = first_7.values
print(f"\nX shape: {X.shape}")
print("X array:")
print(X)

# Compute matrix-matrix multiplication between transpose of X and X
XTX = X.T @ X
print(f"\nXTX shape: {XTX.shape}")
print("XTX:")
print(XTX)

# Invert XTX
XTX_inv = np.linalg.inv(XTX)
print("\nXTX inverse:")
print(XTX_inv)

# Create array y
y = np.array([1100, 1300, 800, 900, 1000, 1100, 1200])
print(f"\ny: {y}")

# Multiply the inverse of XTX with the transpose of X, then multiply by y
w = XTX_inv @ X.T @ y
print(f"\nw: {w}")

# Sum of all elements
w_sum = w.sum()
print(f"Q7. Sum of all elements in w: {w_sum}")

print("\n=== SUMMARY OF ANSWERS ===")
print(f"Q1. Pandas version: {pd.__version__}")
print(f"Q2. Records count: {records_count}")
print(f"Q3. Fuel types: {fuel_types}")
print(f"Q4. Columns with missing values: {columns_with_missing}")
print(f"Q5. Max fuel efficiency (Asia): {max_fuel_efficiency_asia}")
if new_median > original_median:
    median_change = "Yes, it increased"
elif new_median < original_median:
    median_change = "Yes, it decreased"
else:
    median_change = "No"
print(f"Q6. Median changed: {median_change}")
print(f"Q7. Sum of weights: {w_sum}")