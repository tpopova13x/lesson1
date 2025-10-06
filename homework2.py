import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_squared_error
import warnings
warnings.filterwarnings('ignore')

print("=== HOMEWORK 2 ANSWERS ===\n")

# Load the dataset
df = pd.read_csv('car_fuel_efficiency.csv')

# Use only specified columns
columns = ['engine_displacement', 'horsepower', 'vehicle_weight', 'model_year', 'fuel_efficiency_mpg']
df = df[columns].copy()

print("Dataset shape:", df.shape)
print("Columns:", df.columns.tolist())

# EDA - Look at fuel_efficiency_mpg variable
print("\nEDA - Fuel Efficiency Distribution:")
print(df['fuel_efficiency_mpg'].describe())

# Check for skewness (long tail)
skewness = df['fuel_efficiency_mpg'].skew()
print(f"Skewness: {skewness:.3f}")
if skewness > 1:
    print("The distribution has a long right tail")
elif skewness < -1:
    print("The distribution has a long left tail")
else:
    print("The distribution is approximately symmetric")

# Question 1: Column with missing values
print("\n=== Question 1 ===")
missing_values = df.isnull().sum()
print("Missing values per column:")
for col, missing in missing_values.items():
    print(f"  {col}: {missing}")

missing_column = missing_values[missing_values > 0].index[0]
print(f"Q1. Column with missing values: {missing_column}")

# Question 2: Median horsepower
print("\n=== Question 2 ===")
median_hp = df['horsepower'].median()
print(f"Q2. Median horsepower: {median_hp}")

# Prepare and split dataset
print("\n=== Data Preparation ===")

# Shuffle with seed 42
df_shuffled = df.sample(frac=1, random_state=42).reset_index(drop=True)

# Split features and target
X = df_shuffled.drop('fuel_efficiency_mpg', axis=1)
y = df_shuffled['fuel_efficiency_mpg']

# Split 60/20/20
X_temp, X_test, y_temp, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
X_train, X_val, y_train, y_val = train_test_split(X_temp, y_temp, test_size=0.25, random_state=42)  # 0.25 * 0.8 = 0.2

print(f"Train set: {X_train.shape}")
print(f"Validation set: {X_val.shape}")
print(f"Test set: {X_test.shape}")

# Question 3: Missing value strategies
print("\n=== Question 3 ===")

# Option 1: Fill with 0
X_train_0 = X_train.fillna(0)
X_val_0 = X_val.fillna(0)

# Option 2: Fill with mean (computed from training set only)
mean_value = X_train[missing_column].mean()
X_train_mean = X_train.fillna(mean_value)
X_val_mean = X_val.fillna(mean_value)

print(f"Mean value for {missing_column}: {mean_value:.3f}")

# Train models and compute RMSE
def compute_rmse(y_true, y_pred):
    return np.sqrt(mean_squared_error(y_true, y_pred))

# Model with 0 filling
lr_0 = LinearRegression()
lr_0.fit(X_train_0, y_train)
y_pred_0 = lr_0.predict(X_val_0)
rmse_0 = compute_rmse(y_val, y_pred_0)

# Model with mean filling
lr_mean = LinearRegression()
lr_mean.fit(X_train_mean, y_train)
y_pred_mean = lr_mean.predict(X_val_mean)
rmse_mean = compute_rmse(y_val, y_pred_mean)

print(f"RMSE with 0 filling: {round(rmse_0, 2)}")
print(f"RMSE with mean filling: {round(rmse_mean, 2)}")

if round(rmse_0, 2) < round(rmse_mean, 2):
    print("Q3. Better option: With 0")
elif round(rmse_0, 2) > round(rmse_mean, 2):
    print("Q3. Better option: With mean")
else:
    print("Q3. Both are equally good")

# Question 4: Regularized regression
print("\n=== Question 4 ===")

# Use 0 filling for this question
X_train_reg = X_train.fillna(0)
X_val_reg = X_val.fillna(0)

r_values = [0, 0.01, 0.1, 1, 5, 10, 100]
rmse_scores = []

for r in r_values:
    if r == 0:
        # Use LinearRegression for r=0
        model = LinearRegression()
    else:
        # Use Ridge regression
        model = Ridge(alpha=r)
    
    model.fit(X_train_reg, y_train)
    y_pred = model.predict(X_val_reg)
    rmse = compute_rmse(y_val, y_pred)
    rmse_scores.append(round(rmse, 2))
    print(f"r={r}: RMSE = {round(rmse, 2)}")

best_r = r_values[np.argmin(rmse_scores)]
print(f"Q4. Best r value: {best_r}")

# Question 5: Different seeds
print("\n=== Question 5 ===")

seeds = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
rmse_by_seed = []

for seed in seeds:
    # Split with different seed
    X_temp_s, X_test_s, y_temp_s, y_test_s = train_test_split(X, y, test_size=0.2, random_state=seed)
    X_train_s, X_val_s, y_train_s, y_val_s = train_test_split(X_temp_s, y_temp_s, test_size=0.25, random_state=seed)
    
    # Fill missing values with 0
    X_train_s = X_train_s.fillna(0)
    X_val_s = X_val_s.fillna(0)
    
    # Train model without regularization
    lr = LinearRegression()
    lr.fit(X_train_s, y_train_s)
    y_pred_s = lr.predict(X_val_s)
    rmse_s = compute_rmse(y_val_s, y_pred_s)
    rmse_by_seed.append(rmse_s)
    print(f"Seed {seed}: RMSE = {round(rmse_s, 2)}")

std_rmse = np.std(rmse_by_seed)
print(f"Q5. Standard deviation of RMSE scores: {round(std_rmse, 3)}")

# Question 6: Final model with seed 9
print("\n=== Question 6 ===")

# Split with seed 9
X_temp_9, X_test_9, y_temp_9, y_test_9 = train_test_split(X, y, test_size=0.2, random_state=9)
X_train_9, X_val_9, y_train_9, y_val_9 = train_test_split(X_temp_9, y_temp_9, test_size=0.25, random_state=9)

# Combine train and validation
X_combined = pd.concat([X_train_9, X_val_9], axis=0).reset_index(drop=True)
y_combined = pd.concat([y_train_9, y_val_9], axis=0).reset_index(drop=True)

# Fill missing values with 0
X_combined = X_combined.fillna(0)
X_test_9 = X_test_9.fillna(0)

# Train model with r=0.001
ridge_final = Ridge(alpha=0.001)
ridge_final.fit(X_combined, y_combined)
y_pred_final = ridge_final.predict(X_test_9)
rmse_final = compute_rmse(y_test_9, y_pred_final)

print(f"Q6. RMSE on test dataset: {round(rmse_final, 3)}")

print("\n=== SUMMARY OF ANSWERS ===")
print(f"Q1. Column with missing values: {missing_column}")
print(f"Q2. Median horsepower: {median_hp}")
if round(rmse_0, 2) < round(rmse_mean, 2):
    print("Q3. Better filling option: With 0")
elif round(rmse_0, 2) > round(rmse_mean, 2):
    print("Q3. Better filling option: With mean")
else:
    print("Q3. Both options are equally good")
print(f"Q4. Best r value: {best_r}")
print(f"Q5. Standard deviation: {round(std_rmse, 3)}")
print(f"Q6. Final RMSE: {round(rmse_final, 3)}")