# ============================================================
# Startup Profit Prediction using Multiple Linear Regression
# Author : Aswin Balaji PM
# Dataset: 50 Startups (New York, California, Florida)
# Tools  : Python, Pandas, NumPy, Scikit-learn, Matplotlib
# ============================================================

# Step 1: Import Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

# ============================================================
# Step 2: Load Dataset
# ============================================================
df = pd.read_csv('profit_analysis_Data_set.csv')

print("=" * 50)
print("         STARTUP PROFIT PREDICTION")
print("=" * 50)

print("\n--- Dataset Overview ---")
print(f"Total Records : {df.shape[0]}")
print(f"Total Features: {df.shape[1]}")
print("\nFirst 5 Rows:")
print(df.head())

print("\nMissing Values:")
print(df.isnull().sum())

print("\nBasic Statistics:")
print(df.describe())

# ============================================================
# Step 3: Data Preprocessing
# ============================================================

# Encode 'State' column (text → numbers)
le = LabelEncoder()
df['State'] = le.fit_transform(df['State'])

# State encoding info
print("\nState Encoding:")
print("  California = 0")
print("  Florida    = 1")
print("  New York   = 2")

# Define Features (X) and Target (y)
X = df[['RD_Spend', 'Administration', 'Marketing_Spend', 'State']]
y = df['Profit']

# ============================================================
# Step 4: Check Correlation
# ============================================================
print("\n--- Correlation with Profit ---")
corr = df[['RD_Spend', 'Administration', 'Marketing_Spend', 'Profit']].corr()['Profit'].drop('Profit')
for col, val in corr.items():
    print(f"  {col}: {val:.4f}")

# ============================================================
# Step 5: Train/Test Split (80% Train, 20% Test)
# ============================================================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print(f"\nTraining Samples: {X_train.shape[0]}")
print(f"Testing Samples : {X_test.shape[0]}")

# ============================================================
# Step 6: Train the Model
# ============================================================
model = LinearRegression()
model.fit(X_train, y_train)

print("\n--- Model Coefficients ---")
for feature, coef in zip(X.columns, model.coef_):
    print(f"  {feature}: {coef:.4f}")
print(f"  Intercept: {model.intercept_:.4f}")

# ============================================================
# Step 7: Predictions
# ============================================================
y_pred = model.predict(X_test)

print("\n--- Actual vs Predicted Profit ---")
results = pd.DataFrame({
    'Actual Profit'   : y_test.values,
    'Predicted Profit': y_pred.round(2)
})
print(results.reset_index(drop=True).to_string())

# ============================================================
# Step 8: Model Evaluation
# ============================================================
r2   = r2_score(y_test, y_pred)
mae  = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))

print("\n" + "=" * 50)
print("         MODEL PERFORMANCE")
print("=" * 50)
print(f"  R² Score (Accuracy) : {r2*100:.2f}%")
print(f"  Mean Absolute Error : ${mae:,.2f}")
print(f"  RMSE                : ${rmse:,.2f}")
print("=" * 50)

# ============================================================
# Step 9: Predict for Given Inputs (From Assignment)
# ============================================================
print("\n--- Predicting Profit for New Inputs ---")

new_inputs = pd.DataFrame({
    'RD_Spend'       : [21892.92, 23940.93],
    'Administration' : [81910.77, 96489.63],
    'Marketing_Spend': [164270.70, 137001.10],
    'State'          : [1, 0]   # Florida=1, California=0
})

predictions = model.predict(new_inputs)

print(f"\n  Input 1:")
print(f"    R&D Spend      : $21,892.92")
print(f"    Administration : $81,910.77")
print(f"    Marketing Spend: $164,270.70")
print(f"    Predicted Profit: ${predictions[0]:,.2f}")

print(f"\n  Input 2:")
print(f"    R&D Spend      : $23,940.93")
print(f"    Administration : $96,489.63")
print(f"    Marketing Spend: $137,001.10")
print(f"    Predicted Profit: ${predictions[1]:,.2f}")

# ============================================================
# Step 10: Visualizations
# ============================================================

# --- Plot 1: Actual vs Predicted ---
plt.figure(figsize=(8, 5))
plt.scatter(y_test, y_pred, color='steelblue', edgecolors='white', s=90, zorder=3)
plt.plot([y_test.min(), y_test.max()],
         [y_test.min(), y_test.max()], 'r--', linewidth=2, label='Perfect Prediction')
plt.xlabel('Actual Profit ($)')
plt.ylabel('Predicted Profit ($)')
plt.title('Actual vs Predicted Profit')
plt.legend()
plt.tight_layout()
plt.savefig('plot1_actual_vs_predicted.png', dpi=150)
plt.show()
print("\nSaved: plot1_actual_vs_predicted.png")

# --- Plot 2: R&D Spend vs Profit ---
plt.figure(figsize=(8, 5))
plt.scatter(df['RD_Spend'], y, color='darkorange', edgecolors='white', s=80)
plt.xlabel('R&D Spend ($)')
plt.ylabel('Profit ($)')
plt.title('R&D Spend vs Profit (Strongest Driver)')
plt.tight_layout()
plt.savefig('plot2_rd_vs_profit.png', dpi=150)
plt.show()
print("Saved: plot2_rd_vs_profit.png")

# --- Plot 3: Marketing Spend vs Profit ---
plt.figure(figsize=(8, 5))
plt.scatter(df['Marketing_Spend'], y, color='green', edgecolors='white', s=80)
plt.xlabel('Marketing Spend ($)')
plt.ylabel('Profit ($)')
plt.title('Marketing Spend vs Profit')
plt.tight_layout()
plt.savefig('plot3_marketing_vs_profit.png', dpi=150)
plt.show()
print("Saved: plot3_marketing_vs_profit.png")

# --- Plot 4: Feature Correlation Bar Chart ---
plt.figure(figsize=(8, 5))
features = ['RD_Spend', 'Administration', 'Marketing_Spend']
correlations = [0.9729, 0.2007, 0.7478]
colors = ['green' if c > 0.5 else 'orange' for c in correlations]
plt.bar(features, correlations, color=colors, edgecolor='white', width=0.5)
plt.ylabel('Correlation with Profit')
plt.title('Feature Correlation with Profit')
plt.ylim(0, 1.1)
for i, v in enumerate(correlations):
    plt.text(i, v + 0.02, f'{v:.2f}', ha='center', fontweight='bold')
plt.tight_layout()
plt.savefig('plot4_feature_correlation.png', dpi=150)
plt.show()
print("Saved: plot4_feature_correlation.png")

# --- Plot 5: State-wise Average Profit ---
# Re-load original for state names
df2 = pd.read_csv('profit_analysis_Data_set.csv')
state_profit = df2.groupby('State')['Profit'].mean().sort_values(ascending=False)

plt.figure(figsize=(8, 5))
plt.bar(state_profit.index, state_profit.values,
        color=['steelblue', 'darkorange', 'green'],
        edgecolor='white', width=0.5)
plt.ylabel('Average Profit ($)')
plt.title('State-wise Average Profit')
for i, v in enumerate(state_profit.values):
    plt.text(i, v + 1000, f'${v:,.0f}', ha='center', fontweight='bold', fontsize=10)
plt.tight_layout()
plt.savefig('plot5_state_profit.png', dpi=150)
plt.show()
print("Saved: plot5_state_profit.png")

print("\n" + "=" * 50)
print("   ANALYSIS COMPLETE!")
print(f"   Final Model Accuracy: {r2*100:.2f}%")
print("=" * 50)
