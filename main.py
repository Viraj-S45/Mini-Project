import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import MinMaxScaler

# Use TkAgg for interactive plotting (on Linux)
matplotlib.use("TkAgg")

# Load dataset
df = pd.read_csv("dataset.csv")

# Convert wide-format years (1990-2019 as columns) to long-format
df_melted = df.melt(id_vars=["Country", "Country_Code", "Level", "Region"], 
                     var_name="Year", value_name="Life Expectancy")

# Convert Year and Life Expectancy to numeric
df_melted["Year"] = pd.to_numeric(df_melted["Year"], errors="coerce")
df_melted["Life Expectancy"] = pd.to_numeric(df_melted["Life Expectancy"], errors="coerce")

# Handle missing values
df_melted["Life Expectancy"] = df_melted.groupby("Country")["Life Expectancy"].transform(lambda x: x.fillna(x.median()))
df_melted.dropna(inplace=True)

# Ensure Year is an integer
df_melted["Year"] = df_melted["Year"].astype(int)

# Normalize Life Expectancy for ML applications
scaler = MinMaxScaler()
df_melted["Life Expectancy Normalized"] = scaler.fit_transform(df_melted[["Life Expectancy"]])

# Step 7: Machine Learning for Prediction

## 1️⃣ Prepare Data for Regression Model
selected_country = "India"  # Change to any country of interest
country_data = df_melted[df_melted["Country"] == selected_country]

X = country_data[["Year"]]
y = country_data["Life Expectancy"]

# Split into training and testing sets (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

## 2️⃣ Train Regression Model
model = LinearRegression()
model.fit(X_train, y_train)

# Make predictions on test data
y_pred = model.predict(X_test)

# Model evaluation
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("\nModel Performance Metrics:")
print(f"Mean Absolute Error (MAE): {mae:.2f}")
print(f"Mean Squared Error (MSE): {mse:.2f}")
print(f"R-squared (R2 Score): {r2:.2f}")

## 3️⃣ Predict Future Life Expectancy
future_years = np.array([[2030], [2040], [2050]])
future_predictions = model.predict(future_years)

print("\nPredicted Life Expectancy for Future Years:")
for year, prediction in zip(future_years.flatten(), future_predictions):
    print(f"Year {year}: {prediction:.2f}")

# Plot predictions vs actual data
plt.figure(figsize=(10, 6))
plt.scatter(X_test, y_test, label="Actual Data", color="blue")
plt.plot(X_test, y_pred, label="Predicted Data", color="red", linestyle="dashed")
plt.xlabel("Year")
plt.ylabel("Life Expectancy")
plt.title(f"Life Expectancy Prediction for {selected_country}")
plt.legend()
plt.grid()
plt.show()
