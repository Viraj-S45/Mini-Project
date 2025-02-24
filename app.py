import streamlit as st
import pandas as pd
import plotly.express as px
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures

# Load and preprocess dataset
df = pd.read_csv("dataset.csv")

# Convert data to long format for better visualization
df_melted = df.melt(id_vars=["Country", "Country_Code", "Level", "Region"], 
                     var_name="Year", value_name="Life Expectancy")

# Convert Year and Life Expectancy to numeric
df_melted["Year"] = pd.to_numeric(df_melted["Year"], errors="coerce")
df_melted["Life Expectancy"] = pd.to_numeric(df_melted["Life Expectancy"], errors="coerce")

# Title of the dashboard
st.title("Human Life Expectancy Dashboard")

## 1️⃣ Country Selection Dropdown
country = st.selectbox("Select a Country", df_melted["Country"].unique())

# Filter data for selected country
country_data = df_melted[df_melted["Country"] == country]

# Take the average for each year to remove multiple values
country_data = country_data.groupby("Year", as_index=False)["Life Expectancy"].mean()

# Line chart of Life Expectancy over time
st.subheader(f"Life Expectancy Trend for {country}")
fig = px.line(country_data, x="Year", y="Life Expectancy", title=f"Life Expectancy Trend in {country}")
st.plotly_chart(fig)

## 2️⃣ Predict Future Life Expectancy
st.subheader("Predict Future Life Expectancy")

# Train a polynomial regression model for each country separately
if len(country_data) > 5:  # Ensure sufficient data points
    poly = PolynomialFeatures(degree=2)
    X = country_data[["Year"]]
    y = country_data["Life Expectancy"].dropna()  # Drop NaNs in target
    X = X.loc[y.index]  # Ensure X and y have the same indices
    X_poly = poly.fit_transform(X)
    model = LinearRegression()
    model.fit(X_poly, y)

    # Allow user to input future years
    future_year = st.slider("Select a Future Year", min_value=2025, max_value=2100, value=2030, step=5)
    future_year_df = pd.DataFrame([[future_year]], columns=["Year"])  # Ensure DataFrame format
    future_pred = model.predict(poly.transform(future_year_df))  # Keep same format as training data

    st.write(f"**Predicted Life Expectancy in {future_year}: {future_pred[0]:.2f} years**")

## 3️⃣ Country Ranking Section
st.subheader("Country Rankings by Life Expectancy")

# Year Slider (1990 - 2040)
selected_year = st.slider("Select a Year", min_value=1990, max_value=2040, value=2019)

# Predict rankings per country separately
ranking_data = df_melted[df_melted["Year"] == selected_year].dropna().copy()
if selected_year > 2019:
    ranking_data = df_melted.groupby(["Country", "Country_Code"], as_index=False)[["Year", "Life Expectancy"]].mean()
    ranking_data = ranking_data.dropna()
    predictions = []
    for country in ranking_data["Country"].unique():
        country_df = df_melted[df_melted["Country"] == country]
        if len(country_df) > 5:
            poly = PolynomialFeatures(degree=2)
            X = country_df[["Year"]]
            y = country_df["Life Expectancy"].dropna()
            X = X.loc[y.index]
            X_poly = poly.fit_transform(X)
            model = LinearRegression()
            model.fit(X_poly, y)
            future_year_df = pd.DataFrame([[selected_year]], columns=["Year"])
            future_pred = model.predict(poly.transform(future_year_df))
            predictions.append(future_pred[0])
        else:
            # Use the last known life expectancy if prediction isn't possible
            predictions.append(country_df["Life Expectancy"].iloc[-1] if not country_df.empty else np.nan)
    ranking_data["Life Expectancy"] = predictions
    ranking_data["Year"] = selected_year

# Apply rolling average to smooth predictions
ranking_data["Life Expectancy"] = ranking_data["Life Expectancy"].rolling(window=3, min_periods=1).mean()

# Sort by life expectancy (highest first)
ranking_data = ranking_data.sort_values(by="Life Expectancy", ascending=False).reset_index(drop=True)
ranking_data.index += 1

# Search Function
search_country = st.text_input("Search for a Country")
if search_country:
    result = ranking_data[ranking_data["Country"].str.contains(search_country, case=False, na=False)]
    if not result.empty:
        st.write(f"**{search_country} Rank in {selected_year}: {result.index[0]} (Life Expectancy: {result['Life Expectancy'].values[0]:.2f})**")
    else:
        st.write("Country not found!")

st.dataframe(ranking_data[["Country", "Life Expectancy"]].rename(columns={"Life Expectancy": f"Life Expectancy in {selected_year}"}))

## 4️⃣ Compare Life Expectancy Between Two Countries
st.subheader("Compare Two Countries")

# Select two countries for comparison
country_1 = st.selectbox("Select First Country", df_melted["Country"].unique(), index=0)
country_2 = st.selectbox("Select Second Country", df_melted["Country"].unique(), index=1)

# Filter data for the selected countries
data_1 = df_melted[df_melted["Country"] == country_1].groupby("Year", as_index=False)["Life Expectancy"].mean()
data_2 = df_melted[df_melted["Country"] == country_2].groupby("Year", as_index=False)["Life Expectancy"].mean()

# Create comparison plot
fig_compare = px.line(title="Life Expectancy Comparison")
fig_compare.add_scatter(x=data_1["Year"], y=data_1["Life Expectancy"], mode="lines", name=country_1)
fig_compare.add_scatter(x=data_2["Year"], y=data_2["Life Expectancy"], mode="lines", name=country_2)

st.plotly_chart(fig_compare)

## 5️⃣ Interactive World Map
st.subheader("Life Expectancy Across the World")

# Get latest year data
latest_year = df_melted["Year"].max()
latest_data = df_melted[df_melted["Year"] == latest_year].dropna()

# Create world map
fig_map = px.choropleth(
    latest_data,
    locations="Country_Code",
    color="Life Expectancy",
    hover_name="Country",
    title=f"Life Expectancy Around the World ({latest_year})",
    color_continuous_scale=px.colors.sequential.Plasma,
    projection="natural earth"
)
st.plotly_chart(fig_map)
