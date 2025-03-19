import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from prophet import Prophet
from mlxtend.frequent_patterns import apriori, association_rules
from mlxtend.preprocessing import TransactionEncoder

# Step 1: Load & Clean the Data
file_path = "retail_sales_dataset.csv"
df = pd.read_csv(file_path)

# Standardize column names (remove extra spaces)
df.columns = df.columns.str.strip()

# Convert 'Date' to datetime format
df['Date'] = pd.to_datetime(df['Date'])

# Remove duplicates & missing values
df.drop_duplicates(inplace=True)
df.dropna(inplace=True)

# Ensure 'Total Amount' is numeric
df['Total Amount'] = pd.to_numeric(df['Total Amount'], errors='coerce')
df.dropna(subset=['Total Amount'], inplace=True)
df['Total Amount'] = df['Total Amount'].astype(int)

# Step 2: Exploratory Data Analysis (EDA)
print("\nBasic Data Info:")
print(df.info())

# Find the Top 5 Sales Days
top_sales_days = df.groupby('Date')['Total Amount'].sum().nlargest(5)
print("\nTop 5 Sales Days:\n", top_sales_days)

# Sales by Weekday
df['Weekday'] = df['Date'].dt.day_name()
weekday_sales = df.groupby('Weekday')['Total Amount'].sum()

plt.figure(figsize=(8, 5))
weekday_sales.reindex(['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']).plot(kind='bar', color='orange')
plt.title("Total Sales by Weekday")
plt.xlabel("Day of the Week")
plt.ylabel("Total Sales ($)")
plt.xticks(rotation=45)
plt.show()

# Monthly Sales Trend
df['YearMonth'] = df['Date'].dt.to_period('M').astype(str)
sales_trend = df.groupby('YearMonth')['Total Amount'].sum().reset_index()

plt.figure(figsize=(12, 6))
sns.lineplot(data=sales_trend, x='YearMonth', y='Total Amount', marker='o')
plt.title("Monthly Sales Trend")
plt.xlabel("Year-Month")
plt.ylabel("Total Sales ($)")
plt.xticks(rotation=45)
plt.show()

# Step 3: Customer Insights & Segmentation
# Customer Lifetime Value (CLV)
clv = df.groupby('Customer ID').agg(
    total_spent=('Total Amount', 'sum'),
    total_orders=('Transaction ID', 'count'),
    avg_order_value=('Total Amount', 'mean')
)
clv['CLV'] = clv['avg_order_value'] * clv['total_orders']
print("\nTop 10 Customers by CLV:\n", clv.sort_values(by="CLV", ascending=False).head(10))

# Churn Prediction (Customers inactive > 90 days)
latest_date = df['Date'].max()
df['Days Since Last Purchase'] = (latest_date - df.groupby('Customer ID')['Date'].transform('max')).dt.days
churned_customers = df[df['Days Since Last Purchase'] > 90]['Customer ID'].unique()
print("\nTotal Churned Customers:", len(churned_customers))

# Step 4: Market Basket Analysis (Product Affinities)
basket = df.groupby(['Transaction ID', 'Product Category'])['Quantity'].sum().unstack().fillna(0)
basket = basket.applymap(lambda x: 1 if x > 0 else 0)

frequent_itemsets = apriori(basket, min_support=0.05, use_colnames=True)
rules = association_rules(frequent_itemsets, metric="lift", min_threshold=1.2)
print("\nTop Product Associations:\n", rules[['antecedents', 'consequents', 'support', 'confidence', 'lift']].head(10))

# Step 5: Discount Effectiveness Analysis
median_price = df['Price per Unit'].median()
df['Discount Applied'] = df['Price per Unit'] < median_price

discount_sales = df.groupby('Discount Applied')['Total Amount'].sum()

plt.figure(figsize=(8, 5))
discount_sales.plot(kind='bar', color=['red', 'green'])
plt.title("Effect of Discounts on Sales")
plt.xticks(ticks=[0, 1], labels=['No Discount', 'Discount Applied'], rotation=0)
plt.ylabel("Total Sales ($)")
plt.show()

# Step 6: Inventory Demand Forecasting (Improved Prophet Model)
category_forecast = df.groupby(['Date', 'Product Category'])['Quantity'].sum().reset_index()
category_forecast = category_forecast.groupby('Date').sum().reset_index()

category_forecast['y'] = np.log1p(category_forecast['Quantity'])
category_forecast.rename(columns={'Date': 'ds'}, inplace=True)

model = Prophet()
model.fit(category_forecast)

future = model.make_future_dataframe(periods=26, freq='W')
forecast = model.predict(future)
forecast['yhat'] = np.expm1(forecast['yhat'])

plt.figure(figsize=(12, 6))
plt.plot(category_forecast['ds'], np.expm1(category_forecast['y']), label="Actual Demand", marker='o')
plt.plot(forecast['ds'], forecast['yhat'], label="Predicted Demand", linestyle="dashed", color='blue')
plt.fill_between(forecast['ds'], np.expm1(forecast['yhat_lower']), np.expm1(forecast['yhat_upper']), alpha=0.2, color='gray')
plt.legend()
plt.title("Improved Inventory Demand Forecasting")
plt.xlabel("Date")
plt.ylabel("Product Demand")
plt.show()

# Step 7: Sales Forecasting with Prophet (Fixing Overlapping Data)
df_forecast = df.resample('W', on='Date').sum().reset_index()[['Date', 'Total Amount']]
df_forecast.rename(columns={'Date': 'ds', 'Total Amount': 'y'}, inplace=True)

model = Prophet(
    seasonality_mode='multiplicative',
    yearly_seasonality=True,
    weekly_seasonality=True,
    daily_seasonality=False
)
model.add_seasonality(name='monthly', period=30.5, fourier_order=5)
model.fit(df_forecast)

future = model.make_future_dataframe(periods=26, freq='W')
forecast = model.predict(future)

plt.figure(figsize=(12, 6))
plt.plot(df_forecast['ds'], df_forecast['y'], label="Actual Sales", marker='o', alpha=0.7)
plt.plot(forecast['ds'], forecast['yhat'], label="Predicted Sales", linestyle="dashed", color='blue')
plt.fill_between(forecast['ds'], forecast['yhat_lower'], forecast['yhat_upper'], alpha=0.2, color='gray')
plt.ylim(0, df_forecast['y'].max() * 1.2)
plt.legend()
plt.title("Improved Sales Forecast (Prophet Model)")
plt.xlabel("Date")
plt.ylabel("Total Sales ($)")
plt.show()


# Save cleaned data for Tableau
df.to_csv("cleaned_sales_data.csv", index=False)

# Save Forecasted Sales Data
df_forecast[['ds', 'y']].to_csv("sales_forecast.csv", index=False)

# Save Inventory Demand Forecasting Data
category_forecast[['ds', 'y']].to_csv("inventory_forecast.csv", index=False)

print("✅ Cleaned data saved as 'cleaned_sales_data.csv'")
print("✅ Sales forecast saved as 'sales_forecast.csv'")
print("✅ Inventory forecast saved as 'inventory_forecast.csv'")


