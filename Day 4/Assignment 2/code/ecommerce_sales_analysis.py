"""Assignment 2 — E-Commerce Sales Analysis."""
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

BASE = Path(__file__).resolve().parents[1]
DATA, OUTPUT = BASE / "data", BASE / "output"
OUTPUT.mkdir(exist_ok=True)

df = pd.read_csv(DATA / "ecommerce_orders_raw.csv")
print("Shape:", df.shape)
print(df.info())
print("Missing values:\n", df.isna().sum())
print("Duplicate orders:", df.duplicated(subset="Order_ID").sum())
for column in ["City", "Product_Category", "Payment_Method"]:
    print(f"\n{column}:\n", df[column].value_counts(dropna=False))

df["Rating"] = df["Rating"].fillna(df["Rating"].median())
df["City"] = df["City"].str.strip().str.title()
df = df[df["Quantity"] > 0].drop_duplicates(subset="Order_ID").copy()
df["Order_Date"] = pd.to_datetime(df["Order_Date"])
df["Gross_Amount"] = df["Quantity"] * df["Unit_Price"]
df["Discount_Amount"] = df["Gross_Amount"] * df["Discount"] / 100
df["Net_Amount"] = df["Gross_Amount"] - df["Discount_Amount"]
df["Rating_Category"] = df["Rating"].map({1: "Poor", 2: "Poor", 3: "Average", 4: "Good", 5: "Excellent"})
df["Year"] = df["Order_Date"].dt.year
df["Month"] = df["Order_Date"].dt.to_period("M").astype(str)
df["Day"] = df["Order_Date"].dt.day
df["Day_of_Week"] = df["Order_Date"].dt.day_name()

revenue_by_category = df.groupby("Product_Category")["Net_Amount"].sum().sort_values(ascending=False)
monthly_revenue = df.groupby("Month")["Net_Amount"].sum()
summary = {
    "total_revenue": df["Net_Amount"].sum(), "total_orders": df["Order_ID"].nunique(),
    "average_order_value": df.groupby("Order_ID")["Net_Amount"].sum().mean(),
    "top_city": df.groupby("City")["Net_Amount"].sum().idxmax(),
    "payment_method": df["Payment_Method"].mode().iat[0],
    "highest_revenue_month": monthly_revenue.idxmax(),
    "highest_aov_category": df.groupby("Product_Category")["Net_Amount"].mean().idxmax(),
}
print("\nBusiness summary:\n", pd.Series(summary))
print("\nTop products:\n", df.groupby("Product")["Net_Amount"].sum().nlargest(10))
print("\nTop customers:\n", df.groupby("Customer_ID")["Net_Amount"].sum().nlargest(10))
print("\nAverage ratings:\n", df.groupby("Product_Category")["Rating"].mean())

sns.set_theme(style="whitegrid")
ax = revenue_by_category.plot.bar(figsize=(10, 5), title="Revenue by Product Category", ylabel="Net Revenue")
plt.tight_layout(); plt.savefig(OUTPUT / "revenue_by_category.png", dpi=150); plt.close()
ax = monthly_revenue.plot.line(marker="o", figsize=(10, 5), title="Monthly Revenue", ylabel="Net Revenue")
plt.xticks(rotation=45); plt.tight_layout(); plt.savefig(OUTPUT / "monthly_revenue.png", dpi=150); plt.close()
df.to_csv(OUTPUT / "ecommerce_orders_cleaned.csv", index=False)
