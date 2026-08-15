"""Assignment 4 — Customer, Product & Order Analysis."""
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

BASE = Path(__file__).resolve().parents[1]
DATA, OUTPUT = BASE / "data", BASE / "output"
OUTPUT.mkdir(exist_ok=True)
customers = pd.read_csv(DATA / "customers.csv")
products = pd.read_csv(DATA / "products.csv")
orders = pd.read_csv(DATA / "orders.csv")
for name, frame in {"customers": customers, "products": products, "orders": orders}.items():
    print(f"{name}: {frame.shape}\nMissing:\n{frame.isna().sum()}\nDuplicates: {frame.duplicated().sum()}")

orders["Order_Date"] = pd.to_datetime(orders["Order_Date"])
customers = customers.drop_duplicates(subset="Customer_ID")
products = products.drop_duplicates(subset="Product_ID")
orders = orders.drop_duplicates(subset="Order_ID")
valid_customer_orders = orders["Customer_ID"].isin(customers["Customer_ID"]).mean() * 100
valid_product_orders = orders["Product_ID"].isin(products["Product_ID"]).mean() * 100
print(f"Valid customer references: {valid_customer_orders:.2f}%\nValid product references: {valid_product_orders:.2f}%")

df = orders.merge(customers, on="Customer_ID", how="inner", validate="many_to_one")
df = df.merge(products, on="Product_ID", how="inner", validate="many_to_one")
df["Order_Value"] = df["Price"] * df["Quantity"]
df["Month"] = df["Order_Date"].dt.to_period("M").astype(str)
df["Age_Group"] = pd.cut(df["Age"], [17, 30, 45, 60, float("inf")], labels=["18–30", "31–45", "46–60", "61+"])

customer_spend = df.groupby(["Customer_ID", "Customer_Name"], as_index=False)["Order_Value"].sum().sort_values("Order_Value", ascending=False)
customer_spend["Spending_Rank"] = customer_spend["Order_Value"].rank(method="dense", ascending=False).astype(int)
monthly_revenue = df.groupby("Month")["Order_Value"].sum()
print("Total revenue:", df["Order_Value"].sum())
print("Average order value:", df["Order_Value"].mean())
print("Revenue by category:\n", df.groupby("Category")["Order_Value"].sum().sort_values(ascending=False))
print("Revenue by city:\n", df.groupby("City")["Order_Value"].sum().sort_values(ascending=False))
print("Top customers:\n", customer_spend.head(10))
print("Top products:\n", df.groupby("Product_Name")["Order_Value"].sum().nlargest(10))
print("Most purchased product:", df.groupby("Product_Name")["Quantity"].sum().idxmax())
print("Highest revenue month:", monthly_revenue.idxmax())
print("Revenue by age group:\n", df.groupby("Age_Group", observed=True)["Order_Value"].sum())
print("Revenue by gender:\n", df.groupby("Gender")["Order_Value"].sum())
print("Top 10% customers:\n", customer_spend.head(max(1, int(len(customer_spend) * .10))))
print("Most profitable category / city:\n", df.groupby(["City", "Category"])["Order_Value"].sum().groupby(level=0).idxmax())
print("Popular product / category:\n", df.groupby(["Category", "Product_Name"])["Quantity"].sum().groupby(level=0).idxmax())

sns.set_theme(style="whitegrid")
df.groupby("Category")["Order_Value"].sum().sort_values().plot.barh(figsize=(10,5), title="Revenue by Category", xlabel="Revenue")
plt.tight_layout(); plt.savefig(OUTPUT / "revenue_by_category.png", dpi=150); plt.close()
monthly_revenue.plot(figsize=(10,5), marker="o", title="Monthly Revenue", ylabel="Revenue")
plt.xticks(rotation=45); plt.tight_layout(); plt.savefig(OUTPUT / "monthly_revenue.png", dpi=150); plt.close()
df.to_csv(OUTPUT / "consolidated_orders.csv", index=False)
customer_spend.to_csv(OUTPUT / "customer_spending_ranking.csv", index=False)
