"""Assignment 3 — IoT Sensor Data Analysis."""
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

BASE = Path(__file__).resolve().parents[1]
DATA, OUTPUT = BASE / "data", BASE / "output"
OUTPUT.mkdir(exist_ok=True)
df = pd.read_csv(DATA / "iot_sensor_data_raw.csv")
df["Timestamp"] = pd.to_datetime(df["Timestamp"])
df = df.sort_values("Timestamp").reset_index(drop=True)
sensor_columns = ["Temperature", "Humidity", "Pressure", "Vibration", "Battery_Level"]
print("Shape:", df.shape, "\nMissing (%):\n", (df[sensor_columns].isna().mean() * 100).round(2))
df[sensor_columns] = df.groupby("Device_ID")[sensor_columns].transform(lambda x: x.fillna(x.median()))
df[sensor_columns] = df[sensor_columns].fillna(df[sensor_columns].median())

# Thresholds are based on operational risk: temperature >80°C, vibration >5, battery <20%.
df["Battery_Status"] = pd.cut(df["Battery_Level"], [-float("inf"), 20, 50, float("inf")], labels=["Critical", "Moderate", "Healthy"], right=False)
df["Temperature_Status"] = pd.cut(df["Temperature"], [-float("inf"), 70, 80, float("inf")], labels=["Normal", "Warning", "Critical"], right=False)
df["Vibration_Status"] = pd.cut(df["Vibration"], [-float("inf"), 3, 5, float("inf")], labels=["Normal", "Warning", "Critical"], right=False)
df["Machine_Health"] = "Normal"
warning = (df["Temperature_Status"] == "Warning") | (df["Vibration_Status"] == "Warning") | (df["Battery_Status"] == "Moderate")
critical = (df["Temperature_Status"] == "Critical") | (df["Vibration_Status"] == "Critical") | (df["Battery_Status"] == "Critical")
df.loc[warning, "Machine_Health"] = "Warning"
df.loc[critical, "Machine_Health"] = "Critical"
df["Hour"] = df["Timestamp"].dt.hour

print("\nAverage temperature by hour:\n", df.groupby("Hour")["Temperature"].mean())
print("\nDevice summary:\n", df.groupby("Device_ID").agg(avg_temperature=("Temperature","mean"), avg_vibration=("Vibration","mean"), max_temperature=("Temperature","max"), min_battery=("Battery_Level","min")))
critical_devices = df[df["Machine_Health"] == "Critical"].groupby("Device_ID").size()
print("\nDevices critical >5 times:\n", critical_devices[critical_devices > 5])
print("\nPriority (% warning/critical):\n", df.assign(at_risk=df["Machine_Health"].ne("Normal")).groupby("Device_ID")["at_risk"].mean().mul(100).sort_values(ascending=False))

sns.set_theme(style="whitegrid")
df.groupby("Hour")["Temperature"].mean().plot(figsize=(10,5), marker="o", title="Average Temperature by Hour", ylabel="Temperature")
plt.tight_layout(); plt.savefig(OUTPUT / "average_temperature_by_hour.png", dpi=150); plt.close()
pd.crosstab(df["Location"], df["Machine_Health"]).plot.bar(stacked=True, figsize=(10,5), title="Machine Health by Factory")
plt.tight_layout(); plt.savefig(OUTPUT / "machine_health_by_factory.png", dpi=150); plt.close()
df.to_csv(OUTPUT / "iot_sensor_data_cleaned.csv", index=False)
