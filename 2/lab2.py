import polars as pl
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import Ridge, Lasso
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error, mean_absolute_percentage_error
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from dateutil.relativedelta import relativedelta
from datetime import datetime
import warnings
import os

warnings.filterwarnings('ignore')

save_dir = os.path.dirname(os.path.abspath(__file__)) if '__file__' in globals() else os.getcwd()

lf = pl.scan_csv("day.csv")
df = lf.select([
    pl.col("dteday").str.to_date("%Y-%m-%d").alias("Date"),
    pl.col("cnt").cast(pl.Float64).alias("Target"),
    pl.col("temp").cast(pl.Float64),
    pl.col("hum").cast(pl.Float64),
    pl.col("windspeed").cast(pl.Float64)
]).collect()

print(df.describe())

plt.figure(figsize=(12, 5))
plt.plot(df["Date"].to_numpy(), df["Target"].to_numpy(), label='Target')
plt.legend()
plt.savefig(os.path.join(save_dir, "1_timeseries.png"), bbox_inches='tight')
plt.show()

corr = df.select(pl.exclude("Date")).to_pandas().corr()
print(corr)

plt.figure(figsize=(6, 5))
sns.heatmap(corr, annot=True, cmap="coolwarm", fmt=".2f")
plt.savefig(os.path.join(save_dir, "2_correlation.png"), bbox_inches='tight')
plt.show()

df_feat = df.sort("Date").with_columns([
    pl.col("Date").dt.month().alias("month"),
    pl.col("Date").dt.weekday().alias("dayofweek"),
    pl.col("Target").shift(1).alias("lag_1"),
    pl.col("Target").shift(7).alias("lag_7")
]).drop_nulls()

split_date = datetime(2011, 7, 1).date()
train = df_feat.filter(pl.col("Date") < split_date)
test = df_feat.filter(pl.col("Date") >= split_date)

features = ["temp", "hum", "windspeed", "month", "dayofweek", "lag_1", "lag_7"]

X_train, y_train = train.select(features).to_numpy(), train.get_column("Target").to_numpy()
X_test, y_test = test.select(features).to_numpy(), test.get_column("Target").to_numpy()

models = {
    "Ridge": Ridge(),
    "Lasso": Lasso(),
    "GradientBoosting": GradientBoostingRegressor(random_state=42)
}

trained_pipes = {}
for name, m in models.items():
    pipe = Pipeline([("scaler", StandardScaler()), ("model", m)])
    pipe.fit(X_train, y_train)
    trained_pipes[name] = pipe

results = []
for months in range(1, 13):
    end_date = split_date + relativedelta(months=months)
    test_subset = test.filter(pl.col("Date") < end_date)
    
    if test_subset.height == 0:
        continue
        
    X_sub = test_subset.select(features).to_numpy()
    y_sub = test_subset.get_column("Target").to_numpy()
    
    for name, pipe in trained_pipes.items():
        y_pred = pipe.predict(X_sub)
        results.append([
            name, months, 
            r2_score(y_sub, y_pred),
            np.sqrt(mean_squared_error(y_sub, y_pred)),
            mean_absolute_error(y_sub, y_pred),
            mean_squared_error(y_sub, y_pred),
            mean_absolute_percentage_error(y_sub, y_pred) * 100
        ])

res_df = pd.DataFrame(results, columns=["Model", "Months", "R2", "RMSE", "MAE", "MSE", "MAPE"])
print(res_df.to_string(index=False))

plt.figure(figsize=(10, 5))
sns.lineplot(data=res_df, x="Months", y="R2", hue="Model", marker="o")
plt.grid()
plt.savefig(os.path.join(save_dir, "3_r2_vs_months.png"), bbox_inches='tight')
plt.show()

y_pred_gb = trained_pipes["GradientBoosting"].predict(X_test)

plt.figure(figsize=(12, 5))
plt.plot(test["Date"].to_numpy(), y_test, label="Fact", color="black", alpha=0.6)
plt.plot(test["Date"].to_numpy(), y_pred_gb, label="Predict (GB)", color="red", alpha=0.8)
plt.legend()
plt.savefig(os.path.join(save_dir, "4_fact_vs_predict.png"), bbox_inches='tight')
plt.show()