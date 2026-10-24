# 🚇 Delhi Metro Footfall — EDA & Machine Learning (v2: full-year update)

An end-to-end data project on a **synthetic Delhi Metro ridership dataset**: exploratory data analysis, unsupervised learning, and regression models (including ensembles) that predict daily footfall per station.

> **This is v2 of my earlier Delhi Metro EDA project.** v1 analysed only four months (Jan–Apr 2024) and stopped at EDA. v2 uses the **whole of 2024**, rebuilds the EDA with seasonality, festivals and disruptions, and adds a **full machine-learning pipeline**.

---

## What changed from v1 to v2

| | v1 | v2 |
|---|---|---|
| Period | Jan–Apr 2024 (120 days) | **Full year 2024 (366 days)** |
| Rows / columns | ~11.8k / 28 | **35,893 raw (35,868 after de-duplication) / 37** |
| Weather | 5 conditions (winter and spring only) | **12 conditions** (heatwave, rain, thunderstorm, fog …) |
| Seasonality | none | Summer-vacation dip, monsoon dip, festive-season peak, ~4 % yearly growth |
| Events | festival flag only | **Festivals, a July protest near Connaught Place with station closures, security closures, planned maintenance, waterlogging, signal failures, Kanwar Yatra diversions, pollution curbs** |
| Station attributes | re-randomised every day (a flaw) | **Fixed per station**, plus a station-type column |
| Analysis | EDA only | **EDA + clustering + anomaly detection + regression models + ensembles** |

---

## Key findings (EDA)

- **Interchange stations carry ~3.3× the traffic of regular stations** (median ~33,300 vs ~10,000 riders/day). They are 32 % of records but ~60 % of all footfall.
- **Strong seasonal cycle:** median footfall bottoms out in June (~9,900) and peaks in Oct–Nov (~13,800) — a ~40 % swing.
- **The weekend effect depends on station type:** commuter stations drop ~40 %, market/leisure stations barely drop, campus stations drop ~65 %.
- **Bad weather (heatwave, rain, thunderstorm, fog) cuts footfall ~15 %** versus good-weather days of the same month. AQI has no independent effect (the raw link is a seasonal coincidence).
- **Closures show up as near-zero outliers that the usual IQR rule misses** — a per-station baseline ratio catches them. In the simulated July protest, Rajiv Chowk fell from ~36,000 to 16 and 24 riders, while neighbouring stations such as New Delhi gained riders (spillover).

## Machine-learning results

Target: `Total_Footfall` per station per day. The data is split **by day** (292 days train / 74 days test) to avoid leakage, and models are trained on `log(1 + footfall)`.

| Model | R² | MAE (riders) | WAPE |
|---|---|---|---|                                                    
|Voting ensemble | 0.963 | 1496.702 | 8.411 |
|Linear Regression (log target) | 0.962 | 1508.180 | 8.476 |
|Ridge (log target) | 0.962 | 1510.602 | 8.489 |
|Gradient Boosting  | 0.962 | 1516.640 | 8.523 |
|XGBoost | 0.961 | 1521.397 | 8.550 |
|Random Forest | 0.961 | 1532.281 | 8.611 |
|Decision Tree | 0.954 | 1641.917 | 9.227 |
|Linear Regression (raw target) | 0.951 | 1782.293 |10.016 |
|Baseline 2: station typical value | 0.925 | 2174.629 |12.221 |
|Baseline 1: overall average  |-0.004 | 9858.094 |55.400 |
*(WAPE = total absolute error ÷ total actual riders. Results from `notebooks/02_ml_simple.ipynb`.)*

**What I learned**
- Taking the **log of the target** mattered more than the choice of algorithm (9.9 % → 8.4 % for the same linear model).
- **Ensembles tied with simple linear regression** rather than beating it: the synthetic data has ~8 % random noise that no model can predict, so every good model sits near the ceiling.
- In the harder **forward-in-time test** (train Jan–Sep, predict Oct–Dec), combining models helped: voting ensemble **11.7 %** vs Ridge 13.3 % vs baseline 15.9 %. All models under-predicted the unseen festive season (see chart below).
- **Unsupervised learning:** K-Means grouped stations into 4 types (busy hubs, market/airport, residential, far-out), and anomaly detection recovered most closures without labels.
---

## Repository structure

```
delhi-metro-footfall/
├── README.md
├── data/
│   └── delhi_metro_ridership_dataset_v2.csv     # v2: full year 2024
├── notebooks/
│   ├── delhi-metro-ridership-v2-eda.ipynb         # updated EDA on the full year   (add your EDA notebook here)
│   ├── ml-model-delhi-metro-footfall.ipynb       # ML: baselines → linear → trees → ensembles → K-Means
└── report/                      # project report (add the PDF here)
```

## Limitations

- Synthetic data: relationships are cleaner than in real ridership data.
- One year of data cannot teach a model about a second Diwali or year-on-year growth.
- Event information is only known in advance for scheduled events; unplanned ones would need live data.
- Some zones contain a single station, so zone-level conclusions rest on one station.

⭐ If this was useful, consider starring the repo. Feedback and suggestions are welcome — open an issue or message me.
