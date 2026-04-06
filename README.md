# Amazon Delivery Project Code README

## File
- `FinalProjectMSE433.ipynb`

## What the code does
The notebook builds a full analysis pipeline for the Amazon Delivery Dataset. It:
- uploads a `.csv` file or Kaggle `.zip` file from Google Colab
- loads the delivery data into pandas
- cleans text and datetime fields
- creates operational features such as pickup delay, distance, order hour, weekday, weekend flag, peak-period flags, service-level flags, and a distance–traffic interaction
- removes geographic outliers where estimated delivery distance is greater than 100 km
- creates descriptive plots and summary tables
- trains and compares 3 models:
  - Baseline Median
  - Linear Regression
  - HistGradientBoostingRegressor
- evaluates model performance using MAE, RMSE, and R²
- runs cross-validation on the predictive models
- computes permutation importance for the best model
- exports figures, CSV outputs, and a summary JSON file
- zips all outputs into one downloadable file

## Required Python packages
Install or import these packages before running:

```python
import os
import json
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, KFold, cross_validate
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler, OrdinalEncoder
from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error
from sklearn.linear_model import LinearRegression
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.inspection import permutation_importance
from google.colab import files
```

## Best environment to run it
This notebook is written for **Google Colab**:

```python
from google.colab import files
uploaded = files.upload()
files.download(zip_name)
```

## Input file expected by the notebook
Upload either:
- the raw dataset as a `.csv`, or
- a `.zip` file containing the `.csv`

The notebook automatically picks the first uploaded `.csv` or `.zip` file.

## Required dataset columns
The code expects these columns:
- `Order_ID`
- `Agent_Age`
- `Agent_Rating`
- `Store_Latitude`
- `Store_Longitude`
- `Drop_Latitude`
- `Drop_Longitude`
- `Order_Date`
- `Order_Time`
- `Pickup_Time`
- `Weather`
- `Traffic`
- `Vehicle`
- `Area`
- `Delivery_Time`
- `Category`

## How to run the notebook
1. Open the notebook in Google Colab.
2. Run the import cell.
3. When prompted, upload the dataset file as either `.csv` or `.zip`.
4. Run the rest of the notebook from top to bottom.
5. Wait for the notebook to finish generating figures, tables, and model outputs.
6. The notebook will create a folder called `amazon_delivery_project_outputs`.
7. At the end, it will create and download:
   - `amazon_delivery_project_outputs.zip`

## Main folders and files created
The notebook creates:
- `amazon_delivery_project_outputs/`
- `amazon_delivery_project_outputs/figures/`

### Figures saved
- `01_delivery_time_distribution.png`
- `02_delivery_by_traffic.png`
- `03_distance_vs_delivery.png`
- `04_average_delivery_by_hour.png`
- `05_average_delivery_by_weekday.png`
- `06_delivery_by_vehicle.png`
- `07_delivery_by_area.png`
- `08_pickup_delay_effect.png`
- `09_traffic_hour_heatmap.png`
- `10_control_chart.png`
- `11_model_comparison.png`
- `12_top_predictors.png`
- `13_error_by_traffic.png`
- `14_actual_vs_predicted.png`
- `15_residual_distribution.png`
- `16_service_level_150_by_traffic.png`
- `17_highest_delay_segments.png`
- `18_cv_rmse_comparison.png`

### CSV files saved
- `sla_by_traffic.csv`
- `sla_by_weather.csv`
- `worst_segments.csv`
- `pilot_kpis.csv`
- `traffic_summary.csv`
- `vehicle_summary.csv`
- `area_summary.csv`
- `weather_summary.csv`
- `pickup_delay_summary.csv`
- `model_results.csv`
- `cv_results.csv`
- `top_predictors.csv`
- `error_by_traffic.csv`
- `segment_summary.csv`
- `prediction_diagnostics_sample.csv`

### Other files saved
- `summary_metrics.json`
- `amazon_delivery_project_outputs.zip`

## Core logic inside the notebook
### 1. Data loading
The notebook uses a helper function to read either a CSV directly or a CSV inside a ZIP file.

### 2. Cleaning
It strips whitespace, converts malformed missing values to nulls, parses dates/times, and removes geographic outliers.

### 3. Feature engineering
It creates variables such as:
- `Pickup_Delay_Min`
- `Distance_KM`
- `Order_Hour`
- `Order_Minute`
- `Weekday`
- `Month`
- `Is_Weekend`
- `Peak_Lunch`
- `Peak_Dinner`
- service-level indicator flags
- `Distance_x_Traffic`

### 4. Modeling
The notebook compares:
- a dummy baseline model
- a linear regression model
- a histogram gradient boosting model

### 5. Evaluation
The code reports:
- MAE
- RMSE
- R²
- cross-validation metrics
- permutation importance
- diagnostic plots

## Notes
- The notebook is meant to be run top to bottom in one session.
- If you run it outside Colab, you must replace the file upload and download lines with local file paths.
- If column names differ from the expected dataset format, the notebook will fail unless you rename the columns first.
