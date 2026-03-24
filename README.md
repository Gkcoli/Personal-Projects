# Personal-Projects

Self Projects

---

## Housing Price Analysis

A personal end-to-end data science project that predicts California housing
prices from neighbourhood and structural features.

### Project structure

```
housing-price-analysis/
├── data/
│   └── housing.csv              # Bundled dataset (20 000 rows)
├── src/
│   ├── data_preprocessing.py    # Load, clean & feature-engineer the data
│   ├── eda.py                   # EDA helpers – summary stats & plots
│   └── model.py                 # Train, evaluate & persist models
├── tests/
│   └── test_pipeline.py         # pytest test suite (22 tests)
├── outputs/                     # Auto-generated plots (created by main.py)
├── models/                      # Saved model artefacts (created by main.py)
├── main.py                      # Full pipeline entry-point
└── requirements.txt
```

### Pipeline overview

| Step | What happens |
|------|--------------|
| **Data loading** | Read `data/housing.csv` (MedInc, HouseAge, AveRooms, …) |
| **Feature engineering** | Add RoomsPerHousehold, BedroomsPerRoom, PopulationPerHousehold |
| **Outlier removal** | Drop rows where MedianHouseValue ≥ 5.0 (capped values) |
| **EDA** | Summary stats, histograms, correlation heatmap, geo-scatter, scatter-vs-target |
| **Modelling** | Compare Ridge Regression, Random Forest, Gradient Boosting |
| **Evaluation** | RMSE, MAE, R² on 20 % hold-out set |
| **Best model** | Predicted-vs-actual plot, residuals, feature importances, saved to `models/` |

### Quick start

```bash
# 1 – install dependencies
cd housing-price-analysis
pip install -r requirements.txt

# 2 – run the full pipeline
python main.py

# 3 – run tests
python -m pytest tests/ -v
```

### Sample results

| Model | RMSE | MAE | R² |
|-------|------|-----|-----|
| Ridge Regression | 0.512 | 0.407 | 0.846 |
| Random Forest | 0.508 | 0.403 | 0.848 |
| **Gradient Boosting** | **0.488** | **0.385** | **0.861** |

All generated plots are saved to the `outputs/` directory automatically.
