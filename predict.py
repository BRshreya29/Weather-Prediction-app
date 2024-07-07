import pandas as pd
import pickle
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error
import matplotlib.pyplot as plt

# Load Data
weather = pd.read_csv("weather.csv", index_col="DATE")

# Handling Missing Values
null_pct = weather.apply(pd.isnull).sum() / weather.shape[0]
valid_columns = weather.columns[null_pct < 0.05]

# New data set with columns usable for prediction model
weather = weather[valid_columns].copy()
weather.columns = weather.columns.str.lower()
weather = weather.ffill()

# Change index to datetime
weather.index = pd.to_datetime(weather.index)

# Add target column
weather["target"] = weather.shift(-1)["tmax"]
weather = weather.ffill()

# Initialize Ridge model
rr = Ridge(alpha=.1)

# Define the predictors (only basic features)
predictors = ["prcp", "snow", "snwd", "tmax", "tmin"]

# Backtest function
def backtest(weather, model, predictors, start=3650, step=90):
    all_predictions = []

    for i in range(start, weather.shape[0], step):
        train = weather.iloc[:i, :]
        test = weather.iloc[i:(i + step), :]

        model.fit(train[predictors], train["target"])

        preds = model.predict(test[predictors])

        preds = pd.Series(preds, index=test.index)
        combined = pd.concat([test["target"], preds], axis=1)

        combined.columns = ["actual", "prediction"]

        combined["diff"] = (combined["prediction"] - combined["actual"]).abs()

        all_predictions.append(combined)

    return pd.concat(all_predictions)

# Make predictions
predictions = backtest(weather, rr, predictors)

# Calculate mean absolute error
mae = mean_absolute_error(predictions["actual"], predictions["prediction"])
print(f"Mean Absolute Error: {mae}")

# Plot prediction differences
predictions["diff"].round().value_counts().sort_index().plot()
plt.show()

# Save the Ridge regression model to a file
with open('weather_predictor.pkl', 'wb') as file:
    pickle.dump(rr, file)