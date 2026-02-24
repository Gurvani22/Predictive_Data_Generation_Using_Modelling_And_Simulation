import numpy as np
import pandas as pd
from gekko import GEKKO

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor



# Logistic Growth Simulation using GEKKO


def logistic_model(growth_rate, carrying_capacity):
    model = GEKKO(remote=False)
    model.time = np.linspace(0, 6, 60)

    population = model.Var(value=5, lb=0)

    model.Equation(population.dt() ==
                   growth_rate * population *
                   (1 - population / carrying_capacity))

    model.options.IMODE = 4
    model.options.SOLVER = 1

    try:
        model.solve(disp=False)
        pop_values = population.value
        return {
            "final": pop_values[-1],
            "maximum": max(pop_values),
            "average": np.mean(pop_values)
        }
    except:
        return None



# Generate Dataset


bounds = {
    "growth_rate": (0.05, 1.8),
    "capacity": (40, 600)
}

data_rows = []

for _ in range(1200):   # slightly more attempts
    r_val = np.random.uniform(*bounds["growth_rate"])
    k_val = np.random.uniform(*bounds["capacity"])

    output = logistic_model(r_val, k_val)

    if output:
        data_rows.append([
            r_val,
            k_val,
            output["final"],
            output["maximum"],
            output["average"]
        ])

dataset = pd.DataFrame(
    data_rows,
    columns=["r", "K", "final_pop", "max_pop", "avg_pop"]
)

print(dataset.head())



# Train-Test Split


features = dataset[["r", "K"]]
target = dataset["final_pop"]

X_train, X_test, y_train, y_test = train_test_split(
    features, target, test_size=0.25, random_state=101
)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)



# Model Training & Evaluation


regressors = [
    ("Linear", LinearRegression()),
    ("Ridge", Ridge(alpha=1.0)),
    ("Lasso", Lasso(alpha=0.01)),
    ("DecisionTree", DecisionTreeRegressor(max_depth=5)),
    ("RandomForest", RandomForestRegressor(n_estimators=100)),
    ("GradientBoost", GradientBoostingRegressor()),
    ("SVR", SVR()),
    ("KNN", KNeighborsRegressor(n_neighbors=5))
]

performance = []

for label, reg in regressors:
    reg.fit(X_train, y_train)
    predictions = reg.predict(X_test)

    r2 = r2_score(y_test, predictions)
    mse = mean_squared_error(y_test, predictions)
    rmse = np.sqrt(mse)

    performance.append({
        "Model": label,
        "R2": r2,
        "RMSE": rmse,
        "MSE": mse
    })

results_df = pd.DataFrame(performance)
results_df = results_df.sort_values(by="R2", ascending=False)

print("\nModel Performance Comparison:\n")
print(results_df)

print("\nTop Performing Model:\n")
print(results_df.iloc[0])
