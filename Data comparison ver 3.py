import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import make_pipeline

# Load data
exp = pd.read_csv("experiment.csv")
sim = pd.read_csv("simulation.csv")

# Sort data
exp = exp.sort_values("time")
sim = sim.sort_values("time")

# Interpolate simulation data onto experimental time points
sim_interp = np.interp(exp["time"], sim["time"], sim["velocity"])

# Error metrics
error = abs(exp["velocity"] - sim_interp)
avg_error = error.mean()
max_error = error.max()
rmse = np.sqrt(np.mean(error**2))

print(f"Average Error = {avg_error:.2f}")
print(f"Max Error = {max_error:.2f}")
print(f"RMSE = {rmse:.2f}")

# Maximum error point
max_error_index = error.idxmax()
max_error_time = exp["time"].iloc[max_error_index]
exp_value = exp["velocity"].iloc[max_error_index]
sim_interp_value = sim_interp[max_error_index]

# Machine learning fit
X = exp["time"].values.reshape(-1, 1)
y = exp["velocity"].values

model = make_pipeline(
    PolynomialFeatures(degree=3),
    LinearRegression()
)

model.fit(X, y)
y_pred = model.predict(X)

# Conclusion
if avg_error < 0.1:
    conclusion = "Relatively good agreement"
elif avg_error < 0.3:
    conclusion = "Moderate deviation observed"
else:
    conclusion = "Significant deviation observed"

print(f"Conclusion: {conclusion}")

# Plot
plt.figure(figsize=(8, 5))

plt.plot(exp["time"], exp["velocity"], marker="o", label="Experiment")
plt.plot(sim["time"], sim["velocity"], marker="s", label="Simulation")
plt.plot(exp["time"], sim_interp, linestyle="--", marker="x", label="Simulation (Interpolated)")
plt.plot(exp["time"], y_pred, linestyle=":", marker="^", label="Polynomial Regression Fit")

# Highlight max error
plt.plot(max_error_time, exp_value, marker="o", color="red")
plt.plot(max_error_time, sim_interp_value, marker="o", color="red")
plt.plot(
    [max_error_time, max_error_time],
    [exp_value, sim_interp_value],
    color="red",
    linestyle="--",
    label="Maximum Error"
)

plt.annotate(
    f"Max Error = {max_error:.2f}",
    xy=(max_error_time, exp_value),
    xytext=(max_error_time, exp_value + 0.1),
    arrowprops=dict(arrowstyle="->")
)

plt.text(
    0.05, 0.95,
    f"Avg Error = {avg_error:.2f}\nRMSE = {rmse:.2f}\nMax Error = {max_error:.2f}\n{conclusion}",
    transform=plt.gca().transAxes,
    fontsize=10,
    verticalalignment="top",
    bbox=dict(facecolor="white", alpha=0.7)
)

plt.xlabel("Time (s)")
plt.ylabel("Velocity (m/s)")
plt.title("Experiment vs Simulation with Polynomial Regression Fit")
plt.grid()
plt.legend()
plt.tight_layout()
plt.savefig("comparison_plot_polyfeature3_ver3.png", dpi=300)
plt.show()