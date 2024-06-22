import pandas as pd
import numpy as np
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score

# Read data from the CSV file
data = pd.read_csv('data.csv')
time = data['time']
concentration = data['concentration']

# Scale the data to avoid overflow issues
time_scaled = time / max(time)
concentration_scaled = concentration / max(concentration)

# Define the model functions for the fits
def linear_model(t, a, b):
    return a * t + b

def exponential_model(t, a, b):
    return a * np.exp(b * t)

def logarithmic_model(t, a, b):
    return a * np.log(t + 1) + b  # Adding 1 to avoid log(0)

# Perform non-linear least squares fitting
# Linear regression
params_linear, _ = curve_fit(linear_model, time_scaled, concentration_scaled)
predicted_linear = linear_model(time_scaled, *params_linear)
r2_linear = r2_score(concentration_scaled, predicted_linear)

# Exponential regression
initial_params_exponential = [1, 0.1]  # Initial guess for the parameters
params_exponential, _ = curve_fit(exponential_model, time_scaled, concentration_scaled, p0=initial_params_exponential)
predicted_exponential = exponential_model(time_scaled, *params_exponential)
r2_exponential = r2_score(concentration_scaled, predicted_exponential)

# Logarithmic regression
initial_params_logarithmic = [1, 1]  # Initial guess for the parameters
params_logarithmic, _ = curve_fit(logarithmic_model, time_scaled, concentration_scaled, p0=initial_params_logarithmic)
predicted_logarithmic = logarithmic_model(time_scaled, *params_logarithmic)
r2_logarithmic = r2_score(concentration_scaled, predicted_logarithmic)

# Save the results to a file
with open('fit_results.txt', 'w') as file:
    file.write("Lineare Regression:\n")
    file.write(f"Parameter: a={params_linear[0]}, b={params_linear[1]}\n")
    file.write(f"R²: {r2_linear}\n\n")

    file.write("Exponentielle Regression:\n")
    file.write(f"Parameter: a={params_exponential[0]}, b={params_exponential[1]}\n")
    file.write(f"R²: {r2_exponential}\n\n")

    file.write("Logarithmische Regression:\n")
    file.write(f"Parameter: a={params_logarithmic[0]}, b={params_logarithmic[1]}\n")
    file.write(f"R²: {r2_logarithmic}\n")

# Create the plots (optional)
plt.figure(figsize=(10, 6))
plt.scatter(time, concentration, label='Daten', color='black')
plt.plot(time, predicted_linear * max(concentration), label='Linear', color='red')
plt.plot(time, predicted_exponential * max(concentration), label='Exponential', color='blue')
plt.plot(time, predicted_logarithmic * max(concentration), label='Logarithmisch', color='green')
plt.xlabel('Zeit')
plt.ylabel('Konzentration')
plt.legend()
plt.title('Fits der Konzentrationsdaten')
plt.savefig('fits_plot.png')
plt.show()
