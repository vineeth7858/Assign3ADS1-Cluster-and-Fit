import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
os.environ["OMP_NUM_THREADS"] = "1"
import warnings
warnings.filterwarnings("ignore", category=UserWarning)
import sklearn.preprocessing as pp
import sklearn.metrics as skmet
from sklearn import cluster
import matplotlib.cm as cm



df_land = pd.read_csv("BBData.csv")
print(df_land.describe())
# The plan is to use 2000 and 2020 for clustering. Countries with one NaN are 
df_land = df_land[(df_land["2005"].notna()) & (df_land["2020"].notna())]
df_land = df_land.reset_index(drop=True)
# extract 2000
growth = df_land[["Country Name", "2005"]].copy()
# and calculate the growth over 60 years
growth["Growth"] = 100.0/60.0 * (df_land["2020"]-df_land["2005"]) / df_land["2005"]
print(growth.describe())
print()
print(growth.dtypes)

plt.figure(figsize=(8, 8))
plt.scatter(growth["2005"], growth["Growth"])
plt.xlabel("Arable land(hect per person),2005")
plt.ylabel("Growth per year [%]")
plt.show()

# create a scaler object
scaler = pp.RobustScaler()
# and set up the scaler
# extract the columns for clustering
df_ex = growth[["2005", "Growth"]]
scaler.fit(df_ex)
# apply the scaling
norm = scaler.transform(df_ex)
plt.figure(figsize=(8, 8))
plt.scatter(norm[:, 0], norm[:, 1])
plt.xlabel("Arable land(hect per person),2005")
plt.ylabel("Growth per year [%]")
plt.show()

def one_silhoutte(xy, n):
    # set up the clusterer with the number of expected clusters
    kmeans = cluster.KMeans(n_clusters=n, n_init=20)
    # Fit the data, results are stored in the kmeans object
    kmeans.fit(xy) # fit done on x,y pairs
    labels = kmeans.labels_
    # calculate the silhoutte score
    score = (skmet.silhouette_score(xy, labels))
    return score

#calculate silhouette score for 2 to 10 clusters
for ic in range(2, 11):
    score = one_silhoutte(norm, ic)
    print(f"The silhouette score for {ic: 3d} is {score: 7.4f}")

# set up the clusterer with the number of expected clusters
kmeans = cluster.KMeans(n_clusters=3, n_init=20)
# Fit the data, results are stored in the kmeans object
kmeans.fit(norm) # fit done on x,y pairs
# extract cluster labels
labels = kmeans.labels_
# extract the estimated cluster centres and convert to original scales
cen = kmeans.cluster_centers_
cen = scaler.inverse_transform(cen)
xkmeans = cen[:, 0]
ykmeans = cen[:, 1]
plt.figure(figsize=(8.0, 8.0))
# plot data with kmeans cluster number
plt.scatter(growth["2005"], growth["Growth"], 10, labels, marker="o", cmap=cm.rainbow)
# show cluster centres
plt.scatter(xkmeans, ykmeans, 45, "k", marker="d")
plt.xlabel("Arable land(hect per person),2005")
plt.ylabel("Growth per year [%]")
plt.show()

print(cen)

growth2 = growth[labels==0].copy()
print(growth2.describe())

df_ex = growth2[["2005", "Growth"]]
scaler.fit(df_ex)
# apply the scaling
norm = scaler.transform(df_ex)
plt.figure(figsize=(8, 8))
plt.scatter(norm[:, 0], norm[:, 1])
plt.xlabel("Arable land(hect per person),2005")
plt.ylabel("Growth per year [%]")
plt.show()


# set up the clusterer with the number of expected clusters
kmeans = cluster.KMeans(n_clusters=3, n_init=20)
# Fit the data, results are stored in the kmeans object
kmeans.fit(norm) # fit done on x,y pairs
# extract cluster labels
labels = kmeans.labels_
# extract the estimated cluster centres and convert to original scales
cen = kmeans.cluster_centers_
cen = scaler.inverse_transform(cen)
xkmeans = cen[:, 0]
ykmeans = cen[:, 1]
plt.figure(figsize=(8.0, 8.0))
# plot data with kmeans cluster number
plt.scatter(growth2["2005"], growth2["Growth"], 10, labels, marker="o", cmap=cm.rainbow)
# show cluster centres
plt.scatter(xkmeans, ykmeans, 45, "k", marker="d")
plt.xlabel("Arable land(hect per person),2005")
plt.ylabel("Growth per year [%]")
plt.show()


# Define the derivative and error propagation functions
def deriv(x, func, coeffs, ip):
    scale = 1e-6
    delta = np.zeros_like(coeffs, dtype=np.float64)  # Ensure delta is float64
    val = scale * np.abs(coeffs[ip])
    delta[ip] = val
    coeffs_plus = coeffs.copy()
    coeffs_plus[ip] += delta[ip]
    coeffs_minus = coeffs.copy()
    coeffs_minus[ip] -= delta[ip]
    f_plus = func(x, coeffs_plus)
    f_minus = func(x, coeffs_minus)
    return (f_plus - f_minus) / (2 * val)

def error_prop(x, func, coeffs, covar):
    var = np.zeros_like(x, dtype=np.float64)  # Ensure var is float64
    for i in range(len(coeffs)):
        for j in range(len(coeffs)):
            var += deriv(x, func, coeffs, i) * deriv(x, func, coeffs, j) * covar[i, j]
    return np.sqrt(var)

# Polynomial function for use with deriv and adapted_error_prop
def polynomial_function(x, coeffs):
    return np.polyval(coeffs, x)

# Load the data from the provided CSV file
file_path = 'BBData_Uk.csv'
data = pd.read_csv(file_path)

# Extract the years and fertility rates
years = np.array(data.columns[1:], dtype=np.float64)
fertility_rates = data.iloc[0, 1:].values.astype(np.float64)

# Normalize the years for the polynomial fitting process
normalized_years = years - np.min(years)

# Fit the polynomial and calculate the covariance matrix
degree_coeffs, cov_matrix = np.polyfit(normalized_years, fertility_rates, 3, cov=True)

# Future years for prediction
future_years_normalized = np.arange(normalized_years[-1] + 1, normalized_years[-1] + 11, dtype=np.float64)

# Calculate future predictions using the polynomial function
future_fertility_rates = polynomial_function(future_years_normalized, degree_coeffs)

# Calculate the confidence intervals for future predictions
confidence_intervals_future = error_prop(future_years_normalized, polynomial_function, degree_coeffs, cov_matrix)

# Plot the original data, polynomial fit, and future predictions with confidence intervals
plt.figure(figsize=(12, 6))
plt.plot(years, fertility_rates, 'b-', label='Original Data')
plt.plot(years, np.polyval(degree_coeffs, normalized_years), 'r-', label='Polynomial Fit')

# Set the y-axis limits to the range of the original data plus some padding
plt.ylim([min(fertility_rates), 70])

# Plot future predictions with confidence intervals
future_years_actual = years[-1] + (future_years_normalized - normalized_years[-1])
plt.plot(future_years_actual, future_fertility_rates, 'r--', label='Future Predictions (Degree 3)')
plt.fill_between(future_years_actual, 
                 future_fertility_rates - confidence_intervals_future, 
                 future_fertility_rates + confidence_intervals_future, 
                 color='red', alpha=0.2, label='95% Confidence Interval')

# Add labels and title
plt.xlabel('Year')
plt.ylabel('Fixed broadband subscriptions (per 100 people)')
plt.title('Polynomial Fit and Future Predictions with Confidence Intervals')
plt.legend(loc='upper left')
plt.show()
