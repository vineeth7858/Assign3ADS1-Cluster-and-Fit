import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import warnings
import sklearn.preprocessing as pp
import sklearn.metrics as skmet
from sklearn import cluster
import matplotlib.cm as cm
from scipy.optimize import curve_fit

def Getdata(filename):
    """
    Reads a CSV file and returns a DataFrame.

    Parameters:
    filename (str): The path to the CSV file to be read.

    Returns:
    pd.DataFrame: A DataFrame containing the data read from the CSV file.
    """
    df = pd.read_csv(filename)
    return df

def polynomial_fit(x, a, b, c, d):
    """
    Calculates the value of a cubic polynomial at given x.

    Parameters:
    x (number or array): The value(s) at which the polynomial is evaluated.
    a, b, c, d (number): Coefficients of the cubic polynomial.

    Returns:
    number or array: The value of the cubic polynomial at x.
    """
    return a * x**3 + b * x**2 + c * x + d

def error_range(x, f, params, cov_matrix):
    """
    Calculates the error range for a  polynomial function and its parameters.

    Parameters:
    x (number or array): The input value(s) at which the error is evaluated.
    f (function): The polynomial function for which the error is calculated.
    params (list or array): Coefficients of the polynomial.
    cov_matrix (array): The covariance matrix of the polynomial parameters.

    Returns:
    numpy.ndarray: The calculated error values corresponding to each x.
    """
    var = np.zeros_like(x)
    for i in range(len(params)):
        deriv1 = derivative(x, f, params, i)
        for j in range(len(params)):
            deriv2 = derivative(x, f, params, j)
            var += deriv1 * deriv2 * cov_matrix[i, j]
    return np.sqrt(var)

def derivative(x, f, params, index):
    """
    Calculates the derivative of a polynomial function 
    with respect to one of its coefficients.

    Parameters:
    x (number or array): The value(s) at which the derivative is calculated.
    f (function): The polynomial function for which the derivative calculated.
    params (list or array): Coefficients of the polynomial.
    index (int): The index of the coefficient.

    Returns:
    numpy.ndarray or number: The derivative of the polynomial function.
    """
    val = 1e-6
    delta = np.zeros_like(params)
    delta[index] = val * abs(params[index])
    up = params + delta
    low = params - delta
    diff = 0.5 * (f(x, *up) - f(x, *low))
    return diff / (val * abs(params[index]))

def one_silhouette(xy, num_clusters):
    """
    Computes the silhouette score for a given clustering of 2D data.

    Parameters:
    xy (array): 2D data points.
    num_clusters (int): The number of clusters for k-means clustering.

    Returns:
    float: The silhouette score for the clustering.
    """
    kmeans = cluster.KMeans(n_clusters=num_clusters, n_init=20)
    kmeans.fit(xy)
    labels = kmeans.labels_
    score = skmet.silhouette_score(xy, labels)
    return score

Fb_Data = Getdata("BBData.csv")
print(Fb_Data.describe())
# use 2005 and 2020 for clustering. Countries with one NaN are avoided
Fb_Data = Fb_Data[(Fb_Data["2005"].notna()) & (Fb_Data["2020"].notna())]
warnings.filterwarnings("ignore", category=UserWarning)
Fb_Data = Fb_Data.reset_index(drop=True)
# extract 2005
growth = Fb_Data[["Country Name", "2005"]].copy()
# and calculate the growth over 15 years
growth["Growth"] = 100.0/15.0 * (Fb_Data["2020"]-
                                 Fb_Data["2005"]) / Fb_Data["2005"]
print(growth.describe())
print()
print(growth.dtypes)

plt.figure(figsize=(8, 8))
plt.scatter(growth["2005"], growth["Growth"])
plt.xlabel("Fixed broadband subscriptions (per 100 people),2005")
plt.ylabel("Growth per year in percentage")
plt.show()


# create a scaler object
scaler = pp.RobustScaler()
# and set up the scaler
# extract the columns for clustering
Fb_Clust = growth[["2005", "Growth"]]
scaler.fit(Fb_Clust)
# apply the scaling
FB_norm = scaler.transform(Fb_Clust)
plt.figure(figsize=(8, 8))
plt.scatter(FB_norm[:, 0], FB_norm[:, 1])
plt.xlabel("Fixed broadband subscriptions (per 100 people),2005")
plt.ylabel("Growth per year in percentage")
plt.show()


#calculate silhouette score for 2 to 10 clusters
for i in range(2, 11):
    SScore = one_silhouette(FB_norm, i)
    print(f"The silhouette score for {i: 3d} is {SScore: 7.4f}")

# set up the clusterer with the number of expected clusters
kmeans = cluster.KMeans(n_clusters=3, n_init=20)
# Fit the data, results are stored in the kmeans object
kmeans.fit(FB_norm) # fit done on x,y pairs
# extract cluster labels
labels = kmeans.labels_
# extract the estimated cluster centres and convert to original scales
cen = kmeans.cluster_centers_
cen = scaler.inverse_transform(cen)
xkmeans = cen[:, 0]
ykmeans = cen[:, 1]
plt.figure(figsize=(8.0, 8.0))
# plot data with kmeans cluster number
plt.scatter(growth["2005"], growth["Growth"], 10, labels, marker="o", 
            cmap=cm.rainbow, label = 'Data Points')
# show cluster centres
plt.scatter(xkmeans, ykmeans, 45, "k", marker="d", label = 'Cluster Centers')
plt.xlabel("Fixed broadband subscriptions (per 100 people),2005")
plt.ylabel("Growth per year in percentage")
plt.legend()
plt.show()

print(cen)

growth2 = growth[labels==0].copy()
print(growth2.describe())

df_clust2 = growth2[["2005", "Growth"]]
scaler.fit(df_clust2)
# apply the scaling
FB_norm2 = scaler.transform(df_clust2)
plt.figure(figsize=(8, 8))
plt.scatter(FB_norm2[:, 0], FB_norm2[:, 1])
plt.xlabel("Fixed broadband subscriptions (per 100 people),2005")
plt.ylabel("Growth per year in percentage")
plt.show()



# set up the clusterer with the number of expected clusters
kmeans = cluster.KMeans(n_clusters=3, n_init=20)
# Fit the data, results are stored in the kmeans object
kmeans.fit(FB_norm2) # fit done on x,y pairs
# extract cluster labels
labels = kmeans.labels_
# extract the estimated cluster centres and convert to original scales
cen = kmeans.cluster_centers_
cen = scaler.inverse_transform(cen)
xkmeans = cen[:, 0]
ykmeans = cen[:, 1]
plt.figure(figsize=(8.0, 8.0))
# plot data with kmeans cluster number
plt.scatter(growth2["2005"], growth2["Growth"], 10, labels, 
            marker="o", cmap=cm.rainbow, label = 'Data Points')
# show cluster centres
plt.scatter(xkmeans, ykmeans, 45, "k", marker="d", label = 'Cluster Centers')
plt.xlabel("Fixed broadband subscriptions (per 100 people),2005")
plt.ylabel("Growth per year in percentage")
plt.legend()
plt.show()


# Code for Fitting Europe Arable land Data
# Load and transpose Europe land data
FB_UK_Data = Getdata('BBData_Uk.csv')
FB_UK_Data_T = FB_UK_Data.T

# Cleaning the transposed data
FB_UK_Data_T.columns = ['Connection']
FB_UK_Data_T = FB_UK_Data_T.drop('Year')
FB_UK_Data_T.reset_index(inplace=True)
FB_UK_Data_T.rename(columns={'index': 'Year'}, inplace=True)
FB_UK_Data_T['Year'] = FB_UK_Data_T['Year'].astype(int)
FB_UK_Data_T['Connection'] = FB_UK_Data_T['Connection'].astype(float)

# Appending to x and y values for modeling
x_val = FB_UK_Data_T['Year'].values.astype(float)
y_val = FB_UK_Data_T['Connection'].values.astype(float)

# Fitting the polynomial model to the data
popt, pcov = curve_fit(polynomial_fit, x_val, y_val)

# Calculate error ranges for original data
y_err = error_range(x_val, polynomial_fit, popt, pcov)

# Predict for future years and predict values
fut_x = np.arange(max(x_val) + 1, 2031)
fut_y = polynomial_fit(fut_x, *popt)

# Calculate error ranges for predictions
y_fut_err = error_range(fut_x, polynomial_fit, popt, pcov)

# Plotting the fitting data and predicted data
plt.figure(figsize=(10, 6))
plt.plot(x_val, y_val, 'g-', label='Past values')
plt.plot(x_val, polynomial_fit(x_val, *popt), 'b-',
         label='Polynomial fit')
plt.fill_between(x_val, polynomial_fit(x_val, *popt) -
                 y_err, polynomial_fit(x_val, *popt) + y_err, 
            color='lightblue',alpha=0.5, label='Error range of Past values')
plt.plot(fut_x, fut_y, 'b--', label='Future values')
plt.fill_between(fut_x, fut_y - y_fut_err, fut_y +
                 y_fut_err, color='lightblue',
                 alpha=0.5, label='Error range of Future values')
plt.title('Ploynomial Fit & Predicting Future for Fixed Broadband  for UK')
plt.xlabel('Year')
plt.ylabel('Fixed broadband subscriptions (per 100 people)')
plt.legend()
plt.show()
