import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from sklearn.cluster import KMeans
import scipy.stats as stats 

# Saving the files in variables
file_path_data = 'API_19_DS2_en_csv_v2_6300757.csv'
file_country_metadata = 'Metadata_Country_API_19_DS2_en_csv_v2_6300757.csv'
file_indicator_metadata = 'Metadata_Indicator_API_19_DS2_en_csv_v2_6300757.csv'

# Convert these datas into dataframes
data = pd.read_csv(file_path_data, skiprows=4)
country_metadata = pd.read_csv(file_country_metadata)
indicator_metadata = pd.read_csv(file_indicator_metadata)

columns_for_clustering = ['1960', '2022']
data_for_clustering = data[columns_for_clustering].dropna()

kmeans = KMeans(n_clusters=3)  
kmeans.fit(data_for_clustering)
cluster_labels = kmeans.labels_
cluster_centers = kmeans.cluster_centers_

data_for_clustering['Cluster'] = cluster_labels
#print(data_for_clustering)

plt.figure(figsize=(8, 6))

for cluster_num in range(3):
    cluster_data = data_for_clustering[data_for_clustering['Cluster'] == 
                                       cluster_num]
    plt.bar(cluster_num, cluster_data['1960'].mean(), label=f'Cluster \
            {cluster_num}: 1960', alpha=0.7)
    plt.bar(cluster_num + 0.35, cluster_data['2022'].mean(), 
            label=f'Cluster {cluster_num}: 2022', alpha=0.7)

plt.xlabel('Cluster Number')
plt.ylabel('Average Indicator Value')
plt.title('Average Indicator Values for Each Cluster in 1960 and 2022')
plt.legend()
plt.xticks(np.arange(3) + 0.35 / 2, labels=[f'Cluster {i}' for i in range(3)])
plt.grid(axis='y')
plt.show()


cluster_analysis = data_for_clustering.groupby('Cluster').agg({'1960': 'mean', 
                                                               '2022': 'mean'})
print(cluster_analysis)
for cluster_num in range(len(cluster_analysis)):
    cluster_mean_1960 = cluster_analysis.loc[cluster_num, '1960']
    cluster_mean_2022 = cluster_analysis.loc[cluster_num, '2022']
    print(f"Cluster {cluster_num}:")
    print(f"  Mean value of Indicator in 1960: {cluster_mean_1960:.2f}")
    print(f"  Mean value of Indicator in 2022: {cluster_mean_2022:.2f}")

years = np.array(data.columns[5:])
values = np.array(data.iloc[0, 5:])
cleaned_values = np.array([float(value) 
                            if str(value).replace('.', '', 1).isdigit() 
                            else np.nan for value in values])
valid_years = years[~np.isnan(cleaned_values)]
valid_cleaned_values = cleaned_values[~np.isnan(cleaned_values)]

def exponential_model(x, a, b):
    return a * np.exp(b * (x - int(valid_years[0])))

popt, pcov = curve_fit(exponential_model, valid_years.astype(int), 
                        valid_cleaned_values.astype(float))

def err_ranges(popt, pcov, alpha=0.05):
    perr = np.sqrt(np.diag(pcov))
    z_score = np.abs(stats.norm.ppf(alpha / 2))
    lower = popt - z_score * perr
    upper = popt + z_score * perr
    return lower, upper

lower, upper = err_ranges(popt, pcov)
future_years = np.arange(int(valid_years[0]), int(valid_years[0]) + 20)
predicted_values = exponential_model(future_years, *popt)
lower_bound, upper_bound = exponential_model(future_years, *lower), \
                            exponential_model(future_years, *upper)

cluster_analysis = data_for_clustering.groupby('Cluster').agg({'1960': 'mean', 
                                                               '2022': 'mean'})
print(cluster_analysis)

for cluster_num in range(len(cluster_analysis)):
    cluster_mean_1960 = cluster_analysis.loc[cluster_num, '1960']
    cluster_mean_2022 = cluster_analysis.loc[cluster_num, '2022']
    print(f"Cluster {cluster_num}:")
    print(f"  Mean value of Indicator in 1960: {cluster_mean_1960:.2f}")
    print(f"  Mean value of Indicator in 2022: {cluster_mean_2022:.2f}")

plt.show()


plt.figure(figsize=(10, 6))
plt.plot(future_years, predicted_values, color='red', label='Exponential Fit')
plt.fill_between(future_years, lower_bound, upper_bound, alpha=0.3, 
                 color='orange', label='Confidence Intervals')
plt.plot(valid_years.astype(int), valid_cleaned_values.astype(float), 'bo', 
         label='Data Points')
plt.xlabel('Years')
plt.ylabel('Values')
plt.title('Exponential Curve Fitting with Confidence Intervals')
plt.legend()
plt.grid(True)
plt.text(2020, 500, f'Equation: y = {popt[0]:.2f} * exp({popt[1]:.2f} * \
                               (x - {int(valid_years[0])}))', fontsize=10)

plt.figure(figsize=(10, 6))
plt.plot(future_years, predicted_values, color='red', label='Exponential Fit')
plt.fill_between(future_years, lower_bound, upper_bound, alpha=0.3, 
                 color='orange', label='Confidence Intervals')
plt.xlabel('Years')
plt.ylabel('Values')
plt.title('Exponential Curve Fitting with Confidence Intervals')
plt.legend()
plt.grid(True)
plt.show()

# Calling data again because prevoius data has been changed
file_country_metadata = 'Metadata_Country_API_19_DS2_en_csv_v2_6300757.csv'
country_metadata = pd.read_csv(file_country_metadata)
selected_columns = ['Region', 'IncomeGroup', 'SpecialNotes']  
subset_data = country_metadata[selected_columns]
numeric_columns = subset_data.select_dtypes(include=['float64', 
                                                     'int64']).columns.tolist()
subset_data[numeric_columns] = subset_data[numeric_columns]. \
                               fillna(subset_data[numeric_columns].mean())
non_numeric_columns = subset_data.select_dtypes(exclude=['float64', 
                                                    'int64']).columns.tolist()
subset_data[non_numeric_columns] = subset_data[non_numeric_columns]. \
                                   fillna("Unknown")
subset_data_encoded = pd.get_dummies(subset_data)
num_clusters = 3 
kmeans = KMeans(n_clusters=num_clusters)
subset_data['Cluster'] = kmeans.fit_predict(subset_data_encoded)

plt.figure(figsize=(10, 6))
scatter = plt.scatter(subset_data[selected_columns[0]], 
                      subset_data[selected_columns[1]], 
                      c=subset_data['Cluster'], cmap='plasma', 
                      edgecolors='black')
plt.xlabel('Region')
plt.ylabel('Income Group')
plt.xticks(rotation=45)
plt.title('Clustering Countries')
plt.colorbar(scatter, label='Cluster')
plt.grid(True)
plt.show()


# Github: https://github.com/Faraz291/ADS1-assignment3
