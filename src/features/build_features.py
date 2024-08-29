import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from DataTransformation import LowPassFilter, PrincipalComponentAnalysis
from TemporalAbstraction import NumericalAbstraction
from FrequencyAbstraction import FourierTransformation
from sklearn.cluster import KMeans


# --------------------------------------------------------------
# Load data
# --------------------------------------------------------------

df = pd.read_pickle("../../data/interim/02_outliers_removed_chauvenets.pkl")

predictor_columns = list(df.columns[:6])

plt.style.use("fivethirtyeight")
plt.rcParams["figure.figsize"] = (20, 5)
plt.rcParams["figure.dpi"] = 200
plt.rcParams["lines.linewidth"] = 2


# --------------------------------------------------------------
# Dealing with missing values (imputation)
# --------------------------------------------------------------

df.info()

for col in predictor_columns:
    df[col] = df[col].interpolate()

df.info()

# --------------------------------------------------------------
# Calculating set duration
# --------------------------------------------------------------

df[df["set"]==25]["acc_y"]

# calculate mean of a set duration
 
duration = df[df["set"]==25].index[-1] - df[df["set"]==25].index[0]

# loop over all sets

for s in df["set"].unique():
    start = df[df["set"]==s].index[0]  
    stop = df[df["set"]==s].index[-1]
    duration = stop - start
    df.loc[(df["set"]==s), "duration"] = duration.seconds

duration_df = df.groupby(["category"])["duration"].mean()

duration_df.iloc[0] / 5
duration_df.iloc[1] / 10

# --------------------------------------------------------------
# Butterworth lowpass filter (noise)
# --------------------------------------------------------------
df_lowpass = df.copy()

lowpass = LowPassFilter()

fs = 1000 / 200
cutoff = 1.3

df_lowpass = lowpass.low_pass_filter(df_lowpass, "acc_y",fs,cutoff, order=5)

subset = df_lowpass[df_lowpass["set"]==45]
print(subset["label"][0])

fig, ax = plt.subplots(nrows=2, sharex=True, figsize=(20,10))
ax[0].plot(subset["acc_y"].reset_index(drop=True), label="raw data")
ax[1].plot(subset["acc_y_lowpass"].reset_index(drop=True), label="butterworth filter")
ax[0].legend(loc="upper center", bbox_to_anchor=(0.5, 1.15), fancybox=True, shadow=True)
ax[1].legend(loc="upper center", bbox_to_anchor=(0.5, 1.15), fancybox=True, shadow=True)

# filter all columns

for col in predictor_columns:
    df_lowpass = lowpass.low_pass_filter(df_lowpass, col,fs,cutoff, order=5)
    df_lowpass[col] = df_lowpass[col + "_lowpass"]
    del df_lowpass[col + "_lowpass"]



# --------------------------------------------------------------
# Principal component analysis PCA
# --------------------------------------------------------------

df_pca = df_lowpass.copy()
pca = PrincipalComponentAnalysis()

#Determine the explained variance of columns

pc_values = pca.determine_pc_explained_variance(df_pca, predictor_columns)

plt.figure(figsize=(10,10))
plt.plot(range(1, len(predictor_columns) + 1), pc_values)
plt.xlabel("Principal component Number")
plt.ylabel("Explained Variance")
plt.show()

# Determine the optimal number of principal components
# using the Elbow Technique

df_pca = pca.apply_pca(df_pca, predictor_columns, 3)

subset = df_pca[df_pca["set"]== 35]
subset[["pca_1","pca_2","pca_3"]].plot()


# --------------------------------------------------------------
# Sum of squares attributes
# --------------------------------------------------------------

df_squared = df_pca.copy()

acc_r = df_squared["acc_x"] ** 2 + df_squared["acc_y"] + df_squared["acc_z"] ** 2
gyr_r = df_squared["gyr_x"] ** 2 + df_squared["gyr_y"] + df_squared["gyr_z"] ** 2

df_squared["acc_r"] = np.sqrt(acc_r)
df_squared["gyr_r"] = np.sqrt(gyr_r)

subset = df_squared[df_squared["set"]==14]

subset[["acc_r", "gyr_r"]].plot(subplots=True)

# --------------------------------------------------------------
# Temporal abstraction
# --------------------------------------------------------------

df_temporal = df_squared.copy()

numAbs = NumericalAbstraction()

predictor_columns = predictor_columns + ["acc_r", "gyr_r"]

#windows size
ws = int(1000/200)

for col in predictor_columns: 
    df_temporal = numAbs.abstract_numerical(df_temporal, [col], ws, "mean")
    df_temporal = numAbs.abstract_numerical(df_temporal, [col], ws, "std")

#Store by set

df_temporal_list = []

for s in df_temporal["set"].unique():
    subset = df_temporal[df_temporal["set"] == s].copy()
    for col in predictor_columns: 
        df_temporal = numAbs.abstract_numerical(df_temporal, [col], ws, "mean")
        df_temporal = numAbs.abstract_numerical(df_temporal, [col], ws, "std")
    df_temporal_list.append(subset)

df_temporal = pd.concat(df_temporal_list)

# plot one sample

subset[["acc_y", "acc_y_temp_mean_ws_5", "acc_y_temp_std_ws_5"]]
# --------------------------------------------------------------
# Frequency features
# --------------------------------------------------------------

df_frecuency = df_temporal.copy().reset_index()

freabs = FourierTransformation()

fs = int(1000/200) 
ws = int(2800/200)

df_frecuency = freabs.abstract_frequency(data_table=df_frecuency, cols=["acc_y"], window_size=ws, sampling_rate=fs)

df_freq_list = []

for s in df_frecuency["set"].unique():
    print("Applying Fourier Transformations to set {s}")
    subset = df_frecuency[df_frecuency["set"]==s].reset_index(drop=True).copy()
    subset = freabs.abstract_frequency(subset, predictor_columns, ws, fs)
    df_freq_list.append(subset)

df_frecuency = pd.concat(df_freq_list).set_index("epoch (ms)", drop=True)

# --------------------------------------------------------------
# Dealing with overlapping windows
# --------------------------------------------------------------

df_frecuency = df_frecuency.dropna()

df_frecuency = df_frecuency.iloc[::2]


# --------------------------------------------------------------
# Clustering
# --------------------------------------------------------------

df_cluster = df_frecuency.copy()
cluster_columns = ["acc_x","acc_y","acc_z"]
k_values = range(2,10)
inertias = []

for k in k_values:
    subset = df_cluster[cluster_columns]
    kmeans = KMeans(n_clusters=k, n_init=20, random_state=0)
    cluster_labels = kmeans.fit_predict(subset)
    inertias.append(kmeans.inertia_)


kmeans = KMeans(n_clusters=5, n_init=20, random_state=0)
subset = df_cluster[cluster_columns]
df_cluster["cluster"] = kmeans.fit_predict(subset)

#Plot clusters
fig = plt.figure(figsize=(15,15))
ax = fig.add_subplot(projection="3d")
for c in df_cluster["cluster"].unique():
    subset = df_cluster[df_cluster["cluster"]==c]
    ax.scatter(subset["acc_x"], subset["acc_y"], subset["acc_z"], label=c)
ax.set_xlabel("X axis")
ax.set_ylabel("Y axis")
ax.set_zlabel("Z axis")
plt.legend()
plt.show()

# --------------------------------------------------------------
# Export dataset
# --------------------------------------------------------------

df_cluster.to_pickle("../../data/interim/03_data_features.pkl")