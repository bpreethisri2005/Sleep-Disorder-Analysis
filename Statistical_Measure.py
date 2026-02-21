import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import chi2_contingency
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import scipy.stats as stats
from scipy.spatial import distance
from scipy.spatial.distance import pdist, squareform
import gower
from sklearn.preprocessing import LabelEncoder

file_path = r"C:\Users\User\OneDrive\Documents\DataMining Proj\subset_reduced_dataset.xlsx"
df = pd.read_excel(file_path)
df_AF = df.iloc[:, 0:6]
df_selected = df.iloc[:, 0:6]

print("----- Central Tendency Measures -----")

#Mean
print("\nMean values:")
print(df.mean(numeric_only=True))

#Median
print("\nMedian values:")
print(df.median(numeric_only=True))

# Mode
print("\nMode values:")
print(df.mode(numeric_only=True).iloc[0])

print("----- Range, Quartiles & IQR -----")

# Range (max - min)
print("\nRange (Max - Min):")
range_values = df.max(numeric_only=True) - df.min(numeric_only=True)
print(range_values)

# Quartiles
print("\nQuartiles (Q1, Q2, Q3):")
Q1 = df.quantile(0.25, numeric_only=True)
Q2 = df.quantile(0.50, numeric_only=True)  # same as Median
Q3 = df.quantile(0.75, numeric_only=True)
print("Q1:\n", Q1)
print("Q2 (Median):\n", Q2)
print("Q3:\n", Q3)

# Interquartile Range (IQR = Q3 - Q1)
print("\nInterquartile Range (IQR):")
IQR = Q3 - Q1
print(IQR)

print("----- Five Number Summary -----")

five_num_summary = pd.DataFrame({
    "Minimum": df.min(numeric_only=True),
    "Q1 (25%)": df.quantile(0.25, numeric_only=True),
    "Median (Q2)": df.median(numeric_only=True),
    "Q3 (75%)": df.quantile(0.75, numeric_only=True),
    "Maximum": df.max(numeric_only=True)
})

print(five_num_summary)

#Boxplot and Outliers
numeric_cols = df.select_dtypes(include=['number']).columns

print("----- Outlier Detection using IQR -----")
outliers_dict = {}

for col in numeric_cols:
    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    
    # Find outliers
    outliers = df[(df[col] < lower_bound) | (df[col] > upper_bound)][col]
    outliers_dict[col] = outliers.values

    print(f"\n{col}:")
    print(f"Lower Bound = {lower_bound}, Upper Bound = {upper_bound}")
    print(f"Outliers: {outliers.values if len(outliers) > 0 else 'None'}")

# Boxplots for all numeric columns
plt.figure(figsize=(12, 6))
df[numeric_cols].boxplot()
plt.title("Boxplots with Outliers (shown as dots)")
plt.xticks(rotation=45)
plt.show()

#Variance and Standard deviation
print("----- Variance -----")
print(df_AF.var(numeric_only=True))

print("\n----- Standard Deviation -----")
print(df_AF.std(numeric_only=True))

#Covariance and Correlation
print("----- Covariance Matrix -----")
print(df_AF.cov())

print("\n----- Correlation Matrix -----")
print(df_AF.corr())

#Chi Square
contingency_table = pd.crosstab(df.iloc[:,0], df.iloc[:,1])
chi2, p, dof, expected = chi2_contingency(contingency_table)

print("Chi-Square Test Results:")
print("Chi2 Value:", chi2)
print("p-value:", p)
print("Degrees of Freedom:", dof)

# Generate quantile plots
for column in df_selected.columns:
    data = df_selected[column].dropna()
    quantiles = np.linspace(0, 1, len(data))
    sorted_data = np.sort(data)
    
    plt.figure(figsize=(8, 5))
    plt.plot(quantiles, sorted_data, 'o', markersize=3)
    plt.title(f"Quantile Plot for {column}")
    plt.xlabel("Quantiles (0 to 1)")
    plt.ylabel(column)
    plt.grid(True)
    plt.show()

#Q-Q plot
for col in df_AF.columns:
    stats.probplot(df_AF[col], dist="norm", plot=plt)
    plt.title(f"Q–Q Plot for {col}")
    plt.show()

#Histogram
df_AF.hist(figsize=(12, 8), bins=15)
plt.suptitle("Histograms of A–F Columns")
plt.show()

#Scatter Plot
pd.plotting.scatter_matrix(df_AF, figsize=(12, 8), diagonal='hist')
plt.suptitle("Scatter Matrix Plot")
plt.show()

#Pearson and Cosine Similarity
print("----- Pearson Correlation -----")
print(df_AF.corr(method='pearson'))

print("\n----- Cosine Similarity -----")
cos_sim = cosine_similarity(df_AF.T)  # Transposed for column-wise similarity
cosine_df = pd.DataFrame(cos_sim, index=df_AF.columns, columns=df_AF.columns)
print(cosine_df)

#Minkowshki distance
row1 = df_AF.iloc[0]
row2 = df_AF.iloc[1]

minkowski_dist = distance.minkowski(row1, row2, p=3) 
print("Minkowski Distance (p=3) between Row1 & Row2:", minkowski_dist)

#Disimilarity measures
euclidean = distance.euclidean(row1, row2)
manhattan = distance.cityblock(row1, row2)
supremum = distance.chebyshev(row1, row2)

print("Euclidean Distance:", euclidean)
print("Manhattan Distance:", manhattan)
print("Supremum (Chebyshev) Distance:", supremum)

# Euclidean Dissimilarity Matrix
euclidean_matrix = squareform(pdist(df_AF, metric='euclidean'))
print("Euclidean Dissimilarity Matrix:\n", euclidean_matrix)

# Manhattan Dissimilarity Matrix
manhattan_matrix = squareform(pdist(df_AF, metric='cityblock'))
print("\nManhattan Dissimilarity Matrix:\n", manhattan_matrix)

# Supremum (Chebyshev) Dissimilarity Matrix
supremum_matrix = squareform(pdist(df_AF, metric='chebyshev'))
print("\nSupremum (Chebyshev) Dissimilarity Matrix:\n", supremum_matrix)

#SIMMILARITY AND DISSIMILARITY FOR(ORDINAL,NOMINAL,BINARY)
df_nominal = df_selected.astype(str).apply(LabelEncoder().fit_transform)
def simple_matching(u, v):
    return np.sum(u != v) / len(u)
nominal_dist = squareform(pdist(df_nominal.values, metric=simple_matching))
print("\nNominal/Categorical (Simple Matching):")
print(pd.DataFrame(nominal_dist).round(2))

df_binary = df_selected.applymap(lambda v: 1 if v > df_selected.median().median() else 0)
binary_dist = squareform(pdist(df_binary.values, metric="jaccard"))
print("\nBinary (Jaccard Distance):")
print(pd.DataFrame(binary_dist).round(2))

df_ordinal = df_selected.rank(method="average") / df_selected.rank(method="average").max()
ordinal_dist = squareform(pdist(df_ordinal.values, metric="euclidean"))
print("\nOrdinal (Euclidean on Ranks):")
print(pd.DataFrame(ordinal_dist).round(2))

#Mixed Attribute
# Compute Gower distance (dissimilarity) matrix
gower_dist = gower.gower_matrix(df_selected)

# Round for better readability
gower_dist = np.round(gower_dist, 2)
print("\n---Mixed Attributes---")
print(pd.DataFrame(gower_dist))