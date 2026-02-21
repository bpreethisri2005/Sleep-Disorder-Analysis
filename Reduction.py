import pandas as pd
import numpy as np
from sklearn.feature_selection import VarianceThreshold
from sklearn.decomposition import PCA

# Load transformed dataset
transformed_file = r"C:\Users\User\OneDrive\Documents\DataMining Proj\transformed_dataset.xlsx"
df = pd.read_excel(transformed_file)

# ---------------------
# 1. Subset Reduction (Variance + Correlation)
# ---------------------
selector = VarianceThreshold(threshold=0.01)  # Remove low variance features
reduced_data = selector.fit_transform(df.select_dtypes(include=['int64','float64']))

# Get selected feature names
selected_features = df.select_dtypes(include=['int64','float64']).columns[selector.get_support()]

df_subset = pd.DataFrame(reduced_data, columns=selected_features)

# Correlation check
corr_matrix = df_subset.corr().abs()
upper_triangle = corr_matrix.where(~np.tril(np.ones(corr_matrix.shape)).astype(bool))
to_drop = [column for column in upper_triangle.columns if any(upper_triangle[column] > 0.90)]
df_subset.drop(columns=to_drop, inplace=True)

# Save subset reduced dataset
subset_file = r"C:\Users\User\OneDrive\Documents\DataMining Proj\subset_reduced_dataset.xlsx"
df_subset.to_excel(subset_file, index=False)

# ---------------------
# 2. PCA Reduction
# ---------------------
pca = PCA(n_components=5)  # Keep top 5 components (adjustable)
pca_data = pca.fit_transform(df_subset)

df_pca = pd.DataFrame(pca_data, columns=[f"PC{i+1}" for i in range(5)])

# Save PCA reduced dataset
pca_file = r"C:\Users\User\OneDrive\Documents\DataMining Proj\pca_reduced_dataset.xlsx"
df_pca.to_excel(pca_file, index=False)

# ---------------------
# 3. Print Summary
# ---------------------
print(" Data reduction complete!")
print(" Subset Reduction saved as:", subset_file)
print(" PCA Reduction saved as:", pca_file)
print("\nExplained Variance Ratio (PCA):", pca.explained_variance_ratio_)
