import pandas as pd

data = pd.read_csv('./emnist-balanced-train.csv', header=None)
labels = data.values[:, 0]

print("Min label:", labels.min())
print("Max label:", labels.max())
print("Unique labels:", sorted(set(labels)))

print("Columns:", data.columns)
print(data.head())