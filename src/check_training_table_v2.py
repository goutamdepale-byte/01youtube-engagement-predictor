import pandas as pd

training_table = pd.read_csv("C:/Users/Lenovo/youtube_enagagement_project/data/processed/training_table_v2.csv")

print("Shape:", training_table.shape)
print()
print("Category distribution:")
print(training_table["category_id"].value_counts())
print()
print("Target engagement rate stats:")
print(training_table["target_engagement_rate"].describe())
print()
print("Nulls per column:")
print(training_table.isnull().sum())
