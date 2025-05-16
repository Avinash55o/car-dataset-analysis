import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

# LOAD DATASET
df = pd.read_csv('../data/cars.csv')

# REMOVE THE $ SIGN FROM THE MSRP ADN INVOICE
df["MSRP"] = df["MSRP"].replace('[\$,]', '', regex=True).astype(float)
df["Invoice"] = df["Invoice"].replace('[\$,]', '', regex=True).astype(float)



# BASIC ANALYSIS
print("First 5 rows:\n", df.head())
print("\nSummary Stats:\n", df.describe())
print("\nShape of the dataframe:\n", df.shape)
print("\nMissing Values:\n", df.isnull().sum())


# CORRELATION MATRIX
print("\nCorrelation:\n", df.corr(numeric_only=True))


# TOP 5 EXPENSIVE CAR
top5 = df.sort_values(by='MSRP', ascending=False).head(5)
print("\nTop 5 Most Expensive Cars:\n", top5[['Make', 'Model', 'MSRP']])

# MSRP VS HORSEPOWER
plt.figure(figsize=(8,5))
plt.scatter(df['Horsepower'], df['MSRP'], alpha=0.6)
plt.title('MSRP vs Horsepower')
plt.xlabel('Horsepower')
plt.ylabel('MSRP ($)')
plt.grid(True)
plt.savefig('../output/plots/msrp_vs_horsepower.png')
plt.close()

# DISTRIBUTION OF CITY MPG
plt.figure(figsize=(8,5))
df['MPG_City'].hist(bins=20, color='skyblue', edgecolor='black')
plt.title('Distribution of City MPG')
plt.xlabel('City MPG')
plt.ylabel('Number of Cars')
plt.savefig('../output/plots/city_mpg_distribution.png')
plt.close()


# AVERAGE MSRP BY ORIGIN
plt.figure(figsize=(10, 6))
df.groupby('Origin')['MSRP'].mean().plot(kind='bar', color='skyblue', edgecolor='black')
plt.title('Average MSRP by Origin')
plt.xlabel('Origin')
plt.ylabel('Average MSRP ($)')
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig('../output/plots/average_msrp_by_origin.png')
plt.close()


#COMPARISON WEIGHT HORSEPOWER AND MPG CITY
sample_df = df.sample(40, random_state=1)
plt.figure(figsize=(15,8))
bar_width = 0.25
r1 = np.arange(len(sample_df))
r2 = [x + bar_width for x in r1]
r3 = [x + bar_width for x in r2]
plt.bar(r1, sample_df['Weight'], color='b', width=bar_width, edgecolor='grey', label='Weight')
plt.bar(r2, sample_df['Horsepower'], color='g', width=bar_width, edgecolor='grey', label='Horsepower')
plt.bar(r3, sample_df['MPG_City'], color='r', width=bar_width, edgecolor='grey', label='MPG_City')
plt.xlabel('Car Index', fontweight='bold')
plt.xticks([r + bar_width for r in range(len(sample_df))], sample_df.index, rotation=90)
plt.title('Comparison of Weight, Horsepower, and MPG_City')
plt.legend()
plt.tight_layout()
plt.savefig('../output/plots/comparison_weight_horsepower_mpg_city.png')
plt.show()

# COUNT THE NO OF CARS BY ORIGIN
origin_counts = df['Origin'].value_counts()
plt.figure(figsize=(8, 8))
plt.pie(origin_counts, labels=origin_counts.index, autopct='%1.1f%%', startangle=140, colors=['skyblue', 'orange', 'lightgreen'])
plt.title('Distribution of Cars by Origin')
plt.tight_layout()
plt.savefig('../output/plots/distribution_of_cars_by_origin.png')
plt.show()