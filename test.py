import numpy
import pandas as pd

print (numpy.__path__)


df = pd.DataFrame({'Animal': ['Falcon', 'Falcon',
                              'Parrot', 'Parrot'], 
                              'Max Speed': [380., 370., 24., 26.]})

grouped = df.groupby(["Animal", "Max Speed"])

for group, df_group in grouped:
    print(group)
    print(df_group.head())

start_year = 2019
end_year = 2020

years = [x + start_year for x in range(end_year - start_year + 1)]
print(years)


years = [2019, 2020]

for i, year in years:
    print(i)
    print(years)