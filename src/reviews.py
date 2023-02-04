import numpy as np
import pandas as pd
import json

with open('Books_5.json', 'r') as f:
    i = 0
    data = {}

    for line in f:
        a = json.loads(line)

        if 'reviewText' not in a.keys():
            continue

        else:
            if a['verified'] == False:
                continue

            elif int(a['reviewTime'].split()[-1]) < 2013:
                continue

            elif i < 5000000:
                data[i] = json.loads(line)
                i += 1
                print(i)
            else:
                break
         
df = pd.DataFrame.from_dict(data, orient='index')
df['time'] = pd.to_datetime(df['reviewTime'], infer_datetime_format=True)
df.drop(['reviewTime', 'reviewerID', 'style', 'reviewerName', 'unixReviewTime', 'vote', 'image', 'summary', 'verified'], axis = 1, inplace= True)
df['year'] = df['time'].dt.year


df_2013 = df[df['year'] == 2013]
df_2013.to_csv('df_2013.csv')

df_2014 = df[df['year'] == 2014]
df_2014.to_csv('df_2014.csv')

df_2015 = df[df['year'] == 2015]
df_2015.to_csv('df_2015.csv')

df_2016 = df[df['year'] == 2016]
df_2016.to_csv('df_2016.csv')

df_2017 = df[df['year'] == 2017]
df_2017.to_csv('df_2017.csv')

df_2018 = df[df['year'] == 2018]
df_2018.to_csv('df_2018.csv')
