
import json
import xmltodict
import os
import pandas as pd

CURRENT_DATA_DIR='/Users/aymanehachcham/Downloads/Diachronic Data/NYT'
FILE='TheNewYorkTimes.1985.xml'

with open(os.path.join(CURRENT_DATA_DIR, FILE)) as xml_file:
    data_dict = xmltodict.parse(xml_file.read())

data = data_dict['records']['record']
print(data[100]['fulltext'])


# data_dictionary = {}
#
# for i in range(len(data)):
#     data_dictionary[i] = data[i]
#
# df = pd.DataFrame.from_dict(data_dictionary, orient='index')
#
# print(df.iloc[1]['fulltext'])


