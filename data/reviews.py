import ijson
import zipfile 
import pandas as pd

with zipfile.ZipFile("Books_5.json.zip", "r") as z:

    i = 0
    data = {}

    with open("Books_5.json", "rb") as f:
        for record in ijson.items(f, "item"):

            reviewTime = record["reviewTime"]
            asin = record["asin"]
            reviewerName = record['reviewerName']
            reviewText = record['reviewText']
            summary = record['summary']

            data[i] = [reviewTime, asin, reviewerName, reviewText, summary]
            i += 1



df = pd.DataFrame.from_dict(data, orient='index')

df.to_csv(index=False)





