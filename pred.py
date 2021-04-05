import pandas as pd
import mlflow


# Python program to convert text
# file to JSON


import json


# the file to be converted to
# json format
filename = 'article.txt'

# dictionary where the lines from
# text will be stored
dict1 = {}

# first get all lines from file
with open(filename, 'r') as f:
    lines = f.readlines()

# remove spaces
lines = [line.strip() for line in lines]

# finally, write lines in the file
with open(filename, 'w') as f:
    f.writelines(lines)


with open(filename) as fh:

    for line in fh.readlines():


    # reads each line and trims of extra the spaces
        # and gives only the valid words
        #command, description = line.strip().split(None, 1)

        dict1["text"] = line.strip()

#print(dict1)


logged_model = 'file:///Users/vishwanath/Projects/hf_mlflow/mlruns/0/1149283738fe4a8e8af79e6fdac975e8/artifacts/model'

# Load model as a PyFuncModel.
loaded_model = mlflow.pyfunc.load_model(logged_model)



# Predict on a Pandas DataFrame.
summary = loaded_model.predict(pd.DataFrame(dict1, index=[0]))
print(summary['name'][0]['summary_text'])


 
