from transformers import pipeline
import pandas as pd
import os
import mlflow
from mlflow import log_artifact
from mlflow.models import ModelSignature
import json

#installed keras and tensorflow packages

# 
# Example invocation:
# curl -X POST -H "Content-Type:application/json; format=pandas-split"
#     --data '{"columns":["text"],"data":[["This is meh weather"], ["This is great weather"]]}' http://127.0.0.1:5000/invocations
# Response:
# [{"text": "This is meh weather", "label": "NEGATIVE", "score": 0.753343403339386},
#  {"text": "This is great wuther", "label": "POSITIVE", "score": 0.9984261989593506}]

# Define the model class


class SentimentAnalysis(mlflow.pyfunc.PythonModel):
    '''
    Any MLflow Python model is expected to be loadable as a python_function model.
    '''
    def __init__(self):
        from transformers import pipeline
        self.nlp = pipeline('sentiment-analysis')

    def do_nlp_fnx(self, row):
        s = self.nlp(row['text'])[0]
        return [s['label'], s['score']]

    def predict(self, context, model_input):
        print('model_input=' + str(model_input), flush=True)
        model_input[['label', 'score']] = model_input.apply(
            self.do_nlp_fnx, axis=1, result_type='expand')
        return model_input

# construct and log model


inp = json.dumps([{'name': 'text', 'type': 'string'}])
outp = json.dumps([{'name': 'text', 'type': 'string'}, {
                  'name': 'label', 'type': 'string'}, {'name': 'score', 'type': 'double'}])
signature = ModelSignature.from_dict({'inputs': inp, 'outputs': outp})

with mlflow.start_run():
    mlflow.pyfunc.log_model('model', loader_module=None, data_path=None, code_path=None,
                            conda_env=None, python_model=SentimentAnalysis(),
                            artifacts=None, registered_model_name=None, signature=signature,
                            input_example=None, await_registration_for=0)




