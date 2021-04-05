# from transformers import pipeline

# # Open and read the article
# f = open("article.txt", "r", encoding="utf8")
# to_tokenize = f.read()

# # Initialize the HuggingFace summarization pipeline
# summarizer = pipeline("summarization")
# summarized = summarizer(to_tokenize, min_length=75, max_length=100)

# # Print summarized text
# print(summarized)

from transformers import pipeline
import pandas as pd
import os
import mlflow
from mlflow import log_artifact
from mlflow.models import ModelSignature
import json
from mlflow.tracking import MlflowClient


class Summarizer(mlflow.pyfunc.PythonModel):
    '''
    Any MLflow Python model is expected to be loadable as a python_function model.
    '''
    def __init__(self):
        from transformers import pipeline, AutoTokenizer, AutoModelForSeq2SeqLM

        #self.summarize = pipeline('summarization',model="t5-small", tokenizer="t5-small")

        tokenizer = AutoTokenizer.from_pretrained(
            "deep-learning-analytics/wikihow-t5-small")

        self.summarize = AutoModelForSeq2SeqLM.from_pretrained(
            "deep-learning-analytics/wikihow-t5-small")

        

    def summarize_article(self, row):
        # s = self.summarize(row['text'],min_length=75, max_length=500)[0]
        s = self.summarize(row['text'], min_length=75, max_length=500)[0]
        
        return [s]

    def predict(self, context, model_input):
        #print('model_input=' + str(model_input), flush=True)
        model_input[['name']] = model_input.apply(
            self.summarize_article, axis=1, result_type='expand')
        return model_input






input = json.dumps([{'name': 'text', 'type': 'string'}])
output = json.dumps([{'name': 'text', 'type': 'string'}])
signature = ModelSignature.from_dict({'inputs': input, 'outputs': output})

# experiment_name = "text_summarizer"
# print("experiment_name:", experiment_name)
# mlflow.set_experiment(experiment_name)
# client = mlflow.tracking.MlflowClient()
# experiment_id = client.get_experiment_by_name(experiment_name).experiment_id
# print("experiment_id:", experiment_id)



mlflow.set_tracking_uri("")
tracking_uri = mlflow.get_tracking_uri()
print("Current tracking uri: {}".format(tracking_uri))


with mlflow.start_run(run_name="hf_summarizer") as run:
    print(run.info.run_id)
    print("mlflow models serve -m runs:/" + run.info.run_id + "/model --no-conda")
    mlflow.pyfunc.log_model('model', loader_module=None, data_path=None, code_path=None,
                            conda_env=None, python_model=Summarizer(),
                            artifacts=None, registered_model_name=None, signature=signature,
                            input_example=None, await_registration_for=0)

    #mlflow.register_model("runs:/" + run.info.run_id + "/model", "summarizer")

# curl -X POST -H "Content-Type:application/json; format=pandas-split" --data '{"columns":["text"],"data":[["H.P.Lovecraft wrote his best books in Masachusettes."]]}' http://127.0.0.1: 5000/invocations

# curl -X POST -H "Content-Type:application/json; format=pandas-split" --data '{"columns":["text"],"data":[@article.txt]}' http: // 127.0.0.1: 5000/invocations

# mlflow server --backend-store-uri sqlite:///mlflow.db --default-artifact-root ./mlrun --host 0.0.0.0
