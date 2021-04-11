from transformers import pipeline
import pandas as pd
import os
import mlflow
from mlflow import log_artifact
from mlflow.models import ModelSignature
import json
from mlflow.tracking import MlflowClient

# Wrapper class for the sentiment analysis task pipeline using the boiler plate code


class Summarizer(mlflow.pyfunc.PythonModel):
    '''
    Any MLflow Python model is expected to be loadable as a python_function model.
    '''

    def __init__(self):
        from transformers import pipeline, AutoTokenizer, AutoModelWithLMHead

        self.tokenizer = AutoTokenizer.from_pretrained(
            "deep-learning-analytics/wikihow-t5-small")

        self.summarize = AutoModelWithLMHead.from_pretrained(
            "deep-learning-analytics/wikihow-t5-small")

    def summarize_article(self, row):
        tokenized_text = self.tokenizer.encode(row[0], return_tensors="pt")

        # T5-small model trained on Wikihow All data set.
        # model was trained for 3 epochs using a batch size of 16 and learning rate of 3e-4.
        # Max_input_lngth is set as 512 and max_output_length is 150.
        s = self.summarize.generate(
            tokenized_text,
            max_length=150,
            num_beams=2,
            repetition_penalty=2.5,
            length_penalty=1.0,
            early_stopping=True)

        s = self.tokenizer.decode(s[0], skip_special_tokens=True)
        return [s]

    def predict(self, context, model_input):
        model_input[['name']] = model_input.apply(
            self.summarize_article)

        return model_input


# Input and Output formats
input = json.dumps([{'name': 'text', 'type': 'string'}])
output = json.dumps([{'name': 'text', 'type': 'string'}])
# Load model from spec
signature = ModelSignature.from_dict({'inputs': input, 'outputs': output})

#MLFlow Operations
mlflow.set_tracking_uri("")
tracking_uri = mlflow.get_tracking_uri()
print("Current tracking uri: {}".format(tracking_uri))

# Start tracking
with mlflow.start_run(run_name="hf_summarizer") as run:
    print(run.info.run_id)
    runner = run.info.run_id
    print("mlflow models serve -m runs:/" +
          run.info.run_id + "/model --no-conda")
    mlflow.pyfunc.log_model('model', loader_module=None, data_path=None, code_path=None,
                            conda_env=None, python_model=Summarizer(),
                            artifacts=None, registered_model_name=None, signature=signature,
                            input_example=None, await_registration_for=0)

#curl -X POST -H "Content-Type:application/json; format=pandas-split" --data '{"columns":["text"],"data":[["H.P.Lovecraft wrote his best books in Masachusettes."]]}' http://127.0.0.1: 5000/invocations


#curl -X POST -H "Content-Type:application/json; format=pandas-split" --data '{"columns":["text"],"data":[["Howard Phillips Lovecraft (US: /ˈlʌvkræft/; August 20, 1890 – March 15, 1937) was an American writer of weird and horror fiction, who is known for his creation of what became the Cthulhu Mythos.[n 1]Born in Providence, Rhode Island, Lovecraft spent most of his life in New England. He was born into affluence, but his family's wealth dissipated soon after the death of his grandfather. In 1913, he wrote a critical letter to a pulp magazine that ultimately led to his involvement in pulp fiction.H.P.Lovecraft wrote his best books in Masachusettes."]]}' http://127.0.0.1: 5000/invocations
