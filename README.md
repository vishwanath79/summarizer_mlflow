# Text Summarizer on Hugging Face with mlflow

## Table of Contents

- [About](#about)
- [Sample Invocation](#sampleinvocation)


## About <a name = "about"></a>

Uses the Hugging Face pipeline to run predictions i.e text summarization on a block of text.

Details here: https://vishsubramanian.me/hugging-face-with-mlflow/



## Sample invocation <a name = "sampleinvocation"></a>

curl -X POST -H "Content-Type:application/json; format=pandas-split" --data '{"columns":["text"],"data":[["Sample Text"]]}' http://127.0.0.1:5000/invocations

or run predict_text.py




