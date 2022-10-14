import pandas as pd
import torch
import time
import numpy as np
from load_nlp_model_helper import predict_batch

model_dir = "models/model/"

model = torch.load(model_dir + "predictor.pt")
model.eval()

tokenizer = torch.load(model_dir + "tokenizer.pt")
labels = list(pd.read_csv(model_dir + "labels.csv", header=None)[0].astype("str").values)


text = ["natural gas price increases as the supply situation worsens", "apples fall from trees in autumn"]

predictions = predict_batch(model, tokenizer, text, labels)
print(predictions)
