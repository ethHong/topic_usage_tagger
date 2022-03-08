from sre_parse import Tokenizer
import numpy as np
from load_data import *
import streamlit as st
import pickle
import torch

import requests
import json
from stqdm import stqdm


device = "cuda:0" if torch.cuda.is_available() else "cpu"

from transformers import AutoTokenizer
from transformers import AutoModelForSequenceClassification

tokenizer = AutoTokenizer.from_pretrained("facebook/bart-large-mnli")
nli_model = (
    AutoModelForSequenceClassification.from_pretrained(
        "facebook/bart-large-mnli"
    ).cuda()
    if torch.cuda.is_available()
    else AutoModelForSequenceClassification.from_pretrained("facebook/bart-large-mnli")
)


def get_prob(sequence, label):
    premise = sequence
    hypothesis = f"This example is {label}."

    # run through model pre-trained on MNLI
    x = tokenizer.encode(
        premise, hypothesis, return_tensors="pt", truncation_strategy="only_first"
    )
    logits = nli_model(x.to(device))[0]

    # we throw away "neutral" (dim 1) and take the probability of
    # "entailment" (2) as the probability of the label being true
    entail_contradiction_logits = logits[:, [0, 2]]
    probs = entail_contradiction_logits.softmax(dim=1)
    prob_label_is_true = probs[:, 1]
    return prob_label_is_true[0].item()


def get_taggs(sequence, labels, thred=0.5):
    out = []
    for i in stqdm(range(len(labels))):
        l = labels[i]
        temp = get_prob(sequence, l)
        if temp >= thred:
            out.append((l, temp))
    out = sorted(out, key=lambda x: x[1], reverse=True)
    return out