import csv
import glob
import json
import random
import logging
import os
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from enum import Enum
from typing import List, Optional, Union

import tqdm
import numpy as np

import torch
from transformers import (
    WEIGHTS_NAME,
    AdamW,
    AutoConfig,
    AutoModelForMaskedLM,
    AutoTokenizer,
)

def evaluate_standard(preds, labels, scoring_method):

    # The accuracy, precision, recall and F1 scores to return
    acc, prec, recall, f1 = 0.0, 0.0, 0.0, 0.0

    ########################################################
    # TODO: Please finish the standard evaluation metrics.
    # You need to compute the accuracy, precision, recall
    # and F1 score for the predictions and gold labels.
    # Please also make your sci-kit learn scores are computed
    # using `scoring_method` for the `average` argument.
    acc = accuracy_score(labels, preds)
    prec = precision_score(labels, preds, average=scoring_method)
    recall = recall_score(labels, preds, average=scoring_method)
    f1 = f1_score(labels, preds, average=scoring_method)
    # End of TODO
    ########################################################

    return acc, prec, recall, f1

def pairwise_accuracy(guids, preds, labels):

    acc = 0.0  # The accuracy to return.
    
    ########################################################
    # TODO: Please finish the pairwise accuracy computation.
    # Hint: Utilize the `guid` as the `guid` for each
    # statement coming from the same complementary
    # pair is identical. You can simply pair the these
    # predictions and labels w.r.t the `guid`. 
    total_pairs = int(len(guids) / 2)
    correct = [0] * total_pairs
    for i in range(len(guids)):
        correct[guids[i]] += 1 if preds[i] == labels[i] else 0
    
    correct_pairs = 0
    for i in range(len(correct)):
        correct_pairs += 1 if correct[i] == 2 else 0
    
    acc = correct_pairs/total_pairs
    # End of TODO
    ########################################################
     
    return acc

if __name__ == "__main__":

    # Unit-testing the pairwise accuracy function.
    guids = [0, 0, 1, 1, 2, 2, 3, 3]
    preds = np.asarray([0, 0, 1, 0, 0, 1, 1, 1])
    labels = np.asarray([1, 0, 1, 1, 0, 1, 1, 1])
    acc, prec, rec, f1 = evaluate_standard(preds, labels, "binary")
    pair_acc = pairwise_accuracy(guids, preds, labels)

    if acc == 0.75 and prec == 1.0 and round(rec,2) == 0.67 and f1 == 0.8:
        print("Your `evaluate_standard` function is correct!")
    else:
        raise NotImplementedError("Your `evaluate_standard` function is INCORRECT!")

    if pair_acc == 0.5:
        print("Your `pairwise_accuracy` function is correct!")
    else:
        raise NotImplementedError("Your `pairwise_accuracy` function is INCORRECT!")
