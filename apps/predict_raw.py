import numpy as np
from PIL import Image, ImageOps
from fastai.vision import open_image, load_learner, image, torch
import torch
import os

def infer_raw(img):

    models = os.listdir('models_raw/')
    preds = []
    prep_probs = []
    model_names = []
    for m in models:
        model = load_learner('models_raw/'+m+'/')
        pred_class = model.predict(img)[0]
        pred_prob = round(torch.max(model.predict(img)[2]).item()*100)
        preds.append(pred_class)
        prep_probs.append(pred_prob)
        model_names.append(m)
    return (preds,prep_probs,model_names)


def infer_enhanced(img):

    models = os.listdir('models_enhanced/')
    preds = []
    prep_probs = []
    model_names = []
    for m in models:
        model = load_learner('models_enhanced/'+m+'/')
        pred_class = model.predict(img)[0]
        pred_prob = round(torch.max(model.predict(img)[2]).item()*100)
        preds.append(pred_class)
        prep_probs.append(pred_prob)
        model_names.append(m)
    return (preds,prep_probs,model_names)