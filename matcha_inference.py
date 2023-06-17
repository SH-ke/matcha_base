from tqdm.auto import tqdm
import matplotlib.pyplot as plt
import os, sys, cv2, json, glob, csv
from log_stdout import *

from random import shuffle

# import wandb
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torch.cuda.amp import GradScaler, autocast

from transformers import (
    AutoProcessor,
    Pix2StructConfig,
    Pix2StructForConditionalGeneration,
    get_linear_schedule_with_warmup,
)

import albumentations as A
from albumentations.pytorch import ToTensorV2

import warnings
warnings.simplefilter("ignore")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

BOS_TOKEN = "<|BOS|>"
X_START = "<x_start>"
X_END = "<x_end>"
Y_START = "<y_start>"
Y_END = "<y_end>"

new_tokens = [
    "<line>",
    "<vertical_bar>",
    "<scatter>",
    "<dot>",
    "<horizontal_bar>",
    X_START,
    X_END,
    Y_START,
    Y_END,
    BOS_TOKEN,
]




from PIL import Image
import requests
from transformers import AutoProcessor, Pix2StructForConditionalGeneration

# model = Pix2StructForConditionalGeneration.from_pretrained("archive",ignore_mismatched_sizes=True).to(device)
model = Pix2StructForConditionalGeneration.from_pretrained("google/matcha-base",ignore_mismatched_sizes=True).to(device)

# model.load_state_dict(torch.load("./models/pew/pytorch_model.bin"))
model.load_state_dict(torch.load("./models/ckpt_bak/epoch_2.pt"))
processor = AutoProcessor.from_pretrained("google/matcha-base",)
processor.tokenizer.add_tokens(new_tokens)
# model.resize_token_embeddings(len(processor.tokenizer))

path = "models/benetech-making-graphs-accessible/test/images/00dcf883a459.jpg"
image = Image.open(path)

inputs = processor(text="Generate underlying data table of the figure below",images=image, return_tensors="pt")

# autoregressive generation
generated_ids = model.generate(**inputs.to(device), max_new_tokens=50)
generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
print(generated_text)

