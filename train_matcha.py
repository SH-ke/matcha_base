import numpy as np
import pandas as pd
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

# %%
Config = {
    'IMAGE_DIR': '/kaggle/input/benetech-making-graphs-accessible/train/images/',
    
    'MAX_PATCHES': 1024,
    'MODEL_NAME': 'google/matcha-base',
    'IMG_SIZE': (256, 256),
    'MAX_LEN': 256,
    'LR': 3e-5,
    'NB_EPOCHS': 20,
    'TRAIN_BS': 3,
    'VALID_BS': 3,
    'ALL_SAMPLES': int(1e+100),
    '_wandb_kernel': 'tanaym',
}

# %% [markdown]
# ### About W&B:
# <center><img src="https://i.imgur.com/gb6B4ig.png" width="400" alt="Weights & Biases"/></center><br>
# <p style="text-align:center">WandB is a developer tool for companies turn deep learning research projects into deployed software by helping teams track their models, visualize model performance and easily automate training and improving models.
# We will use their tools to log hyperparameters and output metrics from your runs, then visualize and compare results and quickly share findings with your colleagues.<br><br></p>

# %% [markdown]
# To login to W&B, you can use below snippet.
# 
# ```python
# from kaggle_secrets import UserSecretsClient
# user_secrets = UserSecretsClient()
# wb_key = user_secrets.get_secret("WANDB_API_KEY")
# 
# wandb.login(key=wb_key)
# ```
# Make sure you have your W&B key stored as `WANDB_API_KEY` under Add-ons -> Secrets
# 
# You can view [this](https://www.kaggle.com/ayuraj/experiment-tracking-with-weights-and-biases) notebook to learn more about W&B tracking.
# 
# If you don't want to login to W&B, the kernel will still work and log everything to W&B in anonymous mode.

# %%
# def wandb_log(**kwargs):
#     for k, v in kwargs.items():
#         wandb.log({k: v})

# # Start W&B logging
# # W&B Login
# from kaggle_secrets import UserSecretsClient
# user_secrets = UserSecretsClient()
# wb_key = user_secrets.get_secret("WANDB_API_KEY")

# wandb.login(key=wb_key)

# run = wandb.init(
#     project='pytorch',
#     config=Config,
#     group='multi_modal',
#     job_type='train',
# )

# %%
# Let's add chart types as special tokens and a special BOS token
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

# %% [markdown]
# Just resizing and graph image normalization as augments for now!

# %%
def augments():
    return A.Compose([
        A.Resize(width=Config['IMG_SIZE'][0], height=Config['IMG_SIZE'][1]),
        A.Normalize(
            mean=[0, 0, 0],
            std=[1, 1, 1],
            max_pixel_value=255,
        ),
        ToTensorV2(),
    ])

# %%
class BeneTechDataset(Dataset):
    def __init__(self, dataset, processor, augments=None):
        self.dataset = dataset
        self.processor = processor
        self.augments = augments

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        item = self.dataset[idx]
        image = cv2.imread(item['image'][1:])
        if self.augments:
            # image = self.augments(image = np.array(image))['image']
            # print(image)
            image = self.augments(image=image)['image']
        header_texts = "Generate underlying data table of the figure below:"
        encoding = self.processor(
            images=image,
            # header_text = header_texts,
            return_tensors="pt", 
            add_special_tokens=True, 
            max_patches=Config['MAX_PATCHES']
        )
        # encoding["header_text"] = header_texts
        encoding = {k:v.squeeze() for k,v in encoding.items()}
        encoding["text"] = item["label"]
        return encoding

# %% [markdown]
# If you make changes to the vocab or anything else in the below cell, please don't forget to resize model token embeddings (as done below).

# %%
def get_model(extra_tokens=new_tokens):
    # processor = AutoProcessor.from_pretrained(Config['MODEL_NAME'])
    processor = AutoProcessor.from_pretrained(Config['MODEL_NAME'],is_vqa=False)
    model = Pix2StructForConditionalGeneration.from_pretrained(Config['MODEL_NAME'])
    processor.image_processor.size = {
        "height": Config['IMG_SIZE'][0],
        "width": Config['IMG_SIZE'][1],
    }

    processor.tokenizer.add_tokens(extra_tokens)
    model.resize_token_embeddings(len(processor.tokenizer))
    return processor, model

# %%
def collator(batch):
    new_batch = {"flattened_patches":[], "attention_mask":[]}
    texts = [item["text"] for item in batch]
    # header_texts = [item["header_text"] for item in batch]
    text_inputs = processor(
        text=texts, 
        padding="max_length", 
        truncation=True, 
        return_tensors="pt", 
        add_special_tokens=True, 
        max_length=Config['MAX_LEN']
    )
    header = processor(
        text="Generate underlying data table of the figure below", 
        padding="max_length", 
        truncation=True, 
        return_tensors="pt", 
        add_special_tokens=True, 
        max_length=Config['MAX_LEN']
    )
    new_batch["labels"] = text_inputs.input_ids
    new_batch["decoder_input_ids"] = header.input_ids
    for item in batch:
        new_batch["flattened_patches"].append(item["flattened_patches"])
        new_batch["attention_mask"].append(item["attention_mask"])
    new_batch["flattened_patches"] = torch.stack(new_batch["flattened_patches"])
    new_batch["attention_mask"] = torch.stack(new_batch["attention_mask"])

    return new_batch

# %%
def train_one_epoch(model, processor, train_loader, optimizer, scaler):
    """
    Trains the model on all batches for one epoch with NVIDIA's AMP
    """
    model.train()
    avg_loss = 0
    with autocast():
        setup_logging('epoch')
        __logger = logging.getLogger('logger')
        __logger.setLevel(logging.DEBUG)
        for idx, batch in tqdm(enumerate(train_loader), unit_scale=True, dynamic_ncols=True, file=sys.stdout):
            labels = batch.pop("labels").to('cuda')
            flattened_patches = batch.pop("flattened_patches").to('cuda')
            attention_mask = batch.pop("attention_mask").to('cuda')

            outputs = model(
                flattened_patches=flattened_patches,
                attention_mask=attention_mask,
                labels=labels
            )
            loss = outputs.loss
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)
            avg_loss += loss.item()
            # print(loss.item())
            __logger.info('loss item {}'.format(loss.item()))


    avg_loss = avg_loss / len(train_loader)
    
    # print(f"Average training loss: {avg_loss:.4f}")
#     wandb_log(train_loss=avg_loss)
    return avg_loss

@torch.no_grad()
def valid_one_epoch(model, processor, valid_loader):
    """
    Validates the model on all batches (in val set) for one epoch
    """
    model.eval()
    avg_loss = 0
    prog_bar = tqdm(enumerate(valid_loader), total=len(valid_loader))
    for idx, batch in prog_bar:
        labels = batch.pop("labels").to('cuda')
        flattened_patches = batch.pop("flattened_patches").to('cuda')
        attention_mask = batch.pop("attention_mask").to('cuda')
        
        outputs = model(
            flattened_patches=flattened_patches,
            attention_mask=attention_mask,
            labels=labels
        )
        
        loss = outputs.loss
        prog_bar.set_description(f"loss: {loss.item():.4f}")
        # wandb_log(val_step_loss=loss.item())
        avg_loss += loss.item()
        
    avg_loss = avg_loss / len(valid_loader)
    print(f"Average validation loss: {avg_loss:.4f}")
    # wandb_log(val_loss=avg_loss)
    return avg_loss

# %%
def fit(model, processor, train_loader, valid_loader, optimizer, scaler):
    """
    A nice function that binds it all together and reminds me of Keras days from 2018 :)
    """
    best_val_loss = int(1e+5)
    __csv_file_name = "./loggings/{}-{}_log_file.csv".format("avg_loss",
                                                    datetime.datetime.utcnow().isoformat().replace(":", "-"))
    with open(__csv_file_name, mode='a', newline='') as f:
        for epoch in range(Config['NB_EPOCHS']):
            csvw = csv.writer(f)
            print(epoch)
        
            if epoch == 0:
                print(f"if 0 epoch = {epoch}")
                csvw.writerow(["EPOCH", "AVG_LOSS"])
                print(["EPOCH", "AVG_LOSS"])
            # print(f"{'='*20} Epoch: {epoch+1} / {Config['NB_EPOCHS']} {'='*20}")
            _ = train_one_epoch(model, processor, train_loader, optimizer, scaler)
            val_avg_loss = valid_one_epoch(model, processor, valid_loader)
            csvw.writerow([epoch, val_avg_loss])
            print([epoch, val_avg_loss])
        
        if val_avg_loss < best_val_loss:
            best_val_loss = val_avg_loss
            print(f"Saving best model so far with loss: {best_val_loss:.4f}")
            torch.save(model.state_dict(), f"pix2struct_base_benetech.pt")
    print(f"Best model with val_loss: {best_val_loss:.4f}")

# %%
# Training cell
if __name__ == "__main__":
    # Read the processed JSON file
    with open("data.json", "r") as fl:
        dataset = json.load(fl)['data']
        
    # Shuffle the dataset and select however samples you want for training
    shuffle(dataset)
    dataset = dataset[:Config['ALL_SAMPLES']]
    
    # We are splitting the data naively for now
    split = 0.90
    train_samples = int(len(dataset) * split)
    train_ds = dataset[:train_samples+1]
    valid_ds = dataset[train_samples:]
    
    # Yeah all that
    processor, model = get_model()
    model.to('cuda')
#     wandb.watch(model)
    optimizer = torch.optim.Adam(params=model.parameters(), lr=Config['LR'])
    
    # Load the data into Datasets and then make DataLoaders for training
    train_dataset = BeneTechDataset(train_ds, processor, augments=augments())
    train_dataloader = DataLoader(train_dataset, shuffle=True, batch_size=Config['TRAIN_BS'], collate_fn=collator)
    
    valid_dataset = BeneTechDataset(valid_ds, processor, augments=augments())
    valid_dataloader = DataLoader(valid_dataset, shuffle=False, batch_size=Config['VALID_BS'], collate_fn=collator)
    
    nb_train_steps = int(train_samples / Config['TRAIN_BS'] * Config['NB_EPOCHS'])
    
    # Print out the data sizes we are training on
    print(f"Training on {len(train_ds)} samples, Validating on {len(valid_ds)} samples")
    
    # Train the model now
    fit(
        model=model,
        processor=processor,
        train_loader=train_dataloader,
        valid_loader=valid_dataloader,
        optimizer=optimizer,
        scaler=GradScaler(),
    )

# %% [markdown]
# 

# %%
# Once training is done, 
# wandb.finish()

# %% [markdown]
# <center>
# <img src="https://img.shields.io/badge/Upvote-If%20you%20like%20my%20work-07b3c8?style=for-the-badge&logo=kaggle">
# </center>


