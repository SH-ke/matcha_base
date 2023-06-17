#!/usr/bin/env python
# coding: utf-8

# # Donut with additional hyperparameter tuning
# 
# ### Notes: This could still absolutely improve with better post text processing. As I have not put a ton of thought into that peice of things yet. I would imagine this is not the final solution for the donut model and there are likely several other creative ways to boost preformance, especially with regaurd to horizontal or vertical bar plots.
# 
# ### A special thank you to @nbroad (Nicholas Broad) for his insights using donut! Go check out his work and discussions

# In[1]:


import re 
from pathlib import Path
from typing import List
from functools import partial
import sys
from transformers import (
    DonutProcessor,
    VisionEncoderDecoderConfig,
    VisionEncoderDecoderModel,
)
import torch
from torch.utils.data import DataLoader
import numpy as np
import pandas as pd
from datasets import Dataset
from datasets import Image as ds_img
from tqdm.notebook import tqdm

import warnings
warnings.filterwarnings("ignore")
sys.path.append("E:/Xuke/kaggle/Benetech")
torch.cuda.is_available()


# In[2]:


class CFG:
    
    test_grayscale = True
    debug_clean = False
    
    batch_size = 4
    image_path = "benetech-making-graphs-accessible/test/images"
    max_length = 512
    model_dir = "archive"

BOS_TOKEN = "<|BOS|>"
X_START = "<x_start>"
X_END = "<x_end>"
Y_START = "<y_start>"
Y_END = "<y_end>"

PLACEHOLDER_DATA_SERIES = "0;0"
PLACEHOLDER_CHART_TYPE = "line"


# In[3]:


def clean_preds(x: List[str], y: List[str]):
    """
    This function cleans the x and y values predicted by Donut.

    Because it is a generative model, it can insert any character in the 
    model's vocabulary into the prediction string. This function primarily removes
    characters that prevent a number from being cast to a float.

    Example:

    x = ["11", "12", "1E", "14", "15"]
    y = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday"]

    # float("1E") will throw an error

    new_x, new_y = clean_preds(x, y)

    new_x = ["11", "12", "13", "14", "15"]
    new_y = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday"]

    Args:
        x (List[str]): The x values predicted by Donut.
        y (List[str]): The y values predicted by Donut.

    Returns:
        x (List[str]): The cleaned x values.
        y (List[str]): The cleaned y values.
    """
    
    def clean(str_list):
        
        new_list = []
        for temp in str_list:
            if "." not in temp:
                dtype = int
            else:
                dtype = float
            try:
                # First try removing whitespace e.g. float("10 0") will fail
                temp = dtype(re.sub("\s", "", temp))
            except ValueError:

                # remove everything that isn't a digit, period, negative sign, or the letter e
                # It could be "1e-5" or "-0.134"

                temp = re.sub(r"[^0-9\.\-eE]", "", temp)

                if len(temp) == 0:
                    temp = 0
                else:
                    multiple_periods = len(re.findall(r"\.", temp)) > 1
                    multiple_negative_signs = len(re.findall(r"\-", temp)) > 1
                    multiple_e = len(re.findall(r"[eE]", temp)) > 1

                    # Put negative sign in from of it all
                    if multiple_negative_signs:
                        temp = "-" + re.sub(r"\-", "", temp)

                    # Keep first period if there are multiple
                    if multiple_periods:
                        chunks = temp.split(".")
                        try:
                            temp = chunks[0] + "." + "".join(chunks[1:])
                        except IndexError:
                            temp = "".join(chunks)
                    
                    # Keep last e in case it is "e1e-5"
                    if multiple_e:
                        while temp.lower().startswith("e"):
                            temp = temp[1:]
                        
                        while temp.lower().endswith("e"):
                            temp = temp[:-1]
                            
                        chunks = temp.split("e")
                        try:
                            temp = chunks[0:-1] + "e" + "".join(chunks[-1])
                        except IndexError:
                            temp = "".join(chunks)
                try:
                    temp = dtype(temp)
                except ValueError:
                    temp = 0
                    
            new_list.append(temp)

        return new_list

    all_x_chars = "".join(x)
    all_y_chars = "".join(y)

    frac_num_x = len(re.sub(r"[^\d]", "", all_x_chars)) / len(all_x_chars)
    frac_num_y = len(re.sub(r"[^\d]", "", all_y_chars)) / len(all_y_chars)
    
    print(frac_num_x, frac_num_y)

    if CFG.debug_clean:
        print(f"x before clean (len={len(x)})", x)
        print(f"y before clean (len={len(y)})", y)

    if frac_num_x >= 0.5:
        x = clean(x)
    else:
        x = [s.strip() for s in x]
    
    
    if frac_num_y >= 0.5:
        y = clean(y)
    else:
        y = [s.strip() for s in y]
        
    if CFG.debug_clean:
        print(f"x after clean (len={len(x)})", x)
        print(f"y after clean (len={len(x)})", x)

    return x, y
    

def string2preds(pred_string: str):
    """
    Convert the prediction string from Donut to a chart type and x and y values.

    Checks to make sure the special tokens are present and that the x and y values are not empty.
    Will truncate the list of values to the smaller length of the two lists. This is because the 
    lengths of the x and y values must be the same to earn any points.

    Args:
        pred_string (str): The prediction string from Donut.

    Returns:
        chart_type (str): The chart type predicted by Donut.
        x (List[str]): The x values predicted by Donut.
        y (List[str]): The y values predicted by Donut.
    """

    if "<dot>" in pred_string:
        chart_type = "dot"
    elif "<horizontal_bar>" in pred_string:
        chart_type = "horizontal_bar"
    elif "<vertical_bar>" in pred_string:
        chart_type = "vertical_bar"
    elif "<scatter>" in pred_string:
        chart_type = "scatter"
    elif "<line>" in pred_string:
        chart_type = "line"
    else:
        return "vertical_bar", [], []
    
    
    if not all([x in pred_string for x in [X_START, X_END, Y_START, Y_END]]):
        return chart_type, [], []
    
    pred_string = re.sub(r"<one>", "1", pred_string)

    x = pred_string.split(X_START)[1].split(X_END)[0].split(";")
    y = pred_string.split(Y_START)[1].split(Y_END)[0].split(";")

    if len(x) == 0 or len(y) == 0:
        return chart_type, [], []

    x, y = clean_preds(x, y)

    return chart_type, x, y


# In[4]:


image_dir = Path(CFG.image_path)
images = list(image_dir.glob("*.jpg"))

ds = Dataset.from_dict(
    {"image_path": [str(x) for x in images], "id": [x.stem for x in images]}
).cast_column("image_path", ds_img())

def preprocess(examples, processor):
    pixel_values = []

    for sample in examples["image_path"]:
        arr = np.array(sample)
        
        # There are some grayscale images that were making this fail
        # This prevents that.
        if len(arr.shape) == 2:
            print("Changing grayscale to 3 channel format")
            print(arr.shape)
            arr = np.stack([arr]*3, axis=-1)
        
        pixel_values.append(processor(arr, random_padding=True).pixel_values)
        
        
    return {
        "pixel_values": torch.tensor(np.vstack(pixel_values)),
    }

model = VisionEncoderDecoderModel.from_pretrained(CFG.model_dir)
model.eval()

device = torch.device("cuda:0")

model.to(device)
decoder_start_token_id = model.config.decoder_start_token_id
processor = DonutProcessor.from_pretrained(CFG.model_dir)

ids = ds["id"]
ds.set_transform(partial(preprocess, processor=processor))

data_loader = DataLoader(
    ds, batch_size=CFG.batch_size, shuffle=False
)


all_generations = []
for batch in tqdm(data_loader):
    pixel_values = batch["pixel_values"].to(device)

    batch_size = pixel_values.shape[0]

    decoder_input_ids = torch.full(
        (batch_size, 1),
        decoder_start_token_id,
        device=pixel_values.device,
    )

    try:
        outputs = model.generate(
            pixel_values,
            decoder_input_ids=decoder_input_ids,
            max_length=CFG.max_length,
            early_stopping=True,
            pad_token_id=processor.tokenizer.pad_token_id,
            eos_token_id=processor.tokenizer.eos_token_id,
            use_cache=True,
            num_beams=2,       #1 int    (1 - 10)
            temperature=.9,     #1 float  (0 -  ) less div - more div
            top_k=1,           #1 int    (1 -  ) less div - more div
            top_p=.4,           #1 float (0 - 1) more div - less div
            return_dict_in_generate=True,
        )

        all_generations.extend(processor.batch_decode(outputs.sequences))
        
    except:
        all_generations.extend([""]*batch_size)
        
chart_types, x_preds, y_preds = [], [], []
for gen in all_generations:

    try:
        chart_type, x, y = string2preds(gen)
        new_chart_type = chart_type
        x_str = ";".join(list(map(str, x)))
        y_str = ";".join(list(map(str, y)))

    except Exception as e:
        print("Failed to convert to string:", gen)
        print(e)
        new_chart_type = PLACEHOLDER_CHART_TYPE
        x_str = PLACEHOLDER_DATA_SERIES
        y_str = PLACEHOLDER_DATA_SERIES
            
    if len(x_str) == 0:
        x_str = PLACEHOLDER_DATA_SERIES
    if len(y_str) == 0:
        y_str = PLACEHOLDER_DATA_SERIES
    
    chart_types.append(new_chart_type)
    x_preds.append(x_str)
    y_preds.append(y_str)
        

sub_df = pd.DataFrame(
    data={
        "id": [f"{id_}_x" for id_ in ids] + [f"{id_}_y" for id_ in ids],
        "data_series": x_preds + y_preds,
        "chart_type": chart_types * 2,
    }
)

sub_df.to_csv("submission.csv", index=False)


# In[5]:


display(sub_df)


# In[ ]:




