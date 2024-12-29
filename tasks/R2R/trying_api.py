# import numpy as np
import cv2

# import json
import math
import torch

# import time
# import networkx as nx
from PIL import Image
# from lavis.models import load_model_and_preprocess
# # from tasks.R2R.utils import load_nav_graphs
# # from utils import load_nav_graphs
# from tqdm import tqdm
# import requests


import numpy as np
# import cv2
# # import requests 
from transformers import AutoModelForCausalLM 
from transformers import AutoProcessor

import os
import sys
import logging
print(sys.path)
sys.path.append('/home/huang/Desktop/again_mp/Matterport3DSimulator/build') # For MatterSim
print(sys.path)
# sys.path.append('../Matterport3DSimulator/tasks/R2R')

import MatterSim

os.environ["MATTERPORT_DATA_DIR"] = "/data/v1/scans/"
os.environ['HF_HOME'] = '/data/cache/'

LEN_HISTORY = 1

WIDTH = 1024
HEIGHT = 1024
VFOV = 60
HFOV = VFOV*WIDTH/HEIGHT
TEXT_COLOR = [230, 40, 40]
ERROR_MARGIN = 3

## To avoid CUDA out of memory
torch.cuda.empty_cache()

sim = MatterSim.Simulator()
sim.setCameraResolution(WIDTH, HEIGHT)
sim.setCameraVFOV(math.radians(VFOV))
sim.setDepthEnabled(False)
sim.setDiscretizedViewingAngles(True)
sim.setRenderingEnabled(False)
sim.initialize()

model_id = "microsoft/Phi-3-vision-128k-instruct" 

model = AutoModelForCausalLM.from_pretrained(model_id, device_map="cuda", trust_remote_code=True, torch_dtype="auto",
     _attn_implementation='flash_attention_2', cache_dir=os.environ['HF_HOME']) # use _attn_implementation='eager' to disable flash attention
# _attn_implementation='flash_attention_2'

processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)

def open_image(image_path):
    state = sim.getState()[0]
    rgb = np.array(state.rgb, copy=False)
    cv2.imwrite(image_path, rgb)
    raw_img = Image.open(image_path).convert("RGB")
    raw_img.save(image_path)
    print("image saved to ", image_path)
    return raw_img

sim.newEpisode(['2t7WUuJeko7'], ['1e6b606b44df4a6086c0f97e826d4d15'], [0], [0])
print("newEp")

image_folder = "/home/huang/Desktop/again_mp/Matterport3DSimulator/me_testing/"
image_path_list = ['current_view_'+str(i)+'.jpg' for i in range(LEN_HISTORY)]

image_list = [Image.open(image_folder + p) for p in image_path_list]
image = Image.open("/home/huang/Desktop/vln_is_fun/phi3/2t7WUuJeko7/matterport_skybox_images/1e6b606b44df4a6086c0f97e826d4d15_skybox4_sami.jpg")



for i in range(LEN_HISTORY):
    print("STEP!!", i)
    state = sim.getState()[0]
    for key in dir(state):
        print(key, getattr(state, key))

    current_view = open_image("me_testing/current_view_"+str(i)+".jpg")

    sim.makeAction([0], [0.5], [0])
# messages = [{"role": "user", "content": "<|image_1|>\n<|image_2|>\n<|image_3|>\n<|image_4|>\nYou are navigating a room, and the pictures are what you see as you are walking around the room. The last picture is where you at right now. The blue numbers are the possible directions to go. If you want to find a bed in the house, which direction should you go? Please answer '1' or '2'."}] 

messages = [{"role": "user", "content": "<|image_1|>\nYou are navigating a room, and the pictures are what you see as you are walking around the room. The last picture is where you at right now. The blue numbers are the possible directions to go. If you want to find a bed in the house, which direction should you go? Please answer '1' or '2'."}] 
    
while 1:
    print(messages)
    prompt = processor.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

    inputs = processor(prompt, image_list, return_tensors="pt").to("cuda:0") 

    generation_args = { 
        "max_new_tokens": 500, 
        "temperature": 0.0, 
        "do_sample": False, 
    } 

    generate_ids = model.generate(**inputs, eos_token_id=processor.tokenizer.eos_token_id, **generation_args) 

    # remove input tokens 
    generate_ids = generate_ids[:, inputs['input_ids'].shape[1]:]
    response = processor.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0] 

    print("ASSISTANT:", response)
    messages.append(dict({"role": "assistant", "content": response}))

    user_input = input("USER: ")
    if user_input == "q":
        break
    messages.append(dict({"role": "user", "content": user_input}))
