''' Agents: stop/random/shortest/seq2seq  '''

import json
import os
import sys
import numpy as np
import random
import time

import torch
import torch.nn as nn
import torch.distributions as D
from torch.autograd import Variable
from torch import optim
import torch.nn.functional as F

from env import R2RBatch
from utils import padding_idx

### Modify the path below to the directory your MP3D build
sys.path.append('/home/huang/Desktop/again_mp/Matterport3DSimulator/build') # For MatterSim
sys.path.append('/home/huang/Desktop/again_mp/Matterport3DSimulator/')
import MatterSim
import math
import cv2
from PIL import Image
from transformers import AutoModelForCausalLM 
from transformers import AutoProcessor
import yaml
os.environ['HF_HOME'] = '/data/cache/' ## Save hf models here

class BaseAgent(object):
    ''' Base class for an R2R agent to generate and save trajectories. '''

    def __init__(self, env, results_path):
        print("init BaseAgent")
        self.env = env
        self.results_path = results_path
        random.seed(1)
        self.results = {}
        self.losses = [] # For learning agents

    def write_results(self):
        output = [{'instr_id':k, 'trajectory': v} for k,v in self.results.items()]
        with open(self.results_path, 'w') as f:
            json.dump(output, f)

    def rollout(self):
        ''' Return a list of dicts containing instr_id:'xx', path:[(viewpointId, heading_rad, elevation_rad)]  '''
        raise NotImplementedError

    @staticmethod
    def get_agent(name):
        return globals()[name+"Agent"]

    def test(self):
        self.env.reset_epoch()
        self.losses = []
        self.results = {}
        # We rely on env showing the entire batch before repeating anything
        #print('Testing %s' % self.__class__.__name__)
        looped = False
        while True:
            for traj in self.rollout():
                if traj['instr_id'] in self.results:
                    looped = True
                else:
                    self.results[traj['instr_id']] = traj['path']
            if looped:
                break


class StopAgent(BaseAgent):
    ''' An agent that doesn't move! '''

    def rollout(self):
        traj = [{
            'instr_id': ob['instr_id'],
            'path': [(ob['viewpoint'], ob['heading'], ob['elevation'])]
        } for ob in self.env.reset()]
        return traj




class RandomAgent(BaseAgent):
    ''' An agent that picks a random direction then tries to go straight for
        five viewpoint steps and then stops. '''

    def rollout(self):
        obs = self.env.reset()
        traj = [{
            'instr_id': ob['instr_id'],
            'path': [(ob['viewpoint'], ob['heading'], ob['elevation'])]
        } for ob in obs]

        self.steps = random.sample(range(-11,1), len(obs))
        ended = [False] * len(obs)
        for t in range(30):
            actions = []
            for i,ob in enumerate(obs):
                if self.steps[i] >= 5:
                    actions.append((0, 0, 0)) # do nothing, i.e. end
                    ended[i] = True
                elif self.steps[i] < 0:
                    actions.append((0, 1, 0)) # turn right (direction choosing)
                    self.steps[i] += 1
                elif len(ob['navigableLocations']) > 1:
                    actions.append((1, 0, 0)) # go forward
                    self.steps[i] += 1
                else:
                    actions.append((0, 1, 0)) # turn right until we can go forward
            obs = self.env.step(actions)
            # print("actions", actions)
            # print("TRAJ", traj)
            # print("LEN!", len(obs), len(actions), len(traj))
            for i,ob in enumerate(obs):
                if not ended[i]:
                    traj[i]['path'].append((ob['viewpoint'], ob['heading'], ob['elevation']))
            # print("TRAJ afterwards", traj)
        return traj

class PhiAgent(BaseAgent):
    ''' An agent by simply prompting Phi-3 '''

    def __init__(self, env, results_path):
        super().__init__(env, results_path) 
        ## Phi 3
        print("init PhiAgent")

        ## Empty cache to avoid CUDA out of memory
        torch.cuda.empty_cache()

        self.model_id = "microsoft/Phi-3-vision-128k-instruct" 
        self.model = AutoModelForCausalLM.from_pretrained(self.model_id, device_map="cuda", trust_remote_code=True, torch_dtype="auto",
            _attn_implementation='flash_attention_2', cache_dir=os.environ['HF_HOME']) # use _attn_implementation='eager' to disable flash attention
        self.processor = AutoProcessor.from_pretrained(self.model_id, trust_remote_code=True)
        self.temp_save_image_counter = 0 ## just for temporarily output the current_view image. You can ignore this.
        self.traj_code = "-"
        # self.messages = []
        self.image_list = []
    # def dump(self, obj, level=0):
    #     for a in dir(obj):
    #         if a[0] == '_': continue
    #         val = getattr(obj, a)
    #         print("---", a)
    #         if a == 'rgb':
    #             print('  '*level, a, np.array(val, copy=False))
    #         elif isinstance(val, (int, float, str, list, dict, set)):
    #             print('  '*level, a, val)
    #         else:
    #             self.dump(val, level=level+1)
    #     return
    def open_image(self, image_path): ## I added state_rgb
        state = self.env.env.sim.getState()[0]

        rgb = np.array(state.rgb, copy=False)
        cv2.imwrite(image_path, rgb)
        raw_img = Image.open(image_path).convert("RGB")
        raw_img.save(image_path)
        # print("image saved to", image_path)
        ## Just for printing
        if self.temp_save_image_counter < 50:
            raw_img.save(f"unused_images/{str(self.temp_save_image_counter)}.jpg")
            self.temp_save_image_counter += 1
        return raw_img
    
    def parse_response(self, response, is_forward_available):
        if "Move forward" in response:
            if is_forward_available:
                self.traj_code = self.traj_code + "F"
                return (1, 0, 0)
            self.traj_code = self.traj_code + "f"
            return (0, 1, 0) ## The default is to turn right
            
        elif "Turn left" in response:
            self.traj_code = self.traj_code + "L"
            return (0, -1, 0)
        elif "Turn right" in response:
            self.traj_code = self.traj_code + "R"
            return (0, 1, 0)
        elif "Stop" in response:
            self.traj_code = self.traj_code + "S"
            return (0, 0, 0)
        elif "Look up" in response:
            self.traj_code = self.traj_code + "U"
            return (0, 0, 1)
        elif "Look down" in response:
            self.traj_code = self.traj_code + "D"
            return (0, 0, -1)
        else:
            # print("noo!! The response does not follow the format.", response)
            self.traj_code = self.traj_code + "r"
            return (0, 1, 0) ## The default is to turn right
            

    def run_inference(self, ob, is_forward_available, teacher):

        # prompt_template = """
        # You are navigating in a room. Please follow the instruction:
        # {instruction}
        # The below image is what you see at your location:
        # <|image_1|>
        # The actions available are:
        
        # Please select one option above, in the format RESPONSE: <NUMBER>. <ACTION>.
        # RESPONSE:"""
        if is_forward_available:
            options = """
            1. Turn left
            2. Turn right
            3. Stop
            4. Look up
            5. Look down
            6. Move forward
            """
        else:
            options = """1. Turn left
            2. Turn right
            3. Stop
            4. Look up
            5. Look down"""

        prompt_template = """
        {images}
        The above images are the scenes you see along the way, and the last one is your current view. Please follow the instruction:
        {instruction} 
        The actions available are:
        {options}
        
        Please take an action in the format RESPONSE: <NUMBER>. <ACTION>.
        RESPONSE:"""

        # prompt_template = """
        # <|image_1|>
        # <|image_2|>
        # <|image_3|>
        # The above image is what you see. Please follow the instruction:
        # {instruction} 
        # The actions available are:
        # {options}
        
        # Please take an action in the format RESPONSE: <NUMBER>. <ACTION>.
        # RESPONSE:"""

        current_view = self.open_image("current_view.jpg")
        self.image_list.append(Image.open("current_view.jpg"))
        if len(self.image_list) > 5:
            self.image_list = self.image_list[1:]
        images = "\n".join([f"<|image_{str(i+1)}|>" for i in range(len(self.image_list))])
        # self.messages.append({"role": "user", "content": prompt_template.format(instruction = ob['instructions_minimal'], options = options, image_id = str(len(self.image_list)))})
        # image_path_list = ["current_view.jpg"]
        # image_list = [Image.open(p) for p in image_path_list]
        messages = [
            {"role": "system", "content": "You are a navigation agent who follows instruction to move in an indoor environment with the least action steps. You can move around the room by going forward or changing the direction you're facing."},
            {"role": "user", "content": prompt_template.format(instruction = ob['instructions'], options = options, images= images)}
        ]
        # messages = [
        #     {"role": "system", "content": "You are a navigation agent who follows instruction to move in an indoor environment with the least action steps. You can move around the room by going forward or changing the direction you're facing."},
        #     {"role": "user", "content": "<|image_1|>\nWhat is in the image?"}
        # ] 

        
        prompt = self.processor.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = self.processor(prompt, self.image_list, return_tensors="pt").to("cuda:0")
        # prompt = self.processor.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        # inputs = self.processor(prompt, image_list, return_tensors="pt").to("cuda:0") 
        generation_args = { 
            "max_new_tokens": 16, 
            "temperature": 0.0, 
            "do_sample": False, 
        } 
        generate_ids = self.model.generate(**inputs, eos_token_id=self.processor.tokenizer.eos_token_id, **generation_args)
        # remove input tokens 
        generate_ids = generate_ids[:, inputs['input_ids'].shape[1]:]
        response = self.processor.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0] 
        # print(response)
        # self.messages.append(dict({"role": "assistant", "content": response}))
        return response

    def rollout(self):
        #print(type(self.env)) # <class 'env.R2RBatch'>
        # print(self.env.env.sim)
        # self.sim.newEpisode(['2t7WUuJeko7'], ['1e6b606b44df4a6086c0f97e826d4d15'], [0], [0])
        # print("newEp")
        obs = self.env.reset()
        traj = [{
            'instr_id': ob['instr_id'],
            'path': [(ob['viewpoint'], ob['heading'], ob['elevation'])]
        } for ob in obs]
        # print(self.env.data[0])
        # print(obs)
        assert(len(obs) == 1)
#         [{'instr_id': '4370_0', 'scan': 'mJXqzFtmKg4', 'viewpoint': '37c2223d40cb4aedb1563e5e0c3a53e1', 'viewIndex': 13, 'heading': 0.52359877
# 55982988, 'elevation': 0.0, 'feature': None, 'step': 0, 'navigableLocations': [<MatterSim.ViewPoint object at 0x7f4f96476df0>, <Matter
# Sim.ViewPoint object at 0x7f4f96476e30>], 'instructions_minimal': 'Turn around and walk behind the back of the couch outside. Once you go past
#  the chair on the end, stop in front of the next large window on your right past the door. ', 'teacher': (0, 1, 0)}]
        # print("TRAJ in the beginning")
        # print(traj)
        
        ended = [False] * len(obs)
        done = 0
        self.traj_code = '-'

        # self.messages = [
        #     {"role": "system", "content": "You are a navigation agent who follows instruction to move in an indoor environment with the least action steps. You can move around the room by going forward or changing the direction you're facing."},
        # ]
        self.image_list = [] 
        for t in range(30): ## Go at most 30 steps
            if done: break
            actions = []
            for i,ob in enumerate(obs):
                model_response = self.run_inference(ob, len(ob['navigableLocations']) > 1, ob['teacher'])
                model_action = self.parse_response(model_response, len(ob['navigableLocations']) > 1)
                actions.append(model_action) # turn right
                if t == 29 or model_action == (0, 0, 0):
                    ended[i] = True
                    done = 1
            obs = self.env.step(actions)
            
            for i,ob in enumerate(obs):
                if not ended[i]:
                    traj[i]['path'].append((ob['viewpoint'], ob['heading'], ob['elevation']))
        # print("traj afterwards", *traj)
        # assert(0)
        print("Phi", ob['instr_id'], self.traj_code)
        return traj


class ShortestAgent(BaseAgent):
    ''' An agent that always takes the shortest path to goal. '''

    def rollout(self):
        obs = self.env.reset()
        traj = [{
            'instr_id': ob['instr_id'],
            'path': [(ob['viewpoint'], ob['heading'], ob['elevation'])]
        } for ob in obs]
        ended = np.array([False] * len(obs))
        while True:
            actions = [ob['teacher'] for ob in obs]
            obs = self.env.step(actions)
            for i,a in enumerate(actions):
                if a == (0, 0, 0):
                    ended[i] = True
            for i,ob in enumerate(obs):
                if not ended[i]:
                    traj[i]['path'].append((ob['viewpoint'], ob['heading'], ob['elevation']))
            if ended.all():
                break
        return traj


class Seq2SeqAgent(BaseAgent):
    ''' An agent based on an LSTM seq2seq model with attention. '''

    # For now, the agent can't pick which forward move to make - just the one in the middle
    model_actions = ['left', 'right', 'up', 'down', 'forward', '<end>', '<start>', '<ignore>']
    env_actions = [
      (0,-1, 0), # left
      (0, 1, 0), # right
      (0, 0, 1), # up
      (0, 0,-1), # down
      (1, 0, 0), # forward
      (0, 0, 0), # <end>
      (0, 0, 0), # <start>
      (0, 0, 0)  # <ignore>
    ]
    feedback_options = ['teacher', 'argmax', 'sample']

    def __init__(self, env, results_path, encoder, decoder, episode_len=20):
        super(Seq2SeqAgent, self).__init__(env, results_path)
        self.encoder = encoder
        self.decoder = decoder
        self.episode_len = episode_len
        self.losses = []
        self.criterion = nn.CrossEntropyLoss(ignore_index = self.model_actions.index('<ignore>'))

    @staticmethod
    def n_inputs():
        return len(Seq2SeqAgent.model_actions)

    @staticmethod
    def n_outputs():
        return len(Seq2SeqAgent.model_actions)-2 # Model doesn't output start or ignore

    def _sort_batch(self, obs):
        ''' Extract instructions from a list of observations and sort by descending
            sequence length (to enable PyTorch packing). '''

        seq_tensor = np.array([ob['instr_encoding'] for ob in obs])
        seq_lengths = np.argmax(seq_tensor == padding_idx, axis=1)
        seq_lengths[seq_lengths == 0] = seq_tensor.shape[1] # Full length

        seq_tensor = torch.from_numpy(seq_tensor)
        seq_lengths = torch.from_numpy(seq_lengths)

        # Sort sequences by lengths
        seq_lengths, perm_idx = seq_lengths.sort(0, True)
        sorted_tensor = seq_tensor[perm_idx]
        mask = (sorted_tensor == padding_idx)[:,:seq_lengths[0]]

        return Variable(sorted_tensor, requires_grad=False).long().cuda(), \
               mask.byte().cuda(), \
               list(seq_lengths), list(perm_idx)

    def _feature_variable(self, obs):
        ''' Extract precomputed features into variable. '''
        feature_size = obs[0]['feature'].shape[0]
        features = np.empty((len(obs),feature_size), dtype=np.float32)
        for i,ob in enumerate(obs):
            features[i,:] = ob['feature']
        return Variable(torch.from_numpy(features), requires_grad=False).cuda()

    def _teacher_action(self, obs, ended):
        ''' Extract teacher actions into variable. '''
        a = torch.LongTensor(len(obs))
        for i,ob in enumerate(obs):
            # Supervised teacher only moves one axis at a time
            ix,heading_chg,elevation_chg = ob['teacher']
            if heading_chg > 0:
                a[i] = self.model_actions.index('right')
            elif heading_chg < 0:
                a[i] = self.model_actions.index('left')
            elif elevation_chg > 0:
                a[i] = self.model_actions.index('up')
            elif elevation_chg < 0:
                a[i] = self.model_actions.index('down')
            elif ix > 0:
                a[i] = self.model_actions.index('forward')
            elif ended[i]:
                a[i] = self.model_actions.index('<ignore>')
            else:
                a[i] = self.model_actions.index('<end>')
        return Variable(a, requires_grad=False).cuda()

    def rollout(self):
        obs = np.array(self.env.reset())
        batch_size = len(obs)

        # Reorder the language input for the encoder
        seq, seq_mask, seq_lengths, perm_idx = self._sort_batch(obs)
        perm_obs = obs[perm_idx]

        # Record starting point
        traj = [{
            'instr_id': ob['instr_id'],
            'path': [(ob['viewpoint'], ob['heading'], ob['elevation'])]
        } for ob in perm_obs]

        # Forward through encoder, giving initial hidden state and memory cell for decoder
        ctx,h_t,c_t = self.encoder(seq, seq_lengths)

        # Initial action
        a_t = Variable(torch.ones(batch_size).long() * self.model_actions.index('<start>'),
                    requires_grad=False).cuda()
        ended = np.array([False] * batch_size) # Indices match permuation of the model, not env

        # Do a sequence rollout and calculate the loss
        self.loss = 0
        env_action = [None] * batch_size
        for t in range(self.episode_len):

            f_t = self._feature_variable(perm_obs) # Image features from obs
            h_t,c_t,alpha,logit = self.decoder(a_t.view(-1, 1), f_t, h_t, c_t, ctx, seq_mask)
            # Mask outputs where agent can't move forward
            for i,ob in enumerate(perm_obs):
                if len(ob['navigableLocations']) <= 1:
                    logit[i, self.model_actions.index('forward')] = -float('inf')

            # Supervised training
            target = self._teacher_action(perm_obs, ended)
            self.loss += self.criterion(logit, target)

            # Determine next model inputs
            if self.feedback == 'teacher':
                a_t = target                # teacher forcing
            elif self.feedback == 'argmax':
                _,a_t = logit.max(1)        # student forcing - argmax
                a_t = a_t.detach()
            elif self.feedback == 'sample':
                probs = F.softmax(logit, dim=1)
                m = D.Categorical(probs)
                a_t = m.sample()            # sampling an action from model
            else:
                sys.exit('Invalid feedback option')

            # Updated 'ended' list and make environment action
            for i,idx in enumerate(perm_idx):
                action_idx = a_t[i].item()
                if action_idx == self.model_actions.index('<end>'):
                    ended[i] = True
                env_action[idx] = self.env_actions[action_idx]

            obs = np.array(self.env.step(env_action))
            perm_obs = obs[perm_idx]

            # Save trajectory output
            for i,ob in enumerate(perm_obs):
                if not ended[i]:
                    traj[i]['path'].append((ob['viewpoint'], ob['heading'], ob['elevation']))

            # Early exit if all ended
            if ended.all():
                break

        self.losses.append(self.loss.item() / self.episode_len)
        return traj

    def test(self, use_dropout=False, feedback='argmax', allow_cheat=False):
        ''' Evaluate once on each instruction in the current environment '''
        if not allow_cheat: # permitted for purpose of calculating validation loss only
            assert feedback in ['argmax', 'sample'] # no cheating by using teacher at test time!
        self.feedback = feedback
        if use_dropout:
            self.encoder.train()
            self.decoder.train()
        else:
            self.encoder.eval()
            self.decoder.eval()
        super(Seq2SeqAgent, self).test()

    def train(self, encoder_optimizer, decoder_optimizer, n_iters, feedback='teacher'):
        ''' Train for a given number of iterations '''
        assert feedback in self.feedback_options
        self.feedback = feedback
        self.encoder.train()
        self.decoder.train()
        self.losses = []
        for iter in range(1, n_iters + 1):
            encoder_optimizer.zero_grad()
            decoder_optimizer.zero_grad()
            self.rollout()
            self.loss.backward()
            encoder_optimizer.step()
            decoder_optimizer.step()

    def save(self, encoder_path, decoder_path):
        ''' Snapshot models '''
        torch.save(self.encoder.state_dict(), encoder_path)
        torch.save(self.decoder.state_dict(), decoder_path)

    def load(self, encoder_path, decoder_path):
        ''' Loads parameters (but not training state) '''
        self.encoder.load_state_dict(torch.load(encoder_path))
        self.decoder.load_state_dict(torch.load(decoder_path))
