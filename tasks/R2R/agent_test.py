''' Agents: stop/random/shortest/seq2seq  '''

import json
import re
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

###
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
    def __init__(self, env, results_path):
        super().__init__(env, results_path) 
        ## Empty cache to avoid CUDA out of memory
        torch.cuda.empty_cache()
        self.model_id = "microsoft/Phi-3-vision-128k-instruct" 
        self.model = AutoModelForCausalLM.from_pretrained(self.model_id, device_map="cuda", trust_remote_code=True, torch_dtype="auto",
            _attn_implementation='flash_attention_2', cache_dir=os.environ['HF_HOME']) # use _attn_implementation='eager' to disable flash attention
        self.processor = AutoProcessor.from_pretrained(self.model_id, trust_remote_code=True)
        self.image_history = []
        self.traj_code = "-"
        # self.next_action = ""

    def open_image(self, image_path): ## I added state_rgb
        state = self.env.env.sim.getState()[0]

        rgb = np.array(state.rgb, copy=False)
        cv2.imwrite(image_path, rgb)
        raw_img = Image.open(image_path).convert("RGB")
        raw_img.save(image_path)
        # print("image saved to", image_path)
        ## Just for printing
        # if self.temp_save_image_counter < 50:
        #     raw_img.save(f"unused_images/{str(self.temp_save_image_counter)}.jpg")
        #     self.temp_save_image_counter += 1
        return raw_img

    def parse_response(self, ob, response):
        can_forward = len(ob['navigableLocations']) > 1
        if "move forward" in response:
            if can_forward:
                self.traj_code = self.traj_code + "F"
                return (1, 0, 0)
            self.traj_code = self.traj_code + "f"
            return (0, 1, 0)  # The default is to turn right
        elif "turn left" in response:
            self.traj_code = self.traj_code + "L"
            return (0, -1, 0)
        elif "turn right" in response:
            self.traj_code = self.traj_code + "R"
            return (0, 1, 0)
        elif "stop" in response:
            self.traj_code = self.traj_code + "S"
            return (0, 0, 0)
        elif "look up" in response:
            self.traj_code = self.traj_code + "U"
            return (0, 0, 1)
        elif "look down" in response:
            self.traj_code = self.traj_code + "D"
            return (0, 0, -1)
        else:
            self.traj_code = self.traj_code + "r"
            return (0, 1, 0)  # The default is to turn right


    def run(self, ob):
        can_forward = len(ob['navigableLocations']) > 1
        current_view = self.open_image("current_view.jpg")
        self.image_history.append(Image.open("current_view.jpg"))

        actions = ["turn left", "turn right", "stop", "look up", "look down"]
        if can_forward: actions.append("move forward")
        actions = "\n".join(f"{i+1}. {action}" for i, action in enumerate(actions))

        image_history = self.image_history[-3:]

        image_template = '\n'.join([f'<|image_{str(i+1)}|>' for i in range(len(image_history))])
        response_history = '\n\n'.join(self.response_history)
        system_prompt = "You are a navigation agent who follows instruction to move in an indoor environment with the least action steps. You can move around the room by going forward or changing the direction you're facing."
        user_prompt = "\n".join([
            f"{image_template}",
            f"The images above show scenes you encounter along the way, and the last one represents your current view.",
            f"",
            f"Please follow the instruction:",
            f"{ob['instructions_goal_only']}",
            # f"Find the staircase upward.",
            f"",
            f"The actions available are:",
            f"{actions}",
            f"",
            f"Here is your action history:",
            f"{response_history}",
            f"",
            # f"Please take an action in the format ACTION: <NUMBER>. <ACTION>. Only stop if you think you complete the instruction",
            f"Stop if you complete the instruction. Describe the observation first. Then conclude one action at the end in the format:",
            f"",
            f"OBSERVATION: <OBSERVATION>",
            f"ACTION: <ACTION>",
        ])

        print(user_prompt)

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]
        # print(messages)

        prompt = self.processor.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = self.processor(prompt, image_history, return_tensors="pt").to("cuda:0")
        generation_args = { 
            "max_new_tokens": 256, 
            "temperature": 0.0, 
            "do_sample": False, 
        } 
        generate_ids = self.model.generate(**inputs, eos_token_id=self.processor.tokenizer.eos_token_id, **generation_args)
        generate_ids = generate_ids[:, inputs['input_ids'].shape[1]:]
        response = self.processor.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0] 
        print(response)
        # pattern = r"OBSERVE:\s*([\d.]+\s*[^.]+)\.\s*ACTION:\s*(.+)"
        # pattern = r"ACTION \s*([\d.]+\s*[^.]+)"
        pattern = r"ACTION\s*([\w.]+:[^\n]*)"
        matches = re.search(pattern, response)
        action = matches.group(1) if matches else response
        self.response_history.append(response.replace("\n\n", "\n"))
        # self.next_action = next_action
        return action

    def rollout(self):
        # reset all
        obs = self.env.reset()
        traj = [{
            'instr_id': ob['instr_id'],
            'path': [(ob['viewpoint'], ob['heading'], ob['elevation'])]
        } for ob in obs]
        self.image_history = []
        self.response_history = []
        # self.next_action = ""
        self.traj_code = "-"
        done = False
        ended = [False] * len(obs)

        for t in range(30):
            if done: break
            # if obs[0]['instr_id'].startswith("2390"): break
            actions = []
            for ob in obs:
                response = self.run(ob)
                action = self.parse_response(ob, response)
                actions.append(action)
                # print(f"FINAL ACTION: {action}\n=======")
                if action == (0, 0, 0):
                    done = True
                    ended[i] = True

            obs = self.env.step(actions)
            
            for i, ob in enumerate(obs):
                if not ended[i]:
                    traj[i]['path'].append((ob['viewpoint'], ob['heading'], ob['elevation']))
            # time.sleep(1)
            # input("Press any key to continue...")
        print("phi", ob['instr_id'], self.traj_code)
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
