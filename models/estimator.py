from turtle import forward
from sympy import im
import torch 
from torch import nn 
from .model import TICModel 
from .beam_search import * 

class ConfidenceEstimator(nn.Module): 
    def __init__(self, n_hidden, vocab_size):
        super().__init__() 
        self.fc = nn.Linear(n_hidden, vocab_size) 

    def forward(self, input): 
        return self.fc(input)


class ConfidenceEstimationFramework(nn.Module): 
    def __init__(self, config):
        super().__init__() 
        self.captioner = TICModel(config=config, return_decoder_states=True) 
        self.estimator = ConfidenceEstimator(n_hidden=config.n_embd, vocab_size=config.vocab_size) 
        self.sigmod = nn.Sigmoid() 
    
    def forward(self, images, seq): 
        output = self.captioner(images, seq)
        caption_logits, hidden_states = output[0], output[1][-1]   # (bsz, seq_len, model_d)
        caption_prob = self.sigmod(caption_logits) 
        confidence_prob = self.sigmod(self.estimator(hidden_states)) 
        return caption_prob, confidence_prob 


    def beam_search(self, visual, beam_size: int, out_size=1,
                    return_logits=False, **kwargs): 
        self.captioner.language_decoder.return_states = False 
        bs = BeamSearch(self.captioner, self.captioner.max_generation_length, self.captioner.eos_idx, beam_size)
        return bs.apply(visual, out_size, return_logits, **kwargs)