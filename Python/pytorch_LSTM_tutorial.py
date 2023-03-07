'''
This tutorial is from the side below.
https://pytorch.org/tutorials/beginner/nlp/sequence_models_tutorial.html
'''

import torch

torch.manual_seed(1)

lstm  = torch.nn.LSTM(3,3)
input = [torch.randn(1,3) for _ in range(5)]

print(input) # [tensor([[-0.5525,  0.6355, -0.3968]]), 
             #  tensor([[-0.6571, -1.6428,  0.9803]]), 
             #  tensor([[-0.0421, -0.8206,  0.3133]]), 
             #  tensor([[-1.1352,  0.3773, -0.2824]]),
             #  tensor([[-2.5667, -1.4303,  0.5009]])]
print(lstm)