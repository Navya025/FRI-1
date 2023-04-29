import torch
from ResNet import ResNet

# create an instance of the ResNet class
model = ResNet()

# load the saved model state dictionary
model.load_state_dict(torch.load('/home/bwilab/resnet/model.pt'))

# set the model to evaluation mode
model.eval()

# use the loaded model for inference or further training
