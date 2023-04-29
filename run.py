import torch
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
from strhub.models.parseq.system import PARSeq
from strhub.data.module import SceneTextDataModule

# Load model and image transforms
model = PARSeq.load_from_checkpoint('outputs/parseq/2023-04-26_16-02-14/checkpoints/last.ckpt')

#parseq = torch.hub.load('baudm/parseq', 'parseq', pretrained=True).eval()
img_transform = SceneTextDataModule.get_transform(model.hparams.img_size)

img = Image.open('./numbers/ahg_down_2_1.jpeg').convert('RGB')
# Preprocess. Model expects a batch of images with shape: (B, C, H, W)
img = img_transform(img).unsqueeze(0)

#logits = parseq(img)
logits = model(img)
logits.shape  # torch.Size([1, 26, 95]), 94 characters + [EOS] symbol

# Greedy decoding
pred = logits.softmax(-1)
#label, confidence = parseq.tokenizer.decode(pred)
label, confidence = model.tokenizer.decode(pred)
print('Decoded label = {}'.format(label[0]))

# RESNET MODEL

#define the cutoff
threshold = 0.75;

# Load the trained model
model = models.resnet18(pretrained=False)
model.fc = torch.nn.Linear(512, 2)
model.load_state_dict(torch.load('ResNet/ResNetModel.pt'))
model.eval()

# Load and preprocess a single image
img = Image.open('./ahg_up_1_4.jpeg')
transform = transforms.Compose([transforms.Resize(256),
                                transforms.CenterCrop(224),
                                transforms.ToTensor(),
                                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                     std=[0.229, 0.224, 0.225])])
img_tensor = transform(img)
img_tensor = img_tensor.unsqueeze(0)

# Get the predicted class probabilities
with torch.no_grad():
    output = model(img_tensor)
    probs = torch.nn.functional.softmax(output, dim=1)

# Check if both activation levels are below threshold
if probs[0][0] < threshold and probs[0][1] < threshold:
    direction = 'stop'
else:
    # Get the predicted class label
    _, predicted = torch.max(output, 1)
    if predicted == 0:
        direction = 'down'
    else:
        direction = 'up'

floor = label[0]

if direction != 'stop':
    print('The elevator is moving ' + direction + ' to floor ' + floor)
else:
    print('The elevator is stopped on floor ' + floor)