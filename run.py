import torch
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