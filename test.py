import os
import argparse
import torch
import numpy as np
import gptmodel
from torchvision.transforms import ToTensor
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
from utils import load_config, load_vqgan, preprocess_vqgan

parser = argparse.ArgumentParser(description='Test a VQGAN-GPT')
parser.add_argument('--data_path', default='', type=str, help='data path')
parser.add_argument('--model_dir', default='', type=str, help='Cache path for the stored model')
parser.add_argument('--vqconfig_path', default="/scratch/eo41/visual-recognition-memory/vqgan_pretrained_models/imagenet_16x16_16384.yaml", type=str, help='vq config path')
parser.add_argument('--vqmodel_path', default="/scratch/eo41/visual-recognition-memory/vqgan_pretrained_models/imagenet_16x16_16384.ckpt", type=str, help=' vq model path')
parser.add_argument('--batch_size', default=1, type=int, help='batch size')
parser.add_argument('--num_workers', default=4, type=int, help='number of workers')
parser.add_argument('--gpt_config', default='GPT_gimel', type=str, help='name of GPT config', choices=['GPT_alef', 'GPT_bet', 'GPT_gimel', 'GPT_dalet'])
parser.add_argument('--vocab_size', default=16384, type=int, help='vocabulary size')
parser.add_argument('--block_size', default=255, type=int, help='context size')
parser.add_argument('--n_trials', default=100, type=int, help='number of trials (40 for Konkle, 100 for Brady experiments)')
parser.add_argument('--save_name', default='', type=str, help='informative string for saving')

args = parser.parse_args()
print(args)

model_files = os.listdir(args.model_dir)
model_files.sort()
test_losses = np.zeros((len(model_files), args.n_trials))
model_idx = 0

# load vqgan model to encode images
vq_config = load_config(args.vqconfig_path, display=True)
vq_model = load_vqgan(vq_config, ckpt_path=args.vqmodel_path)
vq_model = vq_model.cuda()
print('Loaded VQ encoder.')

# data pipeline
dataset = ImageFolder(args.data_path, ToTensor())
data_loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True)
print('Data loaded: dataset contains {} images, and takes {} evaluation iterations per epoch.'.format(len(dataset), len(data_loader)))

# set up model
mconf = gptmodel.__dict__[args.gpt_config](args.vocab_size, args.block_size)
model = gptmodel.GPT(mconf)

for model_file in model_files:
    print("Loading model:", model_file)
    checkpoint = torch.load(os.path.join(args.model_dir, model_file))
    model.load_state_dict(checkpoint['model_state_dict'])

    if torch.cuda.is_available():
        model = model.cuda()

    model.eval()

    unreduced_losses = []

    with torch.no_grad():
        for _, (images, _) in enumerate(data_loader):
            with torch.no_grad():
                images = preprocess_vqgan(images.cuda())
                _, _, [_, _, indices] = vq_model.encode(images)
                indices = indices.reshape(images.size(0), -1)
                
            # forward prop
            _, _, unreduced_loss = model(indices[:, :-1], indices[:, 1:])  # first output returns logits, last one returns unreduced losses
            unreduced_losses.append(unreduced_loss.cpu().numpy())
        
        unreduced_losses = np.concatenate(unreduced_losses)
        test_losses[model_idx, :] = unreduced_losses

    model_idx += 1

print('Itemized test losses shape:', test_losses.shape)
np.save('{}.npy'.format(args.save_name), test_losses)