import torch
import os
import json
import glob
import pickle
import torchvision.utils as vutils
import sys
sys.path.append('../')
#from models.sngan import Generator

model_name = None

if model_name == None:
    model_name = input('model name: ')

model_generator = f'../result/{model_name}/generator.pkl'
fake_image_path = f'../result/{model_name}'
model_dir = f'../result/{model_name}/models'
config_file = f'../result/{model_name}/.config.json'
batch_size = 100
n_channel = 100
n_image = 100

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
torch.backends.cudnn.benchmark = True

with open(config_file) as f:
    pre_info = json.load(f)
pre = pre_info['epoch']
try:
    overwrite_models = pre_info['overwrite_models']
    model_saving_interval = pre_info['model_saving_interval']
except:
    files = glob.glob(os.path.join(model_dir, 'generator*.pkl'))
    overwrite_models = (len(files) == 0)
    model_saving_interval = None
if overwrite_models:
    generator = pickle.load(open(model_generator, 'rb')).to(device)
    print('generator:',model_generator)
else:
    if model_saving_interval == None:
        files = [os.path.splitext(os.path.basename(file))[0] for file in files]
        file_num = [int(a.replace('generator', '')) for a in files]
        n = max(file_num)
    else:
        n = (pre-1) + model_saving_interval - (pre-1) % model_saving_interval
    model_generator = os.path.join(model_dir, f'generator{n}.pkl')
    generator = pickle.load(open(model_generator, 'rb')).to(device)
    print('generator:', model_generator)

with torch.no_grad():
    noize = torch.randn(batch_size, n_channel, 1, 1, device=device)
    fake_image = generator(noize)
vutils.save_image(fake_image.detach()[:n_image], os.path.join(fake_image_path, f'generated_images.png'), normalize=True, nrow=10)