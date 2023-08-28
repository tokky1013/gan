import torch
import os
import pickle
import torchvision.utils as vutils
from tqdm import tqdm
import cv2
import sys
sys.path.append('../')
#from models.sngan import Generator

model_name = None

if model_name == None:
    model_name = input('model name: ')
model_path = f'/Users/user/Desktop/MyPython/1_GAN/result/{model_name}/models'
fake_image_path = f'/Users/user/Desktop/MyPython/1_GAN/result/{model_name}'
video_name = f'process.mp4'
temp_folder = './.temp'
batch_size = 100
n_channel = 100

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
torch.backends.cudnn.benchmark = True

if not os.path.exists(temp_folder):
    os.mkdir(temp_folder)

models = os.listdir(model_path)
model_num = [int(model.replace('generator', '').replace('.pkl', '')) for model in models if model.startswith('generator')]
model_num.sort()
image_path = []
print('画像生成中')

with torch.no_grad():
    noize = torch.randn(batch_size, n_channel, 1, 1, device=device)
    for n in tqdm(model_num, total=len(model_num)):
        model = f'generator{n}.pkl'
        path = os.path.join(model_path, model)
        generator = pickle.load(open(path, 'rb')).to(device)
        fake_image = generator(noize)
        vutils.save_image(fake_image.detach(), os.path.join(temp_folder, f'temp{n}.png'), normalize=True, nrow=10)
        image_path.append(os.path.join(temp_folder, f'temp{n}.png'))

print('mp4形式に変換中')
fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # コーデックの指定
video  = cv2.VideoWriter(os.path.join(fake_image_path, video_name), fourcc, 10.0, (662, 662))  # VideoWriter型のオブジェクトを生成、再生速度20fps、サイズは画像と同じにする
for path in tqdm(image_path, total=len(image_path)):
    img = cv2.imread(path)
    n = path.split('/')[-1].replace('temp', '').replace('.png', '')
    cv2.putText(img,
                text='epoch: ' + n,
                org=(15, 20),
                fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                fontScale=0.6,
                color=(255, 255, 255),
                thickness=5,
                lineType=cv2.LINE_4)
    cv2.putText(img,
                text='epoch: ' + n,
                org=(15, 20),
                fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                fontScale=0.6,
                color=(0, 0, 0),
                thickness=2,
                lineType=cv2.LINE_4)
    video.write(img)
    os.remove(path)
video.release()
print('完了')