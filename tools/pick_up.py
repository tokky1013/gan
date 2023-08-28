from PIL import Image
import os

model_name = None
n = None

if n == None:
    n = int(input('n: '))
if model_name == None:
    model_name = input('model name: ')
i = input('epoch: ')
h = input('左から何番目？ ')
u = input('上から何番目？ ')

image_path = f'../result/{model_name}/result_image/generated_images{i}.png'
output_path = f'../result/{model_name}/images'
if not os.path.exists(output_path):
    os.mkdir(output_path)
i = 1
while True:
    file_name = f'icon{i}.png'
    if not os.path.exists(os.path.join(output_path, file_name)):
        break
    i += 1

im = Image.open(image_path)

x = (int(h) - 1)*(n + 2) + 2
y = (int(u) - 1)*(n + 2) + 2
im_crop = im.crop((x, y, x+n, y+n))
im_crop.save(os.path.join(output_path, file_name), quality=95)