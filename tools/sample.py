import os
import sys

folder_path = input('フォルダーをドラッグ&ドロップしてください。\n')
n = int(input('n = '))
if folder_path.endswith(' '):
    folder_path = folder_path[:-1]
files = os.listdir(folder_path)
if 'generator.pkl' in files:
    files.remove('generator.pkl')
if 'discriminator.pkl' in files:
    files.remove('discriminator.pkl')
if 'generated_images.png' in files:
    files.remove('generated_images.png')

folder_name = folder_path.split('/')[-1]
if folder_name == 'result_image':
    file_num = [int(a.replace('generated_images', '').replace('.png', '')) for a in files if not a.startswith('.')]
elif folder_name == 'models':
    file_num = [int(a.replace('generator', '').replace('.pkl', '')) for a in files if not (a.startswith('.') or a.startswith('discriminator'))]
else:
    print(f'フォルダ名{folder_name}は未対応です。')
    sys.exit()
file_num.sort()

del file_num[-1]
del file_num[0]
file_num = [i for i in file_num if i % n != 0]

for n in file_num:
    if folder_name == 'result_image':
        file_name = f'generated_images{n}.png'
    else:
        file_name = f'generator{n}.pkl'
        os.remove(os.path.join(folder_path, file_name))
        file_name = f'discriminator{n}.pkl'
    os.remove(os.path.join(folder_path, file_name))