from torch.utils.data import Dataset
import torchvision.transforms as transforms
from pathlib import Path
from PIL import Image
import random

#64×64のアイコン画像を想定。リサイズなしかつ画像フォルダ直下の画像のみロード。path指定でアイコン以外の画像も読み込み可。
class Icon_Dataset(Dataset):
    def __init__(self, img_dir=None, size=None, transform=None):
        self.img_dir = img_dir
        if self.img_dir == None:
            self.img_dir = '/Users/user/Desktop/MyPython/gan/icon加工済み(size=64)'
        self.img_paths = self._get_img_paths(self.img_dir)
        self.transform = transform
    
    def __getitem__(self, index):
        path = self.img_paths[index]
        img = Image.open(path).convert('RGB')
        if self.transform is not None:
            img = self.transform(img)
        return img
    
    def _get_img_paths(self, img_dir):
        img_dir = Path(img_dir)
        img_paths = [p for p in img_dir.iterdir() if p.suffix in ['.jpg', '.png', '.jpeg']]
        print('n_sample:',len(img_paths))
        return img_paths
    
    def __len__(self):
        return len(self.img_paths)

#任意のフォルダ内のすべての画像をロード。sizeで指定したサイズにリサイズする。基本的にこっちのデータローダーを使うのがいいと思う。
class Image_Dataset(Dataset):
    def __init__(self, transform=None, size=64, img_dir=None):
        self.img_dir = img_dir
        if self.img_dir == None:
            self.img_dir = '/Users/user/Desktop/MyPython/1_データ保管庫/animeface-character-dataset/thumb'
        self.img_paths = self._get_img_paths(self.img_dir)
        self.resize = transforms.Resize(size=(size, size))
        self.transform = transform
    
    def __getitem__(self, index):
        path = self.img_paths[index]
        img = Image.open(path).convert('RGB')
        img = self.resize(img)
        if self.transform is not None:
            img = self.transform(img)
        return img
    
    def _get_img_paths(self, img_dir):
        p = Path(img_dir)
        img_paths = list(p.glob('**/*.*'))
        img_paths = [p for p in img_paths if p.suffix in ['.jpg', '.png', '.jpeg']]
        print('n_sample:',len(img_paths))
        return img_paths
    
    def __len__(self):
        return len(self.img_paths)

#Cycle GAN用のデータセット。指定のフォルダ内にtrainA, trainB, testA, testBがあることを想定。引数のtrainでtrainとtestを切り替え可能
class CycleGan_Dataset(Dataset):
    def __init__(self, img_dir=None, size=256, transform=None, train=True):
        super(Dataset, self).__init__()
        self.img_dir = img_dir
        self.img_paths = self._get_img_paths(self.img_dir, train)
        self.resize = transforms.Resize(size=(size, size))
        self.transform = transform
    
    def __getitem__(self, index):
        pathsA = self.img_paths['pathsA']
        index_A = index % len(pathsA)
        pathA = pathsA[index_A]

        pathsB = self.img_paths['pathsB']
        index_B = random.randint(0, len(pathsB)-1)
        pathB = pathsB[index_B]

        imgA = Image.open(pathA).convert('RGB')
        imgB = Image.open(pathB).convert('RGB')
        imgA = self.resize(imgA)
        imgB = self.resize(imgB)
        if self.transform is not None:
            imgA = self.transform(imgA)
            imgB = self.transform(imgB)
        return imgA, imgB
    
    def _get_img_paths(self, img_dir, train):
        p = Path(img_dir)
        if train:
            img_pathsA = list(p.glob('./trainA/*.*'))
            img_pathsB = list(p.glob('./trainB/*.*'))
            if len(img_pathsA)<=1:
                img_pathsA = list(p.glob('./trainA/*/*.*'))
            if len(img_pathsB)<=1:
                img_pathsB = list(p.glob('./trainB/*/*.*'))
        else:
            img_pathsA = list(p.glob('./testA/*.*'))
            img_pathsB = list(p.glob('./testB/*.*'))
            if len(img_pathsA)<=1:
                img_pathsA = list(p.glob('./testA/*/*.*'))
            if len(img_pathsB)<=1:
                img_pathsB = list(p.glob('./testB/*/*.*'))
        img_pathsA = [p for p in img_pathsA if p.suffix in ['.jpg', '.png', '.jpeg']]
        img_pathsB = [p for p in img_pathsB if p.suffix in ['.jpg', '.png', '.jpeg']]
        self.len_dataset = max(len(img_pathsA), len(img_pathsB))
        print('n_sample:', self.len_dataset)
        return {'pathsA': img_pathsA, 'pathsB': img_pathsB}
    
    def __len__(self):
        return self.len_dataset