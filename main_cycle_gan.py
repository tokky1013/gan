import torchvision.transforms as transforms
import pickle

from gan.learner import learn_cyclegan
from gan.config import *
from gan.datasets import *
from models.cycle_gan import Generator, Discriminator                                                       #学習モデル　モデルを変える時に変える

model_name = 'model name'                                                                                    #学習モデルの名前　モデルを変える時に変える
img_dir = 'Path to training data'         #学習データのパス　データセットを変える時に変える
size = 256
current_dir = os.getcwd()
config = Config_cycle_gan(model_name, current_dir)                                                           #学習モデルの設定　モデルを変える時に変える
train = True

transform = transforms.Compose([transforms.RandomHorizontalFlip(), transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
dataset = CycleGan_Dataset(img_dir=img_dir, size=size, transform=transform, train=train)                       #データセットを変える時に変える

if config.load_models:
    if config.overwrite_models:
        G_AtoB = pickle.load(open(config.model_G_AtoB, 'rb')).to(config.device)
        G_BtoA = pickle.load(open(config.model_G_BtoA, 'rb')).to(config.device)
        D_A = pickle.load(open(config.model_D_A, 'rb')).to(config.device)
        D_B = pickle.load(open(config.model_D_B, 'rb')).to(config.device)
        print('Generator AtoB  :', config.model_G_AtoB)
        print('Generator BtoA  :', config.model_G_BtoA)
        print('Discriminator A :', config.model_D_A)
        print('Discriminator B :', config.model_D_B)
    else:
        files = os.listdir(config.model_dir)
        generators = [int(a.replace('generatorAtoB', '').replace('.pkl', '')) for a in files if a.startswith('generatorAtoB')]
        n = max(generators)
        model_G_AtoB = os.path.join(config.model_dir, f'generatorAtoB{n}.pkl')
        model_G_BtoA = os.path.join(config.model_dir, f'generatorBtoA{n}.pkl')
        model_D_A = os.path.join(config.model_dir, f'discriminatorA{n}.pkl')
        model_D_B = os.path.join(config.model_dir, f'discriminatorB{n}.pkl')
        G_AtoB = pickle.load(open(model_G_AtoB, 'rb')).to(config.device)
        G_BtoA = pickle.load(open(model_G_BtoA, 'rb')).to(config.device)
        D_A = pickle.load(open(model_D_A, 'rb')).to(config.device)
        D_B = pickle.load(open(model_D_B, 'rb')).to(config.device)
        print('Generator AtoB  :', model_G_AtoB)
        print('Generator BtoA  :', model_G_BtoA)
        print('Discriminator A :', model_D_A)
        print('Discriminator B :', model_D_B)
else:
    G_AtoB = Generator(config.res_block).to(config.device)
    G_BtoA = Generator(config.res_block).to(config.device)
    D_A = Discriminator().to(config.device)
    D_B = Discriminator().to(config.device)

learn_cyclegan(config, dataset, G_AtoB, G_BtoA, D_A, D_B)