import torchvision.transforms as transforms
import pickle

from gan.learner import learn
from gan.config import *
from gan.datasets import *
from models.sagan import Generator, Discriminator                                       #モデルを変える時に変える

model_name = 'model name'                                                              #モデルを変える時に変える
img_dir = 'Path to training data'            #データセットを変える時に変える
size = 64          
current_dir = os.getcwd()                                                               #モデルを変える時に変える
config = Config_sagan(model_name, current_dir)                                          #モデルを変える時に変える

transform = transforms.Compose([transforms.RandomHorizontalFlip(), transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
dataset = Image_Dataset(img_dir=img_dir, size=size, transform=transform)                #データセットを変える時に変える

if config.load_models:
    if config.overwrite_models:
        generator = pickle.load(open(config.model_generator, 'rb')).to(config.device)
        discriminator = pickle.load(open(config.model_discriminator, 'rb')).to(config.device)
        print('generator     :',config.model_generator)
        print('discriminator :',config.model_discriminator)
    else:
        files = os.listdir(config.model_dir)
        generators = [int(a.replace('generator', '').replace('.pkl', '')) for a in files if a.startswith('generator')]
        n = max(generators)
        model_generator = os.path.join(config.model_dir, f'generator{n}.pkl')
        model_discriminator = os.path.join(config.model_dir, f'discriminator{n}.pkl')
        generator = pickle.load(open(model_generator, 'rb')).to(config.device)
        discriminator = pickle.load(open(model_discriminator, 'rb')).to(config.device)
        print('generator     :', model_generator)
        print('discriminator :', model_discriminator)
else:
    generator = Generator(config.n_channel, bias=config.bias).to(config.device)
    discriminator = Discriminator(bias=config.bias).to(config.device)

learn(config, dataset, generator, discriminator)