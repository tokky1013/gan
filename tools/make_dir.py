import sys
import os
sys.path.append('../')
from gan.config import Config_dcgan

model_name = input('model name:')
current_dir = '../'
config = Config_dcgan(model_name, current_dir)