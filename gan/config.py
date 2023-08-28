import os
import json
import sys
import torch

class Config:
    def __init__(self, model_name, current_dir):
        self.model_name = model_name
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        os.chdir(current_dir)
        self.model_generator = self.set_path(f'result/{model_name}/generator.pkl')
        self.model_discriminator = self.set_path(f'result/{model_name}/discriminator.pkl')
        self.fake_image_dir = self.set_path(f'result/{model_name}/result_image')
        self.model_dir = self.set_path(f'result/{model_name}/models')
        self.config_file = self.set_path(f'result/{model_name}/.config.json')
        self.set_pre_info()
    
    def set_path(self, path):
        #引数のpathまでに存在しないディレクトリがあればすべて作成し、引数をそのまま返す。
        a = os.path.basename(path)
        if '.' in a[1:]:
            dir = os.path.dirname(path)         #pathがファイルの場合
        else:
            dir = path                          #pathがディレクトリの場合
        if not os.path.exists(dir):
            os.makedirs(dir)
        return path
    
    def set_pre_info(self):
        #config.load_models がTrue の時、Trueのままでいいか確認
        if self.load_models:
            model_files = os.listdir(self.model_dir)
            model_files = [a for a in model_files if a.endswith('.pkl')]
            if len(model_files)==0:
                while True:
                    text = input('モデルが存在しません。初めから学習してよろしいですか？(y/n)\n')
                    if text == 'y':
                        self.load_models = False
                        break
                    elif text == 'n':
                        sys.exit()
                    else:
                        print('yかnを入力してください。')
        else:
            if self.overwrite_models:
                try:
                    model_exists = os.path.exists(self.model_generator) and os.path.exists(self.model_discriminator)
                except:
                    model_exists = os.path.exists(self.model_G_AtoB) and os.path.exists(self.model_G_BtoA) and os.path.exists(self.model_D_A) and os.path.exists(self.model_D_B)
            else:
                model_files = [a for a in os.listdir(self.model_dir) if a.endswith('.pkl')]
                model_exists = len(model_files)!=0

            if model_exists:
                while True:
                    text = input('前回保存したモデルがあります。初めから学習しますか？(y/n)\n')
                    if text == 'y':
                        break
                    elif text == 'n':
                        self.load_models = True
                        break
                    else:
                        print('yかnを入力してください。')

        #jsonファイルを読み込み、pre_infoに代入。ファイルが読み込めないか、self.load_modelsがFalseの時pre_info=None
        pre_info = None
        if self.load_models:
            try:
                with open(self.config_file) as f:
                    pre_info = json.load(f)
            except:
                pass
        if pre_info != None:
            pre, n_epoch = pre_info['epoch'], pre_info['n_epoch']

        #pre_infoがNoneの時、configの情報をjsonファイルに書き出す。Noneじゃない時はconfigに前回の情報を書き込む。
        if pre_info == None:
            self.pre = 0
        elif self.load_config:
            self.pre = pre_info['epoch']
            self.n_epoch = pre_info['n_epoch']
            try:
                self.notification_interval = pre_info['notification_interval']
                self.image_saving_interval = pre_info['image_saving_interval']
                self.model_saving_interval = pre_info['model_saving_interval']
                self.overwrite_models = pre_info['overwrite_models'] 
            except:
                pass
        else:
            self.pre = pre_info['epoch']

        with open(self.config_file, 'w') as f:
            book = {'name': self.model_name,
                    'epoch': self.pre,
                    'n_epoch': self.n_epoch,
                    'notification_interval': self.notification_interval,
                    'image_saving_interval': self.image_saving_interval,
                    'model_saving_interval': self.model_saving_interval,
                    'overwrite_models': self.overwrite_models
                    }
            f.write(json.dumps(book))
        print(f'Epoch:{self.pre+1}/{self.n_epoch}')

class Config_dcgan(Config):
    def __init__(self, model_name, current_dir):
        self.load_models = False
        self.load_config = False
        self.overwrite_models = False
        self.notifi = True
        self.n_epoch = 1000

        self.lr_generaor = 0.0002
        self.lr_discriminator = 0.0002
        self.beta1 = 0.5
        self.beta2 = 0.999
        self.epoch_decay = 100
        self.gen_update_interval = 1
        self.batch_size = 100
        self.n_channel = 100
        self.num_workers = 0
        self.model_saving_interval = 10
        self.image_saving_interval = 10
        self.notification_interval = 0
        self.bias = False
        
        super().__init__(model_name, current_dir)

class Config_sagan(Config):
    def __init__(self, model_name, current_dir):
        self.load_models = True
        self.load_config = True
        self.overwrite_models = False
        self.notifi = True
        self.n_epoch = 1000

        self.lr_generaor = 0.0001
        self.lr_discriminator = 0.0004
        self.beta1 = 0.
        self.beta2 = 0.9
        self.epoch_decay = 0
        self.gen_update_interval = 1
        self.batch_size = 100
        self.n_channel = 100
        self.num_workers = 0
        self.model_saving_interval = 10
        self.image_saving_interval = 10
        self.notification_interval = 0
        self.bias = False
        
        super().__init__(model_name, current_dir)

class Config_sngan(Config):
    def __init__(self, model_name, current_dir):
        self.load_models = True
        self.load_config = True
        self.overwrite_models = False
        self.notifi = True
        self.n_epoch = 1000

        self.lr_generaor = 0.0002
        self.lr_discriminator = 0.0002
        self.beta1 = 0.5
        self.beta2 = 0.9
        self.epoch_decay = 100
        self.gen_update_interval = 5
        self.batch_size = 100
        self.n_channel = 100
        self.num_workers = 0
        self.model_saving_interval = 1
        self.image_saving_interval = 1
        self.notification_interval = 0
        self.bias = False

        super().__init__(model_name, current_dir)

class Config_cycle_gan(Config):
    def __init__(self, model_name, current_dir):
        self.load_models = False
        self.load_config = True
        self.overwrite_models = False
        self.notifi = True
        self.n_epoch = 100

        self.lr_generaor = 0.0002
        self.lr_discriminator = 0.0002
        self.beta1 = 0.5
        self.beta2 = 0.999
        self.lambda_cycle = 10
        self.res_block = 3
        self.epoch_decay = 50
        self.gen_update_interval = 1
        self.batch_size = 1
        self.n_channel = 100
        self.num_workers = 0
        self.model_saving_interval = 1
        self.image_saving_interval = 1
        self.notification_interval = 1
        self.iter_saving_interval = 20

        super().__init__(model_name, current_dir)

if __name__ == '__main__':
    model_name = 'test'
    #model_name = input('model name:')
    current_dir = '../'
    config = Config_cycle_gan(model_name, current_dir)