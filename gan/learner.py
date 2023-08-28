from torch.utils.data import DataLoader
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import LambdaLR

import datetime
import pytz
import os
import requests
from tqdm import tqdm
import json
import pickle
import torchvision.utils as vutils
from itertools import chain

def learn(config, dataset, generator, discriminator):
    dataloader = DataLoader(dataset, batch_size=config.batch_size, shuffle=True, num_workers=config.num_workers, pin_memory=True, drop_last=True)

    criterion = nn.MSELoss()
    optimizerG = optim.Adam(generator.parameters(), lr=config.lr_generaor, betas=(config.beta1, config.beta2), weight_decay=1e-5)
    optimizerD = optim.Adam(discriminator.parameters(), lr=config.lr_discriminator, betas=(config.beta1, config.beta2), weight_decay=1e-5)

    schedulerG = LambdaLR(optimizerG,lr_lambda=loss_scheduler(config.epoch_decay, config.pre, config.n_epoch).f)
    schedulerD = LambdaLR(optimizerD,lr_lambda=loss_scheduler(config.epoch_decay, config.pre, config.n_epoch).f)

    torch.backends.cudnn.benchmark = True

    i = 0
    for epoch in range(config.pre, config.n_epoch):
        for x in tqdm(dataloader, total=len(dataloader)):
            real_image = x.to(config.device)
            real_target = torch.full((config.batch_size,), 1., device=config.device)
            fake_target = torch.full((config.batch_size,), 0., device=config.device)

            if i % config.gen_update_interval == 0:
                set_requires_grad(discriminator, False)
                discriminator.zero_grad()
                generator.zero_grad()

                noize = torch.randn(config.batch_size, config.n_channel, 1, 1, device=config.device)
                fake_image = generator(noize)
                fake_img_tensor = fake_image.detach()
                y = discriminator(fake_image)
                errG = criterion(y, real_target)
                errG.backward(retain_graph=True)
                optimizerG.step()
            
            set_requires_grad(discriminator, True)
            discriminator.zero_grad()
            generator.zero_grad()
            
            y = discriminator(real_image)
            errD_real = criterion(y, real_target)
            D_x = y.mean().item()
            
            y = discriminator(fake_img_tensor)
            errD_fake = criterion(y, fake_target)
            D_G_z = y.mean().item()

            errD = errD_real + errD_fake
            errD.backward(retain_graph=True)
            optimizerD.step()
            i += 1

        schedulerG.step()
        schedulerD.step()

        with open(config.config_file, 'r') as f:
            pre_info = json.load(f)
            config.n_epoch = pre_info['n_epoch']
            try:
                config.notification_interval = pre_info['notification_interval']
                config.image_saving_interval = pre_info['image_saving_interval']
                config.model_saving_interval = pre_info['model_saving_interval']
                config.overwrite_models = pre_info['overwrite_models']
            except:
                pass
        with open(config.config_file, 'w') as f:
            book = {'name': config.model_name,
                    'epoch': epoch+1,
                    'n_epoch': config.n_epoch,
                    'notification_interval': config.notification_interval,
                    'image_saving_interval': config.image_saving_interval,
                    'model_saving_interval': config.model_saving_interval,
                    'overwrite_models': config.overwrite_models,
                    'cycle_gan': False
                    }
            f.write(json.dumps(book))

        if config.overwrite_models:
            pickle.dump(generator, open(config.model_generator, 'wb'))
            pickle.dump(discriminator, open(config.model_discriminator, 'wb'))
        else:
            model_num = epoch + config.model_saving_interval - epoch % config.model_saving_interval
            pickle.dump(generator, open(os.path.join(config.model_dir, f'generator{model_num}.pkl'), 'wb'))
            pickle.dump(discriminator, open(os.path.join(config.model_dir, f'discriminator{model_num}.pkl'), 'wb'))

        if (epoch+1) % config.image_saving_interval == 0 or epoch==0:
            vutils.save_image(fake_img_tensor, os.path.join(config.fake_image_dir, f'generated_images{epoch+1}.png'), normalize=True, nrow=10)
        if config.notification_interval > 0:
            if (epoch+1) % config.notification_interval == 0 and epoch+1 != config.n_epoch+config.pre:
                loss_dict = {'loss_G':errG, 'Loss_D':errD}
                notify(config, loss_dict)

        time = datetime.datetime.now(pytz.timezone('Asia/Tokyo'))
        time = time.strftime('%Y/%m/%d %H:%M:%S')
        print('{} Epoch:{}/{}, Loss_D:{:.3f}, Loss_G:{:.3f}, D(x):{:.3f}, D(G(z)):{:.3f}, lr_G:{}, lr_D:{}\n'.format(time, epoch+1, config.n_epoch, errD.item(), errG.item(), D_x, D_G_z, schedulerG.get_last_lr()[-1], schedulerD.get_last_lr()[-1]))

        if epoch+1 >= config.n_epoch:
            break

    loss_dict = {'loss_G':errG, 'Loss_D':errD}
    notify(config, loss_dict, LastModel=True)

def learn_cyclegan(config, dataset, G_AtoB, G_BtoA, D_A, D_B):
    dataloader = DataLoader(dataset, batch_size=config.batch_size, shuffle=True, num_workers=config.num_workers, pin_memory=True, drop_last=True)

    adv_loss = nn.MSELoss()
    cycle_loss = nn.L1Loss()
    optimizerG = optim.Adam(chain(G_AtoB.parameters(),G_BtoA.parameters()), lr=config.lr_generaor, betas=(config.beta1, config.beta2), weight_decay=1e-5)
    optimizerD_A = optim.Adam(D_A.parameters(), lr=config.lr_discriminator, betas=(config.beta1, config.beta2), weight_decay=1e-5)
    optimizerD_B = optim.Adam(D_B.parameters(), lr=config.lr_discriminator, betas=(config.beta1, config.beta2), weight_decay=1e-5)

    schedulerG = LambdaLR(optimizerG,lr_lambda=loss_scheduler(config.epoch_decay, config.pre, config.n_epoch).f)
    schedulerD_A = LambdaLR(optimizerD_A,lr_lambda=loss_scheduler(config.epoch_decay, config.pre, config.n_epoch).f)
    schedulerD_B = LambdaLR(optimizerD_B,lr_lambda=loss_scheduler(config.epoch_decay, config.pre, config.n_epoch).f)

    torch.backends.cudnn.benchmark = True

    itr = 0
    for epoch in range(config.pre, config.n_epoch):
        for imgA, imgB in tqdm(dataloader, total=len(dataloader)):
            real_A = imgA.to(config.device)
            real_B = imgB.to(config.device)
            optimizerD_A.zero_grad()
            optimizerD_B.zero_grad()

            if itr % config.gen_update_interval == 0:
                set_requires_grad([D_A,D_B], False)
                fake_A, fake_B = G_BtoA(real_B), G_AtoB(real_A)
                rec_A, rec_B = G_BtoA(fake_B), G_AtoB(fake_A)
                optimizerG.zero_grad()
                
                pred_fake_A = D_A(fake_A)
                loss_G_B2A = adv_loss(pred_fake_A, torch.tensor(1.0).expand_as(pred_fake_A).to(config.device))
                pred_fake_B = D_B(fake_B)
                loss_G_A2B = adv_loss(pred_fake_B, torch.tensor(1.0).expand_as(pred_fake_B).to(config.device))
                loss_cycle_A = cycle_loss(rec_A, real_A)
                loss_cycle_B = cycle_loss(rec_B, real_B)

                #identity lossの計算
                Ide_loss_A = cycle_loss(G_BtoA(real_A), real_A)
                Ide_loss_B = cycle_loss(G_AtoB(real_B), real_B)
                Ide_loss = Ide_loss_A*config.lambda_cycle + Ide_loss_B*config.lambda_cycle

                loss_G = loss_G_A2B + loss_G_B2A + loss_cycle_A*config.lambda_cycle + loss_cycle_B*config.lambda_cycle + Ide_loss

                loss_G.backward()
                optimizerG.step()
            
            set_requires_grad([D_A,D_B], True)
            optimizerD_A.zero_grad()
            pred_real_A = D_A(real_A)
            pred_fake_A = D_A(fake_A.detach())
            
            loss_D_A_real = adv_loss(pred_real_A, torch.tensor(1.0).expand_as(pred_real_A).to(config.device))
            loss_D_A_fake = adv_loss(pred_fake_A, torch.tensor(0.0).expand_as(pred_fake_A).to(config.device))
            loss_D_A = (loss_D_A_fake + loss_D_A_real)*0.5
            loss_D_A.backward()
            optimizerD_A.step()

            optimizerD_B.zero_grad()
            pred_real_B = D_B(real_B)
            pred_fake_B = D_B(fake_B.detach())
            
            loss_D_B_real = adv_loss(pred_real_B, torch.tensor(1.0).expand_as(pred_real_B).to(config.device))
            loss_D_B_fake = adv_loss(pred_fake_B, torch.tensor(0.0).expand_as(pred_fake_B).to(config.device))
            loss_D_B = (loss_D_B_fake + loss_D_B_real)*0.5
            loss_D_B.backward()
            optimizerD_B.step()
            itr += 1
            if config.iter_saving_interval > 0:
                if itr%config.iter_saving_interval == 0:
                    if config.overwrite_models:
                        pickle.dump(G_AtoB, open(config.model_G_AtoB, 'wb'))
                        pickle.dump(G_BtoA, open(config.model_G_BtoA, 'wb'))
                        pickle.dump(D_A, open(config.model_D_A, 'wb'))
                        pickle.dump(D_B, open(config.model_D_B, 'wb'))
                    else:
                        model_num = epoch + config.model_saving_interval - epoch % config.model_saving_interval
                        pickle.dump(G_AtoB, open(os.path.join(config.model_dir, f'generatorAtoB{model_num}.pkl'), 'wb'))
                        pickle.dump(G_BtoA, open(os.path.join(config.model_dir, f'generatorBtoA{model_num}.pkl'), 'wb'))
                        pickle.dump(D_A, open(os.path.join(config.model_dir, f'discriminatorA{model_num}.pkl'), 'wb'))
                        pickle.dump(D_B, open(os.path.join(config.model_dir, f'discriminatorB{model_num}.pkl'), 'wb'))

                    image_num = epoch + config.image_saving_interval - epoch % config.image_saving_interval
                    result_image = torch.cat((real_A, fake_B, rec_A, real_B, fake_A, rec_B),0).detach()
                    vutils.save_image(result_image, os.path.join(config.fake_image_dir, f'generated_images{image_num}.png'), normalize=True, nrow=3)

        schedulerG.step()
        schedulerD_A.step()
        schedulerD_B.step()

        with open(config.config_file, 'r') as f:
            pre_info = json.load(f)
            config.n_epoch = pre_info['n_epoch']
            try:
                config.notification_interval = pre_info['notification_interval']
                config.image_saving_interval = pre_info['image_saving_interval']
                config.model_saving_interval = pre_info['model_saving_interval']
                config.iter_saving_interval = pre_info['iter_saving_interval']
                config.overwrite_models = pre_info['overwrite_models']
            except:
                pass
        with open(config.config_file, 'w') as f:
            book = {'name': config.model_name,
                    'epoch': epoch+1,
                    'n_epoch': config.n_epoch,
                    'notification_interval': config.notification_interval,
                    'image_saving_interval': config.image_saving_interval,
                    'model_saving_interval': config.model_saving_interval,
                    'iter_saving_interval': config.iter_saving_interval,
                    'overwrite_models': config.overwrite_models,
                    'cycle_gan': True
                    }
            f.write(json.dumps(book))

        if config.overwrite_models:
            pickle.dump(G_AtoB, open(config.model_G_AtoB, 'wb'))
            pickle.dump(G_BtoA, open(config.model_G_BtoA, 'wb'))
            pickle.dump(D_A, open(config.model_D_A, 'wb'))
            pickle.dump(D_B, open(config.model_D_B, 'wb'))
        else:
            model_num = epoch + config.model_saving_interval - epoch % config.model_saving_interval
            pickle.dump(G_AtoB, open(os.path.join(config.model_dir, f'generatorAtoB{model_num}.pkl'), 'wb'))
            pickle.dump(G_BtoA, open(os.path.join(config.model_dir, f'generatorBtoA{model_num}.pkl'), 'wb'))
            pickle.dump(D_A, open(os.path.join(config.model_dir, f'discriminatorA{model_num}.pkl'), 'wb'))
            pickle.dump(D_B, open(os.path.join(config.model_dir, f'discriminatorB{model_num}.pkl'), 'wb'))

        if (epoch+1) % config.image_saving_interval == 0 or epoch==0:
            result_image = torch.cat((real_A, fake_B, rec_A, real_B, fake_A, rec_B),0).detach()
            vutils.save_image(result_image, os.path.join(config.fake_image_dir, f'generated_images{epoch+1}.png'), normalize=True, nrow=3)
        if config.notification_interval > 0:
            if (epoch+1) % config.notification_interval == 0 and epoch+1 != config.n_epoch+config.pre:
                loss_dict = {'loss_G':loss_G, 'Loss_D_A':loss_D_A, 'Loss_D_B':loss_D_B}
                notify(config, loss_dict)

        time = datetime.datetime.now(pytz.timezone('Asia/Tokyo'))
        time = time.strftime('%Y/%m/%d %H:%M:%S')
        print('{} Epoch:{}/{}, Loss_G:{:.3f}, Loss_D_A:{:.3f}, Loss_D_B:{:.3f}, lr_G:{}, lr_D:{}\n'.format(time, epoch+1, config.n_epoch, loss_G.item(), loss_D_A.item(), loss_D_B.item(), schedulerG.get_last_lr()[-1], schedulerD_A.get_last_lr()[-1]))
        if epoch+1 >= config.n_epoch:
            break
    
    loss_dict = {'loss_G':loss_G, 'Loss_D_A':loss_D_A, 'Loss_D_B':loss_D_B}
    notify(config, loss_dict, LastModel=True)

def notify(config, loss_dict, LastModel=False):
    if config.notifi:
        time = datetime.datetime.now(pytz.timezone('Asia/Tokyo'))
        time = time.strftime('%Y/%m/%d %H:%M:%S')
        files = os.listdir(config.fake_image_dir)
        files = [int(a.replace('generated_images', '').replace('.png', '')) for a in files if not a.startswith('.')]
        n = max(files)

        if LastModel:
            send_contents =  time + f'\n学習が終了しました。\nEpoch: {n}'
        else:
            send_contents =  time + f'\nEpoch: {n}'

        for key, loss in loss_dict.items():
            send_contents += ', {}: {:.3f}'.format(key, loss.item())

        TOKEN = 'Q6nxGlnM2VwEuE6aBMyEgp3qjxNW7Y4qjuj6JQnO9U4'
        api_url = 'https://notify-api.line.me/api/notify'
        TOKEN_dic = {'Authorization': 'Bearer' + ' ' + TOKEN}
        send_dic = {'message': send_contents}
        image_file = os.path.join(config.fake_image_dir, f'generated_images{n}.png')
        binary = open(image_file, mode='rb')
        image_dic = {'imageFile': binary}
        requests.post(api_url, headers=TOKEN_dic, data=send_dic, files=image_dic)

def set_requires_grad(models, requires=False):
    if not isinstance(models,list):
        models = [models]
    for model in models:
        if model is not None:
            for param in model.parameters():
                param.requires_grad = requires

class loss_scheduler():
    def __init__(self, epoch_decay, pre, n_epoch):
        self.epoch_decay = epoch_decay
        self.pre = pre
        self.n_epoch = n_epoch

    def f(self, epoch):
        #ベースの学習率に対する倍率を返す(pytorch仕様)
        if epoch+self.pre<=self.n_epoch-self.epoch_decay or self.epoch_decay<=0:
            return 1
        else:
            scaling = 1 - (epoch+self.pre-self.n_epoch+self.epoch_decay)/float(self.epoch_decay)
            return scaling