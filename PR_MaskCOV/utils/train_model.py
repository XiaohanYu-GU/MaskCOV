#coding=utf8
from __future__ import print_function, division

import os,time,datetime
import numpy as np
from math import ceil
import datetime

import torch
from torch import nn
from torch.autograd import Variable
#from torchvision.utils import make_grid, save_image

from utils.utils import LossRecord, clip_gradient
from models.focal_loss import FocalLoss
from utils.eval_model import eval_turn
from utils.Asoftmax_loss import AngleLoss

from tensorboardX import SummaryWriter

import pdb

def dt():
    return datetime.datetime.now().strftime("%Y-%m-%d-%H_%M_%S")


def train(Config,
          model,
          epoch_num,
          start_epoch,
          optimizer,
          exp_lr_scheduler,
          data_loader,
          save_dir,
          data_size=448,
          savepoint=500,
          checkpoint=1000
          ):
    # savepoint: save without evalution
    # checkpoint: save with evaluation
    writer = SummaryWriter(save_dir)  # event

    step = 0
    eval_train_flag = False
    rec_loss = []
    checkpoint_list = []

    train_batch_size = data_loader['train'].batch_size
    train_epoch_step = data_loader['train'].__len__()
    train_loss_recorder = LossRecord(train_batch_size)

    date_suffix = dt()
    log_file = open(os.path.join(Config.log_folder, 'formal_log_r50_cdrm_%s_%s.log'%(str(data_size), date_suffix)), 'a')

    add_loss = nn.L1Loss()
    get_ce_loss = nn.CrossEntropyLoss()
    get_focal_loss = FocalLoss()
    get_angle_loss = AngleLoss()

    for epoch in range(start_epoch,epoch_num):
        exp_lr_scheduler.step(epoch)
        model.train(True)

        iteration = 0
        loss_epoch = 0
        loss_epoch_ce = 0
        loss_epoch_swap = 0
        loss_epoch_cova = 0
        for batch_cnt, data in enumerate(data_loader['train']):
            step += 1
            loss = 0
            model.train(True)
            if Config.use_backbone:
                inputs, labels, img_names = data
                inputs = Variable(inputs.cuda())
                labels = Variable(torch.from_numpy(np.array(labels)).cuda())

            if Config.use_cdrm:
                inputs, labels, labels_swap, swap_cov, img_names = data

                inputs = Variable(inputs.cuda())
                labels = Variable(torch.from_numpy(np.array(labels)).cuda())
                labels_swap = Variable(torch.from_numpy(np.array(labels_swap)).cuda())
                swap_cov = Variable(torch.from_numpy(np.array(swap_cov)).float().cuda())

            optimizer.zero_grad()

            if inputs.size(0) < 2*train_batch_size:

                outputs = model(inputs, None)  # zhao yang
            else:
                outputs = model(inputs, None)

            if Config.use_focal_loss:
                ce_loss = get_focal_loss(outputs[0], labels)
            else:
                ce_loss = get_ce_loss(outputs[0], labels)  # CrossEntropyLoss

            if Config.use_Asoftmax:
                fetch_batch = labels.size(0)
                if batch_cnt % (train_epoch_step // 5) == 0:
                    angle_loss = get_angle_loss(outputs[3], labels[0:fetch_batch:2], decay=0.9)
                else:
                    angle_loss = get_angle_loss(outputs[3], labels[0:fetch_batch:2])
                loss += angle_loss

            loss += ce_loss

            beta_ = 1
            gamma_ = 0.01 if Config.dataset == 'STCAR' else 1

            if Config.use_cdrm:
                swap_loss = get_ce_loss(outputs[1], labels_swap) * beta_  # adv loss
                loss += swap_loss
                cova_loss = add_loss(outputs[2], swap_cov) * gamma_  # cov loss
                loss += cova_loss

            loss.backward()
            torch.cuda.synchronize()

            optimizer.step()
            torch.cuda.synchronize()

            if Config.use_cdrm:
                print('step: {:-8d} / {:d} loss=ce_loss+swap_loss+cova_loss: {:6.4f} = {:6.4f} + {:6.4f} + {:6.4f} '.format(step, train_epoch_step, loss.detach().item(), ce_loss.detach().item(), swap_loss.detach().item(), cova_loss.detach().item()), flush=True)
            if Config.use_backbone:
                print('step: {:-8d} / {:d} loss=ce_loss+swap_loss+cova_loss: {:6.4f} = {:6.4f} '.format(step, train_epoch_step, loss.detach().item(), ce_loss.detach().item()), flush=True)
            rec_loss.append(loss.detach().item())

            train_loss_recorder.update(loss.detach().item())

            iteration = iteration + 1
            loss_epoch = loss_epoch + loss.detach().item()
            loss_epoch_ce = loss_epoch_ce + ce_loss.detach().item()
            loss_epoch_swap = loss_epoch_swap + swap_loss.detach().item()
            loss_epoch_cova = loss_epoch_cova + cova_loss.detach().item()

        writer.add_scalar('loss', loss_epoch / iteration, (epoch + 1))
        writer.add_scalar('ce_loss', loss_epoch_ce / iteration, (epoch + 1))
        writer.add_scalar('swap_loss', loss_epoch_swap / iteration, (epoch + 1))
        writer.add_scalar('cova_loss', loss_epoch_cova / iteration, (epoch + 1))

        # evaluation & save
        if (epoch + 1) % checkpoint == 0:  # 表示每几个step保存并且val一次模型

            print(32*'-', flush=True)
            print('step: {:d} / {:d} global_step: {:8.2f} train_epoch: {:04d} rec_train_loss: {:6.4f}'.format(step, train_epoch_step, 1.0*step/train_epoch_step, epoch, train_loss_recorder.get_val()), flush=True)
            print('current lr:%s' % exp_lr_scheduler.get_lr(), flush=True)
            if eval_train_flag:
                trainval_acc1, trainval_acc2, trainval_acc3 = eval_turn(model, data_loader['trainval'], 'trainval', epoch, log_file)
                if abs(trainval_acc1 - trainval_acc3) < 0.01:
                    eval_train_flag = False

            val_acc1, val_acc2, val_acc3 = eval_turn(model, data_loader['val'], 'val', epoch, log_file)

            save_path = os.path.join(save_dir, 'weights_epoch%d_step%d_valtop1acc%.4f_valtop3acc%.4f.pth'%((epoch + 1), (batch_cnt + 1), val_acc1, val_acc3))
            torch.cuda.synchronize()
            torch.save(model.state_dict(), save_path)
            print('saved model to %s' % (save_path), flush=True)
            torch.cuda.empty_cache()

        # save only
        savepoint = epoch_num
        if (epoch + 1) % savepoint == 0:
            # train_loss_recorder.update(rec_loss)
            # rec_loss = []

            val_acc1, val_acc2, val_acc3 = eval_turn(model, data_loader['val'], 'val', epoch, log_file)  # zhaoyang

            save_path = os.path.join(save_dir, 'savepoint-last-model-steps%d-valtop1acc%.4f_valtop3acc%.4f.pth'%(step, val_acc1, val_acc3))

            checkpoint_list.append(save_path)
            if len(checkpoint_list) == 6:
                os.remove(checkpoint_list[0])
                del checkpoint_list[0]
            torch.save(model.state_dict(), save_path)
            torch.cuda.empty_cache()


    log_file.close()



