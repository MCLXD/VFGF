import torch
from torch.nn import functional as F
from torch.optim import lr_scheduler
import torch.nn as nn
from os.path import join
from datetime import datetime
from utils import topk_accuracy, ValueMeter
import numpy as np
from warmup_scheduler import GradualWarmupScheduler

# 定义判别器网络
class Discriminator(nn.Module):
    def __init__(self, input_dim):
        super(Discriminator, self).__init__()
        self.fc1 = nn.Linear(input_dim, 128)
        self.fc2 = nn.Linear(128, 1)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.sigmoid(self.fc2(x))
        return x

def log(mode, epoch, loss_meter, accuracy_meter, best_perf=None, green=False):
    if green:
        print('\033[92m', end="")
    print(f"time:[{datetime.now()}]")
    print(
        f"[{mode}] Epoch: {epoch:0.2f}. "
        f"Loss: {loss_meter.value():.2f}. "
        f"Accuracy: {accuracy_meter.value():.2f}% ", end="")

    if best_perf:
        print(f"[best: {best_perf:0.2f}]%", end="")

    print('\033[0m')

def save_model(args, model, epoch, train_perf, perf, best_perf, is_best=False):
    torch.save({'state_dict': model.state_dict(), 'epoch': epoch, 'train_perf': train_perf,
                'perf': perf, 'best_perf': best_perf}, join(args.save_path, args.exp_name + '.pth.tar'))
    if is_best:
        torch.save({'state_dict': model.state_dict(), 'epoch': epoch, 'train_perf': train_perf, 'perf': perf, 'best_perf': best_perf}, join(
            args.save_path, args.exp_name + '_best.pth.tar'))
def train(args, model, loaders, optimizer, epochs, start_epoch, start_best_perf, trainval_logger, writer):
    """Training/Validation code"""
    best_perf = start_best_perf  # to keep track of the best performing epoch
    for epoch in range(start_epoch, epochs):
        # define training and validation meters
        loss_meter = {'training': ValueMeter(), 'validation': ValueMeter()}
        annot_loss_meter = {'training': ValueMeter(), 'validation': ValueMeter()}
        accuracy_meter = {'training': ValueMeter(), 'validation': ValueMeter()}

        for mode in ['training', 'validation']:
            # enable gradients only if training
            with torch.set_grad_enabled(mode == 'training'):
                if mode == 'training':
                    model.train()
                else:
                    model.eval()
                ##print("enumerate(loaders[mode])-0", datetime.now())
                for i, batch in enumerate(loaders[mode]):
                    ##print("enumerate(loaders[mode])-1", datetime.now())
                    x = batch['past_features' if args.task == 'anticipation' else 'action_features']
                    if type(x) == list:
                        x = [xx.to(args.device) for xx in x]
                    else:
                        x = x.to(args.device)
                    #id  past_features  lable[v,n,a]  past_frames  all_sample_features
                    label_temp = batch['label'].long().to(args.device)

                    y = label_temp[:, 2] # 单列
                    #feature_labels = x.contiguous()

                    bs = y.shape[0]  # batch size
                    preds, pred_FGM = model(x)#, verb_embedding, noun_embedding) -> base_srl forward
                    ## anticipation loss
                    if args.pre_train is False:

                        preds = preds[:, -args.S_ant:, :].contiguous()  # n 8 dim
                        linear_preds = preds.view(-1, preds.shape[-1])  # 16(8+8) * 2513a
                        linear_labels = y.view(-1, 1).expand(-1, preds.shape[1]).contiguous().view(-1)  # 16(8+8) 准备交叉熵计算
                        ant_loss = F.cross_entropy(linear_preds, linear_labels)

                        pred_FGM = pred_FGM[:, -args.S_ant:, :].contiguous()  # n 8 dim
                        linear_pred_FGM = pred_FGM.view(-1, pred_FGM.shape[-1])  # 16(8+8) * 2513a
                        ant_loss_FGM = F.cross_entropy(linear_pred_FGM, linear_labels) * 0.1
                    else:
                        ant_loss = F.cross_entropy(preds, y)
                        ant_loss_FGM = 0

                    # 总损失
                    loss = ant_loss + ant_loss_FGM
                    #loss = ant_loss + args.reinforce_verb_weight * reinforce_verb_loss + args.reinforce_noun_weight * reinforce_noun_loss
                    if args.pre_train is False:
                        acc = topk_accuracy(preds[:, -4, :].detach().cpu().numpy(), y.detach().cpu().numpy(), (5,))[0]*100
                    else:
                        acc = topk_accuracy(preds.detach().cpu().numpy(), y.detach().cpu().numpy(), (5,))[0] * 100
                    # store the values in the meters to keep incremental averages
                    loss_meter[mode].add(loss.item(), bs)
                    annot_loss_meter[mode].add(ant_loss.item(), bs)
                    accuracy_meter[mode].add(acc, bs)
                    writer.add_scalar( mode + '/total_loss_iter', loss_meter[mode].value(), i + 1 + len(loaders[mode]) * (epoch - 1))
                    writer.add_scalar( mode + '/annot_loss_iter', annot_loss_meter[mode].value(), i + 1 + len(loaders[mode]) * (epoch - 1))
                    writer.add_scalar( mode + '/accuracy_iter', accuracy_meter[mode].value(), i + 1 + len(loaders[mode]) * (epoch - 1))
                    # if in training mode
                    if mode == 'training':

                        optimizer.zero_grad()
                        loss.backward()
                        torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip_value)
                        optimizer.step()

                    # compute decimal epoch for logging
                    e = epoch + i/len(loaders[mode])

                    # log training during loop
                    if mode == 'training' and i != 0 and i % args.display_every == 0:
                        log(mode, e, loss_meter[mode], accuracy_meter[mode])
                    ##print("enumerate(loaders[mode])-2", datetime.now())

                # log at the end of each epoch
                log(mode, epoch+1, loss_meter[mode], accuracy_meter[mode], max(accuracy_meter[mode].value(), best_perf) if mode == 'validation' else None, green=True)

        trainval_logger.log({
            'epoch': epoch,
            'train_loss': loss_meter['training'].value(),
            'val_loss': loss_meter['validation'].value(),
            'train_accuracy': accuracy_meter['training'].value(),
            'val_accuracy': accuracy_meter['validation'].value(),
        })

        writer.add_scalar( mode + '/total_loss_epoch', loss_meter[mode].value(), epoch)
        writer.add_scalar( mode + '/annot_loss_epoch', annot_loss_meter[mode].value(), epoch)
        writer.add_scalar( mode + '/accuracy_epoch', accuracy_meter[mode].value(), epoch)

        if best_perf < accuracy_meter['validation'].value():
            best_perf = accuracy_meter['validation'].value()
            is_best = True
        else:
            is_best = False

        # save checkpoint at the end of each train/val epoch
        save_model(args, model, epoch+1, accuracy_meter['training'].value(), accuracy_meter['validation'].value(), best_perf, is_best=is_best)
def train_attm(args, model, loaders, optimizer, epochs, start_epoch, start_best_perf, trainval_logger, writer):
    """Training/Validation code"""
    best_perf = start_best_perf  # to keep track of the best performing epoch

    for epoch in range(start_epoch, epochs):
        # define training and validation meters
        loss_meter = {'training': ValueMeter(), 'validation': ValueMeter()}
        annot_loss_meter = {'training': ValueMeter(), 'validation': ValueMeter()}
        accuracy_meter = {'training': ValueMeter(), 'validation': ValueMeter()}

        for mode in ['training', 'validation']:
            # enable gradients only if training
            with torch.set_grad_enabled(mode == 'training'):
                if mode == 'training':
                    model.train()
                else:
                    model.eval()
                for i, batch in enumerate(loaders[mode]):
                    ##print("enumerate(loaders[mode])-1", datetime.now())
                    x = batch['past_features' if args.task == 'anticipation' else 'action_features']
                    if type(x) == list:
                        x = [xx.to(args.device) for xx in x]
                    else:
                        x = x.to(args.device)
                    #id  past_features  lable[v,n,a]  past_frames  all_sample_features
                    label_temp = batch['label'].long().to(args.device)
                    y = label_temp[:, 2] # 单列

                    bs = y.shape[0]  # batch size

                    preds= model(x)#, verb_embedding, noun_embedding) -> base_srl forward
                    ## anticipation loss
                    preds = preds[:, -args.S_ant:, :].contiguous() #n 8 dim
                    linear_preds = preds.view(-1, preds.shape[-1])#16(8+8) * 2513a
                    linear_labels = y.view(-1, 1).expand(-1, preds.shape[1]).contiguous().view(-1) #16(8+8) 准备交叉熵计算
                    ant_loss = F.cross_entropy(linear_preds, linear_labels)

                    # 总损失
                    loss = ant_loss

                    acc = topk_accuracy(preds[:, -4, :].detach().cpu().numpy(), y.detach().cpu().numpy(), (5,))[0]*100
                    #print(acc)
                    # store the values in the meters to keep incremental averages
                    loss_meter[mode].add(loss.item(), bs)
                    annot_loss_meter[mode].add(ant_loss.item(), bs)
                    accuracy_meter[mode].add(acc, bs)

                    writer.add_scalar( mode + '/total_loss_iter', loss_meter[mode].value(), i + 1 + len(loaders[mode]) * (epoch - 1))
                    writer.add_scalar( mode + '/annot_loss_iter', annot_loss_meter[mode].value(), i + 1 + len(loaders[mode]) * (epoch - 1))
                    writer.add_scalar( mode + '/accuracy_iter', accuracy_meter[mode].value(), i + 1 + len(loaders[mode]) * (epoch - 1))

                    # if in training mode
                    if mode == 'training':

                        optimizer.zero_grad()
                        loss.backward()
                        torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip_value)
                        optimizer.step()

                    # compute decimal epoch for logging
                    e = epoch + i/len(loaders[mode])

                    # log training during loop
                    if mode == 'training' and i != 0 and i % args.display_every == 0:
                        log(mode, e, loss_meter[mode], accuracy_meter[mode])
                    ##print("enumerate(loaders[mode])-2", datetime.now())

                # log at the end of each epoch
                log(mode, epoch+1, loss_meter[mode], accuracy_meter[mode], max(accuracy_meter[mode].value(), best_perf) if mode == 'validation' else None, green=True)

        trainval_logger.log({
            'epoch': epoch,
            'train_loss': loss_meter['training'].value(),
            'val_loss': loss_meter['validation'].value(),
            'train_accuracy': accuracy_meter['training'].value(),
            'val_accuracy': accuracy_meter['validation'].value(),
        })

        writer.add_scalar( mode + '/total_loss_epoch', loss_meter[mode].value(), epoch)
        writer.add_scalar( mode + '/annot_loss_epoch', annot_loss_meter[mode].value(), epoch)
        writer.add_scalar( mode + '/accuracy_epoch', accuracy_meter[mode].value(), epoch)

        if best_perf < accuracy_meter['validation'].value():
            best_perf = accuracy_meter['validation'].value()
            is_best = True
        else:
            is_best = False

        # save checkpoint at the end of each train/val epoch
        save_model(args, model, epoch+1, accuracy_meter['training'].value(), accuracy_meter['validation'].value(), best_perf, is_best=is_best)