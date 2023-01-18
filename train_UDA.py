import os

# os.environ["CUDA_VISIBLE_DEVICES"] = "1"
import argparse
from math import ceil
import random
import logging
import numpy as np
import torch
import torch.nn as nn
torch.backends.cudnn.benchmark=True
import torch.utils.data as data
import torch.nn.functional as F
from torch.autograd import Variable
from tqdm import tqdm
from distutils.version import LooseVersion
from tensorboardX import SummaryWriter
import sys
sys.path.append(os.path.abspath('.'))

from utils.eval import Eval
from utils.train_helper import get_model

from datasets.gta5_Dataset import GTA5_Dataset
from datasets.cityscapes_Dataset import City_Dataset, City_DataLoader, inv_preprocess, decode_labels
from datasets.gta5_Dataset import GTA5_DataLoader
from datasets.synthia_Dataset import SYNTHIA_Dataset
from datasets.synthia_Dataset import SYNTHIA_DataLoader
from utils.losses import ARCS_loss
from graphs.models.discriminator import FCDiscriminator
from utils.losses import IW_MaxSquareloss

DEBUG = False

datasets_path={
    'cityscapes': {
        'data_root_path': '/Cityscapes',
        'list_path': '/Cityscapes',
    },

    'gta5': {
        'data_root_path': '/GTA5',
        'list_path': '/GTA5',
    },

    'synthia': {
        'data_root_path': '/Synthia',
        'list_path': '/SYNTHIA',
    }
    }


datasets_prefix = '../../../../datasets'
for dataset in datasets_path.keys():
    for path in datasets_path[dataset].keys():
        if 'list' in path:
            datasets_path[dataset][path] = './datasets' + datasets_path[dataset][path]
        else:
            datasets_path[dataset][path] = datasets_prefix + datasets_path[dataset][path]

def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Unsupported value encountered.')

def memory_check(log_string):
    torch.cuda.synchronize()
    if DEBUG:
        print(log_string)
        print(' peak:', '{:.3f}'.format(torch.cuda.max_memory_allocated() / 1024 ** 3), 'GB')
        print(' current', '{:.3f}'.format(torch.cuda.memory_allocated() / 1024 ** 3), 'GB')


class RCSTrainer:

    def __init__(self, args, cuda=None, train_id="None", logger=None):

        self.args = args
        self.cuda = cuda and torch.cuda.is_available()
        self.device = torch.device('cuda' if self.cuda else 'cpu')
        self.train_id = train_id
        self.logger = logger

        self.current_MIoU = 0
        self.best_MIou = 0
        self.best_source_MIou = 0
        self.current_epoch = 0
        self.current_iter = 0
        self.second_best_MIou = 0

        # set TensorboardX
        self.writer = SummaryWriter(self.args.checkpoint_dir)

        # Metric definition
        self.Eval = Eval(self.args.num_classes)

        # loss definition
        self.loss = nn.CrossEntropyLoss(weight=None, ignore_index=-1)
        self.loss.to(self.device)

        self.bce_loss = torch.nn.MSELoss()
        self.bce_loss.to(self.device)

        # model
        self.model, params = get_model(self.args)
        self.model = nn.DataParallel(self.model)
        self.model.to(self.device)

        # support for FCN8s
        if self.args.backbone == "fcn8s_vgg" and self.args.optim == "SGD":
            self.args.optim = "Adam"
            print('WARNING: FCN8s requires Adam optimizer, but SGD was set. Switching to Adam.')

        if self.args.optim == "SGD":
            self.optimizer = torch.optim.SGD(
                params=params,
                momentum=self.args.momentum,
                weight_decay=self.args.weight_decay
            )
        elif self.args.optim == "Adam":
            self.optimizer = torch.optim.Adam(params, betas=(0.9, 0.99), weight_decay=self.args.weight_decay)
        # dataloader
        if DEBUG: print(
            'DEBUG: Loading training and validation datasets (for UDA only val one is used, but it is overwritten)')
        if self.args.dataset == "cityscapes":
            self.dataloader = City_DataLoader(self.args)
        elif self.args.dataset == "gta5":
            self.dataloader = GTA5_DataLoader(self.args)
        else:
            self.dataloader = SYNTHIA_DataLoader(self.args)


        ### DATASETS ###
        self.logger.info('Adaptation {} -> {}'.format(self.args.source_dataset, self.args.target_dataset))
        self.logger.info('Dataset path {} and {}'.format(self.args.source_data_path, self.args.data_root_path))

        source_data_kwargs = {'data_root_path':args.source_data_path,
                              'list_path':args.source_list_path,
                              'base_size':args.base_size,
                              'crop_size':args.crop_size}
        target_data_kwargs = {'data_root_path': args.data_root_path,
                              'list_path': args.list_path,
                              'base_size': args.target_base_size,
                              'crop_size': args.target_crop_size}
        dataloader_kwargs = {'batch_size':self.args.batch_size,
                             'num_workers':self.args.data_loader_workers,
                             'pin_memory':self.args.pin_memory,
                             'drop_last':True}

        if self.args.source_dataset == 'synthia':
            source_data_kwargs['class_16'] = target_data_kwargs['class_16'] = args.class_16

        source_data_gen = SYNTHIA_Dataset if self.args.source_dataset == 'synthia' else GTA5_Dataset

        if DEBUG: print('DEBUG: Loading training dataset (source)')
        self.source_dataloader = data.DataLoader(source_data_gen(args, split='train', **source_data_kwargs), shuffle=True, **dataloader_kwargs)

        if DEBUG: print('DEBUG: Loading training dataset (target)')
        self.target_dataloader = data.DataLoader(City_Dataset(args, split='train', **target_data_kwargs), shuffle=True, **dataloader_kwargs)
        if DEBUG: print('DEBUG: Loading validation dataset (target)')
        target_data_set = City_Dataset(args, split='test', **target_data_kwargs)
        self.target_val_dataloader = data.DataLoader(target_data_set, shuffle=False, **dataloader_kwargs)

        self.logger.info('Source train nums: {}, \n Target train nums: {}, val nums: {}'.format(len(self.source_dataloader),len(self.target_dataloader), len(self.target_val_dataloader)))

        self.dataloader.val_loader = self.target_val_dataloader
        self.dataloader.valid_iterations = (len(target_data_set) + self.args.batch_size) // self.args.batch_size

        self.ignore_index = -1
        self.current_round = self.args.init_round
        self.round_num = self.args.round_num
        if self.args.backbone == 'vgg16':
            self.model_D = FCDiscriminator(num_classes=1024).train().to(self.device)
        else:
            self.model_D = FCDiscriminator(num_classes=2048).train().to(self.device)
        self.optimizer_D = torch.optim.Adam(self.model_D.parameters(), lr=self.args.lr_D, betas=(0.9, 0.99))
        self.source_label = 0
        self.target_label = 1

        self.ARCS_loss = ARCS_loss(ignore_index=-1,
                                                 num_class=self.args.num_classes, device=self.device)
        self.ARCS_loss.to(self.device)

        self.use_em_loss = self.args.lambda_entropy != 0.
        if self.use_em_loss:
            self.entropy_loss = IW_MaxSquareloss(ignore_index=-1,
                                                 num_class=self.args.num_classes,
                                                 ratio=self.args.IW_ratio)
            self.entropy_loss.to(self.device)


        self.loss_kwargs = {}
        self.clustering_params = {
            'norm_order': args.norm_order
        }
        self.loss_kwargs['clustering_params'] = self.clustering_params
        self.entropy_params = {
            'temp': args.ent_temp
        }
        self.loss_kwargs['entropy_params'] = self.entropy_params

        self.best_MIou, self.best_iter, self.current_iter, self.current_epoch = None, None, None, None

        self.epoch_num = None

    def main(self):
        # display args details
        self.logger.info("Global configuration as follows:")
        for key, val in vars(self.args).items():
            self.logger.info("{:25} {}".format(key, val))

        # choose cuda
        current_device = torch.cuda.current_device()
        self.logger.info("This model will run on {}".format(torch.cuda.get_device_name(current_device)))

        # load pretrained checkpoint
        if self.args.pretrained_ckpt_file is not None:
            if os.path.isdir(self.args.pretrained_ckpt_file):
                self.args.pretrained_ckpt_file = os.path.join(self.args.checkpoint_dir, self.train_id + 'final.pth')
            self.load_checkpoint(self.args.pretrained_ckpt_file)

        if not self.args.continue_training:
            self.best_MIou, self.best_iter, self.current_iter, self.current_epoch = 0, 0, 0, 0

        if self.args.continue_training:
            self.load_checkpoint_con(os.path.join(self.args.checkpoint_dir, self.train_id + 'final.pth'))

        self.iter_max = self.dataloader.num_iterations * self.args.epoch_each_round * self.round_num
        self.logger.info('Iter max: {} \nNumber of iterations: {}'.format(self.iter_max, self.dataloader.num_iterations))

        # train
        self.train_round()
        self.writer.close()


    def train_round(self):
        for r in range(self.current_round, self.round_num):
            self.logger.info("\n############## Begin {}/{} Round! #################\n".format(self.current_round + 1, self.round_num))
            self.logger.info("epoch_each_round: {}".format(self.args.epoch_each_round))

            self.epoch_num = (self.current_round + 1) * self.args.epoch_each_round

            self.train()

            self.current_round += 1

    def train(self):
        for epoch in tqdm(range(self.current_epoch, self.epoch_num),
                          desc="Total {} epochs".format(self.epoch_num)):

            if self.args.epoch_stop is not None and self.current_epoch >= self.args.epoch_stop:
                self.logger.info("iteration arrive {}(early stop)/{}(total step)!".format(self.args.epoch_stop, self.round_num))
                break
            self.train_one_epoch()

            self.current_epoch += 1

            # validate
            PA, MPA, MIoU, FWIoU = self.validate()
            self.writer.add_scalar('PA', PA, self.current_epoch)
            self.writer.add_scalar('MPA', MPA, self.current_epoch)
            self.writer.add_scalar('MIoU', MIoU, self.current_epoch)
            self.writer.add_scalar('FWIoU', FWIoU, self.current_epoch)

            self.current_MIoU = MIoU
            is_best = MIoU > self.best_MIou
            if is_best:
                self.best_MIou = MIoU
                self.best_iter = self.current_iter
                self.logger.info("=>saving a new best checkpoint...")
                self.save_checkpoint(self.train_id+'best.pth')
            else:
                self.logger.info("=> The MIoU of val does't improve.")
                self.logger.info("=> The best MIoU of val is {} at {}".format(self.best_MIou, self.best_iter))
        self.logger.info("=>best_MIou {} at {}".format(self.best_MIou, self.best_iter))
        self.logger.info("=>saving the final checkpoint to " + os.path.join(self.args.checkpoint_dir, self.train_id+'final.pth'))
        self.save_checkpoint(self.train_id+'final.pth')

    def train_one_epoch(self):
        tqdm_epoch = tqdm(zip(self.source_dataloader, self.target_dataloader), total=self.dataloader.num_iterations,
                          desc="Train Round-{}-Epoch-{}-total-{}".format(self.current_round, self.current_epoch + 1, self.epoch_num))

        self.logger.info("Training one epoch...")
        self.Eval.reset()


        # Set the model to be in training mode (for batchnorm and dropout)
        if self.args.freeze_bn:  # default False
            self.model.eval()
            self.logger.info("freeze bacth normalization successfully!")
        else:
            self.model.train()

        ### Logging setup ###
        log_list, log_strings = [None], ['Source_loss_ce']
        if self.use_em_loss:
            log_strings += ['EM_loss']
            log_list += [None]
        log_strings += ['Generate_loss']
        log_list += [None]
        log_strings += [ 'f_dist_source', 'f_dist_target']
        log_list += [None] * 2
        log_strings += ['sep_dis']
        log_list += [None]
        log_strings += ['separation_ent_loss']
        log_list += [None]
        log_strings += ['Discriminator_loss_source','Discriminator_loss_target']
        log_list += [None] * 2
        log_string = 'epoch{}-batch-{}:' + '={:3f}-'.join(log_strings) + '={:3f}'

        batch_idx = 0
        for batch_s, batch_t in tqdm_epoch:

            self.poly_lr_scheduler(optimizer=self.optimizer, init_lr=self.args.lr)
            self.writer.add_scalar('learning_rate', self.optimizer.param_groups[0]["lr"], self.current_iter)
            self.poly_lr_scheduler_D(optimizer_D=self.optimizer_D, init_lr_D=self.args.lr_D)
            self.writer.add_scalar('learning_rate_D', self.optimizer_D.param_groups[0]["lr"], self.current_iter)

            if self.current_iter < 1: memory_check('Start (step)')

            #######################
            # Source forward step #
            #######################
            for param in self.model_D.parameters():
                param.requires_grad = False

            x, y, _ = batch_s
            if self.cuda:
                x, y = Variable(x).to(self.device), Variable(y).to(device=self.device, dtype=torch.long)

            if self.current_iter < 1: memory_check('Dataloader Source')

            ########################
            # model output ->  list of:  1) pred; 2) feat from encoder's output
            pred_and_feat = self.model(x)
            pred_source, feat_source = pred_and_feat
            torch.cuda.empty_cache()
            ########################

            if self.current_iter < 1: memory_check('Model Source')

            ##################################
            # Source supervised optimization #
            ##################################
            y = torch.squeeze(y, 1)

            _, pred_source_index = torch.max(F.softmax(pred_source.detach(), dim=1),dim=1)

            ### mask source
            num_true = (torch.eq(pred_source_index,y))#.to(self.device)
            y_confidence = (((torch.ones_like(num_true))*num_true).float())#.to(self.device)
            y_confidence = F.interpolate(y_confidence.unsqueeze(0), size=feat_source.size()[2:], mode='nearest')
            mask_source = (y_confidence > 0.5).to(self.device)

            loss = self.loss(pred_source, y)
            loss_ = loss
            loss_.backward(retain_graph=True)

            # log
            log_ind = 0
            log_list[log_ind] = loss.item()
            log_ind += 1

            if self.current_iter < 1: memory_check('End Source')

            torch.cuda.empty_cache()
            #######################
            # Target forward step #
            #######################
            x, _, _ = batch_t
            if self.cuda:
                x = Variable(x).to(self.device)
            if self.current_iter < 1: memory_check('Dataloader Target')

            ########################
            # model output ->  list of:  1) pred; 2) feat from encoder's output
            pred_and_feat = self.model(x)  # creates the graph
            pred_target, feat_target = pred_and_feat
            torch.cuda.empty_cache()
            ########################

            if self.current_iter < 1: memory_check('Model Target')

            if self.use_em_loss:
                em_loss = self.args.lambda_entropy * self.entropy_loss(pred_target, F.softmax(pred_target, dim=1))
                em_loss.backward(retain_graph=True)
                if self.current_iter < 1: memory_check('Entropy Loss')
                # log
                log_list[log_ind] = em_loss.item()
                log_ind += 1

            ## Refining
            D_out, D_confidence = self.model_D(feat_target)
            loss_adv = self.bce_loss(D_out, torch.FloatTensor(D_out.data.size()).fill_(self.source_label).to(self.device))
            loss_adv = loss_adv * 0.01
            loss_adv.backward(retain_graph=True)
            mask_target = F.interpolate(D_confidence.detach(), size=feat_target.size()[2:], mode='nearest')
            log_list[log_ind] = loss_adv.item()
            log_ind += 1
            torch.cuda.empty_cache()



            self.loss_kwargs['source_prob'] = F.softmax(pred_source, dim=1)#.detach()
            self.loss_kwargs['target_prob'] = F.softmax(pred_target, dim=1)#.detach()
            self.loss_kwargs['source_feat'] = feat_source
            self.loss_kwargs['target_feat'] = feat_target
            self.loss_kwargs['source_mask'] = mask_source
            self.loss_kwargs['target_mask'] = mask_target
            self.loss_kwargs['smo_coeff'] = args.centroid_smoothing

            if self.args.use_source_gt: self.loss_kwargs['source_gt'] = y

            losses_RCS = self.ARCS_loss(**self.loss_kwargs)

            ### ARCS loss
            sep_dis, f_dist_source, f_dist_target = losses_RCS['sep_dis'], losses_RCS['f_dist_source'], losses_RCS['f_dist_target']

            ## aggregation
            sum_num = (torch.sum(mask_source) + torch.sum(1-mask_target)).item()
            target_rate = torch.sum(1-mask_target).item()/sum_num
            source_rate = torch.sum(mask_source).item()/sum_num
            aggregation_loss= self.args.lambda_agg_loss * (f_dist_target*source_rate + f_dist_source*target_rate)
            aggregation_loss.backward(retain_graph=True)
            if self.current_iter < 1: memory_check('Clustering Loss')
            log_list[log_ind:log_ind + 2] = [f_dist_source.item(), f_dist_target.item()]
            log_ind += 2

            ## separation
            # dis
            separation_dis_loss = self.args.lambda_sep_dis_loss* (-1 * sep_dis)
            separation_dis_loss.backward(retain_graph=True)
            if self.current_iter < 1: memory_check('Separation Loss')
            log_list[log_ind] = sep_dis.item()
            log_ind += 1

            # sim
            separation_ent_loss = self.args.lambda_sep_ent_loss * losses_RCS['ent_loss']
            separation_ent_loss.backward()
            if self.current_iter < 1: memory_check('Orthogonality Loss')
            log_list[log_ind] = losses_RCS['ent_loss'].item()
            log_ind += 1

            self.optimizer.step()

            del x, y, pred_source, pred_target, D_out, D_confidence, mask_source, mask_target
            del aggregation_loss, separation_dis_loss, separation_ent_loss
            del loss_, loss_adv
            torch.cuda.empty_cache()

            for param in self.model_D.parameters():
                param.requires_grad = True
            D_out_source, _ = self.model_D(feat_source.detach())
            loss_D_source = self.bce_loss(D_out_source, torch.FloatTensor(D_out_source.data.size()).fill_(self.source_label).to(self.device))
            loss_D_source = loss_D_source *0.005
            loss_D_source.backward()
            log_list[log_ind] = loss_D_source.item()
            log_ind += 1

            D_out_target, _ = self.model_D(feat_target.detach())
            loss_D_target = self.bce_loss(D_out_target, torch.FloatTensor(D_out_target.data.size()).fill_(self.target_label).to(self.device))
            loss_D_target = loss_D_target *0.005
            loss_D_target.backward()
            log_list[log_ind] = loss_D_target.item()
            log_ind += 1

            self.optimizer_D.step()
            self.optimizer_D.zero_grad()
            self.optimizer.zero_grad()

            torch.cuda.empty_cache()

            if batch_idx % self.args.logging_interval == 0:
                self.logger.info(log_string.format(self.current_epoch, batch_idx, *log_list))
            if batch_idx % self.args.writer_interval == 0:
                for name, elem in zip(log_strings, log_list):
                    self.writer.add_scalar(name, elem, self.current_iter)

            batch_idx += 1

            self.current_iter += 1

            torch.cuda.empty_cache()
            del feat_source, feat_target
            del loss_D_source, loss_D_target

            if self.current_iter < 1: memory_check('End (step)')

        tqdm_epoch.close()

        if self.args.save_inter_model:
            self.logger.info("Saving model of epoch {} ...".format(self.current_epoch))
            self.save_checkpoint(self.train_id + '_epoch{}.pth'.format(self.current_epoch))

    def log_one_train_epoch(self, x, label, argpred, train_loss):
        images_inv = inv_preprocess(x.clone().cpu(), self.args.show_num_images, numpy_transform=self.args.numpy_transform)
        labels_colors = decode_labels(label, self.args.num_classes, self.args.show_num_images)
        preds_colors = decode_labels(argpred, self.args.num_classes, self.args.show_num_images)
        for index, (img, lab, color_pred) in enumerate(zip(images_inv, labels_colors, preds_colors)):
            self.writer.add_image('train/'+ str(index)+'/Images', img, self.current_epoch)
            self.writer.add_image('train/'+ str(index)+'/Labels', lab, self.current_epoch)
            self.writer.add_image('train/'+ str(index)+'/preds', color_pred, self.current_epoch)

        if self.args.class_16:
            PA = self.Eval.Pixel_Accuracy()
            MPA_16, MPA = self.Eval.Mean_Pixel_Accuracy()
            MIoU_16, MIoU = self.Eval.Mean_Intersection_over_Union()
            FWIoU_16, FWIoU = self.Eval.Frequency_Weighted_Intersection_over_Union()
        else:
            PA = self.Eval.Pixel_Accuracy()
            MPA = self.Eval.Mean_Pixel_Accuracy()
            MIoU = self.Eval.Mean_Intersection_over_Union()
            FWIoU = self.Eval.Frequency_Weighted_Intersection_over_Union()

        self.logger.info('\nEpoch:{}, train PA1:{}, MPA1:{}, MIoU1:{}, FWIoU1:{}'.format(self.current_epoch, PA, MPA,
                                                                                       MIoU, FWIoU))
        self.writer.add_scalar('train_PA', PA, self.current_epoch)
        self.writer.add_scalar('train_MPA', MPA, self.current_epoch)
        self.writer.add_scalar('train_MIoU', MIoU, self.current_epoch)
        self.writer.add_scalar('train_FWIoU', FWIoU, self.current_epoch)

        tr_loss = sum(train_loss)/len(train_loss) if isinstance(train_loss, list) else train_loss
        self.writer.add_scalar('train_loss', tr_loss, self.current_epoch)
        tqdm.write("The average loss of train epoch-{}-:{}".format(self.current_epoch, tr_loss))

    def validate(self, mode='val'):
        self.logger.info('\nvalidating one epoch...')
        self.Eval.reset()
        with torch.no_grad():
            tqdm_batch = tqdm(self.dataloader.val_loader, total=self.dataloader.valid_iterations,
                              desc="Val Epoch-{}-".format(self.current_epoch + 1))
            if mode == 'val':
                self.model.eval()

            i = 0

            for x, y, id in tqdm_batch:
                if self.cuda:
                    x, y = x.to(self.device), y.to(device=self.device, dtype=torch.long)

                # model
                pred = self.model(x)[0]
                y = torch.squeeze(y, 1)

                pred = pred.data.cpu().numpy()
                label = y.cpu().numpy()
                argpred = np.argmax(pred, axis=1)

                self.Eval.add_batch(label, argpred)

            #show val result on tensorboard
            images_inv = inv_preprocess(x.clone().cpu(), self.args.show_num_images, numpy_transform=self.args.numpy_transform)
            labels_colors = decode_labels(label, self.args.num_classes, self.args.show_num_images)
            preds_colors = decode_labels(argpred, self.args.num_classes, self.args.show_num_images)
            for index, (img, lab, color_pred) in enumerate(zip(images_inv, labels_colors, preds_colors)):
                self.writer.add_image(str(index)+'/Images', img, self.current_epoch)
                self.writer.add_image(str(index)+'/Labels', lab, self.current_epoch)
                self.writer.add_image(str(index)+'/preds', color_pred, self.current_epoch)

            if self.args.class_16:
                def val_info(Eval, name):
                    PA = Eval.Pixel_Accuracy()
                    MPA_16, MPA_13 = Eval.Mean_Pixel_Accuracy()
                    MIoU_16, MIoU_13 = Eval.Mean_Intersection_over_Union()
                    FWIoU_16, FWIoU_13 = Eval.Frequency_Weighted_Intersection_over_Union()
                    PC_16, PC_13 = Eval.Mean_Precision()
                    print("########## Eval{} ############".format(name))

                    self.logger.info('\nEpoch:{:.3f}, {} PA:{:.3f}, MPA_16:{:.3f}, MIoU_16:{:.3f}, FWIoU_16:{:.3f}, PC_16:{:.3f}'.format(self.current_epoch, name, PA, MPA_16,
                                                                                                MIoU_16, FWIoU_16, PC_16))
                    self.logger.info('\nEpoch:{:.3f}, {} PA:{:.3f}, MPA_13:{:.3f}, MIoU_13:{:.3f}, FWIoU_13:{:.3f}, PC_13:{:.3f}'.format(self.current_epoch, name, PA, MPA_13,
                                                                                                MIoU_13, FWIoU_13, PC_13))
                    self.writer.add_scalar('PA'+name, PA, self.current_epoch)
                    self.writer.add_scalar('MPA_16'+name, MPA_16, self.current_epoch)
                    self.writer.add_scalar('MIoU_16'+name, MIoU_16, self.current_epoch)
                    self.writer.add_scalar('FWIoU_16'+name, FWIoU_16, self.current_epoch)
                    self.writer.add_scalar('MPA_13'+name, MPA_13, self.current_epoch)
                    self.writer.add_scalar('MIoU_13'+name, MIoU_13, self.current_epoch)
                    self.writer.add_scalar('FWIoU_13'+name, FWIoU_13, self.current_epoch)
                    return PA, MPA_13, MIoU_13, FWIoU_13
            else:
                def val_info(Eval, name):
                    PA = Eval.Pixel_Accuracy()
                    MPA = Eval.Mean_Pixel_Accuracy()
                    MIoU = Eval.Mean_Intersection_over_Union()
                    FWIoU = Eval.Frequency_Weighted_Intersection_over_Union()
                    PC = Eval.Mean_Precision()
                    print("########## Eval{} ############".format(name))

                    self.logger.info('\nEpoch:{:.3f}, {} PA1:{:.3f}, MPA1:{:.3f}, MIoU1:{:.3f}, FWIoU1:{:.3f}, PC:{:.3f}'.format(self.current_epoch, name, PA, MPA,
                                                                                                MIoU, FWIoU, PC))
                    self.writer.add_scalar('PA'+name, PA, self.current_epoch)
                    self.writer.add_scalar('MPA'+name, MPA, self.current_epoch)
                    self.writer.add_scalar('MIoU'+name, MIoU, self.current_epoch)
                    self.writer.add_scalar('FWIoU'+name, FWIoU, self.current_epoch)
                    return PA, MPA, MIoU, FWIoU

            PA, MPA, MIoU, FWIoU = val_info(self.Eval, "")
            tqdm_batch.close()

        return PA, MPA, MIoU, FWIoU



    def save_checkpoint(self, filename=None):

        filename = os.path.join(self.args.checkpoint_dir, filename)
        state = {
            'epoch': self.current_epoch + 1,
            'iteration': self.current_iter,
            'state_dict': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'state_dict_D': self.model_D.state_dict(),
            'optimizer_D': self.optimizer_D.state_dict(),
            'best_MIou':self.best_MIou,
            'best_iter':self.best_iter
        }
        torch.save(state, filename)


    def load_checkpoint(self, filename):

        try:
            self.logger.info("Loading checkpoint '{}'".format(filename))
            checkpoint = torch.load(filename)
            if 'state_dict' in checkpoint:
                self.model.load_state_dict(checkpoint['state_dict'])
            else:
                self.model.module.load_state_dict(checkpoint)
            self.logger.info("Owns checkpoint loaded successfully from " + filename)
        except OSError as e:
            self.logger.info("Maybe no checkpoint exists from '{}'. Skipping...".format(self.args.checkpoint_dir))
            self.logger.info("No checkpoint exists from '{}'. Skipping...".format(filename))
            self.logger.info("**First time to train**")

    def load_checkpoint_con(self, filename):
        checkpoint = torch.load(filename)
        self.model.load_state_dict(checkpoint['state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        self.current_iter = checkpoint['iteration']
        self.current_epoch = checkpoint['epoch']-1
        self.best_MIou = checkpoint['best_MIou']
        self.best_iter = None
        self.logger.info('Continue Training')


    def poly_lr_scheduler(self, optimizer, init_lr=None, iter=None,
                            max_iter=None, power=None):
        init_lr = self.args.lr if init_lr is None else init_lr
        iter = self.current_iter if iter is None else iter
        max_iter = self.iter_max if max_iter is None else max_iter
        power = self.args.poly_power if power is None else power
        new_lr = init_lr * (1 - float(iter) / max_iter) ** power
        optimizer.param_groups[0]["lr"] = new_lr
        if len(optimizer.param_groups) == 2:
            optimizer.param_groups[1]["lr"] = 10 * new_lr

    def poly_lr_scheduler_D(self, optimizer_D, init_lr_D=None, iter=None,
                            max_iter=None, power_D=None):
        init_lr = self.args.lr_D if init_lr_D is None else init_lr_D
        iter = self.current_iter if iter is None else iter
        max_iter = self.iter_max if max_iter is None else max_iter
        power = self.args.poly_power_D if power_D is None else power_D

        lr = init_lr * ((1 - float(iter) / max_iter) ** (power))
        optimizer_D.param_groups[0]['lr'] = lr
        if len(optimizer_D.param_groups) > 1:
            optimizer_D.param_groups[1]['lr'] = lr * 10



def add_UDA_train_args(arg_parser):
    # Path related arguments
    arg_parser.add_argument('--data_root_path', type=str, default=None, help="the path to dataset")
    arg_parser.add_argument('--list_path', type=str, default=None, help="the path to data split lists")
    arg_parser.add_argument('-cdir', '--checkpoint_dir', default="./log/train", help="the path to ckpt file")

    # Model related arguments
    arg_parser.add_argument('--backbone', default='resnet101', choices=['resnet101', 'vgg16'], help="backbone encoder")
    arg_parser.add_argument('--bn_momentum', type=float, default=0.1, help="batch normalization momentum")
    arg_parser.add_argument('--imagenet_pretrained', type=str2bool, default=True, help="whether apply imagenet pretrained weights")
    arg_parser.add_argument('--pretrained_ckpt_file', type=str, default=None,
                            help="whether to apply pretrained checkpoint")
    arg_parser.add_argument('--continue_training', type=str2bool, default=False, help="whether to continue training ")
    arg_parser.add_argument('--show_num_images', type=int, default=2, help="show how many images during validate")

    # train related arguments
    arg_parser.add_argument('--seed', default=12345, type=int, help='random seed')
    arg_parser.add_argument('--gpu', type=str, default="0", help=" the num of gpu")
    arg_parser.add_argument('--batch_size_per_gpu', default=1, type=int, help='input batch size')

    # dataset related arguments
    arg_parser.add_argument('--dataset', default='cityscapes', type=str, help='dataset choice')
    arg_parser.add_argument('--source_dataset', default='gta5', type=str, choices=['gta5', 'synthia'], help='source dataset choice')
    arg_parser.add_argument('--source_split', default='train', type=str, help='source datasets split')
    arg_parser.add_argument('--base_size', default="1280,720", type=str, help='crop size of image')
    arg_parser.add_argument('--crop_size', default="1280,720", type=str, help='base size of image')
    arg_parser.add_argument('--target_base_size', default="1280,640", type=str, help='crop size of target image')
    arg_parser.add_argument('--target_crop_size', default="1280,640", type=str, help='base size of target image')
    arg_parser.add_argument('--num_classes', default=19, type=int, help='num class of mask')
    arg_parser.add_argument('--data_loader_workers', default=0, type=int, help='num_workers of Dataloader')
    arg_parser.add_argument('--pin_memory', default=False, type=int, help='pin_memory of Dataloader')
    arg_parser.add_argument('--split', type=str, default='train', help="choose from train/val/test/trainval/all")
    arg_parser.add_argument('--random_mirror', default=True, type=str2bool, help='add random_mirror')
    arg_parser.add_argument('--random_crop', default=False, type=str2bool, help='add random_crop')
    arg_parser.add_argument('--resize', default=True, type=str2bool, help='resize')
    arg_parser.add_argument('--gaussian_blur', default=True, type=str2bool, help='add gaussian_blur')
    arg_parser.add_argument('--numpy_transform', default=True, type=str2bool, help='image transform with numpy style')

    # optimization related arguments
    arg_parser.add_argument('--freeze_bn', type=str2bool, default=False, help="whether freeze BatchNormalization")
    arg_parser.add_argument('--optim', default="SGD", type=str, help='optimizer')
    arg_parser.add_argument('--momentum', type=float, default=0.9)
    arg_parser.add_argument('--weight_decay', type=float, default=5e-4)
    arg_parser.add_argument('--lr', type=float, default=2.5e-4, help="init learning rate ")
    arg_parser.add_argument('--poly_power', type=float, default=0.9, help="poly_power")  
    arg_parser.add_argument('--init_round', type=int, default=0, help='init_round')
    arg_parser.add_argument('--round_num', type=int, default=10, help="num round")
    arg_parser.add_argument('--epoch_each_round', type=int, default=2, help="epoch each round")
    arg_parser.add_argument('--epoch_stop', type=int, default=14, help="num round")
    arg_parser.add_argument('--logging_interval', type=int, default=1000, help="interval in steps for logging")
    arg_parser.add_argument('--writer_interval', type=int, default=30, help="interval in steps for writer")
    arg_parser.add_argument('--save_inter_model', type=str2bool, default=False, help="save model at the end of each epoch or not")


    # clustering
    arg_parser.add_argument('--use_source_gt', default=False, type=str2bool, help='use source label or segmented image for pixel/feature classification')
    arg_parser.add_argument('--centroid_smoothing', default=-1, type=float, help="centroid smoothing coefficient, negative to disable")
    arg_parser.add_argument('--lambda_agg_loss', default=0.06, type=float, help="lambda of clustering loss")
    arg_parser.add_argument('--lambda_sep_dis_loss', default=0.02, type=float, help="lambda of clustering loss")
    arg_parser.add_argument('--norm_order', default=1, type=int, help="norm order of feature clustering loss")
    arg_parser.add_argument('--lambda_sep_ent_loss', default=2.0, type=float, help="lambda of orthogonality _confidence loss") #2.
    arg_parser.add_argument('--ent_temp', default=2., type=float, help="temperature for similarity based-distribution")

    # off-the-shelf entropy loss
    arg_parser.add_argument('--lambda_entropy', type=float, default=0., help="lambda of target loss")
    arg_parser.add_argument('--IW_ratio', type=float, default=0.2, help='the ratio of image-wise weighting factor')
    arg_parser.add_argument('--lr_D', type=float, default=1e-4, help="init learning rate ")
    arg_parser.add_argument('--poly_power_D', type=float, default=0.9, help="poly_power")

    return arg_parser



def init_UDA_args(args):

    args.batch_size = args.batch_size_per_gpu * ceil(len(args.gpu) / 2)

    train_id = str(args.dataset)

    crop_size = args.crop_size.split(',')
    base_size = args.base_size.split(',')
    if len(crop_size) == 1:
        args.crop_size = int(crop_size[0])
        args.base_size = int(base_size[0])
    else:
        args.crop_size = (int(crop_size[0]), int(crop_size[1]))
        args.base_size = (int(base_size[0]), int(base_size[1]))

    target_crop_size = args.target_crop_size.split(',')
    target_base_size = args.target_base_size.split(',')
    if len(target_crop_size) == 1:
        args.target_crop_size = int(target_crop_size[0])
        args.target_base_size = int(target_base_size[0])
    else:
        args.target_crop_size = (int(target_crop_size[0]), int(target_crop_size[1]))
        args.target_base_size = (int(target_base_size[0]), int(target_base_size[1]))

    if not args.continue_training:
        if os.path.exists(args.checkpoint_dir):
            print("checkpoint dir exists, which will be removed")
            import shutil
            shutil.rmtree(args.checkpoint_dir, ignore_errors=True)
        # print(os.getcwd())
        try:
            os.mkdir(args.checkpoint_dir)
        except FileNotFoundError:
            print('Missing parent folder in path:  {}'.format(args.checkpoint_dir))
            exit()

    if args.data_root_path is None:
        args.data_root_path = datasets_path[args.dataset]['data_root_path']
        args.list_path = datasets_path[args.dataset]['list_path']

    args.class_16 = True if args.num_classes == 16 else False
    args.class_13 = True if args.num_classes == 13 else False

    # logger configure
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    fh = logging.FileHandler(os.path.join(args.checkpoint_dir, 'train_log.txt'))
    ch = logging.StreamHandler()
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    fh.setFormatter(formatter)
    ch.setFormatter(formatter)
    logger.addHandler(fh)
    logger.addHandler(ch)

    # set seed
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.random.manual_seed(args.seed)
    torch.backends.cudnn.benchmark = True

    def str2none(l):
        l = [l] if not isinstance(l,list) else l
        for i,el in enumerate(l):
            if el == 'None':
                l[i]=None
        return l if len(l)>1 else l[0]

    def str2float(l):
        for i,el in enumerate(l):
            try: l[i] = float(el)
            except (ValueError,TypeError): l[i] = el
        return l


    return args, train_id, logger



if __name__ == '__main__':
    assert LooseVersion(torch.__version__) >= LooseVersion('1.0.0'), 'PyTorch>=1.0.0 is required'

    file_os_dir = os.path.dirname(os.path.abspath(__file__))

    os.chdir(file_os_dir)
    # os.chdir('..')

    arg_parser = argparse.ArgumentParser()
    arg_parser = add_UDA_train_args(arg_parser)

    args = arg_parser.parse_args()
    args, train_id, logger = init_UDA_args(args)
    # args = init_UDA_args(args)
    args.source_data_path = datasets_path[args.source_dataset]['data_root_path']
    args.source_list_path = datasets_path[args.source_dataset]['list_path']

    args.target_dataset = args.dataset


    train_id = str(args.source_dataset)+"2"+str(args.target_dataset)

    assert (args.source_dataset == 'synthia' and args.num_classes == 16) or (args.source_dataset == 'gta5' and args.num_classes == 19), 'dataset:{0:} - classes:{1:}'.format(args.source_dataset, args.num_classes)

    agent = RCSTrainer(args=args, cuda=True, train_id=train_id, logger=logger)
    agent.main()
