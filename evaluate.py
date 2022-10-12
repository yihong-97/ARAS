import os
import argparse
import random
import logging
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
import numpy as np
from math import ceil
from distutils.version import LooseVersion
from tensorboardX import SummaryWriter

import sys
sys.path.append(os.path.abspath('.'))
from datasets.cityscapes_Dataset import City_DataLoader, inv_preprocess, decode_labels
from datasets.gta5_Dataset import GTA5_DataLoader
from utils.train_helper import get_model

import cv2
from PIL import Image
from torchvision import transforms
from utils.eval import Eval

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


datasets_prefix = '../../../../datasets/seg'
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


class Evaluater():
    def __init__(self, args, cuda=None, train_id=None, logger=None):
        self.args = args
        self.cuda = cuda and torch.cuda.is_available()
        self.device = torch.device('cuda' if self.cuda else 'cpu')

        self.current_MIoU = 0
        self.best_MIou = 0
        self.current_epoch = 0
        self.current_iter = 0
        self.train_id = train_id
        self.logger = logger

        # set TensorboardX
        self.writer = SummaryWriter(self.args.checkpoint_dir)

        # Metric definition
        self.Eval = Eval(self.args.num_classes)

        # loss definition
        self.loss = nn.CrossEntropyLoss(ignore_index= -1)
        self.loss.to(self.device)

        # model
        self.model, params = get_model(self.args)
        self.model = nn.DataParallel(self.model, device_ids=[0])
        self.model.to(self.device)

        # load pretrained checkpoint
        if self.args.pretrained_ckpt_file is not None:
            path1 = os.path.join(*self.args.checkpoint_dir.split('/')[:-1], self.train_id + 'best.pth')
            path2 = self.args.pretrained_ckpt_file
            if os.path.exists(path1):
                pretrained_ckpt_file = path1
            elif os.path.exists(path2):
                pretrained_ckpt_file = path2
            else:
                raise AssertionError("no pretrained_ckpt_file")
            self.load_checkpoint(pretrained_ckpt_file)

        # dataloader
        self.dataloader = City_DataLoader(self.args) if self.args.dataset=="cityscapes" else GTA5_DataLoader(self.args)
        self.dataloader.val_loader = self.dataloader.data_loader
        self.dataloader.valid_iterations = min(self.dataloader.num_iterations, 500)
        self.epoch_num = ceil(self.args.iter_max / self.dataloader.num_iterations)

    def main(self):
        # choose cuda
        if self.cuda:
            current_device = torch.cuda.current_device()
            self.logger.info("This model will run on {}".format(torch.cuda.get_device_name(current_device)))
        else:
            self.logger.info("This model will run on CPU")

        # validate
        self.validate()

        self.writer.close()

    def validate(self):
        self.logger.info('validating one epoch...')
        self.Eval.reset()
        with torch.no_grad():
            tqdm_batch = tqdm(self.dataloader.val_loader, total=self.dataloader.valid_iterations, desc="Val Epoch-{}-".format(self.current_epoch + 1))
            self.model.eval()
            i = 0
            print(self.args.checkpoint_dir+'imglist.txt')
            fw = open(self.args.checkpoint_dir+'imglist.txt', 'a')

            for x, y, id in tqdm_batch:
                i += 1
                if self.cuda:
                    x, y = x.to(self.device), y.to(device=self.device, dtype=torch.long)

                # model
                pred = self.model(x)[0]
                y = torch.squeeze(y, 1)

                if self.args.flip:
                    pred_P = F.softmax(pred, dim=1)
                    def flip(x, dim):
                        dim = x.dim() + dim if dim < 0 else dim
                        inds = tuple(slice(None, None) if i != dim
                                else x.new(torch.arange(x.size(i)-1, -1, -1).tolist()).long()
                                for i in range(x.dim()))
                        return x[inds]
                    x_flip = flip(x, -1)
                    pred_flip = self.model(x_flip)[0]
                    pred_P_flip = F.softmax(pred_flip, dim=1)
                    pred_P_2 = flip(pred_P_flip, -1)
                    pred_c = (pred_P+pred_P_2)/2
                    pred = pred_c.data.cpu().numpy()
                else:
                    pred = pred.data.cpu().numpy()
                label = y.cpu().numpy()

                argpred = np.argmax(pred, axis=1)


                self.Eval.add_batch(label, argpred)
                fw.write(str(id[0].split('/')[-1][:-20]))
                fw.write('\n')
                fw.flush()

                if self.args.save_outputs:
                    preds_colors = decode_labels(argpred, self.args.num_classes, self.args.show_num_images)
                    save_dir = self.args.checkpoint_dir + '/out_im/'
                    if not os.path.exists(save_dir):
                        os.mkdir(save_dir)
                    save_name = save_dir + id[0].split('/')[-1][:-20] + '_pred.png'
                    output_im = transforms.ToPILImage()(preds_colors[0])
                    output_im.save(save_name)

                if i == self.dataloader.valid_iterations:
                    break
                
                if i % 20 ==0 and self.args.image_summary:
                    #show val result on tensorboard
                    images_inv = inv_preprocess(x.clone().cpu(), self.args.show_num_images, numpy_transform=self.args.numpy_transform)
                    labels_colors = decode_labels(label, self.args.num_classes, self.args.show_num_images)
                    preds_colors = decode_labels(argpred, self.args.num_classes, self.args.show_num_images)
                    for index, (img, lab, color_pred) in enumerate(zip(images_inv, labels_colors, preds_colors)):
                        self.writer.add_image('eval/'+ str(index)+'/Images', img, self.current_epoch)
                        self.writer.add_image('eval/'+ str(index)+'/Labels', lab, self.current_epoch)
                        self.writer.add_image('eval/'+ str(index)+'/preds', color_pred, self.current_epoch)
            #show val result on tensorboard
            if self.args.image_summary:
                images_inv = inv_preprocess(x.clone().cpu(), self.args.show_num_images, numpy_transform=self.args.numpy_transform)
                labels_colors = decode_labels(label, self.args.num_classes, self.args.show_num_images)
                preds_colors = decode_labels(argpred, self.args.num_classes, self.args.show_num_images)
                for index, (img, lab, color_pred) in enumerate(zip(images_inv, labels_colors, preds_colors)):
                    self.writer.add_image('0Images/'+str(index), img, self.current_epoch)
                    self.writer.add_image('a'+str(index)+'/Labels', lab, self.current_epoch)
                    self.writer.add_image('a'+str(index)+'/preds', color_pred, self.current_epoch)

            # get eval result
            if self.args.class_16:
                def val_info(Eval, name):
                    PA = Eval.Pixel_Accuracy()
                    MPA_16, MPA_13 = Eval.Mean_Pixel_Accuracy()
                    MIoU_16, MIoU_13 = Eval.Mean_Intersection_over_Union()
                    FWIoU_16, FWIoU_13 = Eval.Frequency_Weighted_Intersection_over_Union()
                    PC_16, PC_13 = Eval.Mean_Precision()
                    print("########## Eval{} ############".format(name))

                    self.logger.info('\nEpoch:{:.3f}, {} PA:{:.3f}, MPA_16:{:.3f}, MIoU_16:{:.3f}, FWIoU_16:{:.3f}, PC_16:{:.3f}'.format(self.current_epoch, name, PA, MPA_16, MIoU_16, FWIoU_16, PC_16))
                    self.logger.info('\nEpoch:{:.3f}, {} PA:{:.3f}, MPA_13:{:.3f}, MIoU_13:{:.3f}, FWIoU_13:{:.3f}, PC_13:{:.3f}'.format(self.current_epoch, name, PA, MPA_13, MIoU_13, FWIoU_13, PC_13))
                    return PA, MPA_16, MIoU_16, FWIoU_16
            else:
                def val_info(Eval, name):
                    PA = Eval.Pixel_Accuracy()
                    MPA = Eval.Mean_Pixel_Accuracy()
                    MIoU = Eval.Mean_Intersection_over_Union()
                    FWIoU = Eval.Frequency_Weighted_Intersection_over_Union()
                    PC = Eval.Mean_Precision()
                    print("########## Eval{} ############".format(name))

                    self.logger.info('\nEpoch:{:.3f}, {} PA1:{:.3f}, MPA1:{:.3f}, MIoU1:{:.4f}, FWIoU1:{:.3f}, PC:{:.3f}'.format(self.current_epoch, name, PA, MPA, MIoU, FWIoU, PC))
                    return PA, MPA, MIoU, FWIoU

            PA, MPA, MIoU, FWIoU = val_info(self.Eval, "")
                
            self.Eval.Print_Every_class_Eval()
            tqdm_batch.close()
            fw.close()

        return PA, MPA, MIoU, FWIoU


    def load_checkpoint(self, filename):
        try:
            self.logger.info("Loading checkpoint '{}'".format(filename))
            checkpoint = torch.load(filename)

            if 'state_dict' in checkpoint:
                self.model.load_state_dict(checkpoint['state_dict'])
            else:
                self.model.module.load_state_dict(checkpoint)
            self.logger.info("Checkpoint loaded successfully from "+filename)

            if 'crop_size' in checkpoint:
                self.args.crop_size = checkpoint['crop_size']
                print(checkpoint['crop_size'], self.args.crop_size)
        except OSError as e:
            self.logger.info("No checkpoint exists from '{}'. Skipping...".format(filename))


def add_train_args(arg_parser):
    # Path related arguments
    arg_parser.add_argument('--data_root_path', type=str, default=None, help="the path to dataset")
    arg_parser.add_argument('--list_path', type=str, default=None, help="the path to data split lists")
    arg_parser.add_argument('-cdir', '--checkpoint_dir', default="./log/train", help="the path to ckpt file")

    # Model related arguments
    arg_parser.add_argument('--backbone', default='resnet101', choices=['resnet101', 'vgg16'], help="backbone encoder")
    arg_parser.add_argument('--bn_momentum', type=float, default=0.1, help="batch normalization momentum")
    arg_parser.add_argument('--imagenet_pretrained', type=str2bool, default=True,
                            help="whether apply imagenet pretrained weights")
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
    arg_parser.add_argument('--base_size', default="1280,720", type=str, help='crop size of image')
    arg_parser.add_argument('--crop_size', default="640,360", type=str, help='base size of image')
    arg_parser.add_argument('--target_base_size', default="1024,512", type=str, help='crop size of target image')
    arg_parser.add_argument('--target_crop_size', default="512,256", type=str, help='base size of target image')
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
    arg_parser.add_argument('--iter_max', type=int, default=250000, help="the maxinum of iteration")
    arg_parser.add_argument('--iter_stop', type=int, default=None, help="the early stop step")
    arg_parser.add_argument('--poly_power', type=float, default=0.9, help="poly_power")


    return arg_parser


def init_args(args):
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

    return args, train_id, logger

if __name__ == '__main__':
    assert LooseVersion(torch.__version__) >= LooseVersion('1.0.0'), 'PyTorch>=1.0.0 is required'

    file_os_dir = os.path.dirname(os.path.realpath(__file__))
    os.chdir(file_os_dir)

    arg_parser = argparse.ArgumentParser()
    arg_parser = add_train_args(arg_parser)
    arg_parser.add_argument('--source_dataset', default='None', type=str, help='source dataset choice')
    arg_parser.add_argument('--flip', type=str2bool, default=False, help="flip")
    arg_parser.add_argument('--image_summary', type=str2bool, default=False, help="image_summary")
    arg_parser.add_argument('-so', '--save_outputs', type=str2bool, default=True, help="image_summary")

    args = arg_parser.parse_args()
    if args.split == "train": args.split = "val"
    if args.checkpoint_dir == "none": args.checkpoint_dir = args.pretrained_ckpt_file + "/eval"
    args, train_id, logger = init_args(args)
    args.batch_size_per_gpu = 2
    args.crop_size = args.target_crop_size
    args.base_size = args.target_base_size


    assert (args.source_dataset == 'synthia' and args.num_classes == 16) or (args.source_dataset == 'gta5' and args.num_classes == 19), 'dataset:{0:} - classes:{1:}'.format(args.source_dataset, args.num_classes)

    agent = Evaluater(args=args, cuda=True, train_id="train_id", logger=logger)
    agent.main()