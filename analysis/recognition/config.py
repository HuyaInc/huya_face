import numpy as np
import os
from easydict import EasyDict as edict

hdfs_root = '~/huya_face'
config = edict()

config.bn_mom = 0.9
config.workspace = 256
config.emb_size = 512
config.ckpt_embedding = True
config.save_fc = True
config.net_se = 0
config.net_act = 'relu'
config.net_unit = 3
config.net_input = 1
config.net_blocks = [1,4,6,2]
config.net_output = 'E'
config.net_multiplier = 1.0
config.valsets_path = ''
config.val_targets = ['lfw', 'sllfw', 'calfw', 'cplfw', 'cfp_ff', 'cfp_fp', 'agedb_30', 'vgg2_fp', 'rfw']
config.ce_loss = True
config.fc7_lr_mult = 1.0
config.fc7_wd_mult = 0.1
config.fc7_no_bias = False
config.max_steps = 0
config.data_rand_mirror = True
config.data_cutoff = False
config.data_color = 0
config.data_images_filter = 0
config.count_flops = True
config.memonger = True # work for resnet
config.float16 = False
config.name_prefix = ''


# network settings
network = edict()

network.r100 = edict()
network.r100.net_name = 'fresnet'
network.r100.num_layers = 100

network.r124 = edict()
network.r124.net_name = 'fresnet'
network.r124.num_layers = 124


network.r100fc = edict()
network.r100fc.net_name = 'fresnet'
network.r100fc.num_layers = 100
network.r100fc.net_output = 'FC'

network.r18 = edict()
network.r18.net_name = 'fresnet'
network.r18.num_layers = 18

network.r50 = edict()
network.r50.net_name = 'fresnet'
network.r50.num_layers = 50

network.r50v1 = edict()
network.r50v1.net_name = 'fresnet'
network.r50v1.num_layers = 50
network.r50v1.net_unit = 1

# dataset settings
dataset = edict()
dataset.emore = edict()
dataset.emore.dataset = 'emore'
dataset.emore.dataset_path = os.path.join(hdfs_root, 'face_datasets/faces_emore')
dataset.emore.num_classes = 86000
dataset.emore.image_shape = (112,112,3)
dataset.emore.val_targets = ['lfw', 'calfw', 'cplfw', 'cfp_ff', 'cfp_fp', 'vgg2_fp', 'agedb_30']

dataset.gan_emore = edict()
dataset.gan_emore.dataset = 'gan_emore'
dataset.gan_emore.dataset_path = os.path.join(hdfs_root, 'face_datasets/faces_emore')
dataset.gan_emore.num_classes = 86000
dataset.gan_emore.image_shape = (112,112,3)
dataset.gan_emore.val_targets = ['lfw', 'calfw', 'cplfw', 'cfp_ff', 'cfp_fp', 'vgg2_fp', 'agedb_30']

dataset.gan_cifar100 = edict()
dataset.gan_cifar100.dataset = 'gan_cifar100'
dataset.gan_cifar100.dataset_path = os.path.join(hdfs_root, 'face_datasets/cifar100_debug')
dataset.gan_cifar100.num_classes = 100
dataset.gan_cifar100.image_shape = (32,32,3)
dataset.gan_cifar100.val_targets = ''


loss = edict()
loss.softmax = edict()
loss.softmax.loss_name = 'softmax'
loss.softmax.loss_s = 5

loss.phi_softmax = edict()
loss.phi_softmax.loss_name = 'phi_softmax'
loss.phi_softmax.loss_s = 32

loss.feat_mom_phi = edict()
loss.feat_mom_phi.loss_name = 'feat_mom_phi'
loss.feat_mom_phi.loss_s = 16.0

loss.theta_phi = edict()
loss.theta_phi.loss_name = 'theta_phi'
loss.theta_phi.loss_s = 64.0
loss.theta_phi.slope = 0.88
loss.theta_phi.margin = 3.0

loss.learn_scale_phi = edict()
loss.learn_scale_phi.loss_name = 'learn_scale_phi'
loss.learn_scale_phi.loss_s = 32


loss.nsoftmax = edict()
loss.nsoftmax.loss_name = 'margin_softmax'
loss.nsoftmax.loss_s = 64.0
loss.nsoftmax.loss_m1 = 1.0
loss.nsoftmax.loss_m2 = 0.0
loss.nsoftmax.loss_m3 = 0.0

loss.arcface = edict()
loss.arcface.loss_name = 'margin_softmax'
loss.arcface.loss_s = 1.0
loss.arcface.loss_m1 = 2.0
loss.arcface.loss_m2 = 0.1
loss.arcface.loss_m3 = 0.0

loss.cosface = edict()
loss.cosface.loss_name = 'margin_softmax'
loss.cosface.loss_s = 64.0
loss.cosface.loss_m1 = 1.0
loss.cosface.loss_m2 = 0.0
loss.cosface.loss_m3 = 0.35

loss.combined = edict()
loss.combined.loss_name = 'margin_softmax'
loss.combined.loss_s = 64.0
loss.combined.loss_m1 = 1.0
loss.combined.loss_m2 = 0.3
loss.combined.loss_m3 = 0.2

loss.triplet = edict()
loss.triplet.loss_name = 'triplet'
loss.triplet.images_per_identity = 5
loss.triplet.triplet_alpha = 0.3
loss.triplet.triplet_bag_size = 7200
loss.triplet.triplet_max_ap = 0.0
loss.triplet.per_batch_size = 60
loss.triplet.lr = 0.05

loss.atriplet = edict()
loss.atriplet.loss_name = 'atriplet'
loss.atriplet.images_per_identity = 5
loss.atriplet.triplet_alpha = 0.35
loss.atriplet.triplet_bag_size = 7200
loss.atriplet.triplet_max_ap = 0.0
loss.atriplet.per_batch_size = 60
loss.atriplet.lr = 0.05

# default settings
default = edict()

# default network
default.network = 'r100'
default.pretrained = ''
default.pretrained_epoch = 1
# default dataset
default.dataset = 'emore'
default.loss = 'feat_mom_phi'
default.frequent = 20
default.verbose = 1000
default.kvstore = 'device'

default.end_epoch = 10000
default.lr = 0.1
default.lr_pretrained = 0.01
default.wd = 0.0005
default.mom = 0.9
default.per_batch_size = 128
default.ckpt = 3
default.lr_steps = '20000,40000,60000'
default.lr_steps_pretrained = '30000,60000,80000'
default.model_teacher = ''
default.models_root = os.path.join(hdfs_root, '/model')


def generate_config(_network, _dataset, _loss):
    for k, v in loss[_loss].items():
      config[k] = v
      if k in default:
        default[k] = v
    for k, v in network[_network].items():
      config[k] = v
      if k in default:
        default[k] = v
    for k, v in dataset[_dataset].items():
      config[k] = v
      if k in default:
        default[k] = v
    config.loss = _loss
    config.network = _network
    config.dataset = _dataset
    config.num_workers = 1
    if 'DMLC_NUM_WORKER' in os.environ:
      config.num_workers = int(os.environ['DMLC_NUM_WORKER'])

