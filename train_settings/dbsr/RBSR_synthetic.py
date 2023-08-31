
# Copyright (c) 2021 Huawei Technologies Co., Ltd.
# Licensed under CC BY-NC-SA 4.0 (Attribution-NonCommercial-ShareAlike 4.0 International) (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode
#
# The code is released for academic research use only. For commercial use, please contact Huawei Technologies Co., Ltd.
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import torch.optim as optim
import dataset as datasets
from data import processing, sampler, DataLoader
# import models.dbsr.dbsrnet as dbsr_nets
from models.RBSR_train import RBSR
import actors.dbsr_actors as dbsr_actors
from trainers import SimpleTrainer
import data.transforms as tfm
from admin.multigpu import MultiGPU
from models.loss.image_quality_v2 import PSNR, PixelWiseError
import torch

def run(settings):
    settings.description = 'Default settings for training DBSR models on synthetic burst dataset '
    settings.batch_size = 16
    settings.num_workers = 2
    settings.multi_gpu = True
    settings.print_interval = 1

    settings.crop_sz = (384 , 384)  # 用384x384的patch size
    settings.burst_sz = 14
    settings.downsample_factor = 4

    settings.burst_transformation_params = {'max_translation': 24.0,
                                            'max_rotation': 1.0,
                                            'max_shear': 0.0,
                                            'max_scale': 0.0,
                                            'border_crop': 24}
    settings.burst_reference_aligned = True
    settings.image_processing_params = {'random_ccm': True, 'random_gains': True, 'smoothstep': True, 'gamma': True, 'add_noise': True}

    zurich_raw2rgb_train = datasets.ZurichRAW2RGB(split='train')
    zurich_raw2rgb_val = datasets.ZurichRAW2RGB(split='test')

    transform_train = tfm.Transform(tfm.ToTensorAndJitter(0.0, normalize=True), tfm.RandomHorizontalFlip())
    transform_val = tfm.Transform(tfm.ToTensorAndJitter(0.0, normalize=True), tfm.RandomHorizontalFlip())

    data_processing_train = processing.SyntheticBurstProcessing(settings.crop_sz, settings.burst_sz,
                                                                settings.downsample_factor,
                                                                burst_transformation_params=settings.burst_transformation_params,
                                                                transform=transform_train,
                                                                image_processing_params=settings.image_processing_params)
    data_processing_val = processing.SyntheticBurstProcessing(settings.crop_sz, settings.burst_sz,
                                                              settings.downsample_factor,
                                                              burst_transformation_params=settings.burst_transformation_params,
                                                              transform=transform_val,
                                                              image_processing_params=settings.image_processing_params)

    # Train sampler and loader
    dataset_train = sampler.RandomImage([zurich_raw2rgb_train], [1],
                                        samples_per_epoch=settings.batch_size * 1000, processing=data_processing_train)
    dataset_val = sampler.RandomImage([zurich_raw2rgb_val], [1],
                                      samples_per_epoch=settings.batch_size * 200, processing=data_processing_val)

    loader_train = DataLoader('train', dataset_train, training=True, num_workers=settings.num_workers,
                              stack_dim=0, batch_size=settings.batch_size)
    loader_val = DataLoader('val', dataset_val, training=False, num_workers=settings.num_workers,
                            stack_dim=0, batch_size=settings.batch_size, epoch_interval=5)
    net = RBSR()

    # Wrap the network for multi GPU training
    if settings.multi_gpu:
        net = MultiGPU(net, dim=0)

    objective = {'rgb': PixelWiseError(metric='l1', boundary_ignore=None), 
                 'psnr': PSNR(boundary_ignore=40)}
    loss_weight = {'rgb': 1.0}
    actor = dbsr_actors.DBSRSyntheticActor(net=net, objective=objective, loss_weight=loss_weight)
    optimizer = torch.optim.AdamW([{'params': actor.net.parameters(), 'lr': 1e-4}],
                           lr=2e-4)
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, 400, eta_min=1e-6) 
    trainer = SimpleTrainer(actor, [loader_train, loader_val], optimizer, settings, lr_scheduler)

    trainer.train(400, load_latest=True, fail_safe=False)
