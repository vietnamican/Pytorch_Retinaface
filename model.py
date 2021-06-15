from torch.utils.data import DataLoader
from torch import nn
from pytorch_lightning.trainer.supporters import CombinedLoader

from models.base import Base
from models import RetinaFace
from layers.modules import MultiBoxLoss
from torch import optim
from layers.functions.prior_box import PriorBox
from data import LaPa, detection_collate, preproc, cfg_mnet, cfg_re50, ConcatDataset, EyeGaze

EYE_STATE_IDX = 0
EYE_GAZE_IDX = 1


class Model(Base):
    def __init__(self, cfg=None, args=None, phase='train'):
        super().__init__()
        self.cfg = cfg
        self.args = args
        self.warmup_epochs = self.cfg['warmup_epochs']
        # self.num_training_steps = num_training_steps
        # self.total_warmup_steps = self.num_training_steps * self.warmup_epochs
        self.model = RetinaFace(cfg, phase)
        self.num_classes = 2
        img_dim = self.cfg['image_size']
        prior_box = PriorBox(cfg, image_size=(img_dim, img_dim))
        if self.cfg['gpu_train']:
            self.priors = prior_box.forward().cuda()
        else:
            self.priors = prior_box.forward()

        self.criterion = MultiBoxLoss(
            self.num_classes, 0.35, True, 0, True, 7, 0.35, False)
        self.eyegaze_criterion = nn.MSELoss()

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
    # def training_step(self, batch, batch_idx, optimizer_idx):
        eyestate_batch, eyegaze_batch = batch
        optimizer_idx = EYE_STATE_IDX
        if optimizer_idx == EYE_STATE_IDX:
            images, targets = eyestate_batch
            out, _ = self.model(images)
            loss_l, loss_c = self.criterion(out, self.priors, targets)
            self.log('train_loc_loss', loss_l, prog_bar=True)
            self.log('train_conf_loss', loss_c, prog_bar=True)
            loss = self.cfg['loc_weight'] * loss_l + loss_c
            self.log('train_loss', loss)
        else:
            images, targets = eyegaze_batch
            _, out = self.model(images)
            loss = self.eyegaze_criterion(out, targets)
            self.log('train_eyegaze_loss', loss, prog_bar=True)

        return loss

    def validation_step(self, batch, batch_idx):
        eyestate_batch, eyegaze_batch = batch
        images, targets = eyestate_batch
        out, _ = self.model(images)
        loss_l, loss_c = self.criterion(out, self.priors, targets)
        self.log('val_loc_loss', loss_l)
        self.log('val_conf_loss', loss_c)
        loss1 = self.cfg['loc_weight'] * loss_l + loss_c
        self.log('val_loss', loss1)
        images, targets = eyegaze_batch
        _, out = self.model(images)
        loss2 = self.eyegaze_criterion(out, targets)
        self.log('val_eyegaze_loss', loss2)

        return loss1 + loss2

    def configure_optimizers(self):
        num_training_samples = len(self.train_dataloader()[0])
        self.total_warmup_steps = self.warmup_epochs * num_training_samples
        lr, momentum, weight_decay = self.args.lr, self.args.momentum, self.args.weight_decay
        max_epochs = self.cfg['max_epochs']
        steps = [round(step*max_epochs) for step in self.cfg['decay_steps']]

        eyestate_paramters = list(self.model.body.stage1.parameters()) + \
            list(self.model.ssh_eyestate.parameters()) + \
            list(self.model.ClassHead.parameters()) + \
            list(self.model.BboxHead.parameters())
        # eyegaze_paramters = list(self.model.body.stage2.parameters()) + \
        #     list(self.model.body.stage3.parameters()) + \
        #     list(self.model.ssh_eyegaze.parameters())

        eyestate_optimizer = optim.SGD(eyestate_paramters, lr=lr,
                                       momentum=momentum, weight_decay=weight_decay)
        # eyegaze_optimizer = optim.SGD(eyegaze_paramters, lr=lr,
        #                               momentum=momentum, weight_decay=weight_decay)

        eyestate_lr_scheduler = optim.lr_scheduler.MultiStepLR(
            eyestate_optimizer, milestones=steps, gamma=0.1)
        # eyegaze_lr_scheduler = optim.lr_scheduler.MultiStepLR(
        #     eyegaze_optimizer, milestones=steps, gamma=0.1)
        return [eyestate_optimizer], [eyestate_lr_scheduler]
        # return [eyestate_optimizer, eyegaze_optimizer], [eyestate_lr_scheduler, eyegaze_lr_scheduler]

    def optimizer_step(self, epoch, batch_idx, optimizer, optimizer_idx, optimizer_closure, on_tpu, using_native_amp, using_lbfgs):
        # warm up lr
        if self.trainer.global_step < self.total_warmup_steps:
            lr_scale = min(1., float(self.trainer.global_step +
                           1.) / self.total_warmup_steps)
            for pg in optimizer.param_groups:
                pg['lr'] = lr_scale * self.args.lr

        # update params
        optimizer.step(closure=optimizer_closure)

    def train_dataloader(self):
        train_batch_size = self.args.train_batch_size
        num_workers = self.args.num_workers
        train_image_dir = '../datasets/data_cleaned/train/images'
        train_label_dir = '../datasets/data_cleaned/train/labels'
        train_ir_image_dirs = [
            '../datasets/ir_cleaned/images/out2',
            '../datasets/ir_cleaned/images/out22',
            '../datasets/ir_cleaned/images/out222',
        ]

        train_ir_label_dirs = [
            '../datasets/ir_cleaned/labels/out2',
            '../datasets/ir_cleaned/labels/out22',
            '../datasets/ir_cleaned/labels/out222',
        ]
        lapatraindataset = LaPa(train_image_dir, train_label_dir,
                                'train', augment=True, preload=True, to_gray=False)
        irtraindataset = LaPa(train_ir_image_dirs, train_ir_label_dirs,
                              'train', augment=True, preload=True, to_gray=False)
        traindataset = ConcatDataset(lapatraindataset, irtraindataset)
        eyestate_loader = DataLoader(traindataset, batch_size=train_batch_size,
                                     pin_memory=True, num_workers=num_workers, shuffle=True, collate_fn=detection_collate)

        eyegaze_dirs = [
            '../datasets/data_gaze/rgb/normal',
            '../datasets/data_gaze/rgb/glance',
            '../datasets/data_gaze/ir/normal',
        ]
        # eyegaze_dir = '../datasets/eyegazedata'
        eyegazetraindataset = EyeGaze(
            eyegaze_dirs, set_type='train', target_size=self.cfg['image_size'], augment=True, preload=True, to_gray=False)
        eyegaze_loader = DataLoader(eyegazetraindataset, batch_size=train_batch_size,
                                    pin_memory=True, num_workers=num_workers, shuffle=True)

        return [eyestate_loader, eyegaze_loader]

    def val_dataloader(self):
        train_batch_size = self.args.val_batch_size
        num_workers = self.args.num_workers
        val_image_dir = '../datasets/data_cleaned/val/images'
        val_label_dir = '../datasets/data_cleaned/val/labels'
        lapavaldataset = LaPa(val_image_dir, val_label_dir, 'val',
                              augment=True, preload=True, to_gray=False)
        valdataset = lapavaldataset
        eyestate_loader = DataLoader(valdataset, batch_size=train_batch_size,
                                     pin_memory=True, num_workers=num_workers, shuffle=True, collate_fn=detection_collate)

        eyegaze_dirs = [
            '../datasets/data_gaze/rgb/normal',
            '../datasets/data_gaze/rgb/glance',
            '../datasets/data_gaze/ir/normal',
        ]
        # eyegaze_dir = '../datasets/eyegazedata'
        eyegazetraindataset = EyeGaze(
            eyegaze_dirs, set_type='val', target_size=self.cfg['image_size'], augment=True, preload=True, to_gray=False)
        eyegaze_loader = DataLoader(eyegazetraindataset, batch_size=train_batch_size,
                                    pin_memory=True, num_workers=num_workers, shuffle=True)

        combined_loaders = CombinedLoader(
            [eyestate_loader, eyegaze_loader], "max_size_cycle")
        return combined_loaders
