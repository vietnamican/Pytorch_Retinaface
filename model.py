from models.base import Base
from models import RetinaFace
from layers.modules import MultiBoxLoss
from torch import optim
from layers.functions.prior_box import PriorBox
from data import LaPa, detection_collate
from torch.utils.data import DataLoader
from pytorch_lightning.trainer.supporters import CombinedLoader

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
        self.using_dataset_index = 0
        device='cuda:0'
        img_dim = self.cfg['image_size']
        prior_box = PriorBox(cfg, image_size=(img_dim, img_dim))
        if self.cfg['gpu_train']:
            self.priors = prior_box.forward().to(device)
        else:
            self.priors = prior_box.forward()

        self.criterion = MultiBoxLoss(
            self.num_classes, 0.35, True, 0, True, 7, 0.35, False)

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        batch1, batch2 = batch
        if self.using_dataset_index == 0:
            batch = batch1
            self.using_dataset_index = 1
        else:
            batch = batch2
            self.using_dataset_index = 0
        images, targets = batch
        out = self.model(images)
        loss_l, loss_c = self.criterion(out, self.priors, targets)
        self.log('train_loc_loss', loss_l, prog_bar=True)
        self.log('train_conf_loss', loss_c, prog_bar=True)
        loss = self.cfg['loc_weight'] * loss_l + loss_c
        self.log('train_loss', loss)
        return loss

    def validation_step(self, batch, batch_idx):
        batch1, batch2 = batch

        batch = batch1
        images, targets = batch
        out = self.model(images)
        loss_l, loss_c = self.criterion(out, self.priors, targets)
        self.log('val_loc_loss', loss_l)
        self.log('val_conf_loss', loss_c)
        loss = self.cfg['loc_weight'] * loss_l + loss_c
        self.log('val_loss', loss)

        batch = batch2
        images, targets = batch
        out = self.model(images)
        loss_l, loss_c = self.criterion(out, self.priors, targets)
        self.log('val_loc_loss', loss_l)
        self.log('val_conf_loss', loss_c)
        loss = self.cfg['loc_weight'] * loss_l + loss_c
        self.log('val_loss', loss)

        return loss

    def configure_optimizers(self):
        lr, momentum, weight_decay = self.args.lr, self.args.momentum, self.args.weight_decay
        max_epochs = self.cfg['max_epochs']
        steps = [round(step*max_epochs) for step in self.cfg['decay_steps']]
        optimizer = optim.SGD(self.model.parameters(), lr=lr,
                              momentum=momentum, weight_decay=weight_decay)
        lr_scheduler = optim.lr_scheduler.MultiStepLR(
            optimizer, milestones=steps, gamma=0.1)
        return [optimizer], [lr_scheduler]

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
        train_lapa_image_dir = '../datasets/LaPa_negpos_fusion_cleaned/train/images/'
        train_ir_image_dirs = [
            '../datasets/ir_negpos/positive/images/out2/',
            '../datasets/ir_negpos/positive/images/out22/',
            '../datasets/ir_negpos/negative/images/out2/',
            '../datasets/ir_negpos/negative/images/out22/',
            '../datasets/dms_video_data_cleaned/'
        ]

        train_batch_size = self.args.train_batch_size
        num_workers = self.args.num_workers

        lapatraindataset = LaPa(train_lapa_image_dir,
                                'train', augment=True, preload=False, to_gray=False)
        irtraindataset = LaPa(train_ir_image_dirs,
                            'train', augment=True, preload=False, to_gray=False)

        lapatrainloader = DataLoader(lapatraindataset, batch_size=train_batch_size,
                                pin_memory=True, num_workers=num_workers, shuffle=True, collate_fn=detection_collate)
        irtrainloader = DataLoader(irtraindataset, batch_size=train_batch_size,
                                pin_memory=True, num_workers=num_workers, shuffle=True, collate_fn=detection_collate)


        self.num_training_samples = len(lapatrainloader)
        self.total_warmup_steps = self.num_training_samples * self.warmup_epochs
        return lapatrainloader, irtrainloader



    def val_dataloader(self):
        val_image_dir = '../datasets/LaPa_negpos_fusion_cleaned/val/images/'
        val_ir_image_dirs = [
            '../datasets/ir_negpos/positive/images/out222/',
            '../datasets/ir_negpos/negative/images/out222/'
        ]

        val_batch_size = self.args.val_batch_size
        num_workers = self.args.num_workers

        lapavaldataset = LaPa(val_image_dir, 'val',
                          augment=True, preload=False, to_gray=False)
        irvaldataset = LaPa(val_ir_image_dirs, 'val',
                            augment=True, preload=False, to_gray=False)

        lapavalloader = DataLoader(lapavaldataset, batch_size=val_batch_size,
                            pin_memory=True, num_workers=num_workers, collate_fn=detection_collate)
        irvalloader = DataLoader(irvaldataset, batch_size=val_batch_size,
                            pin_memory=True, num_workers=num_workers, collate_fn=detection_collate)

        combined_loaders = CombinedLoader([lapavalloader, irvalloader], "max_size_cycle")
        return combined_loaders