import pl_bolts


from models.base import Base
from models import RetinaFace
from layers.modules import MultiBoxLoss
from torch import optim
from layers.functions.prior_box import PriorBox


class Model(Base):
    def __init__(self, cfg=None, args=None, num_training_steps=10 ,phase='train'):
        super().__init__()
        self.cfg = cfg
        self.args = args
        self.warmup_epochs = 5
        self.num_training_steps = num_training_steps
        self.total_warmup_steps = self.num_training_steps * self.warmup_epochs
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

    def training_step(self, batch, batch_idx):
        images, targets = batch
        out = self.model(images)
        loss_l, loss_c = self.criterion(out, self.priors, targets)
        self.log('train_loc_loss', loss_l, prog_bar=True)
        self.log('train_conf_loss', loss_c, prog_bar=True)
        loss = self.cfg['loc_weight'] * loss_l + loss_c
        self.log('train_loss', loss)
        return loss

    def validation_step(self, batch, batch_idx):
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
        optimizer = optim.SGD(self.model.parameters(), lr=lr,
                              momentum=momentum, weight_decay=weight_decay)
        lr_scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[190, 220], gamma=0.1)
        return [optimizer], [lr_scheduler]

    def optimizer_step(self, epoch, batch_idx, optimizer, optimizer_idx, optimizer_closure, on_tpu, using_native_amp, using_lbfgs):
         # warm up lr
        if self.trainer.global_step < self.total_warmup_steps:
            lr_scale = min(1., float(self.trainer.global_step + 1.) / self.total_warmup_steps)
            for pg in optimizer.param_groups:
                pg['lr'] = lr_scale * self.args.lr

        # update params
        optimizer.step(closure=optimizer_closure)