import torch

from models.base import Base
from models import RetinaFace, RetinaFaceWidth, RetinaFaceDistill
from layers.modules import MultiBoxLoss, MultiBoxLossDistill
from torch import optim
from layers.functions.prior_box import PriorBox, PriorBoxDistill
from models.tdkd import TDKD


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
        
    def forward(self, x):
        return self.model(x)

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

class ModelDistill(Model):
    def __init__(self, model=None, cfg=None, args=None, num_training_steps=10 ,phase='train'):
        super().__init__(cfg=cfg, args=args, num_training_steps=num_training_steps, phase=phase)
        self.model = model(cfg, phase)

# class ModelWidth(Model):
#     def __init__(self, cfg=None, args=None, num_training_steps=10 ,phase='train'):
#         super().__init__(cfg=cfg, args=args, num_training_steps=num_training_steps, phase=phase)
#         self.model = RetinaFaceWidth(cfg, phase)

class Distill(Base):
    def __init__(self, teacher_model, student_model, cfg=None, args=None):
        super().__init__()
        self.teacher_model = teacher_model
        self.student_model = student_model
        self.cfg = cfg
        self.args = args
        self.num_classes = 2
        self.criterion = MultiBoxLossDistill(self.num_classes, 0.35, True, 0, True, 7, 0.35, False)
        self.tdkd = TDKD()
        img_dim = self.cfg['image_size']
        prior_box = PriorBoxDistill(cfg, image_size=(img_dim, img_dim))
        with torch.no_grad():
            if self.cfg['gpu_train']:
                self.priors, self.priors_by_layer = prior_box.forward()
                self.priors = self.priors.cuda()
            else:
                self.priors, self.priors_by_layer = prior_box.forward()
        self.freeze_with_prefix('teacher_model')

    def training_step(self, batch, batch_idx):
        images, targets = batch
        out = self.student_model(images)
        teacher_out = self.teacher_model(images)
        loss_l, loss_c, pos, neg = self.criterion(out, self.priors, targets)
        feat_loss, prob_loss = self.tdkd(out, teacher_out, self.priors_by_layer, self.priors, targets, pos, neg)

        d_feat = 1.0
        scale = loss_l.detach() / feat_loss.detach()
        d_prob = 50

        loss_objective = self.cfg['loc_weight'] * loss_l + loss_c
        loss = loss_objective + d_feat * scale * feat_loss + d_prob * prob_loss
        self.log('train_loss', loss)
        self.log('train_loss_objective', loss_objective)
        self.log('train_loss_location', loss_c, prog_bar=True)
        self.log('train_loss_class', loss_c, prog_bar=True)
        self.log('train_loss_feature', feat_loss)
        self.log('train_loss_prob', prob_loss)
        return loss

    def validation_step(self, batch, batch_idx):
        images, targets = batch
        out = self.student_model(images)
        teacher_out = self.teacher_model(images)
        loss_l, loss_c, pos, neg = self.criterion(out, self.priors, targets)
        feat_loss, prob_loss = self.tdkd(out, teacher_out, self.priors_by_layer, self.priors, targets, pos, neg)

        d_feat = 1.0
        scale = loss_l.detach() / feat_loss.detach()
        d_prob = 50

        loss_objective = self.cfg['loc_weight'] * loss_l + loss_c
        loss = loss_objective + d_feat * scale * feat_loss + d_prob * prob_loss
        self.log('val_loss', loss)
        self.log('val_loss_objective', loss_objective)
        self.log('val_loss_location', loss_c)
        self.log('val_loss_class', loss_c)
        self.log('val_loss_feature', feat_loss)
        self.log('val_loss_prob', prob_loss)
        return loss

    def configure_optimizers(self):
        lr, momentum, weight_decay = self.args.lr, self.args.momentum, self.args.weight_decay
        decay1 = self.cfg['distill_decay1']
        decay2 = self.cfg['distill_decay2']
        optimizer = optim.SGD(self.student_model.parameters(), lr=lr,
                              momentum=momentum, weight_decay=weight_decay)
        lr_scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[decay1, decay2], gamma=0.1)
        return [optimizer], [lr_scheduler]

    # def optimizer_step(self, epoch, batch_idx, optimizer, optimizer_idx, optimizer_closure, on_tpu, using_native_amp, using_lbfgs):
    #      # warm up lr
    #     if self.trainer.global_step < self.total_warmup_steps:
    #         lr_scale = min(1., float(self.trainer.global_step + 1.) / self.total_warmup_steps)
    #         for pg in optimizer.param_groups:
    #             pg['lr'] = lr_scale * self.args.lr

    #     # update params
    #     optimizer.step(closure=optimizer_closure)
        
        