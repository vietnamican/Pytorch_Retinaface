from models.fdm import FeatureDistillMask
from models.distill_loss import FeatLoss, ProbLoss
import torch.nn as nn


def conv_dw(inp, oup, stride):
    return nn.Sequential(
#        nn.Conv2d(inp, inp, 3, stride, 1, groups=inp, bias=False),
#        nn.BatchNorm2d(inp),
#        nn.ReLU(inp),
#
        nn.Conv2d(inp, oup, 1, 1, 0, bias=False),
#        nn.BatchNorm2d(oup),
#        nn.ReLU(oup),
    )

class Adapt(nn.Module):
	def __init__(self, inchannels=64):
		super(Adapt, self).__init__()
		self.trans = conv_dw(inchannels, inchannels, 1)

	def forward(self, x):
		x = self.trans(x)
		return x


class TDKD(nn.Module):
	def __init__(self):
		super(TDKD, self).__init__()
		self.feat_loss = FeatLoss()
		self.prob_loss = ProbLoss()
		self.stage = 1

	def forward(self, stu_output, tea_output, priors_by_layer, priors, targets, pos, neg):

		_, _, stu_feat, stu_prob  = stu_output
		_, _, tea_feat, tea_prob  = tea_output
	
		num = int(pos.shape[0])

		pos_part1 = pos[:, :]
		cls_part1 = (neg[:, :] + pos_part1) > 0
		loc_list = pos_part1.view(num, -1, 2).sum(dim = 2) > 0
		cls_list = cls_part1.view(num, -1, 2).sum(dim = 2) > 0


		feat_loss = self.feat_loss(stu_feat, tea_feat, loc_list, cls_list)
		prob_loss = self.prob_loss(stu_prob, tea_prob)

		return feat_loss, prob_loss

