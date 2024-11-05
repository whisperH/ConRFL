import torch
import torch.nn as nn
import torch.nn.functional as F

class CrossEntropyLabelSmooth_weighted(nn.Module):
	"""Cross entropy loss with label smoothing regularizer.

	Reference:
	Szegedy et al. Rethinking the Inception Architecture for Computer Vision. CVPR 2016.
	Equation: y = (1 - epsilon) * y + epsilon / K.

	Args:
		num_classes (int): number of classes.
		epsilon (float): weight.
	"""

	def __init__(self, num_classes, epsilon=0.1):
		super(CrossEntropyLabelSmooth_weighted, self).__init__()
		self.num_classes = num_classes
		self.epsilon = epsilon
		self.logsoftmax = nn.LogSoftmax(dim=1).cuda()

	def forward(self, inputs, targets, weights):
		"""
		Args:
			inputs: prediction matrix (before softmax) with shape (batch_size, num_classes)
			targets: ground truth labels with shape (num_classes)
		"""
		log_probs = self.logsoftmax(inputs)
		targets = torch.zeros_like(log_probs).scatter_(1, targets.unsqueeze(1), 1)
		targets = (1 - self.epsilon) * targets + self.epsilon / self.num_classes
		loss = torch.sum(torch.sum((- targets * log_probs), dim=1).view(-1) * weights)
		return loss


class CrossEntropyLabelSmooth(nn.Module):
	"""Cross entropy loss with label smoothing regularizer.

	Reference:
	Szegedy et al. Rethinking the Inception Architecture for Computer Vision. CVPR 2016.
	Equation: y = (1 - epsilon) * y + epsilon / K.

	Args:
		num_classes (int): number of classes.
		epsilon (float): weight.
	"""

	def __init__(self, num_classes, epsilon=0.1):
		super(CrossEntropyLabelSmooth, self).__init__()
		self.num_classes = num_classes
		self.epsilon = epsilon
		self.logsoftmax = nn.LogSoftmax(dim=1).cuda()

	def forward(self, inputs, targets):
		"""
		Args:
			inputs: prediction matrix (before softmax) with shape (batch_size, num_classes)
			targets: ground truth labels with shape (num_classes)
		"""
		log_probs = self.logsoftmax(inputs)
		targets = torch.zeros_like(log_probs).scatter_(1, targets.unsqueeze(1), 1)
		targets = (1 - self.epsilon) * targets + self.epsilon / self.num_classes
		loss = (- targets * log_probs).mean(0).sum()
		return loss


# class CrossEntropy(nn.Module):
# 	def __init__(self, weight=0.5, eps=0.1, alpha=0.2):
# 		super(CrossEntropy, self).__init__()
# 		self.weight = weight
# 		self.eps = eps
# 		self.alpha=alpha
#
# 	def forward(self, pred_class_logits, gt_classes):
# 		num_classes = pred_class_logits.size(1)
#
# 		if eps >= 0:
# 			smooth_param = self.eps
# 		else:
# 			# Adaptive label smooth regularization
# 			soft_label = F.softmax(pred_class_logits, dim=1)
# 			smooth_param = self.alpha * soft_label[torch.arange(soft_label.size(0)), gt_classes].unsqueeze(1)
#
# 		log_probs = F.log_softmax(pred_class_logits, dim=1)
# 		with torch.no_grad():
# 			targets = torch.ones_like(log_probs)
# 			targets *= smooth_param / (num_classes - 1)
# 			targets.scatter_(1, gt_classes.data.unsqueeze(1), (1 - smooth_param))
#
# 		loss = (-targets * log_probs).sum(dim=1)
#
# 		"""
# 		# confidence penalty
# 		conf_penalty = 0.3
# 		probs = F.softmax(pred_class_logits, dim=1)
# 		entropy = torch.sum(-probs * log_probs, dim=1)
# 		loss = torch.clamp_min(loss - conf_penalty * entropy, min=0.)
# 		"""
#
# 		with torch.no_grad():
# 			non_zero_cnt = max(loss.nonzero().size(0), 1)
#
# 		loss = loss.sum() / non_zero_cnt
#
# 		return loss * self.weight

def cross_entropy_loss(pred_class_logits, gt_classes, eps, alpha=0.2):
	num_classes = pred_class_logits.size(1)

	if eps >= 0:
		smooth_param = eps
	else:
		# Adaptive label smooth regularization
		soft_label = F.softmax(pred_class_logits, dim=1)
		smooth_param = alpha * soft_label[torch.arange(soft_label.size(0)), gt_classes].unsqueeze(1)

	log_probs = F.log_softmax(pred_class_logits, dim=1)
	with torch.no_grad():
		targets = torch.ones_like(log_probs)
		targets *= smooth_param / (num_classes - 1)
		targets.scatter_(1, gt_classes.data.unsqueeze(1), (1 - smooth_param))

	loss = (-targets * log_probs).sum(dim=1)

	"""
    # confidence penalty
    conf_penalty = 0.3
    probs = F.softmax(pred_class_logits, dim=1)
    entropy = torch.sum(-probs * log_probs, dim=1)
    loss = torch.clamp_min(loss - conf_penalty * entropy, min=0.)
    """

	with torch.no_grad():
		non_zero_cnt = max(loss.nonzero().size(0), 1)
		# non_zero_cnt = max(loss.nonzero().size(0), 1)

	loss = loss.sum() / non_zero_cnt

	return loss