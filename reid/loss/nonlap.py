from __future__ import absolute_import

from torch import nn
import numpy as np
from scipy import stats

import torch
import torch.nn.functional as F

# global variable for speeding up the calculation
prev_bs = 0
prior_dist = None


def GaussianKernel(radius, std):
	"""
    Generate a gaussian blur kernel based on the given radius and std.

    Args
    ----------
    radius: int
        Radius of the Gaussian kernel. Center not included.
    std: float
        Standard deviation of the Gaussian kernel.

    Returns
    ----------
    weight: torch.FloatTensor, [2 * radius + 1, 2 * radius + 1]
        Output Gaussian kernel.

    """
	size = 2 * radius + 1
	weight = torch.ones(size, size)
	weight.requires_grad = False
	for i in range(-radius, radius + 1):
		for j in range(-radius, radius + 1):
			dis = (i * i) + (j * j)
			weight[i + radius][j + radius] = np.exp(-dis / (2 * std * std))
	weight = weight / weight.sum()
	return weight

def update_prior_dist(batch_size, alpha, beta, device):
	"""
    Update the samples of prior distribution due to the change of batchsize.

    Args
    ----------
    batch_size: int
        Current batch size.
    alpha: float
        Parameter of Beta distribution.
    beta: float
        Parameter of Beta distribution.

    """
	global prior_dist
	grid_points = torch.arange(1., 2 * batch_size, 2., device=device).float() / (2 * batch_size)
	grid_points_np = grid_points.cpu().numpy()
	grid_points_icdf = stats.beta.ppf(grid_points_np, a=alpha, b=beta)
	prior_dist = torch.tensor(grid_points_icdf, device=device).float().unsqueeze(1)

def shapingloss(assign, radius, std, num_parts, alpha, beta, eps=1e-5):
	"""
    Wasserstein shaping loss for Bernoulli distribution.

    Args
    ----------
    assign: torch.cuda.FloatTensor, [batch_size, num_parts, height, width]
        Assignment map for grouping.
    radius: int
        Radius for the Gaussian kernel.
    std: float
        Standard deviation for the Gaussian kernel.
    num_parts: int
        Number of object parts in the current model.
    alpha: float
        Parameter of Beta distribution.
    beta: float
        Parameter of Beta distribution.
    eps:
        Epsilon for rescaling the distribution.

    Returns
    ----------
    loss: torch.cuda.FloatTensor, [1, ]
        Average Wasserstein shaping loss for the current minibatch.

    """
	global prev_bs, prior_dist
	batch_size = assign.shape[0]

	# Gaussian blur
	if radius == 0:
		assign_smooth = assign
	else:
		weight = GaussianKernel(radius, std)
		weight = weight.contiguous().view(1, 1, 2 * radius + 1, 2 * radius + 1).expand(num_parts, 1, 2 * radius + 1,
																					   2 * radius + 1).cuda()
		assign_smooth = F.conv2d(assign, weight, groups=num_parts)

	# pool the assignment maps into [batch_size, num_parts] for the empirical distribution of part occurence
	part_occ = F.adaptive_max_pool2d(assign_smooth, (1, 1)).squeeze(2).squeeze(2)
	emp_dist, _ = part_occ.sort(dim=0, descending=False)

	# the Beta prior
	if batch_size != prev_bs:
		update_prior_dist(batch_size, alpha, beta, assign.device)

	# rescale the distribution
	emp_dist = (emp_dist + eps).log()
	prior_dist = (prior_dist + eps).log()

	# return the loss
	output_nk = (emp_dist - prior_dist).abs()
	loss = output_nk.mean()
	return loss



class NonlapLoss(nn.Module):
	'''
	Compute Triplet loss augmented with Batch Hard
	Details can be seen in 'In defense of the Triplet Loss for Person Re-Identification'
	'''

	def __init__(self, radius=2, std=0.2, alpha=1, beta=0.001, num_parts=5, eps=1e-5):
		super(NonlapLoss, self).__init__()
		self.radius = radius
		self.std = std
		self.alpha = alpha
		self.beta = beta
		self.num_parts = num_parts
		self.eps = eps

	def forward(self, assign, mode='conv'):
		if mode == 'conv':
			loss = shapingloss(
				assign, self.radius, self.std, self.num_parts,
				self.alpha, self.beta, self.eps
			)
			return loss
		else:
			loss = shapingloss(
				assign, self.radius, self.std, self.num_parts,
				self.alpha, self.beta, self.eps
			)
			return loss

