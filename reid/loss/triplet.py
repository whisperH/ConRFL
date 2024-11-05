from __future__ import absolute_import

import torch
from torch import nn
import torch.nn.functional as F

def softmax_weights(dist, mask):
	max_v = torch.max(dist * mask, dim=1, keepdim=True)[0]
	diff = dist - max_v
	Z = torch.sum(torch.exp(diff) * mask, dim=1, keepdim=True) + 1e-6  # avoid division by zero
	W = torch.exp(diff) * mask / Z
	return W

def euclidean_dist(x, y):
	m, n = x.size(0), y.size(0)
	xx = torch.pow(x, 2).sum(1, keepdim=True).expand(m, n)
	yy = torch.pow(y, 2).sum(1, keepdim=True).expand(n, m).t()
	dist = xx + yy - 2 * torch.matmul(x, y.t())
	dist = dist.clamp(min=1e-12).sqrt()  # for numerical stability
	return dist

def cosine_dist(x, y):
	bs1, bs2 = x.size(0), y.size(0)
	frac_up = torch.matmul(x, y.transpose(0, 1))
	frac_down = (torch.sqrt(torch.sum(torch.pow(x, 2), 1))).view(bs1, 1).repeat(1, bs2) * \
	            (torch.sqrt(torch.sum(torch.pow(y, 2), 1))).view(1, bs2).repeat(bs1, 1)
	cosine = frac_up / frac_down
	return 1-cosine

def _batch_hard(mat_distance, mat_similarity, indice=False):
	sorted_mat_distance, positive_indices = torch.sort(mat_distance + (-9999999.) * (1 - mat_similarity), dim=1, descending=True)
	hard_p = sorted_mat_distance[:, 0]
	hard_p_indice = positive_indices[:, 0]
	sorted_mat_distance, negative_indices = torch.sort(mat_distance + (9999999.) * (mat_similarity), dim=1, descending=False)
	hard_n = sorted_mat_distance[:, 0]
	hard_n_indice = negative_indices[:, 0]
	if(indice):
		return hard_p, hard_n, hard_p_indice, hard_n_indice
	return hard_p, hard_n

def _batch_mid_hard(mat_distance, mat_similarity, indice=False):
	sorted_mat_distance, positive_indices = torch.sort(mat_distance + (-9999999.) * (1 - mat_similarity), dim=1, descending=True)
	hard_p = sorted_mat_distance[:, 1]
	hard_p_indice = positive_indices[:, 1]
	sorted_mat_distance, negative_indices = torch.sort(mat_distance + (9999999.) * (mat_similarity), dim=1, descending=False)
	hard_n = sorted_mat_distance[:, 0]
	hard_n_indice = negative_indices[:, 0]
	if(indice):
		return hard_p, hard_n, hard_p_indice, hard_n_indice
	return hard_p, hard_n


def normalize(x, axis=-1):
	"""Normalizing to unit length along the specified dimension.
    Args:
      x: pytorch Variable
    Returns:
      x: pytorch Variable, same shape as input
    """
	x = 1. * x / (torch.norm(x, 2, axis, keepdim=True).expand_as(x) + 1e-12)
	return x

class TripletLoss(nn.Module):
	'''
	Compute Triplet loss augmented with Batch Hard
	Details can be seen in 'In defense of the Triplet Loss for Person Re-Identification'
	'''

	def __init__(self, margin, normalize_feature=False, mid_hard=False):
		super(TripletLoss, self).__init__()
		self.margin = margin
		self.normalize_feature = normalize_feature
		self.margin_loss = nn.MarginRankingLoss(margin=margin).cuda()
		self.mid_hard = mid_hard

	def forward(self, emb, label, emb_=None):
		if emb_ is None:
			mat_dist = euclidean_dist(emb, emb)
			# mat_dist = cosine_dist(emb, emb)
			assert mat_dist.size(0) == mat_dist.size(1)
			N = mat_dist.size(0)
			mat_sim = label.expand(N, N).eq(label.expand(N, N).t()).float()
			if self.mid_hard:
				dist_ap, dist_an = _batch_mid_hard(mat_dist, mat_sim)
			else:
				dist_ap, dist_an = _batch_hard(mat_dist, mat_sim)
			assert dist_an.size(0)==dist_ap.size(0)
			y = torch.ones_like(dist_ap)
			loss = self.margin_loss(dist_an, dist_ap, y)
			prec = (dist_an.data > dist_ap.data).sum() * 1. / y.size(0)
			return loss, prec
		else:
			mat_dist = euclidean_dist(emb, emb_)
			N = mat_dist.size(0)
			mat_sim = label.expand(N, N).eq(label.expand(N, N).t()).float()
			dist_ap, dist_an = _batch_hard(mat_dist, mat_sim)
			y = torch.ones_like(dist_ap)
			loss = self.margin_loss(dist_an, dist_ap, y)
			return loss

class SoftTripletLoss(nn.Module):

	def __init__(self, margin=None, normalize_feature=False,mid_hard=False):
		super(SoftTripletLoss, self).__init__()
		self.margin = margin
		self.normalize_feature = normalize_feature
		self.mid_hard = mid_hard

	def forward(self, emb1, emb2, label):
		if self.normalize_feature:
			# equal to cosine similarity
			emb1 = F.normalize(emb1)
			emb2 = F.normalize(emb2)

		mat_dist = euclidean_dist(emb1, emb1)
		assert mat_dist.size(0) == mat_dist.size(1)
		N = mat_dist.size(0)
		mat_sim = label.expand(N, N).eq(label.expand(N, N).t()).float()
		if self.mid_hard:
			dist_ap, dist_an, ap_idx, an_idx = _batch_mid_hard(mat_dist, mat_sim, indice=True)
		else:
			dist_ap, dist_an, ap_idx, an_idx = _batch_hard(mat_dist, mat_sim, indice=True)
		assert dist_an.size(0)==dist_ap.size(0)
		triple_dist = torch.stack((dist_ap, dist_an), dim=1)
		triple_dist = F.log_softmax(triple_dist, dim=1)
		if (self.margin is not None):
			loss = (- self.margin * triple_dist[:,0] - (1 - self.margin) * triple_dist[:,1]).mean()
			return loss

		mat_dist_ref = euclidean_dist(emb2, emb2)
		dist_ap_ref = torch.gather(mat_dist_ref, 1, ap_idx.view(N,1).expand(N,N))[:,0]
		dist_an_ref = torch.gather(mat_dist_ref, 1, an_idx.view(N,1).expand(N,N))[:,0]
		triple_dist_ref = torch.stack((dist_ap_ref, dist_an_ref), dim=1)
		triple_dist_ref = F.softmax(triple_dist_ref, dim=1).detach()

		loss = (- triple_dist_ref * triple_dist).mean(0).sum()
		return loss

class SoftTripletLoss_weight(nn.Module):

	def __init__(self, margin=None, normalize_feature=False,mid_hard=False):
		super(SoftTripletLoss_weight, self).__init__()
		self.margin = margin
		self.normalize_feature = normalize_feature
		self.mid_hard = mid_hard

	def forward(self, emb1, emb2, label, weights):
		if self.normalize_feature:
			# equal to cosine similarity
			emb1 = F.normalize(emb1)
			emb2 = F.normalize(emb2)

		mat_dist = euclidean_dist(emb1, emb1)
		assert mat_dist.size(0) == mat_dist.size(1)
		N = mat_dist.size(0)
		mat_sim = label.expand(N, N).eq(label.expand(N, N).t()).float()
		if self.mid_hard:
			dist_ap, dist_an, ap_idx, an_idx = _batch_mid_hard(mat_dist, mat_sim, indice=True)
		else:
			dist_ap, dist_an, ap_idx, an_idx = _batch_hard(mat_dist, mat_sim, indice=True)
		assert dist_an.size(0)==dist_ap.size(0)
		triple_dist = torch.stack((dist_ap, dist_an), dim=1)
		triple_dist = F.log_softmax(triple_dist, dim=1)
		if (self.margin is not None):
			loss = torch.sum((- self.margin * triple_dist[:,0] - (1 - self.margin) * triple_dist[:,1]) * weights)
			return loss

		mat_dist_ref = euclidean_dist(emb2, emb2)
		dist_ap_ref = torch.gather(mat_dist_ref, 1, ap_idx.view(N,1).expand(N,N))[:,0]
		dist_an_ref = torch.gather(mat_dist_ref, 1, an_idx.view(N,1).expand(N,N))[:,0]
		triple_dist_ref = torch.stack((dist_ap_ref, dist_an_ref), dim=1)
		triple_dist_ref = F.softmax(triple_dist_ref, dim=1).detach()

		loss = torch.sum((- torch.sum((triple_dist_ref * triple_dist),dim=1)).view(-1) * weights)
		return loss

# class Triplet_Loss(nn.Module):
# 	'''
# 	Compute Triplet loss augmented with Batch Hard
# 	Details can be seen in 'In defense of the Triplet Loss for Person Re-Identification'
# 	'''
#
# 	def __init__(self, margin, normalize_feature=False, mid_hard=False):
# 		super(Triplet_Loss, self).__init__()
#
# 	def hard_example_mining(self, dist_mat, is_pos, is_neg):
# 		"""For each anchor, find the hardest positive and negative sample.
# 		Args:
# 		  dist_mat: pair wise distance between samples, shape [N, M]
# 		  is_pos: positive index with shape [N, M]
# 		  is_neg: negative index with shape [N, M]
# 		Returns:
# 		  dist_ap: pytorch Variable, distance(anchor, positive); shape [N]
# 		  dist_an: pytorch Variable, distance(anchor, negative); shape [N]
# 		  p_inds: pytorch LongTensor, with shape [N];
# 			indices of selected hard positive samples; 0 <= p_inds[i] <= N - 1
# 		  n_inds: pytorch LongTensor, with shape [N];
# 			indices of selected hard negative samples; 0 <= n_inds[i] <= N - 1
# 		NOTE: Only consider the case in which all labels have same num of samples,
# 		  thus we can cope with all anchors in parallel.
# 		"""
#
# 		assert len(dist_mat.size()) == 2
# 		N = dist_mat.size(0)
#
# 		# `dist_ap` means distance(anchor, positive)
# 		# both `dist_ap` and `relative_p_inds` with shape [N, 1]
# 		dist_ap, relative_p_inds = torch.max(
# 			dist_mat[is_pos].contiguous().view(N, -1), 1, keepdim=True)
# 		# `dist_an` means distance(anchor, negative)
# 		# both `dist_an` and `relative_n_inds` with shape [N, 1]
# 		dist_an, relative_n_inds = torch.min(
# 			dist_mat[is_neg].contiguous().view(N, -1), 1, keepdim=True)
#
# 		# shape [N]
# 		dist_ap = dist_ap.squeeze(1)
# 		dist_an = dist_an.squeeze(1)
#
# 		return dist_ap, dist_an
#
#
# 	def weighted_example_mining(self, dist_mat, is_pos, is_neg):
# 		"""For each anchor, find the weighted positive and negative sample.
# 		Args:
# 		  dist_mat: pytorch Variable, pair wise distance between samples, shape [N, N]
# 		  is_pos:
# 		  is_neg:
# 		Returns:
# 		  dist_ap: pytorch Variable, distance(anchor, positive); shape [N]
# 		  dist_an: pytorch Variable, distance(anchor, negative); shape [N]
# 		"""
# 		assert len(dist_mat.size()) == 2
#
# 		is_pos = is_pos.float()
# 		is_neg = is_neg.float()
# 		dist_ap = dist_mat * is_pos
# 		dist_an = dist_mat * is_neg
#
# 		weights_ap = softmax_weights(dist_ap, is_pos)
# 		weights_an = softmax_weights(-dist_an, is_neg)
#
# 		dist_ap = torch.sum(dist_ap * weights_ap, dim=1)
# 		dist_an = torch.sum(dist_an * weights_an, dim=1)
#
# 		return dist_ap, dist_an
#
#
# 	def triplet_loss(self, embedding, targets, margin, norm_feat, hard_mining):
# 		r"""Modified from Tong Xiao's open-reid (https://github.com/Cysu/open-reid).
# 		Related Triplet Loss theory can be found in paper 'In Defense of the Triplet
# 		Loss for Person Re-Identification'."""
#
# 		if norm_feat: embedding = normalize(embedding, axis=-1)
#
# 		all_embedding = embedding
# 		all_targets = targets
# 		dist_mat = euclidean_dist(embedding, all_embedding)
#
# 		N, M = dist_mat.size()
# 		is_pos = targets.view(N, 1).expand(N, M).eq(all_targets.view(M, 1).expand(M, N).t())
# 		is_neg = targets.view(N, 1).expand(N, M).ne(all_targets.view(M, 1).expand(M, N).t())
#
# 		if hard_mining:
# 			dist_ap, dist_an = self.hard_example_mining(dist_mat, is_pos, is_neg)
# 		else:
# 			dist_ap, dist_an = self.weighted_example_mining(dist_mat, is_pos, is_neg)
#
# 		y = dist_an.new().resize_as_(dist_an).fill_(1)
#
# 		if margin > 0:
# 			loss = F.margin_ranking_loss(dist_an, dist_ap, y, margin=margin)
# 		else:
# 			loss = F.soft_margin_loss(dist_an - dist_ap, y)
# 			# fmt: off
# 			if loss == float('Inf'): loss = F.margin_ranking_loss(dist_an, dist_ap, y, margin=0.3)
# 			# fmt: on
#
# 		return loss
# 	def forward(self, emb, label, emb_=None):
# 		pass


def hard_example_mining(dist_mat, is_pos, is_neg):
	"""For each anchor, find the hardest positive and negative sample.
    Args:
      dist_mat: pair wise distance between samples, shape [N, M]
      is_pos: positive index with shape [N, M]
      is_neg: negative index with shape [N, M]
    Returns:
      dist_ap: pytorch Variable, distance(anchor, positive); shape [N]
      dist_an: pytorch Variable, distance(anchor, negative); shape [N]
      p_inds: pytorch LongTensor, with shape [N];
        indices of selected hard positive samples; 0 <= p_inds[i] <= N - 1
      n_inds: pytorch LongTensor, with shape [N];
        indices of selected hard negative samples; 0 <= n_inds[i] <= N - 1
    NOTE: Only consider the case in which all labels have same num of samples,
      thus we can cope with all anchors in parallel.
    """

	assert len(dist_mat.size()) == 2
	N = dist_mat.size(0)

	# `dist_ap` means distance(anchor, positive)
	# both `dist_ap` and `relative_p_inds` with shape [N, 1]
	dist_ap, relative_p_inds = torch.max(
		dist_mat[is_pos].contiguous().view(N, -1), 1, keepdim=True)
	# `dist_an` means distance(anchor, negative)
	# both `dist_an` and `relative_n_inds` with shape [N, 1]
	dist_an, relative_n_inds = torch.min(
		dist_mat[is_neg].contiguous().view(N, -1), 1, keepdim=True)

	# shape [N]
	dist_ap = dist_ap.squeeze(1)
	dist_an = dist_an.squeeze(1)

	return dist_ap, dist_an


def weighted_example_mining(dist_mat, is_pos, is_neg):
	"""For each anchor, find the weighted positive and negative sample.
    Args:
      dist_mat: pytorch Variable, pair wise distance between samples, shape [N, N]
      is_pos:
      is_neg:
    Returns:
      dist_ap: pytorch Variable, distance(anchor, positive); shape [N]
      dist_an: pytorch Variable, distance(anchor, negative); shape [N]
    """
	assert len(dist_mat.size()) == 2

	is_pos = is_pos.float()
	is_neg = is_neg.float()
	dist_ap = dist_mat * is_pos
	dist_an = dist_mat * is_neg

	weights_ap = softmax_weights(dist_ap, is_pos)
	weights_an = softmax_weights(-dist_an, is_neg)

	dist_ap = torch.sum(dist_ap * weights_ap, dim=1)
	dist_an = torch.sum(dist_an * weights_an, dim=1)

	return dist_ap, dist_an


def triplet_loss(embedding, targets, margin, norm_feat, hard_mining):
	r"""Modified from Tong Xiao's open-reid (https://github.com/Cysu/open-reid).
    Related Triplet Loss theory can be found in paper 'In Defense of the Triplet
    Loss for Person Re-Identification'."""

	if norm_feat: embedding = normalize(embedding, axis=-1)

	# For distributed training, gather all features from different process.

	all_embedding = embedding
	all_targets = targets

	dist_mat = euclidean_dist(embedding, all_embedding)

	N, M = dist_mat.size()
	is_pos = targets.view(N, 1).expand(N, M).eq(all_targets.view(M, 1).expand(M, N).t())
	is_neg = targets.view(N, 1).expand(N, M).ne(all_targets.view(M, 1).expand(M, N).t())

	if hard_mining:
		dist_ap, dist_an = hard_example_mining(dist_mat, is_pos, is_neg)
	else:
		dist_ap, dist_an = weighted_example_mining(dist_mat, is_pos, is_neg)

	y = dist_an.new().resize_as_(dist_an).fill_(1)

	if margin > 0:
		loss = F.margin_ranking_loss(dist_an, dist_ap, y, margin=margin)
	else:
		loss = F.soft_margin_loss(dist_an - dist_ap, y)
		# fmt: off
		if loss == float('Inf'): loss = F.margin_ranking_loss(dist_an, dist_ap, y, margin=0.3)
		# fmt: on

	return loss