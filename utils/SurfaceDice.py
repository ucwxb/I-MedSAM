# -*- coding: utf-8 -*-
import numpy as np
from medpy import metric

def compute_dice_coefficient(mask_gt, mask_pred):
  volume_sum = mask_gt.sum() + mask_pred.sum()
  if volume_sum == 0:
    return np.NaN
  volume_intersect = (mask_gt & mask_pred).sum()
  return 2*volume_intersect / volume_sum
 

def compute_HD_distances(mask_gt, mask_pred):
  if mask_gt.sum() <= 0 or mask_pred.sum() <= 0:
    return 100.0
  hd = metric.binary.hd(mask_pred, mask_gt)
  hd = np.fabs(hd)
  return hd