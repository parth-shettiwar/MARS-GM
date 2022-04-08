import numpy as np
import random
import torch
from torch.utils.data import Dataset

class GRDataset(Dataset):
  def __init__(self, data, u_items_list, u_users_list, u_users_items_list, i_users_list, **kwargs):
    self.data = data
    self.u_items_list = u_items_list
    self.u_users_list = u_users_list
    self.u_users_items_list = u_users_items_list
    self.i_users_list = i_users_list
    self.dataset = ""
    if "dataset" in kwargs and kwargs['dataset'] == "FilmTrust":
      self.dataset = kwargs['dataset']
      self.i_items_list = kwargs['i_items_list']
      self.i_items_users_list = kwargs['i_items_users_list']

  def __getitem__(self, index):
    uid = self.data[index][0]
    iid = self.data[index][1]
    label = self.data[index][2]
    u_items = self.u_items_list[uid]
    u_users = self.u_users_list[uid]
    u_users_items = self.u_users_items_list[uid]
    i_users = self.i_users_list[iid]


    if self.dataset == "FilmTrust":
      i_items = self.i_items_list[iid]
      i_items_users = self.i_items_users_list[iid]

      return (uid, iid, label), u_items, u_users, u_users_items, i_users, i_items, i_items_users

    return (uid, iid, label), u_items, u_users, u_users_items, i_users, None, None

  def __len__(self):
    return len(self.data)

# class GRDatasetFilm(Dataset):
# 	def __init__(self, data, u_items_list, u_users_list, u_users_items_list, i_users_list, i_item):
# 		self.data = data
# 		self.u_items_list = u_items_list
# 		self.u_users_list = u_users_list
# 		self.u_users_items_list = u_users_items_list
# 		self.i_users_list = i_users_list

# 	def __getitem__(self, index):
# 		uid = self.data[index][0]
# 		iid = self.data[index][1]
# 		label = self.data[index][2]
# 		u_items = self.u_items_list[uid]
# 		u_users = self.u_users_list[uid]
# 		u_users_items = self.u_users_items_list[uid]
# 		i_users = self.i_users_list[iid]

# 		return (uid, iid, label), u_items, u_users, u_users_items, i_users

# 	def __len__(self):
# 		return len(self.data)
