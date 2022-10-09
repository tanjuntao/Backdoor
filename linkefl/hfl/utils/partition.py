import torch



class Partition(object):
	""" Dataset-like object, but only access a subset of it. """
	def __init__(self, data, index):
		self.data = data
		# self.index = index
		self.index = list(index)

	def __len__(self):
		return len(self.index)

	def __getitem__(self, index):
		data_idx = self.index[index]
		return self.data[data_idx]
