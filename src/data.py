import torch
from torch.utils import data
import numpy as np

class LocEventData(data.Dataset):
      def __init__(self, times, locs, labels, device=torch.device('cpu')):
            self.len = len(times) 
            self.times = times
            self.locs = locs
            self.y = labels
            self.times = self.times.to(device)
            self.locs = self.locs.to(device)
            self.y = self.y.to(device)

      def __len__(self):
            return self.len

      def __getitem__(self, index):
            return self.times[index], self.locs[index], self.y[index] 
