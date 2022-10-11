import numpy as np
from torch.utils.data import Dataset
import torch

# build mock dataloader
class PAMAP2Dataset(Dataset):
    def __init__(self, data, actions, attributes, action_feats, action_classes):
        super(PAMAP2Dataset, self).__init__()
        self.data = torch.from_numpy(data)
        self.actions = actions
        self.attributes = torch.from_numpy(attributes)
        self.action_feats = torch.from_numpy(action_feats)
        self.target_feat = torch.from_numpy(action_feats[action_classes, :])
        # build action to id mapping dict
        self.n_action = len(self.actions)
        self.action2Id = dict(zip(action_classes, range(self.n_action)))

    def __getitem__(self, ind):
        x = self.data[ind, ...]
        x_mask = np.array([0]) #self.padding_mask[ind, ...]
        target = self.actions[ind]
        y = torch.from_numpy(np.array([self.action2Id[target]]))
        y_feat = self.action_feats[target, ...]
        attr = self.attributes[target, ...]
        return x, y, y_feat, attr, x_mask

    def __len__(self):
        return self.data.shape[0]