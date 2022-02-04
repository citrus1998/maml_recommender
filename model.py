import torch
import torch.nn as nn
import torch.nn.functional as F

class MatrixFactorization(nn.Module):
    def __init__(self):
        super(MatrixFactorization, self).__init__()

    def forward(self, user_ids, item_ids, params):
        user_features = F.embedding(user_ids, params['user_weight'])
        item_features = F.embedding(item_ids, params['item_weight'])

        ratings = (user_features * item_features).sum(dim=1)
        ratings = F.dropout(ratings, p=0.6)

        return ratings, user_features, item_features

