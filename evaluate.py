import torch

from dataloader import BaseExplicitDataLoader
import numpy as np

from utils import mini_batch, merge_dict
from torch import nn
import metrics
import matplotlib.pyplot as plt
import random

def get_label(test_data, pred_data):
    r = []
    for i in range(len(test_data)):
        groundTrue = test_data[i]
        predictTopK = pred_data[i]
        pred = list(map(lambda x: x in groundTrue, predictTopK))
        pred = np.array(pred).astype("float")
        r.append(pred)
    return np.array(r).astype('float')


def recall_precision_ATk(test_data, r, k):
    right_pred = r[:, :k].sum(1)
    precis_n = np.array([k for i in range(len(test_data))])
    recall_n = np.array([len(test_data[i]) for i in range(len(test_data))])
    recall = np.sum(right_pred / recall_n)
    precis = np.sum(right_pred / precis_n)

    return {'recall': recall, 'precision': precis}


def NDCGatK_r(test_data, r, k):
    """
    Normalized Discounted Cumulative Gain
    rel_i = 1 or 0, so 2^{rel_i} - 1 = 1 or 0
    """
    assert len(r) == len(test_data)
    pred_data = r[:, :k]

    test_matrix = np.zeros((len(pred_data), k))
    for i, items in enumerate(test_data):
        length = k if k <= len(items) else len(items)
        test_matrix[i, :length] = 1
    max_r = test_matrix
    idcg = np.sum(max_r * 1. / np.log2(np.arange(2, k + 2)), axis=1)
    dcg = pred_data * (1. / np.log2(np.arange(2, k + 2)))
    dcg = np.sum(dcg, axis=1)
    idcg[idcg == 0.] = 1.
    ndcg = dcg / idcg
    ndcg[np.isnan(ndcg)] = 0.
    return np.sum(ndcg)


class ImplicitTestManager:
    def __init__(
            self,
            model,
            data_loader,
            test_batch_size: int,
            top_k_list: list,
            use_item_pool: bool = False
    ):
        self.model = model
        self.data_loader = data_loader

        self.batch_size: int = test_batch_size
        self.top_k_list: list = top_k_list
        self.top_k_list.sort(reverse=False)
        self.use_item_pool: bool = use_item_pool

    def evaluate(self,type_list) -> dict:
        self.model.eval()

        def _merge_dicts_elements_func(elements_list, **args):
            user_num: int = args['user_num']
            return (np.sum(np.array(elements_list), axis=0) / float(user_num)).tolist()

        test_users = torch.empty(0, dtype=torch.int64).to('cuda')
        test_items = torch.empty(0, dtype=torch.int64).to('cuda')
        test_pre_ratings = torch.empty(0).to('cuda')
        test_ratings = torch.empty(0).to('cuda')
        ndcg_ratings = torch.empty(0).to('cuda')
        ndcg_item = torch.empty(0).to('cuda')
        ut_dict = {}
        pt_dict = {}

        result_dicts_list: list = []

        for batch_idx, (users, items, ratings) in enumerate(self.data_loader):
            users = users.to('cuda')
            items = items.to('cuda')
            ratings = ratings.to('cuda')

            pre_ratings = self.model.predict(users, items)

            for i, u in enumerate(users):
                try:
                    ut_dict[u.item()].append(ratings[i].item())
                    pt_dict[u.item()].append(pre_ratings[i].item())
                except:
                    ut_dict[u.item()] = [ratings[i].item()]
                    pt_dict[u.item()] = [pre_ratings[i].item()]
            test_users = torch.cat((test_users, users))
            test_items = torch.cat((test_items, items))
            test_pre_ratings = torch.cat((test_pre_ratings, pre_ratings))
            test_ratings = torch.cat((test_ratings, ratings))

        test_results = metrics.evaluate(test_pre_ratings, test_ratings, type_list,
                                              users=test_users, items=test_items,
                                              UAUC=(ut_dict, pt_dict))

        return test_results