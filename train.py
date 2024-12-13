import itertools
import json
import math

from torch import nn
import torch
import numpy as np
from scipy.stats import entropy

from dataloader import ImplicitBCELossDataLoaderStaticPopularity
from evaluate import ImplicitTestManager, ExplicitTestManager
from utils import mini_batch, merge_dict, _mean_merge_dict_func, transfer_loss_dict_to_line_str, _show_me_a_list_func
from utils import EarlyStopping, Stop_args, StopVariable
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

class ImplicitTrainManager:
    def __init__(
            self, model,val_evaluator, test_evaluator,device,
            train_norm_data, train_uniform_data,
            val_uniform_data, test_uniform_data,
            batch_size,
            epochs: int,
            cluster_interval: int, evaluate_interval: int, lr: float,
            invariant_coe: float, env_aware_coe: float,  env_coe: float, L2_coe: float,
            L1_coe: float,
            alpha: float = None, use_class_re_weight: bool = False, test_begin_epoch: int = 0,
            begin_cluster_epoch: int = None, stop_cluster_epoch: int = None, cluster_use_random_sort: bool = True,
            use_recommend_re_weight: bool = True
    ):
        self.model = model
        self.val_evaluator = val_evaluator
        self.test_evaluator = test_evaluator
        self.envs_num = self.model.env_num
        self.device: torch.device = device
        self.norm_users_tensor: torch.Tensor = train_norm_data[:, 0]
        self.norm_items_tensor: torch.Tensor = train_norm_data[:, 1]
        self.norm_scores_tensor: torch.Tensor = train_norm_data[:, 2].float()
        self.uniform_users_tensor: torch.Tensor = train_uniform_data[:, 0]
        self.uniform_items_tensor: torch.Tensor = train_uniform_data[:, 1]
        self.uniform_scores_tensor: torch.Tensor = train_uniform_data[:, 2].float()
        self.envs: torch.LongTensor = torch.LongTensor(np.random.randint(0, self.envs_num, train_norm_data.shape[0]))
        self.envs = self.envs.to(device)
        self.cluster_interval: int = cluster_interval
        self.evaluate_interval: int = evaluate_interval
        self.batch_size: int = batch_size
        self.epochs: int = epochs
        self.lr = lr
        self.optimizer = torch.optim.Adam(model.parameters(), lr=self.lr)

        self.recommend_loss_type = nn.BCELoss
        self.cluster_distance_func = nn.BCELoss(reduction='none')
        self.env_loss_type = nn.NLLLoss

        self.invariant_coe = invariant_coe
        self.env_aware_coe = env_aware_coe
        self.env_coe = env_coe
        self.L2_coe = L2_coe
        self.L1_coe = L1_coe

        self.epoch_cnt: int = 0

        self.batch_num = math.ceil(train_norm_data.shape[0] / batch_size)

        self.each_env_count = dict()

        if alpha is None:
            self.alpha = 0.
            self.update_alpha = True
        else:
            self.alpha = alpha
            self.update_alpha = False

        self.use_class_re_weight: bool = use_class_re_weight
        self.use_recommend_re_weight: bool = use_recommend_re_weight
        self.sample_weights: torch.Tensor = torch.Tensor(np.zeros(train_norm_data.shape[0])).to(device)
        self.class_weights: torch.Tensor = torch.Tensor(np.zeros(self.envs_num)).to(device)

        self.test_begin_epoch: int = test_begin_epoch

        self.begin_cluster_epoch: int = begin_cluster_epoch
        self.stop_cluster_epoch: int = stop_cluster_epoch

        self.eps_random_tensor: torch.Tensor = self._init_eps().to(self.device)

        self.cluster_use_random_sort: bool = cluster_use_random_sort

        self.const_env_tensor_list: list = []

        for env in range(self.envs_num):
            envs_tensor: torch.Tensor = torch.LongTensor(np.full(train_norm_data.shape[0], env, dtype=int))
            envs_tensor = envs_tensor.to(self.device)
            self.const_env_tensor_list.append(envs_tensor)

    def _init_eps(self):
        base_eps = 1e-10
        eps_list: list = [base_eps * (1e-1 ** idx) for idx in range(self.envs_num)]
        temp: torch.Tensor = torch.Tensor(eps_list)
        eps_random_tensor: torch.Tensor = torch.Tensor(list(itertools.permutations(temp)))

        return eps_random_tensor

    def train_a_batch(
            self,
            batch_users_tensor: torch.Tensor,
            batch_items_tensor: torch.Tensor,
            batch_scores_tensor: torch.Tensor,
            batch_envs_tensor: torch.Tensor,
            batch_sample_weights: torch.Tensor,
            alpha,
    ) -> dict:

        invariant_score, env_aware_score, env_outputs = self.model(
            batch_users_tensor, batch_items_tensor, batch_envs_tensor, alpha
        )

        if self.use_class_re_weight:
            env_loss = self.env_loss_type(reduction='none')
        else:
            env_loss = self.env_loss_type()

        if self.use_recommend_re_weight:
            recommend_loss = self.recommend_loss_type(reduction='none')
        else:
            recommend_loss = self.recommend_loss_type()

        invariant_loss: torch.Tensor = recommend_loss(invariant_score, batch_scores_tensor)
        env_aware_loss: torch.Tensor = recommend_loss(env_aware_score, batch_scores_tensor)

        envs_loss: torch.Tensor = env_loss(env_outputs, batch_envs_tensor)

        if self.use_class_re_weight:
            envs_loss = torch.mean(envs_loss * batch_sample_weights)

        if self.use_recommend_re_weight:
            invariant_loss = torch.mean(invariant_loss * batch_sample_weights)
            env_aware_loss = torch.mean(env_aware_loss * batch_sample_weights)

        L2_reg: torch.Tensor = self.model.get_L2_reg(batch_users_tensor, batch_items_tensor, batch_envs_tensor)
        L1_reg: torch.Tensor = self.model.get_L1_reg(batch_users_tensor, batch_items_tensor, batch_envs_tensor)

        loss: torch.Tensor = invariant_loss * self.invariant_coe + env_aware_loss * self.env_aware_coe \
                             + envs_loss * self.env_coe + L2_reg * self.L2_coe + L1_reg * self.L1_coe

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        loss_dict: dict = {
            'invariant_loss': float(invariant_loss),
            'env_aware_loss': float(env_aware_loss),
            'envs_loss': float(envs_loss),
            'L2_reg': float(L2_reg),
            'L1_reg': float(L1_reg),
            'loss': float(loss),
        }
        return loss_dict

    def cluster_a_batch(
            self,
            batch_users_tensor: torch.Tensor,
            batch_items_tensor: torch.Tensor,
            batch_scores_tensor: torch.Tensor,
    ) -> torch.Tensor:
        # 此时应该是eval()\
        distances_list: list = []
        for env_idx in range(self.envs_num):
            envs_tensor: torch.Tensor = self.const_env_tensor_list[env_idx][0:batch_users_tensor.shape[0]]
            cluster_pred: torch.Tensor = self.model.cluster_predict(batch_users_tensor, batch_items_tensor,
                                                                      envs_tensor)
            distances: torch.Tensor = self.cluster_distance_func(cluster_pred, batch_scores_tensor)
            distances = distances.reshape(-1, 1)
            distances_list.append(distances)


        each_envs_distances: torch.Tensor = torch.cat(distances_list, dim=1)
        if self.cluster_use_random_sort:
            sort_random_index: np.array = \
                np.random.randint(0, self.eps_random_tensor.shape[0], each_envs_distances.shape[0])
            random_eps: torch.Tensor = self.eps_random_tensor[sort_random_index]
            each_envs_distances = each_envs_distances + random_eps
        new_envs: torch.Tensor = torch.argmin(each_envs_distances, dim=1)

        return new_envs

    def train_a_epoch(self) -> dict:
        self.model.train()
        loss_dicts_list: list = []

        for (batch_index, (
                batch_users_tensor, batch_items_tensor, batch_scores_tensor, batch_envs_tensor, batch_sample_weights
        )) \
                in enumerate(mini_batch(self.batch_size, self.norm_users_tensor,
                                        self.norm_items_tensor, self.norm_scores_tensor, self.envs,
                                        self.sample_weights)):

            if self.update_alpha:
                p = float(batch_index + (self.epoch_cnt + 1) * self.batch_num) / float((self.epoch_cnt + 1)
                                                                                       * self.batch_num)
                self.alpha = 2. / (1. + np.exp(-10. * p)) - 1.

            loss_dict: dict = self.train_a_batch(
                batch_users_tensor=batch_users_tensor,
                batch_items_tensor=batch_items_tensor,
                batch_scores_tensor=batch_scores_tensor,
                batch_envs_tensor=batch_envs_tensor,
                batch_sample_weights=batch_sample_weights,
                alpha=self.alpha,
            )
            loss_dicts_list.append(loss_dict)

        self.epoch_cnt += 1

        mean_loss_dict: dict = merge_dict(loss_dicts_list, _mean_merge_dict_func)

        return mean_loss_dict

    def cluster(self) -> int:
        self.model.eval()

        new_env_tensors_list: list = []

        for (batch_index, (batch_users_tensor, batch_items_tensor, batch_scores_tensor)) \
                in enumerate(
            mini_batch(self.batch_size, self.norm_users_tensor, self.norm_items_tensor, self.norm_scores_tensor)):
            new_env_tensor: torch.Tensor = self.cluster_a_batch(
                batch_users_tensor=batch_users_tensor,
                batch_items_tensor=batch_items_tensor,
                batch_scores_tensor=batch_scores_tensor
            )


            new_env_tensors_list.append(new_env_tensor)

        all_new_env_tensors: torch.Tensor = torch.cat(new_env_tensors_list, dim=0)
        envs_diff: torch.Tensor = (self.envs - all_new_env_tensors) != 0
        diff_num: int = int(torch.sum(envs_diff))
        self.envs = all_new_env_tensors
        return diff_num

    def update_each_env_count(self):
        result_dict: dict = {}
        for env in range(self.envs_num):
            cnt = torch.sum(self.envs == env)
            result_dict[env] = cnt
        self.each_env_count.update(result_dict)

    def stat_envs(self) -> dict:
        result: dict = dict()
        class_rate_np: np.array = np.zeros(self.envs_num)
        for env in range(self.envs_num):
            cnt: int = int(torch.sum(self.envs == env))
            result[env] = cnt
            class_rate_np[env] = min(cnt + 1, self.norm_scores_tensor.shape[0] - 1)

        class_rate_np = class_rate_np / self.norm_scores_tensor.shape[0]
        self.class_weights = torch.Tensor(class_rate_np).to(self.device)
        self.sample_weights = self.class_weights[self.envs]

        return result

    def train(self, silent: bool = False, auto: bool = False):
        test_result_list: list = []
        test_epoch_list: list = []

        cluster_diff_num_list: list = []
        cluster_epoch_list: list = []
        envs_cnt_list: list = []

        loss_result_list: list = []
        train_epoch_index_list: list = []

        temp_eval_result = self.val_evaluator.evaluate(['AUC'])
        test_result_list.append(temp_eval_result)
        test_epoch_list.append(self.epoch_cnt)

        self.stat_envs()

        if not silent and not auto:
            print('test at epoch:', self.epoch_cnt)
            print(transfer_loss_dict_to_line_str(temp_eval_result))

        stopping_args = Stop_args(patience=80, max_epochs=300)
        early_stopping = EarlyStopping(self.model, **stopping_args)

        while self.epoch_cnt < self.epochs:
            temp_loss_dict = self.train_a_epoch()
            train_epoch_index_list.append(self.epoch_cnt)
            loss_result_list.append(temp_loss_dict)
            if not silent and not auto:
                print('train epoch:', self.epoch_cnt)
                print(transfer_loss_dict_to_line_str(temp_loss_dict))

            if (self.epoch_cnt % self.evaluate_interval) == 0 and self.epoch_cnt >= self.test_begin_epoch:
                temp_eval_result: dict = self.val_evaluator.evaluate(['AUC'])
                test_result_list.append(temp_eval_result)
                test_epoch_list.append(self.epoch_cnt)

                if not silent and not auto:
                    print('val at epoch:', self.epoch_cnt)
                    print(transfer_loss_dict_to_line_str(temp_eval_result))

                if early_stopping.check([temp_eval_result['AUC']], self.epoch_cnt):
                    break

            if (self.epoch_cnt % self.cluster_interval) == 0:
                if (self.begin_cluster_epoch is None or self.begin_cluster_epoch <= self.epoch_cnt) \
                        and (self.stop_cluster_epoch is None or self.stop_cluster_epoch > self.epoch_cnt):
                    diff_num: int = self.cluster()
                    cluster_diff_num_list.append(diff_num)
                else:
                    diff_num: int = 0
                    cluster_diff_num_list.append(diff_num)

                envs_cnt: dict = self.stat_envs()

                cluster_epoch_list.append(self.epoch_cnt)
                envs_cnt_list.append(envs_cnt)

                if not silent and not auto:
                    print('cluster at epoch:', self.epoch_cnt)
                    print('diff num:', diff_num)
                    print(transfer_loss_dict_to_line_str(envs_cnt))


        # restore best model
        print('#' * 120)
        print('Loading {}th epoch'.format(early_stopping.best_epoch))
        self.model.load_state_dict(early_stopping.best_state)
        final_eval_result = self.test_evaluator.evaluate(['AUC','Recall_Precision_NDCG@'])

        self.test_evaluator.evaluate_each()
        print('best performance:', format(final_eval_result))
        print('#' * 120)
        return final_eval_result

class BasicImplicitTrainManager:
    def __init__(
            self, model, val_evaluator,test_evaluator,
            device: torch.device,
            train_norm_data, train_uniform_data,
            val_uniform_data, test_uniform_data, batch_size: int,
            epochs: int, evaluate_interval: int, lr: float,
            L2_coe: float, L1_coe: float, test_begin_epoch: int = 0
    ):
        self.model = model
        self.val_evaluator = val_evaluator
        self.test_evaluator = test_evaluator

        self.device: torch.device = device

        self.norm_users_tensor: torch.Tensor = train_norm_data[:, 0]
        self.norm_items_tensor: torch.Tensor = train_norm_data[:, 1]
        self.norm_scores_tensor: torch.Tensor = train_norm_data[:, 2].float()
        self.uniform_users_tensor: torch.Tensor = train_uniform_data[:, 0]
        self.uniform_items_tensor: torch.Tensor = train_uniform_data[:, 1]
        self.uniform_scores_tensor: torch.Tensor = train_uniform_data[:, 2].float()

        # print(self.scores_tensor)

        self.evaluate_interval: int = evaluate_interval
        self.batch_size: int = batch_size
        self.epochs: int = epochs
        self.optimizer: torch.optim.Adam = torch.optim.Adam(model.parameters(), lr=lr)
        self.recommend_loss = nn.BCELoss()

        self.L2_coe = L2_coe
        self.L1_coe = L1_coe

        self.epoch_cnt: int = 0

        self.batch_num = math.ceil(self.norm_users_tensor.shape[0] / batch_size)

        self.test_begin_epoch: int = test_begin_epoch

    def train_a_batch(
            self,
            batch_users_tensor: torch.Tensor,
            batch_items_tensor: torch.Tensor,
            batch_scores_tensor: torch.Tensor,
            *args
    ) -> dict:

        # print()

        score_loss: torch.Tensor = self.model(batch_users_tensor, batch_items_tensor, batch_scores_tensor)
        L2_reg: torch.Tensor = self.model.get_L2_reg(batch_users_tensor, batch_items_tensor)
        L1_reg: torch.Tensor = self.model.get_L1_reg(batch_users_tensor, batch_items_tensor)

        loss: torch.Tensor = score_loss + L2_reg * self.L2_coe + L1_reg * self.L1_coe

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        loss_dict: dict = {
            'score_loss': float(score_loss),
            'L2_reg': float(L2_reg),
            'L1_reg': float(L1_reg),
            'loss': float(loss),
        }
        return loss_dict

    def train_a_epoch(self) -> dict:
        self.model.train()
        loss_dicts_list: list = []

        for (batch_index, (batch_users_tensor, batch_items_tensor, batch_scores_tensor)) \
                in enumerate(mini_batch(self.batch_size, self.norm_users_tensor,
                                        self.norm_items_tensor, self.norm_scores_tensor,)):

            loss_dict: dict = self.train_a_batch(
                batch_users_tensor=batch_users_tensor,
                batch_items_tensor=batch_items_tensor,
                batch_scores_tensor=batch_scores_tensor
            )
            loss_dicts_list.append(loss_dict)

        self.epoch_cnt += 1

        mean_loss_dict: dict = merge_dict(loss_dicts_list, _mean_merge_dict_func)

        return mean_loss_dict

    def train(self, silent: bool = False, auto: bool = False):
        test_result_list: list = []
        test_epoch_list: list = []

        loss_result_list: list = []
        train_epoch_index_list: list = []

        temp_eval_result: dict = self.val_evaluator.evaluate(['AUC'])
        test_result_list.append(temp_eval_result)
        test_epoch_list.append(self.epoch_cnt)

        if not silent and not auto:
            print('test at epoch:', self.epoch_cnt)
            print(transfer_loss_dict_to_line_str(temp_eval_result))

        stopping_args = Stop_args(patience=80, max_epochs=300)
        early_stopping = EarlyStopping(self.model, **stopping_args)

        while self.epoch_cnt < self.epochs:
            temp_loss_dict = self.train_a_epoch()
            train_epoch_index_list.append(self.epoch_cnt)
            loss_result_list.append(temp_loss_dict)
            if not silent and not auto:
                print('train epoch:', self.epoch_cnt)
                print(transfer_loss_dict_to_line_str(temp_loss_dict))

            if (self.epoch_cnt % self.evaluate_interval) == 0 and self.epoch_cnt >= self.test_begin_epoch:
                temp_eval_result: dict = self.val_evaluator.evaluate(['AUC'])
                test_result_list.append(temp_eval_result)
                test_epoch_list.append(self.epoch_cnt)

                if not silent and not auto:
                    print('test at epoch:', self.epoch_cnt)
                    print(transfer_loss_dict_to_line_str(temp_eval_result))

                if early_stopping.check([temp_eval_result['AUC']], self.epoch_cnt):
                    break

        print('#' * 120)
        print('Loading {}th epoch'.format(early_stopping.best_epoch))
        self.model.load_state_dict(early_stopping.best_state)
        final_eval_result = self.test_evaluator.evaluate(['AUC', 'Recall_Precision_NDCG@'])

        print('best performance:', format(final_eval_result))
        print('#' * 120)
        return final_eval_result

class BasicUniformImplicitTrainManager(BasicImplicitTrainManager):
    def __init__(
            self, model,
            val_evaluator,test_evaluator,
            device: torch.device,
            train_norm_data, train_uniform_data,
            val_uniform_data, test_uniform_data,
            batch_size: int,
            epochs: int, evaluate_interval: int, lr: float,
            L2_coe: float, L1_coe: float, test_begin_epoch: int = 0
    ):
        super(BasicUniformImplicitTrainManager, self).__init__(
            model=model, val_evaluator=val_evaluator,test_evaluator=test_evaluator,
            device=device, train_norm_data=train_norm_data, train_uniform_data=train_uniform_data,
            val_uniform_data=val_uniform_data, test_uniform_data=test_uniform_data,
            batch_size=batch_size, epochs=epochs, evaluate_interval=evaluate_interval, lr=lr,
            L2_coe=L2_coe, L1_coe=L1_coe, test_begin_epoch=test_begin_epoch
        )

        self.uniform_user: torch.Tensor = train_uniform_data[:, 0].to(self.device).long()
        self.uniform_item: torch.Tensor = train_uniform_data[:, 1].to(self.device).long()
        self.uniform_score: torch.Tensor = train_uniform_data[:, 2].to(self.device).float()

class ImplicitTrainStaticPopularityManager(ImplicitTrainManager):
    def __init__(
            self, model, evaluator: ImplicitTestManager,
            device: torch.device, data_loader: ImplicitBCELossDataLoaderStaticPopularity,
            training_data: torch.Tensor, batch_size: int,
            epochs: int, cluster_interval: int,  evaluate_interval: int, lr: float,
            invariant_coe: float, env_aware_coe: float, env_coe: float, L2_coe: float, L1_coe: float,
            static_pop_interval: int,
            alpha: float = None, use_class_re_weight: bool = False, test_begin_epoch: int = 0,
            begin_cluster_epoch: int = None, stop_cluster_epoch: int = None, cluster_use_random_sort: bool = True,
            use_recommend_re_weight: bool = True
    ):
        super(ImplicitTrainStaticPopularityManager, self).__init__(
            model=model, evaluator=evaluator, device=device, training_data=training_data, batch_size=batch_size,
            epochs=epochs, cluster_interval=cluster_interval, evaluate_interval=evaluate_interval,
            lr=lr, invariant_coe=invariant_coe, env_aware_coe=env_aware_coe, env_coe=env_coe, L2_coe=L2_coe,
            L1_coe=L1_coe, alpha=alpha, use_class_re_weight=use_class_re_weight, test_begin_epoch=test_begin_epoch,
            begin_cluster_epoch=begin_cluster_epoch, stop_cluster_epoch=stop_cluster_epoch,
            cluster_use_random_sort=cluster_use_random_sort, use_recommend_re_weight=use_recommend_re_weight
        )
        self.training_np: np.array = training_data.cpu().numpy()
        self.static_pop_interval: int = static_pop_interval

        self.data_loader: ImplicitBCELossDataLoaderStaticPopularity = data_loader

    def static_pop(self):
        users_cnt_weight_result: dict = dict()
        items_cnt_weight_result: dict = dict()

        users_normalize_cnt_weight_result: dict = dict()
        items_normalize_cnt_weight_result: dict = dict()

        users_cnt_result: dict = dict()
        items_cnt_result: dict = dict()

        users_normalize_cnt_result: dict = dict()
        items_normalize_cnt_result: dict = dict()

        pair_cnt_add_result: dict = dict()
        pair_normalize_cnt_multiply_result: dict = dict()

        for env in range(self.envs_num):
            select_indexes_np: np.array = (self.envs == env).cpu().numpy()
            select_samples: np.array = self.training_np[select_indexes_np]

            users_id: np.array = select_samples[:, 0].reshape(-1)
            items_id: np.array = select_samples[:, 1].reshape(-1)

            users_id_unique: np.array = np.unique(users_id)
            items_id_unique: np.array = np.unique(items_id)

            users_cnt_weight_mean = np.mean(self.data_loader.query_users_inter_cnt(users_id))
            items_cnt_weight_mean = np.mean(self.data_loader.query_items_inter_cnt(items_id))
            users_cnt_weight_result[env] = users_cnt_weight_mean
            items_cnt_weight_result[env] = items_cnt_weight_mean

            users_normalize_cnt_weight_mean = np.mean(self.data_loader.query_users_inter_cnt_normalize(users_id))
            items_normalize_cnt_weight_mean = np.mean(self.data_loader.query_items_inter_cnt_normalize(items_id))
            users_normalize_cnt_weight_result[env] = users_normalize_cnt_weight_mean
            items_normalize_cnt_weight_result[env] = items_normalize_cnt_weight_mean

            users_cnt_mean = np.mean(self.data_loader.query_users_inter_cnt(users_id_unique))
            items_cnt_mean = np.mean(self.data_loader.query_items_inter_cnt(items_id_unique))
            users_cnt_result[env] = users_cnt_mean
            items_cnt_result[env] = items_cnt_mean

            users_normalize_cnt_mean = np.mean(self.data_loader.query_users_inter_cnt_normalize(users_id_unique))
            items_normalize_cnt_mean = np.mean(self.data_loader.query_items_inter_cnt_normalize(items_id_unique))
            users_normalize_cnt_result[env] = users_normalize_cnt_mean
            items_normalize_cnt_result[env] = items_normalize_cnt_mean

            pair_cnt_add_mean = np.mean(self.data_loader.query_pairs_cnt_add(users_id, items_id))
            pair_normalize_cnt_multiply_mean \
                = np.mean(self.data_loader.query_pairs_cnt_normalize_multiply(users_id, items_id))
            pair_cnt_add_result[env] = pair_cnt_add_mean
            pair_normalize_cnt_multiply_result[env] = pair_normalize_cnt_multiply_mean

        final_result: dict = {
            'users_cnt_weight_result': users_cnt_weight_result,
            'items_cnt_weight_result': items_cnt_weight_result,
            'users_normalize_cnt_weight_result': users_normalize_cnt_weight_result,
            'items_normalize_cnt_weight_result': items_normalize_cnt_weight_result,
            'users_cnt_result': users_cnt_result,
            'items_cnt_result': items_cnt_result,
            'users_normalize_cnt_result': users_normalize_cnt_result,
            'items_normalize_cnt_result': items_normalize_cnt_result,
            'pair_cnt_add_result': pair_cnt_add_result,
            'pair_normalize_cnt_multiply_result': pair_normalize_cnt_multiply_result
        }

        return final_result

    def final_cluster_stat(self, colors_list: list):
        assert len(colors_list) == self.envs_num
        color_result: list = []
        user_cnt_result: list = []
        item_cnt_result: list = []
        user_cnt_normalize_result: list = []
        item_cnt_normalize_result: list = []

        for env in range(self.envs_num):
            select_indexes_np: np.array = (self.envs == env).cpu().numpy()
            select_samples: np.array = self.training_np[select_indexes_np]

            users_id: np.array = select_samples[:, 0].reshape(-1)
            items_id: np.array = select_samples[:, 1].reshape(-1)

            users_cnt: list = self.data_loader.query_users_inter_cnt(users_id).tolist()
            items_cnt: list = self.data_loader.query_items_inter_cnt(items_id).tolist()
            users_cnt_normalize: list = self.data_loader.query_users_inter_cnt_normalize(users_id).tolist()
            items_cnt_normalize: list = self.data_loader.query_items_inter_cnt_normalize(items_id).tolist()
            temp_color = [colors_list[env]] * select_samples.shape[0]
            color_result += temp_color
            user_cnt_result += users_cnt
            item_cnt_result += items_cnt
            user_cnt_normalize_result += users_cnt_normalize
            item_cnt_normalize_result += items_cnt_normalize

        return user_cnt_result, item_cnt_result, user_cnt_normalize_result, item_cnt_normalize_result, color_result

    def train(self, silent: bool = False, auto: bool = False):
        test_result_list: list = []
        test_epoch_list: list = []

        cluster_diff_num_list: list = []
        cluster_epoch_list: list = []
        envs_cnt_list: list = []

        loss_result_list: list = []
        train_epoch_index_list: list = []

        temp_eval_result: dict = self.evaluator.evaluate()
        test_result_list.append(temp_eval_result)
        test_epoch_list.append(self.epoch_cnt)

        stat_result_dicts_list: list = []
        stat_epoch_list: list = []

        self.stat_envs()

        if not silent and not auto:
            print('test at epoch:', self.epoch_cnt)
            print(transfer_loss_dict_to_line_str(temp_eval_result))

        while self.epoch_cnt < self.epochs:
            temp_loss_dict = self.train_a_epoch()
            train_epoch_index_list.append(self.epoch_cnt)
            loss_result_list.append(temp_loss_dict)
            if not silent and not auto:
                print('train epoch:', self.epoch_cnt)
                print(transfer_loss_dict_to_line_str(temp_loss_dict))

            if (self.epoch_cnt % self.evaluate_interval) == 0 and self.epoch_cnt >= self.test_begin_epoch:
                temp_eval_result: dict = self.evaluator.evaluate()
                test_result_list.append(temp_eval_result)
                test_epoch_list.append(self.epoch_cnt)

                if not silent and not auto:
                    print('test at epoch:', self.epoch_cnt)
                    print(transfer_loss_dict_to_line_str(temp_eval_result))

            if (self.epoch_cnt % self.cluster_interval) == 0:

                if (self.begin_cluster_epoch is None or self.begin_cluster_epoch <= self.epoch_cnt) \
                        and (self.stop_cluster_epoch is None or self.stop_cluster_epoch > self.epoch_cnt):
                    diff_num: int = self.cluster()
                    cluster_diff_num_list.append(diff_num)
                else:
                    diff_num: int = 0
                    cluster_diff_num_list.append(diff_num)

                envs_cnt: dict = self.stat_envs()

                cluster_epoch_list.append(self.epoch_cnt)
                envs_cnt_list.append(envs_cnt)

                if not silent and not auto:
                    print('cluster at epoch:', self.epoch_cnt)
                    print('diff num:', diff_num)
                    print(transfer_loss_dict_to_line_str(envs_cnt))

            if (self.epoch_cnt % self.static_pop_interval) == 0:
                pop_result: dict = self.static_pop()
                pop_json_str: str = json.dumps(pop_result, indent=4)

                stat_epoch_list.append(self.epoch_cnt)
                stat_result_dicts_list.append(pop_result)

                if not silent and not auto:
                    print('pop stat at epoch:', self.epoch_cnt)
                    print(pop_json_str)

        merged_stat_result: dict = merge_dict(stat_result_dicts_list, _show_me_a_list_func)
        inner_merged_stat_result: dict = dict()

        for key in merged_stat_result.keys():
            temp_dict_list: list = merged_stat_result[key]
            merged_temp_dict: dict = merge_dict(temp_dict_list, _show_me_a_list_func)
            inner_merged_stat_result[key] = merged_temp_dict

        return (loss_result_list, train_epoch_index_list), \
               (test_result_list, test_epoch_list), \
               (cluster_diff_num_list, envs_cnt_list, cluster_epoch_list), \
               (inner_merged_stat_result, stat_epoch_list)

class ImplicitTrainManager_KD:
    def __init__(
            self, T_model,S_model,
            T_evaluator,S_evaluator,device,
            training_data, batch_size,batch_size_KD,
            T_epochs: int, S_epochs: int, gama,
            cluster_interval: int,  evaluate_interval: int, lr: float,lr_KD,
            invariant_coe: float, env_aware_coe: float, variant_coe: float,env_coe: float, L2_coe: float, L1_coe: float,
            alpha: float = None, use_class_re_weight: bool = False, test_begin_epoch: int = 0,
            begin_cluster_epoch: int = None, stop_cluster_epoch: int = None, cluster_use_random_sort: bool = True,
            use_recommend_re_weight: bool = True
    ):
        self.T_model = T_model
        self.S_model = S_model
        self.T_evaluator = T_evaluator
        self.S_evaluator = S_evaluator
        self.envs_num = self.T_model.env_num
        self.device: torch.device = device
        self.users_tensor: torch.Tensor = training_data[:, 0]
        self.items_tensor: torch.Tensor = training_data[:, 1]
        self.scores_tensor: torch.Tensor = training_data[:, 2].float()
        self.envs: torch.LongTensor = torch.LongTensor(np.random.randint(0, self.envs_num, training_data.shape[0]))
        self.envs = self.envs.to(device)
        self.cluster_interval: int = cluster_interval
        self.evaluate_interval: int = evaluate_interval
        self.batch_size: int = batch_size
        self.batch_size_KD: int = batch_size_KD
        self.T_epochs: int = T_epochs
        self.S_epochs: int = S_epochs
        self.lr = lr
        self.lr_KD = lr_KD
        self.T_optimizer = torch.optim.Adam(T_model.parameters(), lr=self.lr)
        self.S_optimizer = torch.optim.Adam(S_model.parameters(), lr=self.lr_KD)
        self.gama = gama

        self.recommend_loss_type = nn.BCELoss
        self.cluster_distance_func = nn.BCELoss(reduction='none')
        self.env_loss_type = nn.NLLLoss

        self.invariant_coe = invariant_coe
        self.env_aware_coe = env_aware_coe
        self.variant_coe = variant_coe
        self.env_coe = env_coe
        self.L2_coe = L2_coe
        self.L1_coe = L1_coe

        self.epoch_cnt: int = 0

        self.batch_num = math.ceil(training_data.shape[0] / batch_size)

        self.each_env_count = dict()

        if alpha is None:
            self.alpha = 0.
            self.update_alpha = True
        else:
            self.alpha = alpha
            self.update_alpha = False

        self.use_class_re_weight: bool = use_class_re_weight
        self.use_recommend_re_weight: bool = use_recommend_re_weight
        self.sample_weights: torch.Tensor = torch.Tensor(np.zeros(training_data.shape[0])).to(device)
        self.class_weights: torch.Tensor = torch.Tensor(np.zeros(self.envs_num)).to(device)

        self.test_begin_epoch: int = test_begin_epoch

        self.begin_cluster_epoch: int = begin_cluster_epoch
        self.stop_cluster_epoch: int = stop_cluster_epoch

        self.eps_random_tensor: torch.Tensor = self._init_eps().to(self.device)

        self.cluster_use_random_sort: bool = cluster_use_random_sort

        self.const_env_tensor_list: list = []

        for env in range(self.envs_num):
            envs_tensor: torch.Tensor = torch.LongTensor(np.full(training_data.shape[0], env, dtype=int))
            envs_tensor = envs_tensor.to(self.device)
            self.const_env_tensor_list.append(envs_tensor)

    def _init_eps(self):
        base_eps = 1e-10
        eps_list: list = [base_eps * (1e-1 ** idx) for idx in range(self.envs_num)]
        temp: torch.Tensor = torch.Tensor(eps_list)
        eps_random_tensor: torch.Tensor = torch.Tensor(list(itertools.permutations(temp)))

        return eps_random_tensor

    def train_a_batch(
            self,
            batch_users_tensor: torch.Tensor,
            batch_items_tensor: torch.Tensor,
            batch_scores_tensor: torch.Tensor,
            batch_envs_tensor: torch.Tensor,
            batch_sample_weights: torch.Tensor,
            alpha,
    ) -> dict:

        invariant_score,variant_score, env_aware_score, env_outputs = self.T_model(
            batch_users_tensor, batch_items_tensor, batch_envs_tensor, alpha
        )

        if self.use_class_re_weight:
            env_loss = self.env_loss_type(reduction='none')
        else:
            env_loss = self.env_loss_type()

        if self.use_recommend_re_weight:
            recommend_loss = self.recommend_loss_type(reduction='none')
        else:
            recommend_loss = self.recommend_loss_type()


        invariant_loss: torch.Tensor = recommend_loss(invariant_score, batch_scores_tensor)
        variant_loss: torch.Tensor = recommend_loss(variant_score, batch_scores_tensor)
        env_aware_loss: torch.Tensor = recommend_loss(env_aware_score, batch_scores_tensor)

        envs_loss: torch.Tensor = env_loss(env_outputs, batch_envs_tensor)

        if self.use_class_re_weight:
            envs_loss = torch.mean(envs_loss * batch_sample_weights)

        if self.use_recommend_re_weight:
            invariant_loss = torch.mean(invariant_loss * batch_sample_weights)
            variant_loss = torch.mean(variant_loss * batch_sample_weights)
            env_aware_loss = torch.mean(env_aware_loss * batch_sample_weights)

        L2_reg: torch.Tensor = self.T_model.get_L2_reg(batch_users_tensor, batch_items_tensor, batch_envs_tensor)
        L1_reg: torch.Tensor = self.T_model.get_L1_reg(batch_users_tensor, batch_items_tensor, batch_envs_tensor)


        loss: torch.Tensor = invariant_loss * self.invariant_coe + env_aware_loss * self.env_aware_coe + variant_loss*self.variant_coe \
                             + envs_loss * self.env_coe + L2_reg * self.L2_coe + L1_reg * self.L1_coe

        self.T_optimizer.zero_grad()
        loss.backward()
        self.T_optimizer.step()

        loss_dict: dict = {
            'invariant_loss': float(invariant_loss),
            'variant_loss':float(variant_loss),
            'env_aware_loss': float(env_aware_loss),
            'envs_loss': float(envs_loss),
            'L2_reg': float(L2_reg),
            'L1_reg': float(L1_reg),
            'loss': float(loss),
        }
        return loss_dict

    def train_a_batch_KD(
            self,
            batch_users_tensor: torch.Tensor,
            batch_items_tensor: torch.Tensor,
            batch_scores_tensor: torch.Tensor,
            batch_envs_tensor: torch.Tensor,
            batch_sample_weights: torch.Tensor,
            alpha,
            gama
    ) -> dict:

        invariant_score, variant_score, env_aware_score, env_outputs = self.T_model(batch_users_tensor, batch_items_tensor, batch_envs_tensor, alpha)
        stu_score = self.S_model(batch_users_tensor, batch_items_tensor)

        invariant_loss = nn.MSELoss()(invariant_score, batch_scores_tensor)
        variant_loss = nn.MSELoss()(variant_score, batch_scores_tensor)

        if self.use_recommend_re_weight:
            invariant_loss = torch.mean(invariant_loss * batch_sample_weights)
            variant_loss = torch.mean(variant_loss * batch_sample_weights)

        W_inv = torch.pow(invariant_loss, gama) / (torch.pow(invariant_loss, gama) +  torch.pow(variant_loss, gama))
        W_var = torch.pow(variant_loss, gama) / (torch.pow(invariant_loss, gama) + torch.pow(variant_loss, gama))

        y_star = W_inv * invariant_score + W_var*variant_score

        student_loss = nn.MSELoss(reduction='mean')(stu_score,y_star)
        L2_reg: torch.Tensor = self.S_model.get_L2_reg(batch_users_tensor, batch_items_tensor)
        L1_reg: torch.Tensor = self.S_model.get_L1_reg(batch_users_tensor, batch_items_tensor)

        loss = student_loss*15 + L2_reg * 0.1 + L1_reg * 0.01

        self.S_optimizer.zero_grad()
        loss.backward()
        self.S_optimizer.step()

        loss_dict: dict = {
            'student_loss': float(student_loss),
            'L2_reg': float(L2_reg),
            'L1_reg': float(L1_reg),
            'loss': float(loss),
        }
        return loss_dict

    def cluster_a_batch(
            self,
            batch_users_tensor: torch.Tensor,
            batch_items_tensor: torch.Tensor,
            batch_scores_tensor: torch.Tensor,
    ) -> torch.Tensor:

        distances_list: list = []
        for env_idx in range(self.envs_num):
            envs_tensor: torch.Tensor = self.const_env_tensor_list[env_idx][0:batch_users_tensor.shape[0]]
            cluster_pred: torch.Tensor = self.T_model.cluster_predict(batch_users_tensor, batch_items_tensor, envs_tensor)
            distances: torch.Tensor = self.cluster_distance_func(cluster_pred, batch_scores_tensor)
            distances = distances.reshape(-1, 1)

            distances_list.append(distances)


        each_envs_distances: torch.Tensor = torch.cat(distances_list, dim=1)
        if self.cluster_use_random_sort:
            sort_random_index: np.array = \
                np.random.randint(0, self.eps_random_tensor.shape[0], each_envs_distances.shape[0])
            random_eps: torch.Tensor = self.eps_random_tensor[sort_random_index]
            each_envs_distances = each_envs_distances + random_eps

        new_envs: torch.Tensor = torch.argmin(each_envs_distances, dim=1)

        return new_envs

    def train_a_epoch(self) -> dict:
        self.T_model.train()
        loss_dicts_list: list = []

        for (batch_index, (
                batch_users_tensor, batch_items_tensor, batch_scores_tensor, batch_envs_tensor, batch_sample_weights
        )) \
                in enumerate(mini_batch(self.batch_size, self.users_tensor,
                                        self.items_tensor, self.scores_tensor, self.envs, self.sample_weights)):

            if self.update_alpha:
                p = float(batch_index + (self.epoch_cnt + 1) * self.batch_num) / float((self.epoch_cnt + 1)
                                                                                       * self.batch_num)
                self.alpha = 2. / (1. + np.exp(-10. * p)) - 1.

            loss_dict: dict = self.train_a_batch(
                batch_users_tensor=batch_users_tensor,
                batch_items_tensor=batch_items_tensor,
                batch_scores_tensor=batch_scores_tensor,
                batch_envs_tensor=batch_envs_tensor,
                batch_sample_weights=batch_sample_weights,
                alpha=self.alpha,
            )
            loss_dicts_list.append(loss_dict)

        self.epoch_cnt += 1

        mean_loss_dict: dict = merge_dict(loss_dicts_list, _mean_merge_dict_func)

        return mean_loss_dict

    def train_a_epoch_KD(self,gama) -> dict:
        self.S_model.train()
        loss_dicts_list: list = []

        for (batch_index, (
                batch_users_tensor, batch_items_tensor, batch_scores_tensor, batch_envs_tensor, batch_sample_weights
        )) \
                in enumerate(mini_batch(self.batch_size_KD, self.users_tensor,
                                        self.items_tensor, self.scores_tensor, self.envs, self.sample_weights)):

            loss_dict: dict = self.train_a_batch_KD(
                batch_users_tensor=batch_users_tensor,
                batch_items_tensor=batch_items_tensor,
                batch_scores_tensor=batch_scores_tensor,
                batch_envs_tensor=batch_envs_tensor,
                batch_sample_weights=batch_sample_weights,
                alpha=self.alpha,
                gama = gama
            )
            loss_dicts_list.append(loss_dict)

        self.epoch_cnt += 1

        mean_loss_dict: dict = merge_dict(loss_dicts_list, _mean_merge_dict_func)

        return mean_loss_dict

    def cluster(self) -> int:
        self.T_model.eval()

        new_env_tensors_list: list = []

        for (batch_index, (batch_users_tensor, batch_items_tensor, batch_scores_tensor)) \
                in enumerate(mini_batch(self.batch_size, self.users_tensor, self.items_tensor, self.scores_tensor)):

            new_env_tensor: torch.Tensor = self.cluster_a_batch(
                batch_users_tensor=batch_users_tensor,
                batch_items_tensor=batch_items_tensor,
                batch_scores_tensor=batch_scores_tensor
            )


            new_env_tensors_list.append(new_env_tensor)

        all_new_env_tensors: torch.Tensor = torch.cat(new_env_tensors_list, dim=0)
        envs_diff: torch.Tensor = (self.envs - all_new_env_tensors) != 0
        diff_num: int = int(torch.sum(envs_diff))
        self.envs = all_new_env_tensors
        return diff_num

    def update_each_env_count(self):
        result_dict: dict = {}
        for env in range(self.envs_num):
            cnt = torch.sum(self.envs == env)
            result_dict[env] = cnt
        self.each_env_count.update(result_dict)

    def stat_envs(self) -> dict:
        result: dict = dict()
        class_rate_np: np.array = np.zeros(self.envs_num)
        for env in range(self.envs_num):
            cnt: int = int(torch.sum(self.envs == env))
            result[env] = cnt
            class_rate_np[env] = min(cnt + 1, self.scores_tensor.shape[0] - 1)

        class_rate_np = class_rate_np / self.scores_tensor.shape[0]
        self.class_weights = torch.Tensor(class_rate_np).to(self.device)
        self.sample_weights = self.class_weights[self.envs]

        return result

    def train(self, silent: bool = False, auto: bool = False):
        test_result_list: list = []
        test_epoch_list: list = []

        cluster_diff_num_list: list = []
        cluster_epoch_list: list = []
        envs_cnt_list: list = []

        loss_result_list: list = []
        train_epoch_index_list: list = []

        temp_eval_result = self.T_evaluator.evaluate()
        test_result_list.append(temp_eval_result)
        test_epoch_list.append(self.epoch_cnt)

        self.stat_envs()

        if not silent and not auto:
            print('test at epoch:', self.epoch_cnt)
            print(transfer_loss_dict_to_line_str(temp_eval_result))

        stopping_args = Stop_args(patience=40, max_epochs=300)
        early_stopping = EarlyStopping(self.T_model, **stopping_args)
        #train teacher model
        print('***************************************Train Teacher**************************************************')
        while self.epoch_cnt < self.T_epochs:
            temp_loss_dict = self.train_a_epoch()
            train_epoch_index_list.append(self.epoch_cnt)
            loss_result_list.append(temp_loss_dict)
            if not silent and not auto:
                print('train epoch:', self.epoch_cnt)
                print(transfer_loss_dict_to_line_str(temp_loss_dict))

            if (self.epoch_cnt % self.evaluate_interval) == 0 and self.epoch_cnt >= self.test_begin_epoch:
                temp_eval_result: dict = self.T_evaluator.evaluate(['AUC'])
                test_result_list.append(temp_eval_result)
                test_epoch_list.append(self.epoch_cnt)

                if not silent and not auto:
                    print('test at epoch:', self.epoch_cnt)
                    print(transfer_loss_dict_to_line_str(temp_eval_result))

                if early_stopping.check([temp_eval_result['ndcg'][5]], self.epoch_cnt):
                    break

            if (self.epoch_cnt % self.cluster_interval) == 0:

                if (self.begin_cluster_epoch is None or self.begin_cluster_epoch <= self.epoch_cnt) \
                        and (self.stop_cluster_epoch is None or self.stop_cluster_epoch > self.epoch_cnt):
                    diff_num: int = self.cluster()
                    cluster_diff_num_list.append(diff_num)
                else:
                    diff_num: int = 0
                    cluster_diff_num_list.append(diff_num)

                envs_cnt: dict = self.stat_envs()

                cluster_epoch_list.append(self.epoch_cnt)
                envs_cnt_list.append(envs_cnt)

                if not silent and not auto:
                    print('cluster at epoch:', self.epoch_cnt)
                    print('diff num:', diff_num)
                    print(transfer_loss_dict_to_line_str(envs_cnt))

        # restore best model
        print('Loading {}th epoch'.format(early_stopping.best_epoch))
        print('best performance in teacher:',format(early_stopping.best_vals))
        self.T_model.load_state_dict(early_stopping.best_state)
        print('#'*120)

        # train Student model
        test_result_list = []
        test_epoch_list = []
        loss_result_list: list = []
        train_epoch_index_list: list = []
        self.epoch_cnt = 0

        #stopping_args_stu = Stop_args(patience=200, max_epochs=300)
        #early_stopping_stu = EarlyStopping(self.S_model, **stopping_args_stu)
        print('***************************************Train Student***************************************')
        while self.epoch_cnt < self.S_epochs:

            gama = self.gama + self.epoch_cnt/100
            if gama > 2: gama = 2

            temp_loss_dict = self.train_a_epoch_KD(gama=gama)
            train_epoch_index_list.append(self.epoch_cnt)
            loss_result_list.append(temp_loss_dict)
            if not silent and not auto:
                print('train epoch:', self.epoch_cnt)
                print(transfer_loss_dict_to_line_str(temp_loss_dict))

            if (self.epoch_cnt % 1) == 0 and self.epoch_cnt >= self.test_begin_epoch:
                temp_eval_result: dict = self.S_evaluator.evaluate()
                test_result_list.append(temp_eval_result)
                test_epoch_list.append(self.epoch_cnt)

                if not silent and not auto:
                    print('test at epoch:', self.epoch_cnt)
                    print(transfer_loss_dict_to_line_str(temp_eval_result))

        print('best performance in teacher:', format(early_stopping.best_vals))
        return (loss_result_list, train_epoch_index_list), (test_result_list, test_epoch_list)

class UniformImplicitTrainManager_KD:
    def __init__(
            self, T_model,S_model,
            T_val_evaluator, T_test_evaluator,
            S_val_evaluator, S_test_evaluator,device,
            train_norm_data,train_uniform_data,
            val_uniform_data,test_uniform_data,
            batch_size,batch_size_KD,
            T_epochs: int, S_epochs: int, gama,
            cluster_interval: int,  evaluate_interval: int, lr: float,lr_KD,
            invariant_coe: float, env_aware_coe: float, variant_coe: float,env_coe: float, L2_coe: float, L1_coe: float,
            student_coe,stu_L2_coe,stu_L1_coe,
            alpha: float = None, use_class_re_weight: bool = False, test_begin_epoch: int = 0,
            begin_cluster_epoch: int = None, stop_cluster_epoch: int = None, cluster_use_random_sort: bool = True,
            use_recommend_re_weight: bool = True
    ):
        self.T_model = T_model
        self.S_model = S_model
        self.T_val_evaluator = T_val_evaluator
        self.T_test_evaluator = T_test_evaluator
        self.S_val_evaluator = S_val_evaluator
        self.S_test_evaluator = S_test_evaluator
        self.envs_num = self.T_model.env_num
        self.device: torch.device = device
        self.norm_users_tensor: torch.Tensor = train_norm_data[:, 0]
        self.norm_items_tensor: torch.Tensor = train_norm_data[:, 1]
        self.norm_scores_tensor: torch.Tensor = train_norm_data[:, 2].float()
        self.uniform_users_tensor: torch.Tensor = train_uniform_data[:, 0]
        self.uniform_items_tensor: torch.Tensor = train_uniform_data[:, 1]
        self.uniform_scores_tensor: torch.Tensor = train_uniform_data[:, 2].float()
        self.envs: torch.LongTensor = torch.LongTensor(np.random.randint(0, self.envs_num, train_norm_data.shape[0]))
        self.envs = self.envs.to(device)
        self.cluster_interval: int = cluster_interval
        self.evaluate_interval: int = evaluate_interval
        self.batch_size: int = batch_size
        self.batch_size_KD: int = batch_size_KD
        self.T_epochs: int = T_epochs
        self.S_epochs: int = S_epochs
        self.lr = lr
        self.lr_KD = lr_KD
        self.T_optimizer = torch.optim.Adam(T_model.parameters(), lr=self.lr)
        self.S_optimizer = torch.optim.Adam(S_model.parameters(), lr=self.lr_KD)
        self.gama = gama

        self.recommend_loss_type = nn.BCELoss
        self.cluster_distance_func = nn.BCELoss(reduction='none')
        self.env_loss_type = nn.NLLLoss

        self.invariant_coe = invariant_coe
        self.env_aware_coe = env_aware_coe
        self.variant_coe = variant_coe
        self.env_coe = env_coe
        self.L2_coe = L2_coe
        self.L1_coe = L1_coe
        self.student_coe = student_coe
        self.stu_L2_coe = stu_L2_coe
        self.stu_L1_coe = stu_L1_coe

        self.epoch_cnt: int = 0

        self.batch_num = math.ceil(train_norm_data.shape[0] / batch_size)

        self.each_env_count = dict()

        if alpha is None:
            self.alpha = 0.
            self.update_alpha = True
        else:
            self.alpha = alpha
            self.update_alpha = False

        self.use_class_re_weight: bool = use_class_re_weight
        self.use_recommend_re_weight: bool = use_recommend_re_weight
        self.sample_weights: torch.Tensor = torch.Tensor(np.zeros(train_norm_data.shape[0])).to(device)
        self.class_weights: torch.Tensor = torch.Tensor(np.zeros(self.envs_num)).to(device)

        self.test_begin_epoch: int = test_begin_epoch

        self.begin_cluster_epoch: int = begin_cluster_epoch
        self.stop_cluster_epoch: int = stop_cluster_epoch

        self.eps_random_tensor: torch.Tensor = self._init_eps().to(self.device)

        self.cluster_use_random_sort: bool = cluster_use_random_sort

        self.const_env_tensor_list: list = []

        for env in range(self.envs_num):
            envs_tensor: torch.Tensor = torch.LongTensor(np.full(train_norm_data.shape[0], env, dtype=int))
            envs_tensor = envs_tensor.to(self.device)
            self.const_env_tensor_list.append(envs_tensor)

    def _init_eps(self):
        base_eps = 1e-10
        eps_list: list = [base_eps * (1e-1 ** idx) for idx in range(self.envs_num)]
        temp: torch.Tensor = torch.Tensor(eps_list)
        eps_random_tensor: torch.Tensor = torch.Tensor(list(itertools.permutations(temp)))

        return eps_random_tensor

    def train_a_batch(
            self,
            batch_users_tensor: torch.Tensor,
            batch_items_tensor: torch.Tensor,
            batch_scores_tensor: torch.Tensor,
            batch_envs_tensor: torch.Tensor,
            batch_sample_weights: torch.Tensor,
            alpha,
    ) -> dict:

        invariant_score,variant_score, env_aware_score, env_outputs = self.T_model(
            batch_users_tensor, batch_items_tensor, batch_envs_tensor, alpha
        )

        if self.use_class_re_weight:
            env_loss = self.env_loss_type(reduction='none')
        else:
            env_loss = self.env_loss_type()

        if self.use_recommend_re_weight:
            recommend_loss = self.recommend_loss_type(reduction='none')
        else:
            recommend_loss = self.recommend_loss_type()

        invariant_loss: torch.Tensor = recommend_loss(invariant_score, batch_scores_tensor)
        variant_loss: torch.Tensor = recommend_loss(variant_score, batch_scores_tensor)
        env_aware_loss: torch.Tensor = recommend_loss(env_aware_score, batch_scores_tensor)

        envs_loss: torch.Tensor = env_loss(env_outputs, batch_envs_tensor)

        if self.use_class_re_weight:
            envs_loss = torch.mean(envs_loss * batch_sample_weights)

        if self.use_recommend_re_weight:
            invariant_loss = torch.mean(invariant_loss * batch_sample_weights)
            variant_loss = torch.mean(variant_loss * batch_sample_weights)
            env_aware_loss = torch.mean(env_aware_loss * batch_sample_weights)

        L2_reg: torch.Tensor = self.T_model.get_L2_reg(batch_users_tensor, batch_items_tensor, batch_envs_tensor)
        L1_reg: torch.Tensor = self.T_model.get_L1_reg(batch_users_tensor, batch_items_tensor, batch_envs_tensor)


        loss: torch.Tensor = invariant_loss * self.invariant_coe + env_aware_loss * self.env_aware_coe + variant_loss*self.variant_coe \
                             + envs_loss * self.env_coe + L2_reg * self.L2_coe + L1_reg * self.L1_coe

        self.T_optimizer.zero_grad()
        loss.backward()
        self.T_optimizer.step()

        loss_dict: dict = {
            'invariant_loss': float(invariant_loss),
            'variant_loss':float(variant_loss),
            'env_aware_loss': float(env_aware_loss),
            'envs_loss': float(envs_loss),
            'L2_reg': float(L2_reg),
            'L1_reg': float(L1_reg),
            'loss': float(loss),
        }
        return loss_dict

    def train_a_batch_KD(
            self,
            batch_users_tensor: torch.Tensor,
            batch_items_tensor: torch.Tensor,
            batch_scores_tensor: torch.Tensor,
            batch_envs_tensor: torch.Tensor,
            batch_sample_weights: torch.Tensor,
            alpha,
            gama
    ) -> dict:

        invariant_score, variant_score, env_aware_score, env_outputs = self.T_model(batch_users_tensor, batch_items_tensor, batch_envs_tensor, alpha)
        stu_score = self.S_model(batch_users_tensor, batch_items_tensor)

        dis = abs(env_aware_score - invariant_score)
        W_inv = torch.pow(1-dis, gama)
        W_env = 1 - W_inv

        y_star = W_inv * invariant_score + W_env * env_aware_score

        student_loss = nn.MSELoss(reduction='mean')(stu_score,y_star)
        L2_reg: torch.Tensor = self.S_model.get_L2_reg(batch_users_tensor, batch_items_tensor)
        L1_reg: torch.Tensor = self.S_model.get_L1_reg(batch_users_tensor, batch_items_tensor)

        loss = student_loss*self.student_coe + L2_reg * self.stu_L2_coe + L1_reg * self.stu_L1_coe

        self.S_optimizer.zero_grad()
        loss.backward()
        self.S_optimizer.step()

        loss_dict: dict = {
            'student_loss': float(student_loss),
            'L2_reg': float(L2_reg),
            'L1_reg': float(L1_reg),
            'loss': float(loss),
        }
        return loss_dict

    def cluster_a_batch(
            self,
            batch_users_tensor: torch.Tensor,
            batch_items_tensor: torch.Tensor,
            batch_scores_tensor: torch.Tensor,
    ) -> torch.Tensor:
        distances_list: list = []
        for env_idx in range(self.envs_num):
            envs_tensor: torch.Tensor = self.const_env_tensor_list[env_idx][0:batch_users_tensor.shape[0]]
            cluster_pred: torch.Tensor = self.T_model.cluster_predict(batch_users_tensor, batch_items_tensor, envs_tensor)
            distances: torch.Tensor = self.cluster_distance_func(cluster_pred, batch_scores_tensor)
            distances = distances.reshape(-1, 1)
            distances_list.append(distances)

        each_envs_distances: torch.Tensor = torch.cat(distances_list, dim=1)

        if self.cluster_use_random_sort:
            sort_random_index: np.array = \
                np.random.randint(0, self.eps_random_tensor.shape[0], each_envs_distances.shape[0])
            random_eps: torch.Tensor = self.eps_random_tensor[sort_random_index]
            each_envs_distances = each_envs_distances + random_eps

        new_envs: torch.Tensor = torch.argmin(each_envs_distances, dim=1)

        return new_envs

    def train_a_epoch(self) -> dict:
        self.T_model.train()
        loss_dicts_list: list = []

        for (batch_index, (
                batch_users_tensor, batch_items_tensor, batch_scores_tensor, batch_envs_tensor, batch_sample_weights
        )) \
                in enumerate(mini_batch(self.batch_size, self.norm_users_tensor,
                                        self.norm_items_tensor, self.norm_scores_tensor, self.envs, self.sample_weights)):

            if self.update_alpha:
                p = float(batch_index + (self.epoch_cnt + 1) * self.batch_num) / float((self.epoch_cnt + 1)
                                                                                       * self.batch_num)
                self.alpha = 2. / (1. + np.exp(-10. * p)) - 1.

            loss_dict: dict = self.train_a_batch(
                batch_users_tensor=batch_users_tensor,
                batch_items_tensor=batch_items_tensor,
                batch_scores_tensor=batch_scores_tensor,
                batch_envs_tensor=batch_envs_tensor,
                batch_sample_weights=batch_sample_weights,
                alpha=self.alpha,
            )
            loss_dicts_list.append(loss_dict)

        self.epoch_cnt += 1

        mean_loss_dict: dict = merge_dict(loss_dicts_list, _mean_merge_dict_func)

        return mean_loss_dict

    def train_a_epoch_KD(self,gama):
        self.S_model.train()
        loss_dicts_list1: list = []
        loss_dicts_list2: list = []

        for (batch_index, (
                batch_users_tensor, batch_items_tensor, batch_scores_tensor, batch_envs_tensor, batch_sample_weights
        )) \
                in enumerate(mini_batch(self.batch_size_KD, self.norm_users_tensor,
                                        self.norm_items_tensor, self.norm_scores_tensor, self.envs, self.sample_weights)):

            loss_dict: dict = self.train_a_batch_KD(
                batch_users_tensor=batch_users_tensor,
                batch_items_tensor=batch_items_tensor,
                batch_scores_tensor=batch_scores_tensor,
                batch_envs_tensor=batch_envs_tensor,
                batch_sample_weights=batch_sample_weights,
                alpha=self.alpha,
                gama = gama
            )
            loss_dicts_list1.append(loss_dict)

        self.epoch_cnt += 1

        mean_loss_dict1: dict = merge_dict(loss_dicts_list1, _mean_merge_dict_func)

        return mean_loss_dict1

    def cluster(self) -> int:
        self.T_model.eval()

        new_env_tensors_list: list = []

        for (batch_index, (batch_users_tensor, batch_items_tensor, batch_scores_tensor)) \
                in enumerate(mini_batch(self.batch_size, self.norm_users_tensor, self.norm_items_tensor, self.norm_scores_tensor)):

            new_env_tensor: torch.Tensor = self.cluster_a_batch(
                batch_users_tensor=batch_users_tensor,
                batch_items_tensor=batch_items_tensor,
                batch_scores_tensor=batch_scores_tensor
            )


            new_env_tensors_list.append(new_env_tensor)

        all_new_env_tensors: torch.Tensor = torch.cat(new_env_tensors_list, dim=0)
        envs_diff: torch.Tensor = (self.envs - all_new_env_tensors) != 0
        diff_num: int = int(torch.sum(envs_diff))
        self.envs = all_new_env_tensors
        return diff_num

    def update_each_env_count(self):
        result_dict: dict = {}
        for env in range(self.envs_num):
            cnt = torch.sum(self.envs == env)
            result_dict[env] = cnt
        self.each_env_count.update(result_dict)

    def stat_envs(self) -> dict:
        result: dict = dict()
        class_rate_np: np.array = np.zeros(self.envs_num)
        for env in range(self.envs_num):
            cnt: int = int(torch.sum(self.envs == env))
            result[env] = cnt
            class_rate_np[env] = min(cnt + 1, self.norm_scores_tensor.shape[0] - 1)

        class_rate_np = class_rate_np / self.norm_scores_tensor.shape[0]
        self.class_weights = torch.Tensor(class_rate_np).to(self.device)
        self.sample_weights = self.class_weights[self.envs]

        return result

    def train(self, silent: bool = False, auto: bool = False):
        test_result_list: list = []
        test_epoch_list: list = []

        cluster_diff_num_list: list = []
        cluster_epoch_list: list = []
        envs_cnt_list: list = []

        loss_result_list: list = []
        train_epoch_index_list: list = []

        temp_eval_result = self.T_val_evaluator.evaluate(['AUC'])
        test_result_list.append(temp_eval_result)
        test_epoch_list.append(self.epoch_cnt)

        self.stat_envs()

        if not silent and not auto:
            print('test at epoch:', self.epoch_cnt)
            print(transfer_loss_dict_to_line_str(temp_eval_result))

        stopping_args = Stop_args(patience=80, max_epochs=300)
        early_stopping = EarlyStopping(self.T_model, **stopping_args)
        #train teacher model
        print('***************************************Train teacher**************************************************')
        while self.epoch_cnt < self.T_epochs:
            temp_loss_dict = self.train_a_epoch()
            train_epoch_index_list.append(self.epoch_cnt)
            loss_result_list.append(temp_loss_dict)
            if not silent and not auto:
                print('train epoch:', self.epoch_cnt)
                print(transfer_loss_dict_to_line_str(temp_loss_dict))

            if (self.epoch_cnt % self.evaluate_interval) == 0 and self.epoch_cnt >= self.test_begin_epoch:
                temp_eval_result: dict = self.T_val_evaluator.evaluate(['AUC'])
                test_result_list.append(temp_eval_result)
                test_epoch_list.append(self.epoch_cnt)

                if not silent and not auto:
                    print('val at epoch:', self.epoch_cnt)
                    print(transfer_loss_dict_to_line_str(temp_eval_result))

                if early_stopping.check([temp_eval_result['AUC']], self.epoch_cnt):
                    break

            if (self.epoch_cnt % self.cluster_interval) == 0:

                if (self.begin_cluster_epoch is None or self.begin_cluster_epoch <= self.epoch_cnt) \
                        and (self.stop_cluster_epoch is None or self.stop_cluster_epoch > self.epoch_cnt):
                    diff_num: int = self.cluster()
                    cluster_diff_num_list.append(diff_num)
                else:
                    diff_num: int = 0
                    cluster_diff_num_list.append(diff_num)

                envs_cnt: dict = self.stat_envs()

                cluster_epoch_list.append(self.epoch_cnt)
                envs_cnt_list.append(envs_cnt)

                if not silent and not auto:
                    print('cluster at epoch:', self.epoch_cnt)
                    print('diff num:', diff_num)
                    print(transfer_loss_dict_to_line_str(envs_cnt))

        # restore best model
        #print('#' * 120)
        print('Loading {}th epoch'.format(early_stopping.best_epoch))
        self.T_model.load_state_dict(early_stopping.best_state)
        final_eval_result = self.T_test_evaluator.evaluate(['AUC','Recall_Precision_NDCG@'])
        print('best performance in inv teacher:',format(final_eval_result))
        print('*'*120)

        # train Student model
        test_result_list = []
        test_epoch_list = []
        loss_result_list1: list = []
        train_epoch_index_list: list = []
        self.epoch_cnt = 0

        stopping_args = Stop_args(patience=80, max_epochs=300)
        early_stopping = EarlyStopping(self.S_model, **stopping_args)
        print('***************************************Train Student***************************************')
        while self.epoch_cnt < self.S_epochs:

            temp_loss_dict1 = self.train_a_epoch_KD(gama=self.gama)
            train_epoch_index_list.append(self.epoch_cnt)
            loss_result_list1.append(temp_loss_dict1)
            if not silent and not auto:
                print('train epoch:', self.epoch_cnt)
                print(transfer_loss_dict_to_line_str(temp_loss_dict1))

            if (self.epoch_cnt % 1) == 0 and self.epoch_cnt >= self.test_begin_epoch:
                temp_eval_result: dict = self.S_val_evaluator.evaluate(['AUC'])
                test_result_list.append(temp_eval_result)
                test_epoch_list.append(self.epoch_cnt)

                if not silent and not auto:
                    print('val at epoch:', self.epoch_cnt)
                    print(transfer_loss_dict_to_line_str(temp_eval_result))

                if early_stopping.check([temp_eval_result['AUC']], self.epoch_cnt):
                    break
       
        #print('#' * 120)
        print('Loading {}th epoch'.format(early_stopping.best_epoch))
        self.S_model.load_state_dict(early_stopping.best_state)
        final_eval_result = self.S_test_evaluator.evaluate(['AUC','Recall_Precision_NDCG@'])
        print('best performance in student:', format(final_eval_result))
        print('*' * 120)

        return final_eval_result