import copy
import math

from torch import nn
import torch
import numpy as np

from functions import ReverseLayerF

class LinearLogSoftMaxEnvClassifier(nn.Module):
    def __init__(self, factor_dim, env_num):
        super(LinearLogSoftMaxEnvClassifier, self).__init__()
        self.linear_map= nn.Linear(factor_dim, env_num)
        self.classifier_func = nn.LogSoftmax(dim=1)
        self._init_weight()
        self.elements_num = float(factor_dim * env_num)
        self.bias_num = float(env_num)

    def forward(self, invariant_preferences):
        result = self.linear_map(invariant_preferences)
        result = self.classifier_func(result)
        return result

    def get_L1_reg(self):
        return torch.norm(self.linear_map.weight, 1) / self.elements_num \
               + torch.norm(self.linear_map.bias, 1) / self.bias_num

    def get_L2_reg(self):
        return torch.norm(self.linear_map.weight, 2).pow(2) / self.elements_num \
               + torch.norm(self.linear_map.bias, 2).pow(2) / self.bias_num

    def _init_weight(self):
        torch.nn.init.xavier_uniform_(self.linear_map.weight)

class InvPrefImplicit(nn.Module):
    def __init__(
            self, user_num, item_num, env_num, factor_num, reg_only_embed = False,
            reg_env_embed = True
    ):
        super(InvPrefImplicit, self).__init__()
        self.env_num = env_num
        self.factor_num = factor_num

        self.embed_user_invariant = nn.Embedding(user_num, factor_num)
        self.embed_item_invariant = nn.Embedding(item_num, factor_num)

        self.embed_user_env_aware = nn.Embedding(user_num, factor_num)
        self.embed_item_env_aware = nn.Embedding(item_num, factor_num)

        self.embed_env = nn.Embedding(env_num, factor_num)

        self.env_classifier = LinearLogSoftMaxEnvClassifier(factor_num, env_num)
        self.output_func = nn.Sigmoid()

        self.reg_only_embed = reg_only_embed

        self.reg_env_embed = reg_env_embed

        self._init_weight()

    def _init_weight(self):
        nn.init.normal_(self.embed_user_invariant.weight, std=0.01)
        nn.init.normal_(self.embed_item_invariant.weight, std=0.01)
        nn.init.normal_(self.embed_user_env_aware.weight, std=0.01)
        nn.init.normal_(self.embed_item_env_aware.weight, std=0.01)
        nn.init.normal_(self.embed_env.weight, std=0.01)

    def forward(self, users_id, items_id, envs_id, alpha):
        users_embed_invariant = self.embed_user_invariant(users_id)
        items_embed_invariant = self.embed_item_invariant(items_id)

        users_embed_env_aware = self.embed_user_env_aware(users_id)
        items_embed_env_aware = self.embed_item_env_aware(items_id)

        envs_embed = self.embed_env(envs_id)

        invariant_preferences = users_embed_invariant * items_embed_invariant
        variant_preferences = users_embed_env_aware * items_embed_env_aware
        env_aware_preferences = variant_preferences * envs_embed

        invariant_score = self.output_func(torch.sum(invariant_preferences, dim=1))
        variant_score = self.output_func(torch.sum(variant_preferences, dim=1))
        env_aware_mid_score = self.output_func(torch.sum(env_aware_preferences, dim=1))
        env_aware_score = invariant_score * env_aware_mid_score

        reverse_invariant_preferences = ReverseLayerF.apply(invariant_preferences, alpha)
        env_outputs = self.env_classifier(reverse_invariant_preferences)

        return invariant_score.reshape(-1), env_aware_score.reshape(-1), env_outputs.reshape(-1, self.env_num)

    def cal_env_score(self, users_id, items_id):
        users_embed_env_aware = self.embed_user_env_aware(users_id)
        items_embed_env_aware = self.embed_item_env_aware(items_id)

        env_aware_preferences = users_embed_env_aware * items_embed_env_aware

        env_aware_mid_score = self.output_func(torch.sum(env_aware_preferences, dim=1))

        return env_aware_mid_score.reshape(-1)

    def get_users_reg(self, users_id, norm):
        invariant_embed_gmf = self.embed_user_invariant(users_id)
        env_aware_embed_gmf = self.embed_user_env_aware(users_id)
        if norm == 2:
            reg_loss = \
                (env_aware_embed_gmf.norm(2).pow(2) + invariant_embed_gmf.norm(2).pow(2)) \
                / (float(len(users_id)) * float(self.factor_num) * 2)
        elif norm == 1:
            reg_loss = \
                (env_aware_embed_gmf.norm(1) + invariant_embed_gmf.norm(1)) \
                / (float(len(users_id)) * float(self.factor_num) * 2)
        else:
            raise KeyError('norm must be 1 or 2')
        return reg_loss

    def get_items_reg(self, items_id, norm):
        invariant_embed_gmf = self.embed_item_invariant(items_id)
        env_aware_embed_gmf = self.embed_item_env_aware(items_id)
        if norm == 2:
            reg_loss = \
                (env_aware_embed_gmf.norm(2).pow(2) + invariant_embed_gmf.norm(2).pow(2)) \
                / (float(len(items_id)) * float(self.factor_num) * 2)
        elif norm == 1:
            reg_loss = \
                (env_aware_embed_gmf.norm(1) + invariant_embed_gmf.norm(1)) \
                / (float(len(items_id)) * float(self.factor_num) * 2)
        else:
            raise KeyError('norm must be 1 or 2')
        return reg_loss

    def get_envs_reg(self, envs_id, norm):
        embed_gmf = self.embed_env(envs_id)
        if norm == 2:
            reg_loss = embed_gmf.norm(2).pow(2) / (float(len(envs_id)) * float(self.factor_num))
        elif norm == 1:
            reg_loss = embed_gmf.norm(1) / (float(len(envs_id)) * float(self.factor_num))
        else:
            raise KeyError('norm must be 1 or 2')
        return reg_loss

    def get_L2_reg(self, users_id, items_id, envs_id):
        if not self.reg_only_embed:
            result = self.env_classifier.get_L2_reg()
            result = result + (
                    self.get_users_reg(users_id, 2) + self.get_items_reg(items_id, 2))
        else:
            result = self.get_users_reg(users_id, 2) + self.get_items_reg(items_id, 2)

        if self.reg_env_embed:
            result = result + self.get_envs_reg(envs_id, 2)
        # print('L2', result)
        return result

    def get_L1_reg(self, users_id, items_id, envs_id):
        if not self.reg_only_embed:
            result = self.env_classifier.get_L1_reg()
            result = result + (
                    self.get_users_reg(users_id, 1) + self.get_items_reg(items_id, 1))
        else:
            result = self.get_users_reg(users_id, 1) + self.get_items_reg(items_id, 1)
        if self.reg_env_embed:
            result = result + self.get_envs_reg(envs_id, 1)
        return result

    def predict(self, users_id,items_id):
        users_embed_inv = self.embed_user_invariant(users_id)
        items_embed_inv = self.embed_item_invariant(items_id)

        invariant_preferences = users_embed_inv * items_embed_inv
        invariant_score = self.output_func(torch.sum(invariant_preferences, dim=1))

        final_score = invariant_score

        return final_score.reshape(-1)

    def cluster_predict(self, users_id, items_id, envs_id) -> torch.Tensor:
        _,env_aware_score, _ = self.forward(users_id, items_id, envs_id, 0.)
        return env_aware_score

    def allrank(self, users, trained_user_interaction):
        users_embed_inv = self.embed_user_invariant(users)
        items_embed_inv = self.embed_item_invariant.weight

        users_embed_inv = users_embed_inv.unsqueeze(1).repeat(1,items_embed_inv.shape[0], 1)
        items_embed_inv = items_embed_inv.unsqueeze(0).repeat(users_embed_inv.shape[0], 1, 1)

        invariant_preferences = users_embed_inv * items_embed_inv
        preds = self.output_func(torch.sum(invariant_preferences, dim=2))

        for i,u in enumerate(set(users)):
            preds[i,list(trained_user_interaction[u])] = -999.0
        return preds

class InvPrefImplicit_changed(nn.Module):
    def __init__(
            self, user_num, item_num, env_num, factor_num, reg_only_embed = False,
            reg_env_embed = True
    ):
        super(InvPrefImplicit_changed, self).__init__()
        self.env_num = env_num
        self.factor_num = factor_num

        self.embed_user_invariant = nn.Embedding(user_num, factor_num)
        self.embed_item_invariant = nn.Embedding(item_num, factor_num)

        self.embed_user_env_aware = nn.Embedding(user_num, factor_num)
        self.embed_item_env_aware = nn.Embedding(item_num, factor_num)

        self.embed_env = nn.Embedding(env_num, factor_num)

        self.env_classifier = LinearLogSoftMaxEnvClassifier(factor_num, env_num)
        self.output_func = nn.Sigmoid()

        self.reg_only_embed = reg_only_embed

        self.reg_env_embed = reg_env_embed

        self._init_weight()

    def _init_weight(self):
        nn.init.normal_(self.embed_user_invariant.weight, std=0.01)
        nn.init.normal_(self.embed_item_invariant.weight, std=0.01)
        nn.init.normal_(self.embed_user_env_aware.weight, std=0.01)
        nn.init.normal_(self.embed_item_env_aware.weight, std=0.01)
        nn.init.normal_(self.embed_env.weight, std=0.01)

    def forward(self, users_id, items_id, envs_id, alpha):
        users_embed_invariant = self.embed_user_invariant(users_id)
        items_embed_invariant = self.embed_item_invariant(items_id)

        users_embed_env_aware = self.embed_user_env_aware(users_id)
        items_embed_env_aware = self.embed_item_env_aware(items_id)

        envs_embed = self.embed_env(envs_id)

        invariant_preferences = users_embed_invariant * items_embed_invariant
        variant_preferences = users_embed_env_aware * items_embed_env_aware
        env_aware_preferences = variant_preferences * envs_embed

        invariant_score = self.output_func(torch.sum(invariant_preferences, dim=1))
        variant_score = self.output_func(torch.sum(variant_preferences, dim=1))
        env_aware_mid_score = self.output_func(torch.sum(env_aware_preferences, dim=1))
        env_aware_score = invariant_score * env_aware_mid_score

        reverse_invariant_preferences = ReverseLayerF.apply(invariant_preferences, alpha)
        env_outputs = self.env_classifier(reverse_invariant_preferences)

        return invariant_score.reshape(-1), variant_score.reshape(-1), env_aware_score.reshape(-1), env_outputs.reshape(-1, self.env_num)

    def cal_env_score(self, users_id, items_id):
        users_embed_env_aware = self.embed_user_env_aware(users_id)
        items_embed_env_aware = self.embed_item_env_aware(items_id)

        env_aware_preferences = users_embed_env_aware * items_embed_env_aware

        env_aware_mid_score = self.output_func(torch.sum(env_aware_preferences, dim=1))

        return env_aware_mid_score.reshape(-1)

    def get_users_reg(self, users_id, norm):
        invariant_embed_gmf = self.embed_user_invariant(users_id)
        env_aware_embed_gmf = self.embed_user_env_aware(users_id)
        if norm == 2:
            reg_loss = \
                (env_aware_embed_gmf.norm(2).pow(2) + invariant_embed_gmf.norm(2).pow(2)) \
                / (float(len(users_id)) * float(self.factor_num) * 2)
        elif norm == 1:
            reg_loss = \
                (env_aware_embed_gmf.norm(1) + invariant_embed_gmf.norm(1)) \
                / (float(len(users_id)) * float(self.factor_num) * 2)
        else:
            raise KeyError('norm must be 1 or 2')
        return reg_loss

    def get_items_reg(self, items_id, norm):
        invariant_embed_gmf = self.embed_item_invariant(items_id)
        env_aware_embed_gmf = self.embed_item_env_aware(items_id)
        if norm == 2:
            reg_loss = \
                (env_aware_embed_gmf.norm(2).pow(2) + invariant_embed_gmf.norm(2).pow(2)) \
                / (float(len(items_id)) * float(self.factor_num) * 2)
        elif norm == 1:
            reg_loss = \
                (env_aware_embed_gmf.norm(1) + invariant_embed_gmf.norm(1)) \
                / (float(len(items_id)) * float(self.factor_num) * 2)
        else:
            raise KeyError('norm must be 1 or 2')
        return reg_loss

    def get_envs_reg(self, envs_id, norm):
        embed_gmf = self.embed_env(envs_id)
        if norm == 2:
            reg_loss = embed_gmf.norm(2).pow(2) / (float(len(envs_id)) * float(self.factor_num))
        elif norm == 1:
            reg_loss = embed_gmf.norm(1) / (float(len(envs_id)) * float(self.factor_num))
        else:
            raise KeyError('norm must be 1 or 2')
        return reg_loss

    def get_L2_reg(self, users_id, items_id, envs_id):
        if not self.reg_only_embed:
            result = self.env_classifier.get_L2_reg()
            result = result + (
                    self.get_users_reg(users_id, 2) + self.get_items_reg(items_id, 2))
        else:
            result = self.get_users_reg(users_id, 2) + self.get_items_reg(items_id, 2)

        if self.reg_env_embed:
            result = result + self.get_envs_reg(envs_id, 2)
        # print('L2', result)
        return result

    def get_L1_reg(self, users_id, items_id, envs_id):
        if not self.reg_only_embed:
            result = self.env_classifier.get_L1_reg()
            result = result + (
                    self.get_users_reg(users_id, 1) + self.get_items_reg(items_id, 1))
        else:
            result = self.get_users_reg(users_id, 1) + self.get_items_reg(items_id, 1)
        if self.reg_env_embed:
            result = result + self.get_envs_reg(envs_id, 1)
        return result

    def predict(self, users_id,items_id):
        users_embed_inv = self.embed_user_invariant(users_id)
        items_embed_inv = self.embed_item_invariant(items_id)

        invariant_preferences = users_embed_inv * items_embed_inv
        invariant_score = self.output_func(torch.sum(invariant_preferences, dim=1))

        final_score = invariant_score

        return final_score.reshape(-1)

    def cluster_predict(self, users_id, items_id, envs_id) -> torch.Tensor:
        _, _,env_aware_score, _ = self.forward(users_id, items_id, envs_id, 0.)
        return env_aware_score

    def allrank(self, users, trained_user_interaction):
        users_embed_inv = self.embed_user_invariant(users)
        items_embed_inv = self.embed_item_invariant.weight

        users_embed_inv = users_embed_inv.unsqueeze(1).repeat(1,items_embed_inv.shape[0], 1)
        items_embed_inv = items_embed_inv.unsqueeze(0).repeat(users_embed_inv.shape[0], 1, 1)

        invariant_preferences = users_embed_inv * items_embed_inv
        preds = self.output_func(torch.sum(invariant_preferences, dim=2))

        for i,u in enumerate(set(users)):
            preds[i,list(trained_user_interaction[u])] = -999.0
        return preds

class KD_Debias_student(nn.Module):
    def __init__(self, user_num, item_num, factor_num):
        super(KD_Debias_student, self).__init__()
        self.factor_num = factor_num
        self.user_emb = nn.Embedding(user_num, self.factor_num)
        self.item_emb = nn.Embedding(item_num, self.factor_num)
        self.output_func = nn.Sigmoid()

        self._init_weight()

    def _init_weight(self):
        nn.init.normal_(self.user_emb.weight,std=0.01)
        nn.init.normal_(self.item_emb.weight,std=0.01)

    def forward(self, users_id, items_id, ground_truth = None):
        users_emb= self.user_emb(users_id)
        items_emb = self.item_emb(items_id)

        ratings = torch.sum(users_emb * items_emb, dim=1)
        final_ratings = self.output_func(ratings)

        return final_ratings

    def get_users_reg(self, users_id, norm):
        embed_gmf = self.user_emb(users_id)
        if norm == 2:
            reg_loss = embed_gmf.norm(2).pow(2) / (float(len(users_id)) * float(self.factor_num))
            #reg_loss = torch.sum(embed_gmf ** 2)
        elif norm == 1:
            reg_loss = embed_gmf.norm(1) / (float(len(users_id)) * float(self.factor_num))
        else:
            raise KeyError('norm must be 1 or 2')
        return reg_loss

    def get_items_reg(self, items_id, norm):
        embed_gmf = self.item_emb(items_id)
        if norm == 2:
            reg_loss = embed_gmf.norm(2).pow(2) / (float(len(items_id)) * float(self.factor_num))
            #reg_loss = torch.sum(embed_gmf ** 2)
        elif norm == 1:
            reg_loss = embed_gmf.norm(1) / (float(len(items_id)) * float(self.factor_num))
        else:
            raise KeyError('norm must be 1 or 2')
        return reg_loss

    def get_L1_reg(self, users_id, items_id):
        return self.get_users_reg(users_id, 1) + self.get_items_reg(items_id, 1)

    def get_L2_reg(self, users_id, items_id):
        return self.get_users_reg(users_id, 2) + self.get_items_reg(items_id, 2)

    def predict(self, users_id,items_id):
        users_emb = self.user_emb(users_id)
        items_emb = self.item_emb(items_id)

        ratings = self.output_func(torch.sum(users_emb*items_emb, dim=1))

        return ratings