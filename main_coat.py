import json
import random

import global_config
from dataloader import CoatImplicitDataset,Interactions,DataLoader

import torch
import numpy as np
from evaluate import ImplicitTestManager
from model import InvPrefImplicit_changed,KD_Debias_student
from train import UniformImplicitTrainManager_KD
from utils import draw_score_pic, merge_dict, _show_me_a_list_func, draw_loss_pic_one_by_one, query_user, query_str, \
    mkdir, query_int, get_class_name_str, _mean_merge_dict_func, show_me_all_the_fucking_result

torch.cuda.set_device(0)
DEVICE: torch.device = torch.device('cuda')


#best performance in teacher
MODEL_CONFIG: dict = {
    'env_num': 2,
    'factor_num': 40,
    'reg_only_embed': True,
    'reg_env_embed': False
}
#best performance in teacher
TRAIN_CONFIG: dict = {
    "batch_size": 1024,"batch_size_KD": 8196,
    "T_epochs": 300,"S_epochs": 500,
    "cluster_interval": 5,"evaluate_interval": 1,
    "lr": 0.002,"lr_KD":0.005,
    "gama":0.3,
    "invariant_coe": 1,
    "env_aware_coe": 15,
    "variant_coe": 5,
    "env_coe": 0.5,
    "L2_coe": 0.7,
    "L1_coe": 0.3,
    'student_coe': 1,
    'stu_L2_coe': 0.1,
    'stu_L1_coe': 0.01,
    "alpha": 3,
    "use_class_re_weight": True,
    "use_recommend_re_weight": False,
    "test_begin_epoch": 0,
    "begin_cluster_epoch": None,
    "stop_cluster_epoch": None
}
print(TRAIN_CONFIG)
EVALUATE_CONFIG = {
    'top_k_list': [5],
    'test_batch_size': 1024,
    'eval_k': 5,
    'eval_metric': 'ndcg'
}


RANDOM_SEED_LIST = [13051307,17354622,7422964,15522123,18666365,13346594,16769482]
'''
for i in range(10):
   RANDOM_SEED_LIST.append(random.randint(0,20000000))
'''
print(RANDOM_SEED_LIST)
DATASET_PATH = '/coat/'
METRIC_LIST = ['ndcg', 'recall']


def main(
        device: torch.device,
        model_config,
        train_config,
        evaluate_config,
        dataset,
        random_seed,
        silent = True,
        auto = False,
):
    torch.manual_seed(random_seed)
    torch.cuda.manual_seed(random_seed)
    torch.cuda.manual_seed_all(random_seed)
    np.random.seed(random_seed)

    Inv_model = InvPrefImplicit_changed(
        user_num=dataset.user_num,
        item_num=dataset.item_num,
        env_num=model_config['env_num'],
        factor_num=model_config['factor_num'],
        reg_only_embed=model_config['reg_only_embed'],
        reg_env_embed=model_config['reg_env_embed']
    )
    Inv_model = Inv_model.to(device)

    KD_model = KD_Debias_student(
        user_num=dataset.user_num,
        item_num=dataset.item_num,
        factor_num=model_config['factor_num'],
    )
    KD_model = KD_model.to(device)

    val_loader = DataLoader(Interactions(dataset._uniform_val_data, dataset.user_interaction), batch_size=EVALUATE_CONFIG['test_batch_size'],shuffle=False)
    test_loader = DataLoader(Interactions(dataset._uniform_test_data,dataset.user_interaction),batch_size=EVALUATE_CONFIG['test_batch_size'],shuffle=False)

    Inv_val_evaluator = ImplicitTestManager(
        model=Inv_model,
        data_loader=val_loader,
        test_batch_size=evaluate_config['test_batch_size'],
        top_k_list=evaluate_config['top_k_list'],
        use_item_pool=True
    )

    Inv_test_evaluator = ImplicitTestManager(
        model=Inv_model,
        data_loader=test_loader,
        test_batch_size=evaluate_config['test_batch_size'],
        top_k_list=evaluate_config['top_k_list'],
        use_item_pool=True
    )

    KD_val_evaluator = ImplicitTestManager(
        model=KD_model,
        data_loader=val_loader,
        test_batch_size=evaluate_config['test_batch_size'],
        top_k_list=evaluate_config['top_k_list'],
        use_item_pool=True
    )

    KD_test_evaluator = ImplicitTestManager(
        model=KD_model,
        data_loader=test_loader,
        test_batch_size=evaluate_config['test_batch_size'],
        top_k_list=evaluate_config['top_k_list'],
        use_item_pool=True
    )

    train_norm_data = torch.LongTensor(dataset._norm_train_data).to(device)
    train_uniform_data = torch.LongTensor(dataset._uniform_train_data).to(device)
    val_uniform_data = torch.LongTensor(dataset._uniform_val_data).to(device)
    test_uniform_data = torch.LongTensor(dataset._uniform_test_data).to(device)

    train_manager = UniformImplicitTrainManager_KD(
        T_model=Inv_model, S_model=KD_model,
        T_val_evaluator=Inv_val_evaluator, T_test_evaluator=Inv_test_evaluator,
        S_val_evaluator=KD_val_evaluator,S_test_evaluator=KD_test_evaluator,
        gama=train_config['gama'],device=device,
        train_norm_data = train_norm_data,train_uniform_data = train_uniform_data,
        val_uniform_data = val_uniform_data,test_uniform_data = test_uniform_data,
        batch_size=train_config['batch_size'],batch_size_KD=train_config['batch_size_KD'],
        T_epochs=train_config['T_epochs'],S_epochs=train_config['S_epochs'],
        cluster_interval=train_config['cluster_interval'],
        evaluate_interval=train_config['evaluate_interval'],
        lr=train_config['lr'], lr_KD=train_config['lr_KD'],invariant_coe=train_config['invariant_coe'],
        env_aware_coe=train_config['env_aware_coe'], variant_coe=train_config['variant_coe'], env_coe=train_config['env_coe'],
        L2_coe=train_config['L2_coe'], L1_coe=train_config['L1_coe'],
        student_coe = train_config['student_coe'],stu_L2_coe = train_config['stu_L2_coe'],stu_L1_coe = train_config['stu_L1_coe'],
        alpha=train_config['alpha'],use_class_re_weight=train_config['use_class_re_weight'],
        test_begin_epoch=train_config['test_begin_epoch'], begin_cluster_epoch=train_config['begin_cluster_epoch'],
        stop_cluster_epoch=train_config['stop_cluster_epoch'],
        use_recommend_re_weight=train_config['use_recommend_re_weight']
    )

    best_performance = train_manager.train(silent=silent, auto=auto)

    return best_performance


if __name__ == '__main__':

    best_metric_perform_list: list = []
    all_metric_results_list: list = []

    for seed in RANDOM_SEED_LIST:
        print()
        print('Begin seed:', seed)
        dataset = CoatImplicitDataset(
            dataset_path=global_config.DATASET_PATH + DATASET_PATH,seed = seed
        )
        best_perform = main(
            device=DEVICE, model_config=MODEL_CONFIG, train_config=TRAIN_CONFIG,evaluate_config=EVALUATE_CONFIG,
            dataset=dataset,random_seed=seed
        )

        best_metric_perform_list.append(best_perform)

    result_merged_by_metric = merge_dict(best_metric_perform_list, _show_me_a_list_func)
    print('Best perform mean:', np.mean(result_merged_by_metric['ndcg']))
    print('Best perform var:', np.var(result_merged_by_metric['ndcg']))
    print('Best perform std:', np.std(result_merged_by_metric['ndcg']))
    print('Random seed list:', RANDOM_SEED_LIST)