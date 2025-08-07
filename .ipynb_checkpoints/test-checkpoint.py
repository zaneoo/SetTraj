import argparse
import random
import numpy as np
import torch
import os
import importlib
import matplotlib.pyplot as plt
import time

from dataset import TrajectoryDataset
from torch.utils.data import DataLoader
from model import TrajectoryModel
import torch.nn.functional as F

# 参数设置
parser = argparse.ArgumentParser() 
parser.add_argument('--dataset_path', type=str, default='./dataset/')
parser.add_argument('--dataset_name', type=str, default='sdd')
parser.add_argument("--hp_config", type=str, default=None, help='hyper-parameter')
parser.add_argument('--num_works', type=int, default=8)
parser.add_argument('--obs_len', type=int, default=8)
parser.add_argument('--pred_len', type=int, default=12)
parser.add_argument('--seed', type=int, default=1)
parser.add_argument('--gpu', type=str, default='0')
parser.add_argument('--data_scaling', type=list, default=[1.9, 0.4])
parser.add_argument('--checkpoint', type=str, default='./checkpoint/')
parser.add_argument('--model_path', type=str, required=True, help='预训练模型路径')
parser.add_argument('--vis_results', action='store_true', default=False, help='是否可视化预测结果')
parser.add_argument('--save_results', action='store_true', default=False, help='是否保存评估结果')
parser.add_argument('--top_k', type=int, default=20, help='评估时使用的轨迹预测数量')

args = parser.parse_args()

# 设置随机种子
seed = args.seed
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
print(args)

# 设置GPU
os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

# 加载配置
spec = importlib.util.spec_from_file_location("hp_config", args.hp_config)
hp_config = importlib.util.module_from_spec(spec)
spec.loader.exec_module(hp_config)

# 加载测试数据集
test_dataset = TrajectoryDataset(dataset_path=args.dataset_path, dataset_name=args.dataset_name,
                              dataset_type='test', translation=True, rotation=True, 
                              scaling=False, obs_len=args.obs_len)

test_loader = DataLoader(test_dataset, collate_fn=test_dataset.coll_fn, 
                       batch_size=hp_config.batch_size, shuffle=False, num_workers=args.num_works)

# 初始化模型
model = TrajectoryModel(in_size=2, obs_len=args.obs_len, pred_len=args.pred_len, 
                      embed_size=hp_config.model_hidden_dim, enc_num_layers=2, 
                      int_num_layers_list=[1,1], heads=4, forward_expansion=2)
model = model.cuda()

# 加载预训练权重
print(f"加载预训练模型: {args.model_path}")
model.load_state_dict(torch.load(args.model_path))
model.eval()

def vis_predicted_trajectories(obs_traj, gt, pred_trajs, pred_probabilities, min_index, batch_idx=0, sample_idx=0):
    """
    可视化轨迹预测结果
    
    参数:
        obs_traj: 观测轨迹，形状为 [B T_obs 2]
        gt: 真实轨迹，形状为 [B T_pred 2]
        pred_trajs: 预测轨迹，形状为 [B K T_pred 2]
        pred_probabilities: 预测概率，形状为 [B K]
        min_index: 最小误差索引，形状为 [B]
        batch_idx: 批次索引
        sample_idx: 样本索引
    """
    plt.figure(figsize=(10, 8))
    
    for i in range(min(obs_traj.shape[0], 5)):  # 最多显示5个样本
        plt.clf()
        curr_obs = obs_traj[i].cpu().numpy()  # [T_obs 2]
        curr_gt = gt[i].cpu().numpy()
        curr_preds = pred_trajs[i].cpu().numpy()
        
        curr_pros = pred_probabilities[i].cpu().numpy()
        curr_min_index = min_index[i].cpu().numpy()
        
        # 绘制观测轨迹
        obs_x = curr_obs[:, 0]
        obs_y = curr_obs[:, 1]
        plt.plot(obs_x, obs_y, marker='o', color='green', label='Observed', linewidth=2)
        
        # 绘制真实轨迹
        gt_x = np.concatenate((obs_x[-1:], curr_gt[:, 0]))
        gt_y = np.concatenate((obs_y[-1:], curr_gt[:, 1]))
        plt.plot(gt_x, gt_y, marker='o', color='blue', label='Ground Truth', linewidth=2)
        plt.scatter(gt_x[-1], gt_y[-1], marker='*', color='blue', s=300)
        
        # 绘制预测轨迹
        for j in range(curr_preds.shape[0]):
            pred_x = np.concatenate((obs_x[-1:], curr_preds[j][:, 0]))
            pred_y = np.concatenate((obs_y[-1:], curr_preds[j][:, 1]))
            
            if j == curr_min_index:
                plt.plot(pred_x, pred_y, ls='-.', lw=2.0, color='red', label='Best Prediction' if j == 0 else None)
                plt.scatter(pred_x[-1], pred_y[-1], marker='*', color='orange', s=300)
            else:
                plt.plot(pred_x, pred_y, ls='-.', lw=0.5, color='red', alpha=0.3, label='Other Predictions' if j == 1 else None)
                plt.scatter(pred_x[-1], pred_y[-1], marker='*', color='red', s=100, alpha=0.3)
            
            # 显示预测概率
            plt.text(pred_x[-1], pred_y[-1], f"{curr_pros[j]:.2f}", ha='center')
        
        plt.title(f'Trajectory Prediction (Batch {batch_idx}, Sample {i})')
        plt.xlabel('X-coordinate')
        plt.ylabel('Y-coordinate')
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.tight_layout()
        
        # 保存图像
        save_path = f'./fig/{args.dataset_name}'
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        plt.savefig(f'{save_path}/batch_{batch_idx}_sample_{i}_{time.time()}.png')
        
        if i == sample_idx:  # 如果是指定的样本，额外保存一份
            plt.savefig(f'{save_path}/selected_sample_{time.time()}.png')
    
    return

def test():
    """
    测试模型性能
    """
    model.eval()
    
    ade_all = 0  # 平均位移误差
    fde_all = 0  # 最终位移误差
    num_traj = 0  # 轨迹数量
    
    ade_list = []  # 存储所有样本的ADE
    fde_list = []  # 存储所有样本的FDE
    
    for batch_idx, (ped, neis, mask) in enumerate(test_loader):
        ped = ped.cuda()
        neis = neis.cuda()
        mask = mask.cuda()
        
        # 数据处理
        if args.dataset_name == 'eth':
            ped[:, :, 0] = ped[:, :, 0] * args.data_scaling[0]
            ped[:, :, 1] = ped[:, :, 1] * args.data_scaling[1]
        
        ped_obs = ped[:, :args.obs_len]  # 观测轨迹
        gt = ped[:, args.obs_len:]  # 真实轨迹
        neis_obs = neis[:, :, :args.obs_len]  # 邻居观测轨迹
        
        with torch.no_grad():
            num_traj += ped_obs.shape[0]
            
            # 模型前向传播
            pred_trajs, pred_logits = model(ped_obs, neis_obs, mask)
            
            # 预测概率
            pred_probs = F.softmax(pred_logits, dim=-1)
            
            # 重塑预测轨迹 [B, L, T_pred*2] -> [B, L, T_pred, 2]
            pred_trajs = pred_trajs.reshape(pred_trajs.shape[0], pred_trajs.shape[1], gt.shape[1], 2)
            
            # 计算误差
            gt_expanded = gt.unsqueeze(1)  # [B, 1, T_pred, 2]
            
            # L2距离
            distances = torch.norm(pred_trajs - gt_expanded, p=2, dim=-1)  # [B, L, T_pred]
            
            # 平均位移误差
            ade_per_pred = torch.mean(distances, dim=-1)  # [B, L]
            min_ade, min_ade_indices = torch.min(ade_per_pred, dim=-1)  # [B]
            
            # 最终位移误差
            fde_per_pred = distances[:, :, -1]  # [B, L]
            min_fde, min_fde_indices = torch.min(fde_per_pred, dim=-1)  # [B]
            
            # 汇总统计
            ade_all += min_ade.sum().item()
            fde_all += min_fde.sum().item()
            
            # 记录每个样本的指标
            ade_list.extend(min_ade.cpu().numpy())
            fde_list.extend(min_fde.cpu().numpy())
            
            # 可视化预测结果
            if args.vis_results and batch_idx % 10 == 0:  # 每10个批次可视化一次
                # 获取top-k预测概率和索引
                top_k = min(args.top_k, pred_probs.shape[1])
                top_k_probs, top_k_indices = torch.topk(pred_probs, k=top_k, dim=-1)
                
                # 收集top-k预测
                batch_size = pred_trajs.shape[0]
                top_k_trajs = torch.zeros((batch_size, top_k, gt.shape[1], 2), device=pred_trajs.device)
                
                for b in range(batch_size):
                    top_k_trajs[b] = pred_trajs[b, top_k_indices[b]]
                
                # 可视化
                vis_predicted_trajectories(
                    ped_obs, gt, top_k_trajs, top_k_probs, min_fde_indices, batch_idx
                )
    
    # 计算最终指标
    ade = ade_all / num_traj
    fde = fde_all / num_traj
    
    # 输出结果
    print(f"Dataset: {args.dataset_name}")
    print(f"Model: {args.model_path}")
    print(f"评估样本数: {num_traj}")
    print(f"minADE: {ade:.2f}")
    print(f"minFDE: {fde:.2f}")
    
    # 保存结果
    if args.save_results:
        result_path = f'./results/{args.dataset_name}'
        if not os.path.exists(result_path):
            os.makedirs(result_path)
        
        result_file = f'{result_path}/results_{time.strftime("%Y%m%d_%H%M%S")}.txt'
        with open(result_file, 'w') as f:
            f.write(f"测试数据集: {args.dataset_name}\n")
            f.write(f"模型路径: {args.model_path}\n")
            f.write(f"评估样本数: {num_traj}\n")
            f.write(f"minADE: {ade:.4f}\n")
            f.write(f"minFDE: {fde:.4f}\n")
            f.write(f"ADE统计: 平均={np.mean(ade_list):.4f}, 中位数={np.median(ade_list):.4f}, 最大={np.max(ade_list):.4f}, 最小={np.min(ade_list):.4f}\n")
            f.write(f"FDE统计: 平均={np.mean(fde_list):.4f}, 中位数={np.median(fde_list):.4f}, 最大={np.max(fde_list):.4f}, 最小={np.min(fde_list):.4f}\n")
    
    return ade, fde

if __name__ == "__main__":
    test() 