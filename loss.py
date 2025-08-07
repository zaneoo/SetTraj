import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment


class HungarianMatcherLoss(nn.Module):
    def __init__(self, lambda_traj=1.0, lambda_cls=1.0):
        """
        初始化匈牙利匹配损失函数
        
        参数:
            lambda_traj: 轨迹损失的权重
            lambda_cls: 分类损失的权重
        """
        super(HungarianMatcherLoss, self).__init__()
        self.lambda_traj = lambda_traj
        self.lambda_cls = lambda_cls
        self.reg_criterion = nn.SmoothL1Loss(reduction='none')
        self.cls_criterion = nn.CrossEntropyLoss()

    def forward(self, outputs, targets):
        """
        计算基于匈牙利匹配的损失
        
        参数:
            outputs: 字典，包含模型输出
                - pred_trajs: 预测轨迹，形状为 [B, L, T_pred*2]
                - pred_logits: 预测分数/logits，形状为 [B, L]
            targets: 真实轨迹，形状为 [B, T_pred*2]
            
        返回:
            total_loss: 总损失
            loss_dict: 包含各部分损失的字典，用于记录
        """
        pred_trajs = outputs['pred_trajs']  # [B, L, T_pred*2]
        pred_logits = outputs['pred_logits']  # [B, L]
        
        batch_size = pred_trajs.shape[0]
        num_queries = pred_trajs.shape[1]
        
        # 存储每个样本的匹配索引
        batch_indices = []
        
        # 对批次中的每个样本进行匹配
        for b in range(batch_size):
            # 计算轨迹距离代价 (L1距离)
            traj_cost = torch.cdist(
                pred_trajs[b].view(num_queries, -1),  # [L, T_pred*2]
                targets[b].view(1, -1),  # [1, T_pred*2]
                p=1
            ).squeeze(1)  # [L]
            
            # 计算分类代价
            # 将logits转换为概率，然后取负值作为代价
            cls_probs = F.softmax(pred_logits[b], dim=0)  # [L]
            cls_cost = -cls_probs  # [L]
            
            # 结合代价
            cost = self.lambda_traj * traj_cost + self.lambda_cls * cls_cost
            
            # 修复：将1D数组转换为2D数组以适应匈牙利算法
            # 每个预测轨迹对应一行，只有一个目标，所以只有一列
            cost_np = cost.detach().cpu().numpy()
            cost_matrix = cost_np.reshape(num_queries, 1)
            
            # 使用匈牙利算法找到最优匹配
            pred_idx, _ = linear_sum_assignment(cost_matrix)
            
            # 由于只有一个目标，我们直接取代价最小的预测索引
            # 或者也可以使用 min_pred_idx = torch.argmin(cost).item()
            batch_indices.append(pred_idx)
        
        # 计算回归损失
        loss_reg = 0
        for b in range(batch_size):
            pred_idx = batch_indices[b]
            
            # 只对匹配的预测计算回归损失
            # 注意：由于每个样本可能匹配到多个预测，我们需要处理这种情况
            if len(pred_idx) > 0:
                matched_preds = pred_trajs[b, pred_idx]  # [num_matched, T_pred*2]
                matched_targets = targets[b].unsqueeze(0).expand(len(pred_idx), -1)  # [num_matched, T_pred*2]
                
                # 计算SmoothL1Loss
                reg_loss = self.reg_criterion(matched_preds, matched_targets).mean()
                loss_reg += reg_loss
        
        # 计算平均回归损失
        loss_reg = loss_reg / batch_size if batch_size > 0 else 0
        
        # 计算分类损失
        loss_cls = 0
        for b in range(batch_size):
            pred_idx = batch_indices[b]
            
            # 创建one-hot目标向量
            target_cls = torch.zeros_like(pred_logits[b])
            target_cls[pred_idx] = 1.0
            
            # 计算交叉熵损失
            cls_loss = F.binary_cross_entropy_with_logits(
                pred_logits[b], target_cls
            )
            loss_cls += cls_loss
        
        # 计算平均分类损失
        loss_cls = loss_cls / batch_size if batch_size > 0 else 0
        
        # 总损失
        total_loss = loss_reg + loss_cls
        
        # 返回总损失和损失字典
        loss_dict = {
            'loss': total_loss.item(),
            'loss_reg': loss_reg.item(),
            'loss_cls': loss_cls.item()
        }
        
        return total_loss, loss_dict


def build_criterion(lambda_traj=1.0, lambda_cls=1.0):
    """
    构建损失函数
    
    参数:
        lambda_traj: 轨迹损失的权重
        lambda_cls: 分类损失的权重
        
    返回:
        criterion: 损失函数实例
    """
    return HungarianMatcherLoss(lambda_traj, lambda_cls) 