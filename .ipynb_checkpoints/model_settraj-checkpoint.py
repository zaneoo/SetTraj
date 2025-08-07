import torch
import torch.nn as nn

from transformer_encoder import Encoder
from transformer_decoder import Decoder


class TrajectoryModel(nn.Module):

    def __init__(self, in_size, obs_len, pred_len, embed_size, enc_num_layers, int_num_layers_list, heads, forward_expansion):
        super(TrajectoryModel, self).__init__()
        
        # 只处理观测轨迹，不再拼接预测轨迹
        self.embedding = nn.Linear(in_size*obs_len, embed_size)
        
        # 添加可学习的模式查询参数，替代原来的通用运动模式
        L = 150  # 设置模式数量为50
        self.mode_queries = nn.Parameter(torch.randn(L, embed_size))
        
        self.mode_encoder = Encoder(embed_size, enc_num_layers, heads, forward_expansion, islinear=True)
        self.cls_head = nn.Linear(embed_size, 1)
        
        # 保留邻居嵌入层，移除GNN相关层
        self.nei_embedding = nn.Linear(in_size*obs_len, embed_size)
        self.social_decoder = Decoder(embed_size, int_num_layers_list[1], heads, forward_expansion, islinear=False)
        self.reg_head = nn.Linear(embed_size, in_size*pred_len)

    def spatial_interaction(self, ped, neis, mask):
        
        # ped [B K embed_size]
        # neis [B N obs_len 2]  N is the max number of agents of current scene
        # mask [B N N] is used to stop the attention from invalid agents

        neis = neis.reshape(neis.shape[0], neis.shape[1], -1)  # [B N obs_len*2]
        nei_embeddings = self.nei_embedding(neis)  # [B N embed_size]
        
        mask = mask[:, 0:1].repeat(1, ped.shape[1], 1)  # [B K N]
        int_feat = self.social_decoder(ped, nei_embeddings, mask)  # [B K embed_size]

        return int_feat # [B K embed_size]
    
    def forward(self, ped_obs, neis_obs, mask):
        # ped_obs [B obs_len 2]
        # nei_obs [B N obs_len 2]
        # mask [B N N]
        
        batch_size = ped_obs.shape[0]
        
        # 1. 获取观测轨迹嵌入 E_o
        ped_obs_flat = ped_obs.reshape(batch_size, -1)  # [B obs_len*2]
        E_o = self.embedding(ped_obs_flat)  # [B embed_size]
        E_o = E_o.unsqueeze(1)  # [B 1 embed_size]
        
        # 2. 将 mode_queries 扩展到批次维度
        mode_queries = self.mode_queries.unsqueeze(0).expand(batch_size, -1, -1)  # [B L embed_size]
        
        # 3. 将 E_o 与 mode_queries 相加得到编码器输入
        E_o_expanded = E_o.expand(-1, mode_queries.shape[1], -1)  # [B L embed_size]
        encoder_input = E_o_expanded + mode_queries  # [B L embed_size]
        
        # 4. 使用模式级编码器处理
        ped_feat = self.mode_encoder(encoder_input)  # [B L embed_size]
        
        # 5. 社交级解码器处理
        int_feat = self.spatial_interaction(ped_feat, neis_obs, mask)  # [B L embed_size]
        
        # 6. 预测头处理
        scores = self.cls_head(int_feat).squeeze(-1)  # [B L]
        pred_trajs = self.reg_head(int_feat)  # [B L pred_len*2]
        
        # 7. 直接返回所有 L 个预测及其分数
        return pred_trajs, scores