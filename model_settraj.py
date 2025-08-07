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
        
        # Add query_modulator_net (Scheme 1)
        self.query_modulator_net = nn.Sequential(
            nn.Linear(embed_size, L * embed_size // 2),
            nn.ReLU(),
            nn.Linear(L * embed_size // 2, L * embed_size)
        )
        # Initialize the final layer of query_modulator_net for small initial modulations
        self.query_modulator_net[-1].weight.data.normal_(mean=0.0, std=0.01)
        self.query_modulator_net[-1].bias.data.zero_()
        
        self.mode_encoder = Encoder(embed_size, enc_num_layers, heads, forward_expansion, islinear=True)
        self.cls_head = nn.Linear(embed_size, 1)
        
        # 保留邻居嵌入层，移除GNN相关层
        self.nei_embedding = nn.Linear(in_size*obs_len, embed_size)
        self.social_decoder = Decoder(embed_size, int_num_layers_list[1], heads, forward_expansion, islinear=False)
        self.reg_head = nn.Linear(embed_size, in_size*pred_len)

    def spatial_interaction(self, ped, neis, mask):
        
        # ped [B L embed_size] # Changed K to L to reflect mode_queries count
        # neis [B N obs_len 2]  N is the max number of agents of current scene
        # mask [B N N] is used to stop the attention from invalid agents

        neis = neis.reshape(neis.shape[0], neis.shape[1], -1)  # [B N obs_len*2]
        nei_embeddings = self.nei_embedding(neis)  # [B N embed_size]
        
        # The mask for Decoder usually is [Batch, Num_target_tokens, Num_source_tokens]
        # Here, target_tokens are L modes, source_tokens are N neighbors.
        # So, mask should be [B, L, N]
        # Current mask is [B,N,N]. We need to select the relevant part for each of the L modes.
        # Assuming the original mask refers to interactions between the N agents, and we want to mask interactions for each of the L predicted trajectories of the target agent.
        # A common way is to use the target agent's part of the mask, repeated L times.
        # If the input mask is mask[b, i, j] is 1 if agent i CANNOT attend to agent j.
        # Let's assume ped_feat (target) is the first agent implicitly for interaction masking purposes.
        # The original code had: mask = mask[:, 0:1].repeat(1, ped.shape[1], 1) # [B K N] which seems to pick the first row of the N*N mask and repeat it K times.
        # This implies the first agent in neis_obs might be the target agent itself if N includes the target, or the mask is structured such that mask[0,:] are connections to target.
        # Let's keep it simple and assume the original intention for K (now L) copies was correct for the decoder's needs.
        # The decoder expects mask of shape [batch_size, target_seq_len, source_seq_len]
        # Here, target_seq_len is L, source_seq_len is N (num_neighbors)
        # So we need mask [B, L, N]
        # Original was mask[:, 0:1].repeat(1, ped.shape[1], 1) gives [B, L, N]
        # This takes the first row of the agent-agent mask (presumably interactions with agent 0) and repeats it L times.
        # This seems plausible if agent 0 is always the ego/target agent in the context of the mask.
        # The 'mask' variable here refers to the original function argument [B, N, N]
        interaction_mask = mask[:, 0, :].unsqueeze(1).repeat(1, ped.shape[1], 1) # [B, L, N]

        int_feat = self.social_decoder(ped, nei_embeddings, interaction_mask)  # [B L embed_size]

        return int_feat # [B L embed_size]
    
    def forward(self, ped_obs, neis_obs, mask):
        # ped_obs [B obs_len 2]
        # nei_obs [B N obs_len 2]
        # mask [B N N]
        
        batch_size = ped_obs.shape[0]
        L = self.mode_queries.shape[0] # Get L from self.mode_queries
        
        # 1. 获取观测轨迹嵌入 E_o
        ped_obs_flat = ped_obs.reshape(batch_size, -1)  # [B obs_len*2]
        E_o = self.embedding(ped_obs_flat)  # [B embed_size]
        
        # Generate modulations for mode_queries
        modulations_flat = self.query_modulator_net(E_o)  # [B, L * embed_size]
        modulations = modulations_flat.view(batch_size, L, -1)  # [B, L, embed_size]
        
        base_queries_expanded = self.mode_queries.unsqueeze(0).expand(batch_size, -1, -1)  # [B, L, embed_size]
        dynamic_mode_queries = base_queries_expanded + modulations  # [B, L, embed_size]
        
        # Expand E_o to combine with dynamic_mode_queries
        E_o_expanded_for_encoder = E_o.unsqueeze(1).expand(-1, L, -1)  # [B, L, embed_size]
        
        encoder_input = E_o_expanded_for_encoder + dynamic_mode_queries  # [B, L, embed_size]
        
        # 4. 使用模式级编码器处理
        ped_feat = self.mode_encoder(encoder_input)  # [B, L, embed_size]
        
        # 5. 社交级解码器处理
        int_feat = self.spatial_interaction(ped_feat, neis_obs, mask)  # [B, L, embed_size]
        
        # 6. 预测头处理
        scores = self.cls_head(int_feat).squeeze(-1)  # [B, L]
        pred_trajs = self.reg_head(int_feat)  # [B, L, pred_len*2]
        
        # 7. 直接返回所有 L 个预测及其分数
        return pred_trajs, scores