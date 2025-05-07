import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

class TrafalignTransformer(nn.Module):
    def __init__(self, cfg, channels=256, attention_dim=64, num_heads=1, num_random_points=18):
        super(TrafalignTransformer, self).__init__()
        self.num_heads = num_heads
        self.attention_dim = attention_dim
        self.num_random_points = num_random_points
        
        # 定义线性变换矩阵，Q, K, V
        self.W_Q = nn.Linear(channels, attention_dim * num_heads)
        self.W_K = nn.Linear(channels, attention_dim * num_heads)
        self.W_V = nn.Linear(channels, attention_dim * num_heads)
        
        # Position Encoding (Learnable)
        grid_size = cfg['voxelization']['grid_size']
        stride = cfg['model']['deform']['input_stride'][0]
        self.position_encoding = nn.Parameter(torch.randn(1, grid_size[0]//stride, grid_size[1]//stride, channels))
        
        # 输出线性层
        self.fc_out = nn.Linear(attention_dim * num_heads, channels)
        
        # 前馈神经网络部分
        self.ffn = nn.Sequential(
            nn.Linear(channels, channels * 4),
            nn.ReLU(),
            nn.Linear(channels * 4, channels)
        )
        
        # Layer Normalization
        self.norm1 = nn.LayerNorm(channels)
        self.norm2 = nn.LayerNorm(channels)
    
    def forward(self, X, selected_indices, topk = 1,topk_index=None):
        if topk == 1:
            return self.forward_wotopk(X,selected_indices)
        else:
            return self.forward_wtopk(X,selected_indices,topk_index)

    def forward_wtopk(self,X,selected_indices,topk_index):
        batch_size, height, width, channels = X.shape
        num_pixels = topk_index.shape[1]
        X_ = X.clone()

        # 添加 Position Encoding
        pe = rearrange(self.position_encoding,'b h w c -> b (h w) c').repeat(batch_size,1,1)
        X = rearrange(X,'b h w c -> b (h w) c')
        index = topk_index.unsqueeze(-1).expand(-1, -1, X.shape[-1])
        X, pe = X.gather(1,index), pe.gather(1,index)
        X = X + pe  # X 形状为 (batch_size, 50, 60, 256)
        
        # 计算 Q, K, V
        Q = self.W_Q(X)  # 形状 (batch_size, num_pixels, attention_dim * num_heads)
        K = self.W_K(X_)  # 形状 (batch_size, height*width, attention_dim * num_heads)
        V = self.W_V(X_)  # 形状 (batch_size, height*width, attention_dim * num_heads)
        
        # 重新调整形状以适应多头注意力机制
        Q = Q.view(batch_size, num_pixels, self.num_heads, self.attention_dim)  
        K = K.view(batch_size, height*width, self.num_heads, self.attention_dim)  
        V = V.view(batch_size, height*width, self.num_heads, self.attention_dim) 
        
        # 使用选择的索引从 K 和 V 中进行查询
        # selected_keys 和 selected_values 的形状为 (batch_size, num_pixels, num_random_points, num_heads, d)
        selected_keys = torch.gather(K.unsqueeze(2).expand(-1, -1, self.num_random_points, -1, -1), 1, selected_indices.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, -1, self.num_heads, self.attention_dim))
        selected_values = torch.gather(V.unsqueeze(2).expand(-1, -1, self.num_random_points, -1, -1), 1, selected_indices.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, -1, self.num_heads, self.attention_dim))
        
        index_ = topk_index[...,None,None,None].expand(-1, -1, selected_keys.shape[2], selected_keys.shape[3], selected_keys.shape[4])
        selected_keys = torch.gather(selected_keys, 1, index_)
        selected_values = torch.gather(selected_values, 1, index_)

        # 计算注意力分数 (batch_size, num_pixels, num_random_points, num_heads)
        attention_scores = torch.einsum('bihd,bijhd->bijh', Q, selected_keys) / torch.sqrt(torch.tensor(self.attention_dim, dtype=torch.float32))
        
        # 计算注意力权重并进行归一化
        attention_weights = F.softmax(attention_scores, dim=-2)  # 形状 (batch_size, num_pixels, num_random_points, num_heads)
        
        # 计算加权的 Value 加和 (batch_size, num_pixels, num_heads, d)
        context = torch.einsum('bijh,bijhd->bihd', attention_weights, selected_values)
        
        # 重新调整形状回原始尺寸 (batch_size, num_pixels, attention_dim * num_heads)
        context = context.reshape(batch_size, num_pixels, self.num_heads * self.attention_dim)
        
        # 通过输出线性层
        out = self.fc_out(context)  # (batch_size, 3000, channels)
        
        # 残差连接和 Layer Normalization
        X = self.norm1(X + out)
        
        # 前馈神经网络和残差连接
        ffn_out = self.ffn(X)
        out = self.norm2(X + ffn_out)
        
        # 恢复原始形状 (batch_size, 50, 60, channels)
        out_ = rearrange(X_,' b h w c -> b (h w) c')
        outs = out_.scatter_(1,index,out)
        outs = rearrange(outs,' b (h w) c -> b h w c', h = X_.shape[1])

        return outs
        
    def forward_wotopk(self,X,selected_indices):
        batch_size, height, width, channels = X.shape
        num_pixels = height * width
        
        # 添加 Position Encoding
        X = X + self.position_encoding  # X 形状为 (batch_size, 50, 60, 256)
        
        # 计算 Q, K, V
        Q = self.W_Q(X)  # 形状 (batch_size, 50, 60, attention_dim * num_heads)
        K = self.W_K(X)  # 形状 (batch_size, 50, 60, attention_dim * num_heads)
        V = self.W_V(X)  # 形状 (batch_size, 50, 60, attention_dim * num_heads)
        
        # 重新调整形状以适应多头注意力机制
        Q = Q.view(batch_size, num_pixels, self.num_heads, self.attention_dim)  # (batch_size, 3000, num_heads, d)
        K = K.view(batch_size, num_pixels, self.num_heads, self.attention_dim)  # (batch_size, 3000, num_heads, d)
        V = V.view(batch_size, num_pixels, self.num_heads, self.attention_dim)  # (batch_size, 3000, num_heads, d)
        
        # 使用选择的索引从 K 和 V 中进行查询
        # selected_keys 和 selected_values 的形状为 (batch_size, num_pixels, num_random_points, num_heads, d)
        selected_keys = torch.gather(K.unsqueeze(2).expand(-1, -1, self.num_random_points, -1, -1), 1, selected_indices.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, -1, self.num_heads, self.attention_dim))
        selected_values = torch.gather(V.unsqueeze(2).expand(-1, -1, self.num_random_points, -1, -1), 1, selected_indices.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, -1, self.num_heads, self.attention_dim))
                
        # 计算注意力分数 (batch_size, num_pixels, num_random_points, num_heads)
        attention_scores = torch.einsum('bihd,bijhd->bijh', Q, selected_keys) / torch.sqrt(torch.tensor(self.attention_dim, dtype=torch.float32))
        
        # 计算注意力权重并进行归一化
        attention_weights = F.softmax(attention_scores, dim=-2)  # 形状 (batch_size, num_pixels, num_random_points, num_heads)
        
        # 计算加权的 Value 加和 (batch_size, num_pixels, num_heads, d)
        context = torch.einsum('bijh,bijhd->bihd', attention_weights, selected_values)
        
        # 重新调整形状回原始尺寸 (batch_size, num_pixels, attention_dim * num_heads)
        context = context.reshape(batch_size, num_pixels, self.num_heads * self.attention_dim)
        
        # 通过输出线性层
        out = self.fc_out(context)  # (batch_size, 3000, channels)
        
        # 恢复原始形状 (batch_size, 50, 60, channels)
        out = out.view(batch_size, height, width, channels)
        
        # 残差连接和 Layer Normalization
        X = self.norm1(X + out)
        
        # 前馈神经网络和残差连接
        ffn_out = self.ffn(X)
        out = self.norm2(X + ffn_out)
        
        return out,attention_weights