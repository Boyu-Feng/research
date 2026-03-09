import torch
import torch.nn as nn

class HiddenStateDecomposer(nn.Module):
    """
    将hidden state分解为局部信息（next token）和全局信息（future context）
    """
    def __init__(self, hidden_size, dropout=0.1):
        super().__init__()
        self.hidden_size = hidden_size
        
        # 局部信息投影（next token）
        self.local_proj = nn.Linear(hidden_size, hidden_size)
        
        # 全局信息投影（future context）
        self.global_proj = nn.Linear(hidden_size, hidden_size)
        
        # 门控机制，用于动态分配信息
        self.gate = nn.Sequential(
            nn.Linear(hidden_size, hidden_size * 2),
            nn.Sigmoid()
        )
        
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(hidden_size)
        
    def forward(self, hidden_state):
        """
        Args:
            hidden_state: (batch_size, seq_len, hidden_size) 或 (batch_size, hidden_size)
        
        Returns:
            local_state: 局部信息（next token）
            global_state: 全局信息（future context）
        """
        # 归一化输入
        normed_hidden = self.layer_norm(hidden_state)
        
        # 计算门控值
        gates = self.gate(normed_hidden)
        local_gate, global_gate = gates.chunk(2, dim=-1)
        
        # 投影到局部和全局空间
        local_state = self.local_proj(normed_hidden) * local_gate
        global_state = self.global_proj(normed_hidden) * global_gate
        
        # 应用dropout
        local_state = self.dropout(local_state)
        global_state = self.dropout(global_state)
        
        return local_state, global_state
