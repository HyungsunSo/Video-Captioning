import torch
import torch.nn as nn

class PerceiverResampler(nn.Module):
    def __init__(self, dim, depth=6, dim_head=64, heads=8, num_queries=32, embedding_dim=512):
        super().__init__()
        self.latents = nn.Parameter(torch.randn(num_queries, dim))
        # ... (기존 소스코드의 PerceiverResampler 클래스 내용 전체 복사)