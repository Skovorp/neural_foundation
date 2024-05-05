import torch
from torch import nn
from transformers import GPT2Config, GPT2Model


class Encoder(nn.Module):
    def __init__(self, n_filters_time, filter_time_length, pool_time_length, stride_avg_pool, drop_prob, outp_dim, **kwargs):
        super().__init__()
        self.n_filters_time = n_filters_time
        self.conv_stack = nn.Sequential(
            nn.Conv2d(1, n_filters_time, kernel_size=(1, filter_time_length)),
            nn.Conv2d(n_filters_time, n_filters_time, kernel_size=(4, 1)),
            nn.BatchNorm2d(num_features=n_filters_time),
            nn.GELU(),
            nn.AvgPool2d(
                kernel_size=(1, pool_time_length),
                stride=(1, stride_avg_pool)
            ),
            nn.Dropout(p=drop_prob),
            nn.Conv2d(n_filters_time, n_filters_time, kernel_size=(1, 1), stride=(1, 1))
        )
        self.outp_proj = nn.LazyLinear(outp_dim)
        self.transformer_encoder = nn.TransformerEncoderLayer(d_model=n_filters_time, dim_feedforward=512, nhead=8, norm_first=True)
    
        
    def forward(self, x):
        # x -- (batch, chunks, channels, time)
        batch_size, num_chunks, channels, time = x.shape
        x_prep = x.reshape(batch_size * num_chunks, 1, channels, time)
        assert (x_prep[0, 0, :, :] == x[0, 0, :, :]).all(), "reshape in encoder corrupted data"
        x_prep = self.conv_stack(x_prep) # (batch * chunks, n_filters_time, 1, time)
        x_prep = x_prep.squeeze(2) # (batch * chunks, n_filters_time, time)
        x_prep = x_prep.permute(0, 2, 1) # (batch * chunks, time, n_filters_time)
        x_prep = self.transformer_encoder(x_prep) # same
        x_prep = x_prep.reshape(batch_size, num_chunks, -1) # (batch, chunks, ?)
        # x_prep = x_prep.permute(0, 1, 3, 2) # (batch, chunks, n_filters_time, time)
        x_prep = self.outp_proj(x_prep) # (batch, chunks, outp_dim)
        return x_prep
    

class Decoder(nn.Module):
    def __init__(self):
        super().__init__()
        
        configuration = GPT2Config(vocab_size=1)
        self.gpt = GPT2Model(configuration) # (batch, seq_len, emb_dim) -> (batch, seq_len, emb_dim)
    
    def forward(self, x):
        return self.gpt.forward(inputs_embeds=x, return_dict=True).last_hidden_state