
import inspect
import math
from typing import Any, Dict, List, Optional, Tuple, Union
from transformers.modeling_outputs import ModelOutput
from dataclasses import dataclass
import torch
import torch.nn.functional as F
import torch.utils.checkpoint
from torch import nn
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss

# 导入activations.py文件中的ACT2FN
from transformers.activations import ACT2FN
from transformers.cache_utils import DynamicCache
from transformers.modeling_attn_mask_utils import AttentionMaskConverter

from transformers.modeling_outputs import (
    BaseModelOutput,
)

from transformers.modeling_utils import PreTrainedModel
from transformers.utils import (
    add_start_docstrings,
    add_start_docstrings_to_model_forward,
    is_flash_attn_greater_or_equal_2_10,
    logging,
    replace_return_docstrings,
)
from transformers.utils.import_utils import (
    is_causal_conv1d_available,
    is_flash_attn_2_available,
    is_mamba_ssm_available,
)
from .configuration_bme import BMEConfig


if is_flash_attn_2_available():
    from flash_attn import flash_attn_func, flash_attn_varlen_func
    from flash_attn import index_first_axis, pad_input, unpad_input  # noqa

    _flash_supports_window_size = "window_size" in list(inspect.signature(flash_attn_func).parameters)


if is_mamba_ssm_available():
    from mamba_ssm.ops.selective_scan_interface import mamba_inner_fn, selective_scan_fn
    from mamba_ssm.ops.triton.selective_state_update import selective_state_update
else:
    selective_state_update, selective_scan_fn, mamba_inner_fn = None, None, None

if is_causal_conv1d_available():
    from causal_conv1d import causal_conv1d_fn, causal_conv1d_update
else:
    causal_conv1d_update, causal_conv1d_fn = None, None

is_fast_path_available = all(
    (selective_state_update, selective_scan_fn, causal_conv1d_fn, causal_conv1d_update, mamba_inner_fn)
)


class BMERMSNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-6):
        """
        BMERMSNorm is equivalent to T5LayerNorm and LlamaRMSNorm
        """
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states):
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        return self.weight * hidden_states.to(input_dtype)


class Embedding(nn.Module):
    def __init__(self, config: BMEConfig):
        super().__init__()
        # 第一个模态的embedding
        self.eeg_embedding = nn.Embedding(config.eeg_vocab_size, config.hidden_size, padding_idx=config.pad_token_id)
        # 第二个模态的embedding
        self.ecg_embedding = nn.Embedding(config.ecg_vocab_size, config.hidden_size, padding_idx=config.pad_token_id)
        # # 第三个模态的embedding
        # self.speech_embedding = nn.Embedding(config.speech_vocab_size, config.hidden_size, padding_idx=config.pad_token_id)

    def forward(
        self, 
        eeg_input_ids: Optional[torch.Tensor] = None,
        ecg_input_ids: Optional[torch.Tensor] = None,
        # speech_input_ids: Optional[torch.Tensor] = None,
    ):

        eeg_embeds = self.eeg_embedding(eeg_input_ids)
    
        ecg_embeds = self.ecg_embedding(ecg_input_ids)

        # speech_embeds = self.speech_embedding(speech_input_ids)
        
        return eeg_embeds, ecg_embeds


class LognConv1d(nn.Conv1d):
    """
    对数卷积核权重: 调整卷积核的权重, 使其随着位置的增加而逐渐变化, 模拟对数位置编码的效果.
    """
    def __init__(self, in_channels, out_channels, bias, kernel_size, groups, padding, log_scale_init=0.0, dtype=torch.float32):
        super().__init__(in_channels, out_channels, kernel_size, groups=groups, padding=padding, bias=bias, dtype=dtype)
        self.log_scale = nn.Parameter(torch.tensor(log_scale_init, dtype=torch.float32))

        # TODO: 很糟糕的是, Mamba 是调用卷积的权重进行计算的, 并不直接使用这里定义的卷积forward方法. 这意味着我们只能在初始化时调整权重, 而不能在forward方法中调整权重. 我们急需对 Mamba 进行重构, 以便我们可以在forward方法中调整权重.

        # 获取卷积核的权重
        weight = self.weight # [out_channels, in_channels, kernel_size]
        offset = 64
        # 获取out_channels
        out_channels = weight.size(0)
        # 计算logn
        logn = torch.arange(offset+1, offset+out_channels+1, dtype=torch.float32)[-out_channels:] # [out_channels]
        # base 是训练数据的平均序列长度
        base = torch.tensor(256)
        logn = torch.log(logn) / torch.log(base)
        logn[logn < 1.0] = 1.0
        logn[logn > 1.0] *= torch.exp(self.log_scale)
        logn = logn.to(weight.dtype).view(out_channels, 1, 1)
        # 对卷积核的权重进行调整
        self.weight = nn.Parameter(weight * logn)
    
    def forward(
        self, 
        input: torch.Tensor
    ) -> torch.Tensor:
        # 计算卷积
        return F.conv1d(input, self.weight, self.bias, self.stride, self.padding, self.dilation, self.groups)


class MambaCache:
    """
    Arguments:
        config: MambaConfig
        batch_size: int
        dtype: torch.dtype
        device: torch.device

    Attributes:
        seqlen_offset: int
        dtype: torch.dtype
        conv_states: Dict[int, torch.Tensor] # layer_idx -> [batch_size, intermediate_size, conv_kernel_size]
        ssm_states: Dict[int, torch.Tensor] # layer_idx -> [batch_size, intermediate_size, ssm_state_size]
    """

    def __init__(
        self, config: BMEConfig, batch_size: int, dtype: torch.dtype = torch.float16, device: Optional[str] = None
    ):
        self.seqlen_offset = 0
        self.dtype = dtype
        intermediate_size = config.intermediate_size
        ssm_state_size = config.state_size
        conv_kernel_size = config.conv_kernel

        self.conv_states = {
            i: torch.zeros(batch_size, intermediate_size, conv_kernel_size, device=device, dtype=dtype)
            for i in range(config.num_hidden_layers)
        }
        self.ssm_states = {
            i: torch.zeros(batch_size, intermediate_size, ssm_state_size, device=device, dtype=dtype)
            for i in range(config.num_hidden_layers)
        }


class MambaMixer(nn.Module):
    """
    Compute ∆, A, B, C, and D the state space parameters and compute the `contextualized_states`.
    A, D are input independent (see Mamba paper [1] Section 3.5.2 "Interpretation of A" for why A isn't selective)
    ∆, B, C are input-dependent (this is a key difference between Mamba and the linear time invariant S4,
    and is why Mamba is called **selective** state spaces)
    """

    def __init__(self, config: BMEConfig, layer_idx: int):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.ssm_state_size = config.state_size
        self.conv_kernel_size = config.conv_kernel
        self.intermediate_size = config.intermediate_size
        self.time_step_rank = int(config.time_step_rank)
        self.layer_idx = layer_idx
        self.use_conv_bias = config.use_conv_bias
        self.conv1d = LognConv1d(
            in_channels=self.intermediate_size,
            out_channels=self.intermediate_size,
            bias=config.use_conv_bias,
            kernel_size=config.conv_kernel,
            groups=self.intermediate_size,
            padding=config.conv_kernel - 1,
        )

        self.activation = config.hidden_act
        self.act = ACT2FN[config.hidden_act]

        # projection of the input hidden states
        # self.in_proj = nn.Linear(self.hidden_size, self.intermediate_size * 2, bias=config.use_bias)
        self.in_up_proj = nn.Linear(
            self.hidden_size, 
            self.intermediate_size * 2 * 4, 
            bias=config.use_bias 
        )
        self.in_down_proj = nn.Linear(
            self.intermediate_size * 2 * 4, 
            self.intermediate_size * 2,
            bias=config.use_bias
        )
        # selective projection used to make dt, B and C input dependant
        self.x_proj = nn.Linear(self.intermediate_size, self.time_step_rank + self.ssm_state_size * 2, bias=False)
        # time step projection (discretization)
        self.dt_proj = nn.Linear(self.time_step_rank, self.intermediate_size, bias=True)

        # S4D real initialization. These are not discretized!
        # The core is to load them, compute the discrete states, then write the updated state. Keeps the memory bounded
        A = torch.arange(1, self.ssm_state_size + 1, dtype=torch.float32)[None, :]
        A = A.expand(self.intermediate_size, -1).contiguous()

        self.A_log = nn.Parameter(torch.log(A))
        self.D = nn.Parameter(torch.ones(self.intermediate_size))
        # self.out_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=config.use_bias)
        self.out_up_proj = nn.Linear(
            self.intermediate_size, 
            self.intermediate_size * 4, 
            bias=config.use_bias
        )
        self.out_down_proj = nn.Linear(
            self.intermediate_size * 4, 
            self.hidden_size, 
            bias=config.use_bias
        )
        self.use_bias = config.use_bias

    def cuda_kernels_forward(self, hidden_states: torch.Tensor, cache_params: Optional[MambaCache] = None):
        # 1. Gated MLP's linear projection
        # projected_states = self.in_proj(hidden_states).transpose(1, 2)
        projected_states = self.in_down_proj(self.act(self.in_up_proj(hidden_states))).transpose(1, 2)

        if self.training and cache_params is None:  # Doesn't support outputting the states -> used for training
            contextualized_states = mamba_inner_fn(
                projected_states,
                self.conv1d.weight,
                self.conv1d.bias if self.use_conv_bias else None,
                self.x_proj.weight,
                self.dt_proj.weight,
                self.out_proj.weight,
                self.out_proj.bias.float() if self.use_bias else None,
                -torch.exp(self.A_log.float()),
                None,  # input-dependent B
                None,  # input-dependent C
                self.D.float(),
                delta_bias=self.dt_proj.bias.float(),
                delta_softplus=True,
            )

        else:
            hidden_states, gate = projected_states.chunk(2, dim=1)

            # 2. Convolution sequence transformation
            conv_weights = self.conv1d.weight.view(self.conv1d.weight.size(0), self.conv1d.weight.size(2))
            if cache_params is not None and cache_params.seqlen_offset > 0:
                hidden_states = causal_conv1d_update(
                    hidden_states.squeeze(-1),
                    cache_params.conv_states[self.layer_idx],
                    conv_weights,
                    self.conv1d.bias,
                    self.activation,
                )
                hidden_states = hidden_states.unsqueeze(-1)
            else:
                if cache_params is not None:
                    conv_states = nn.functional.pad(
                        hidden_states, (self.conv_kernel_size - hidden_states.shape[-1], 0)
                    )
                    cache_params.conv_states[self.layer_idx].copy_(conv_states)
                hidden_states = causal_conv1d_fn(
                    hidden_states, conv_weights, self.conv1d.bias, activation=self.activation
                )

            # 3. State Space Model sequence transformation
            # 3.a. input varying initialization of time_step, B and C
            ssm_parameters = self.x_proj(hidden_states.transpose(1, 2))
            time_step, B, C = torch.split(
                ssm_parameters, [self.time_step_rank, self.ssm_state_size, self.ssm_state_size], dim=-1
            )
            discrete_time_step = self.dt_proj.weight @ time_step.transpose(1, 2)

            A = -torch.exp(self.A_log.float())
            # 3.c perform the recurrence y ← SSM(A, B, C)(x)
            time_proj_bias = self.dt_proj.bias.float() if hasattr(self.dt_proj, "bias") else None
            if cache_params is not None and cache_params.seqlen_offset > 0:
                scan_outputs = selective_state_update(
                    cache_params.ssm_states[self.layer_idx],
                    hidden_states[..., 0],
                    discrete_time_step[..., 0],
                    A,
                    B[:, 0],
                    C[:, 0],
                    self.D,
                    gate[..., 0],
                    time_proj_bias,
                    dt_softplus=True,
                ).unsqueeze(-1)
            else:
                scan_outputs, ssm_state = selective_scan_fn(
                    hidden_states,
                    discrete_time_step,
                    A,
                    B.transpose(1, 2),
                    C.transpose(1, 2),
                    self.D.float(),
                    gate,
                    time_proj_bias,
                    delta_softplus=True,
                    return_last_state=True,
                )
                if ssm_state is not None and cache_params is not None:
                    cache_params.ssm_states[self.layer_idx].copy_(ssm_state)

            # 4. Final linear projection
            # contextualized_states = self.out_proj(scan_outputs.transpose(1, 2))
            contextualized_states = self.out_down_proj(self.act(self.out_up_proj(scan_outputs.transpose(1, 2))))
        return contextualized_states

    # fmt: off
    def slow_forward(self, input_states, cache_params: Optional[MambaCache]=None):
        batch_size, seq_len, _ = input_states.shape
        dtype = input_states.dtype
        # 1. Gated MLP's linear projection
        projected_states = self.in_down_proj(self.act(self.in_up_proj(input_states))).transpose(1, 2)                   # [batch, 2 * intermediate_size, seq_len]
        hidden_states, gate = projected_states.chunk(2, dim=1)

        # 2. Convolution sequence transformation
        if cache_params is not None:
            ssm_state = cache_params.ssm_states[self.layer_idx].clone()
            if cache_params.seqlen_offset > 0:
                conv_state = cache_params.conv_states[self.layer_idx]                   # [batch, intermediate_size, conv_kernel_size]
                conv_state = torch.roll(conv_state, shifts=-1, dims=-1)
                conv_state[:, :, -1] = hidden_states[:, :, 0]
                cache_params.conv_states[self.layer_idx].copy_(conv_state)
                hidden_states = torch.sum(conv_state * self.conv1d.weight[:, 0, :], dim=-1)
                if self.use_conv_bias:
                    hidden_states += self.conv1d.bias
                hidden_states = self.act(hidden_states).to(dtype).unsqueeze(-1)         # [batch, intermediate_size, 1] : decoding
            else:
                conv_state = nn.functional.pad(
                    hidden_states,
                    (self.conv_kernel_size - hidden_states.shape[-1], 0)
                )
                cache_params.conv_states[self.layer_idx].copy_(conv_state)
                hidden_states = self.act(self.conv1d(hidden_states)[..., :seq_len])     # [batch, intermediate_size, seq_len]
        else:
            ssm_state = torch.zeros(
                (batch_size, self.intermediate_size, self.ssm_state_size),
                device=hidden_states.device, dtype=dtype
            )
            hidden_states = self.act(self.conv1d(hidden_states)[..., :seq_len])         # [batch, intermediate_size, seq_len]

        # 3. State Space Model sequence transformation
        # 3.a. Selection:  [batch, seq_len, self.time_step_rank + self.ssm_state_size * 2]
        ssm_parameters = self.x_proj(hidden_states.transpose(1, 2))
        time_step, B, C = torch.split(
            ssm_parameters, [self.time_step_rank, self.ssm_state_size, self.ssm_state_size], dim=-1
        )
        discrete_time_step = self.dt_proj(time_step)                                    # [batch, seq_len, intermediate_size]
        discrete_time_step = nn.functional.softplus(discrete_time_step).transpose(1, 2) # [batch, intermediate_size, seq_len]

        # 3.b. Discretization: B and C to [batch, seq_len, intermediate_size, ssm_state_size] (SRAM)
        A = -torch.exp(self.A_log.float())                                              # [intermediate_size, ssm_state_size]
        discrete_A = torch.exp(A[None, :, None, :] * discrete_time_step[:, :, :, None]) # [batch, intermediate_size, seq_len, ssm_state_size]
        discrete_B = discrete_time_step[:, :, :, None] * B[:, None, :, :].float()       # [batch, intermediade_size, seq_len, ssm_state_size]
        deltaB_u = discrete_B * hidden_states[:, :, :, None].float()

        # 3.c perform the recurrence y ← SSM(A, B, C)(x)
        scan_outputs = []
        for i in range(seq_len):
            ssm_state = discrete_A[:, :, i, :] * ssm_state + deltaB_u[:, :, i, :]      # [batch, intermediade_size, ssm_state]
            scan_output = torch.matmul(ssm_state.to(dtype), C[:, i, :].unsqueeze(-1))  # [batch, intermediade_size, 1]
            scan_outputs.append(scan_output[:, :, 0])
        scan_output = torch.stack(scan_outputs, dim=-1)                                # [batch, seq_len, intermediade_size]
        scan_output = scan_output + (hidden_states * self.D[None, :, None])
        scan_output = (scan_output * self.act(gate))

        if cache_params is not None:
            cache_params.ssm_states[self.layer_idx].copy_(ssm_state)

        # 4. Final linear projection
        contextualized_states = self.out_down_proj(self.act(self.out_up_proj(scan_output.transpose(1, 2))))             # [batch, seq_len, hidden_size]
        return contextualized_states
    # fmt: on

    def forward(self, hidden_states, cache_params: Optional[MambaCache] = None):
        if is_fast_path_available and "cuda" in self.x_proj.weight.device.type:
            return self.cuda_kernels_forward(hidden_states, cache_params)
        return self.slow_forward(hidden_states, cache_params)
    

class Multimodal_MambaMixer(nn.Module):

    def __init__(self,config: BMEConfig) :
        super().__init__()
        self.eeg_mamba = MambaMixer(config, layer_idx=0)
        self.ecg_mamba = MambaMixer(config, layer_idx=0)
        # self.speech_mamba = MambaMixer(config, layer_idx=0)

    def forward(
        self,
        eeg_hidden_states: torch.Tensor,
        ecg_hidden_states: torch.Tensor,
        # speech_hidden_states: torch.Tensor,
        cache_params: Optional[MambaCache] = None,
    ):
        eeg_contextualized_states = self.eeg_mamba(eeg_hidden_states, cache_params)
        ecg_contextualized_states = self.ecg_mamba(ecg_hidden_states, cache_params)
        # speech_contextualized_states = self.speech_mamba(speech_hidden_states, cache_params)
        return eeg_contextualized_states, ecg_contextualized_states


class MultiModalMultiHeadAggregatedAttention(torch.nn.Module):
    def __init__(self, config: BMEConfig):
        super().__init__()
        self.head_dim = config.hidden_size // config.num_heads
        self.num_heads = config.num_heads
        
        self.eeg_Q = torch.nn.Linear(config.hidden_size, config.num_heads * self.head_dim)
        self.eeg_K = torch.nn.Linear(config.hidden_size, config.num_heads * self.head_dim)
        self.eeg_V = torch.nn.Linear(config.hidden_size, config.num_heads * self.head_dim)

        self.ecg_Q = torch.nn.Linear(config.hidden_size, config.num_heads * self.head_dim)
        self.ecg_K = torch.nn.Linear(config.hidden_size, config.num_heads * self.head_dim)
        self.ecg_V = torch.nn.Linear(config.hidden_size, config.num_heads * self.head_dim)

        # self.speech_Q = torch.nn.Linear(config.hidden_size, config.num_heads * self.head_dim)
        # self.speech_K = torch.nn.Linear(config.hidden_size, config.num_heads * self.head_dim)
        # self.speech_V = torch.nn.Linear(config.hidden_size, config.num_heads * self.head_dim)
        # 聚合三个模态权重 缩放正弦函数
        self.MulitModalAggregatedWeight = torch.nn.Parameter(torch.ones(1, 1, config.hidden_size * 2))
        # self.scale = torch.nn.Parameter(torch.ones(1, 1, config.hidden_size * 2))
        
        self.W_out = torch.nn.Linear(config.num_heads * self.head_dim * 2, config.hidden_size)
    
    def forward(
        self,
        eeg: torch.Tensor,
        ecg: torch.Tensor,
        # speech: torch.Tensor,
    ) -> torch.Tensor:
        batch_size, sequence_length, hidden_dim = eeg.shape

        # eeg ecg speech Q K V
        eeg_Q = self.eeg_Q(eeg).reshape(batch_size, sequence_length, self.num_heads, self.head_dim).transpose(1, 2)
        eeg_K = self.eeg_K(eeg).reshape(batch_size, sequence_length, self.num_heads, self.head_dim).transpose(1, 2)
        eeg_V = self.eeg_V(eeg).reshape(batch_size, sequence_length, self.num_heads, self.head_dim).transpose(1, 2)
        
        ecg_Q = self.ecg_Q(ecg).reshape(batch_size, sequence_length, self.num_heads, self.head_dim).transpose(1, 2)
        ecg_K = self.ecg_K(ecg).reshape(batch_size, sequence_length, self.num_heads, self.head_dim).transpose(1, 2)
        ecg_V = self.ecg_V(ecg).reshape(batch_size, sequence_length, self.num_heads, self.head_dim).transpose(1, 2)

        # speech_Q = self.speech_Q(speech).reshape(batch_size, sequence_length, self.num_heads, self.head_dim).transpose(1, 2)
        # speech_K = self.speech_K(speech).reshape(batch_size, sequence_length, self.num_heads, self.head_dim).transpose(1, 2)
        # speech_V = self.speech_V(speech).reshape(batch_size, sequence_length, self.num_heads, self.head_dim).transpose(1, 2)
        
        offset = 64
        query_length = eeg_Q.size(1)
        key_length = eeg_K.size(1)
        logn = torch.arange(offset+1, offset+key_length+1, dtype=torch.float32, device=eeg.device)[-query_length:]
        base = torch.tensor(2048).to(eeg.device)
        logn = torch.log(logn) / torch.log(base)
        logn[logn < 1.0] = 1.0
        logn = logn.to(eeg.dtype).reshape(1, query_length, 1, 1)

        eeg_Q = eeg_Q * logn
        ecg_Q = ecg_Q * logn
        # speech_Q = speech_Q * logn

        device = eeg.device
        eeg_QK = torch.matmul(eeg_Q, eeg_K.transpose(-2, -1))
        eeg_QK = eeg_QK / torch.sqrt(torch.tensor(eeg_Q.size(-1)).float())
        eeg_QK = eeg_QK.masked_fill(torch.ones(eeg_Q.size(-2), eeg_K.size(-2), dtype=torch.bool).to(device).tril(diagonal=0).logical_not(), float('-inf'))
        A_attention_scores = torch.nn.functional.softmax(eeg_QK, dim=-1, dtype=torch.float32)
        A_output = torch.matmul(A_attention_scores, eeg_V)

        ecg_QK = torch.matmul(ecg_Q, ecg_K.transpose(-2, -1))
        ecg_QK = ecg_QK / torch.sqrt(torch.tensor(ecg_Q.size(-1)).float())
        ecg_QK = ecg_QK.masked_fill(torch.ones(ecg_Q.size(-2), ecg_K.size(-2), dtype=torch.bool).to(device).tril(diagonal=0).logical_not(), float('-inf'))
        B_attention_scores = torch.nn.functional.softmax(ecg_QK, dim=-1, dtype=torch.float32)
        B_output = torch.matmul(B_attention_scores, ecg_V)

        # speech_QK = torch.matmul(speech_Q, speech_K.transpose(-2, -1))
        # speech_QK = speech_QK / torch.sqrt(torch.tensor(speech_Q.size(-1)).float())
        # speech_QK = speech_QK.masked_fill(torch.ones(speech_Q.size(-2), speech_K.size(-2), dtype=torch.bool).to(device).tril(diagonal=0).logical_not(), float('-inf'))
        # C_attention_scores = torch.nn.functional.softmax(speech_QK, dim=-1, dtype=torch.float32)
        # C_output = torch.matmul(C_attention_scores, speech_V)

        # 拼接多头注意力
        eeg_attention = A_output.transpose(1, 2).contiguous().reshape(batch_size, sequence_length, hidden_dim)
        ecg_attention = B_output.transpose(1, 2).contiguous().reshape(batch_size, sequence_length, hidden_dim)
        # speech_attention = C_output.transpose(1, 2).contiguous().reshape(batch_size, sequence_length, hidden_dim)

        
        # 聚合三个模态
        # 在hidden_size维度上拼接
        attention = torch.cat([eeg_attention, ecg_attention], dim=-1)
        # 在最后一个维度上乘以权重
        attention = attention * torch.exp(self.MulitModalAggregatedWeight)
        # / torch.log(self.scale)
        attention = self.W_out(attention)
        return attention


class MLP(torch.nn.Module):
    def __init__(self, config: BMEConfig):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.ffn_hidden_size = self.hidden_size * 4

        self.gate_linear = torch.nn.Linear(self.hidden_size, self.ffn_hidden_size)
        self.gate_act = torch.nn.Sigmoid()
        self.up_linear = torch.nn.Linear(self.hidden_size, self.ffn_hidden_size)
        self.up_act = torch.nn.SiLU()
        self.down_linear = torch.nn.Linear(self.ffn_hidden_size, self.hidden_size)

    def forward(self, x):
        return self.down_linear(self.up_act(self.up_linear(x) * self.gate_act(self.gate_linear(x))))


class BMEPreTrainedModel(PreTrainedModel):
    
    config_class = BMEConfig
    base_model_prefix = "backbone"
    _no_split_modules = ["Multimodal_MambaMixer", "MultiModalMultiHeadAggregatedAttention", "MLP"]
    supports_gradient_checkpointing = True

    def _init_weights(self, module):
        """Initialize the weights."""
        if isinstance(module, MambaMixer):
            module.A_log._no_weight_decay = True
            module.D._no_weight_decay = True

            dt_init_std = self.config.time_step_rank**-0.5 * self.config.time_step_scale
            if self.config.time_step_init_scheme == "constant":
                nn.init.constant_(module.dt_proj.weight, dt_init_std)
            elif self.config.time_step_init_scheme == "random":
                nn.init.uniform_(module.dt_proj.weight, -dt_init_std, dt_init_std)

            dt = torch.exp(
                torch.rand(self.config.intermediate_size)
                * (math.log(self.config.time_step_max) - math.log(self.config.time_step_min))
                + math.log(self.config.time_step_min)
            ).clamp(min=self.config.time_step_floor)
            # # Inverse of softplus: https://github.com/pytorch/pytorch/issues/72759
            inv_dt = dt + torch.log(-torch.expm1(-dt))
            with torch.no_grad():
                module.dt_proj.bias.copy_(inv_dt)
            module.dt_proj.bias._no_reinit = True

        if isinstance(module, nn.Linear):
            if module.bias is not None:
                if not getattr(module.bias, "_no_reinit", False):
                    nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, std=self.config.initializer_range)

        if self.config.rescale_prenorm_residual:
            # Reinitialize selected weights subject to the OpenAI GPT-2 Paper Scheme:
            #   > A modified initialization which accounts for the accumulation on the residual path with model depth. Scale
            #   > the weights of residual layers at initialization by a factor of 1/√N where N is the # of residual layers.
            #   >   -- GPT-2 :: https://openai.com/blog/better-language-models/
            #
            # Reference (Megatron-LM): https://github.com/NVIDIA/Megatron-LM/blob/main/megatron/model/gpt_model.py
            for name, p in module.named_parameters():
                if name in ["out_proj.weight"]:
                    # Special Scaled Initialization --> There are 2 Layer Norms per Transformer Block
                    # Following Pytorch init, except scale by 1/sqrt(2 * n_layer)
                    # We need to reinit p since this code could be called multiple times
                    # Having just p *= scale would repeatedly scale it down
                    nn.init.kaiming_uniform_(p, a=math.sqrt(5))
                    with torch.no_grad():
                        p /= math.sqrt(self.config.num_layers)


class BMEModel(BMEPreTrainedModel):

    def __init__(self, config: BMEConfig):
        super().__init__(config)
        self.config = config

        self.embeddings = Embedding(config)

        self.pre_mixer_layernorm = BMERMSNorm(config.hidden_size)
        self.multi_modal_mixer = Multimodal_MambaMixer(config)

        self.pre_attention_layernorm = BMERMSNorm(config.hidden_size)
        self.attention = MultiModalMultiHeadAggregatedAttention(config)

        self.pre_mlp_layernorm = BMERMSNorm(config.hidden_size)
        self.mlp = MLP(config)

        self.hidden_dropout = nn.Dropout(config.hidden_dropout)

        self.final_layer_norm = BMERMSNorm(config.hidden_size)

        self.post_init()

    def get_input_embeddings(self):
        return self.embeddings
    
    def set_input_embeddings(self, value):
        self.embeddings = value


    def forward(
        self,
        eeg_input_ids: Optional[torch.Tensor] = None,
        ecg_input_ids: Optional[torch.Tensor] = None,
        # speech_input_ids: Optional[torch.Tensor] = None,
        cache_params: Optional[MambaCache] = None,
    ):
        # 1. Embeddings
        eeg_embeds, ecg_embeds = self.embeddings(
            eeg_input_ids=eeg_input_ids, 
            ecg_input_ids=ecg_input_ids, 
            # speech_input_ids=speech_input_ids
        )
        
        # 2. Multi-Modal Mixer
        eeg_residual = eeg_embeds
        ecg_residual = ecg_embeds
        # speech_residual = speech_embeds

        eeg_hidden_states = self.pre_mixer_layernorm(eeg_embeds)
        ecg_hidden_states = self.pre_mixer_layernorm(ecg_embeds)
        # speech_hidden_states = self.pre_mixer_layernorm(speech_embeds)

        eeg_hidden_states, ecg_hidden_states = self.multi_modal_mixer(
            eeg_hidden_states, 
            ecg_hidden_states, 
            # speech_hidden_states, 
            cache_params
        )

        eeg_hidden_states = self.hidden_dropout(eeg_hidden_states) if self.training else eeg_hidden_states
        ecg_hidden_states = self.hidden_dropout(ecg_hidden_states) if self.training else ecg_hidden_states
        # speech_hidden_states = self.hidden_dropout(speech_hidden_states) if self.training else speech_hidden_states

        eeg_hidden_states = eeg_hidden_states + eeg_residual
        ecg_hidden_states = ecg_hidden_states + ecg_residual
        # speech_hidden_states = speech_hidden_states + speech_residual

        # 3. Multi-Modal Attention
        eeg_residual = eeg_hidden_states
        ecg_residual = ecg_hidden_states
        # speech_residual = speech_hidden_states

        eeg_hidden_states = self.pre_attention_layernorm(eeg_hidden_states)
        ecg_hidden_states = self.pre_attention_layernorm(ecg_hidden_states)
        # speech_hidden_states = self.pre_attention_layernorm(speech_hidden_states)

        attention_hidden_states = self.attention(eeg_hidden_states, ecg_hidden_states)

        attention_hidden_states = self.hidden_dropout(attention_hidden_states) if self.training else attention_hidden_states

        attention_hidden_states = attention_hidden_states + eeg_residual + ecg_residual

        # 4. MLP
        residual = attention_hidden_states

        hidden_states = self.pre_mlp_layernorm(attention_hidden_states)

        hidden_states = self.mlp(hidden_states)

        hidden_states = self.hidden_dropout(hidden_states) if self.training else hidden_states

        hidden_states = hidden_states + residual

        hidden_states = self.final_layer_norm(hidden_states)

        return hidden_states


@dataclass
class BMEModelSequenceClassificationOutput(BaseModelOutput):
    """
    BMEModelOutput with optional loss.
    """

    last_hidden_state: torch.FloatTensor = None
    logits: torch.FloatTensor = None
    loss: Optional[torch.FloatTensor] = None


class BMEModelForSequenceClassification(BMEPreTrainedModel):

    def __init__(self, config: BMEConfig):
        super().__init__(config)
        self.num_labels = config.num_labels

        self.bme = BMEModel(config)
        self.classifier = nn.Linear(config.hidden_size, self.config.num_labels)

        self.post_init()

    def forward(
        self,
        eeg_input_ids: Optional[torch.Tensor] = None,
        ecg_input_ids: Optional[torch.Tensor] = None,
        # speech_input_ids: Optional[torch.Tensor] = None,
        cache_params: Optional[MambaCache] = None,
        labels: Optional[torch.Tensor] = None,
    ):
        hidden_states = self.bme(
            eeg_input_ids=eeg_input_ids, 
            ecg_input_ids=ecg_input_ids, 
            # speech_input_ids=speech_input_ids, 
            cache_params=cache_params
        )

        logits = self.classifier(hidden_states[:, -1])

        if labels is not None:
            if self.num_labels == 1:
                loss_fct = nn.MSELoss()
                loss = loss_fct(logits.view(-1), labels.view(-1))
            else:
                loss_fct = nn.CrossEntropyLoss()
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            

        return BMEModelSequenceClassificationOutput(
            last_hidden_state=hidden_states,
            logits=logits,
            loss=loss,
        )

    def get_input_embeddings(self):
        return self.bme.get_input_embeddings()
    
    def set_input_embeddings(self, value):
        self.bme.set_input_embeddings(value)
