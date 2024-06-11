
import math
from transformers.configuration_utils import PretrainedConfig
class BMEConfig(PretrainedConfig):
    model_type = "wbh_mamba"

    def __init__(
        self,
        eeg_vocab_size=4096+2,
        ecg_vocab_size=4096+2,
        speech_vocab_size=4096+2,
        hidden_size=512,
        hidden_dropout=0.1,
        state_size=16,
        num_hidden_layers=2,
        num_heads=32,
        layer_norm_epsilon=1e-5,
        pad_token_id=0,
        bos_token_id=0,
        eos_token_id=0,
        expand=2,
        conv_kernel=4,
        use_bias=False,
        use_conv_bias=True,
        hidden_act="silu",
        initializer_range=0.1,
        residual_in_fp32=True,
        time_step_rank="auto",
        time_step_scale=1.0,
        time_step_min=0.001,
        time_step_max=0.1,
        time_step_init_scheme="random",
        time_step_floor=1e-4,
        rescale_prenorm_residual=False,
        use_cache=True,
        num_labels=2,
        **kwargs,
    ):
        
        self.eeg_vocab_size = eeg_vocab_size
        self.ecg_vocab_size = ecg_vocab_size
        self.speech_vocab_size = speech_vocab_size
        self.hidden_size = hidden_size
        self.hidden_dropout = hidden_dropout
        self.state_size = state_size
        self.num_hidden_layers = num_hidden_layers
        self.layer_norm_epsilon = layer_norm_epsilon
        self.conv_kernel = conv_kernel
        self.expand = expand
        self.intermediate_size = int(expand * self.hidden_size)
        self.num_heads = num_heads
        self.bos_token_id = bos_token_id
        self.eos_token_id = eos_token_id
        self.pad_token_id = pad_token_id
        self.use_bias = use_bias
        self.use_conv_bias = use_conv_bias
        self.hidden_act = hidden_act
        self.initializer_range = initializer_range
        self.time_step_rank = math.ceil(self.hidden_size / 16) if time_step_rank == "auto" else time_step_rank
        self.time_step_scale = time_step_scale
        self.time_step_min = time_step_min
        self.time_step_max = time_step_max
        self.time_step_init_scheme = time_step_init_scheme
        self.time_step_floor = time_step_floor
        self.rescale_prenorm_residual = rescale_prenorm_residual
        self.residual_in_fp32 = residual_in_fp32
        self.use_cache = use_cache
        self.num_labels = num_labels

        super().__init__(bos_token_id=bos_token_id, eos_token_id=eos_token_id, pad_token_id=pad_token_id, **kwargs)
