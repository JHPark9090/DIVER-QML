import torch
from torch import nn
from mup import MuReadout
from utils import mup_utils
from models.diver import DIVER
from models.model_builders import MakeModelIgnoreDataInfoList, CustomIdentity
from models.finetune_builders import MLPProjector, get_dims

import sys
sys.path.append('/home/connectome/justin/DIVER_QML')
from QTSTransformer import QuantumTSTransformer


class ReshapeForQTS(nn.Module):
    """
    Reshapes DIVER output from (B, C, N, d_model) to (B, C*d_model, N)
    for QuantumTSTransformer input.

    - B: batch size
    - C: number of channels
    - N: number of patches (timesteps)
    - d_model: embedding dimension

    Output shape (B, C*d_model, N) where:
    - C*d_model becomes feature_dim
    - N becomes n_timesteps
    """
    def __init__(self):
        super().__init__()

    def forward(self, x, data_info_list=None):
        B, C, N, d_model = x.shape
        # Reshape to (B, C*d_model, N) - treating N as timesteps
        x = x.permute(0, 1, 3, 2)  # (B, C, d_model, N)
        x = x.reshape(B, C * d_model, N)  # (B, C*d_model, N)
        return x

class FineTuneModel(nn.Module):
    def __init__(self, params, task_info_dict): 
        super().__init__()
        self.task_info_dict = task_info_dict 
        base_shape_save_dir = params.model_dir
        width = params.width
        depth = params.depth
        mup = params.mup_weights
        patch_size = params.patch_size
        
        with torch.no_grad():
            self.backbone = DIVER(d_model=width, e_layer=depth, mup=mup, patch_size=patch_size)
            if params.mup_weights:
                def _backbone_builder(w: int, d: int):
                    self.backbone= DIVER(d_model=w, e_layer=d, mup=True, patch_size=patch_size)
                    return self.backbone
                identifier="DIVER_iEEG_FINAL_model" 
                if patch_size==50:
                    identifier += "_patch50"
                mup_utils.apply_mup(target_module=self.backbone, model_builder=_backbone_builder, 
                                    identifier=identifier, width=width, depth=depth,
                                    save_dir=base_shape_save_dir)

        self.feature_extraction_func = lambda x, data_info_list=None : x['token_manager_output']['org_x_position_features'] 
        self.ft_model_input_adapter = CustomIdentity()
        self.ft_core_model = CustomIdentity()
        self.ft_model_output_adapter = CustomIdentity()

    def load_backbone_checkpoint(self, foundation_dir, device='cpu', deepspeed_pth_format=False):
        from utils import checkpoint
        checkpoint.load_model_checkpoint(
                    self.backbone, device, foundation_dir, deepspeed_pth_format=deepspeed_pth_format
        )

    def forward(self, x, data_info_list=None):
        x_features = self.backbone(x, data_info_list=data_info_list, use_mask=False, return_encoder_output=True)
        x_features = self.feature_extraction_func(x_features, data_info_list=data_info_list) 
        finetune_model_input = self.ft_model_input_adapter(x_features, data_info_list=data_info_list)
        finetune_model_output = self.ft_core_model(finetune_model_input, data_info_list=data_info_list)
        finetune_model_output = self.ft_model_output_adapter(finetune_model_output, data_info_list=data_info_list)

        return finetune_model_output

class flatten_linear_finetune(FineTuneModel):
    def __init__(self, params, task_info_dict):
        super().__init__(params, task_info_dict)
        use_mup = params.ft_mup
        in_dim, out_dim = get_dims(params, task_info_dict)

        self.ft_model_input_adapter = MakeModelIgnoreDataInfoList(nn.Flatten())
        self.ft_core_model = MakeModelIgnoreDataInfoList(nn.Linear(in_dim, out_dim) if not use_mup
            else MuReadout(in_dim, out_dim, output_mult=1.0))
        self.ft_model_output_adapter = CustomIdentity()
        
        if params.foundation_dir: 
            self.load_backbone_checkpoint(params.foundation_dir, device='cpu', deepspeed_pth_format=params.deepspeed_pth_format)

class flatten_mlp_finetune(FineTuneModel):
    def __init__(self, params, task_info_dict):
        super().__init__(params, task_info_dict)
        configs = get_dims(params, task_info_dict)

        self.ft_model_input_adapter = MakeModelIgnoreDataInfoList(nn.Flatten())
        self.ft_core_model = MLPProjector(configs)
        self.ft_model_output_adapter = CustomIdentity()

        if params.foundation_dir:
            self.load_backbone_checkpoint(params.foundation_dir, device='cpu', deepspeed_pth_format=params.deepspeed_pth_format)


class flatten_qts_finetune(FineTuneModel):
    """
    Fine-tune model using QuantumTSTransformer as the classification head.

    The DIVER backbone is frozen and its output (B, C, N, d_model) is reshaped
    to (B, C*d_model, N) to be processed by the quantum transformer.
    """
    def __init__(self, params, task_info_dict):
        super().__init__(params, task_info_dict)

        # Get dimensions from task info
        qts_config = get_dims(params, task_info_dict)

        feature_dim = qts_config["feature_dim"]  # C * d_model
        n_timesteps = qts_config["n_timesteps"]  # N (number of patches)
        output_dim = qts_config["output_dim"]    # num_classes

        # QTS hyperparameters from params
        n_qubits = getattr(params, 'qts_n_qubits', 4)
        degree = getattr(params, 'qts_degree', 2)
        n_ansatz_layers = getattr(params, 'qts_n_ansatz_layers', 2)
        qts_dropout = getattr(params, 'qts_dropout', 0.1)

        # Device setup
        device = torch.device(f"cuda:{params.cuda}" if torch.cuda.is_available() else "cpu")

        # Reshape adapter: (B, C, N, d_model) -> (B, C*d_model, N)
        self.ft_model_input_adapter = ReshapeForQTS()

        # Quantum Time Series Transformer as classification head
        self.ft_core_model = MakeModelIgnoreDataInfoList(
            QuantumTSTransformer(
                n_qubits=n_qubits,
                n_timesteps=n_timesteps,
                degree=degree,
                n_ansatz_layers=n_ansatz_layers,
                feature_dim=feature_dim,
                output_dim=output_dim,
                dropout=qts_dropout,
                device=device
            )
        )

        self.ft_model_output_adapter = CustomIdentity()

        # Load pretrained DIVER backbone
        if params.foundation_dir:
            self.load_backbone_checkpoint(
                params.foundation_dir,
                device='cpu',
                deepspeed_pth_format=params.deepspeed_pth_format
            )

        # Freeze the backbone (DIVER-1 pretrained weights)
        if getattr(params, 'frozen', True):
            for param in self.backbone.parameters():
                param.requires_grad = False
            print("DIVER backbone frozen. Only QTSTransformer head will be trained.")