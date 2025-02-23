from dataclasses import dataclass, field

__all__ = ["cVAEConfig"]

@dataclass
class SchedulerParams:
    step_size: int = 1
    gamma: float = 0.9


@dataclass
class Scheduler:
    name: str = "step"
    params: SchedulerParams = field(default_factory=SchedulerParams)


@dataclass
class ModelParams:
    # Model params
    in_channels: int = 4
    model_channels: int = 16
    latent_channels: int = 8
    in_size: tuple[int, ...] = (6, 25, 25)
    channel_mult: list[int] = field(default_factory=lambda: [1, 2, 2, 4])
    z_down: list[int] = field(default_factory=lambda: [1, 2])
    cond_in_size: tuple[int, ...] = (2, 25, 25)
    cond_z_down: list[int] = field(default_factory=lambda: [1])
    attention_resolutions: list[int] = field(default_factory=lambda: [4, 8])
    dropout: float = 0.0
    num_res_blocks: int = 1
    use_gated_conv: bool = True
    gated_conv_heads: int = 16
    # Loss params
    conf_weight: float = 1.0
    offset_weight: float = 0.25
    vae_weight: float = 1.0
    pos_weight: float = 5.0


@dataclass
class Model:
    name: str = "ConditionalVAE"
    checkpoint: str = ""
    params: ModelParams = field(default_factory=ModelParams)


@dataclass
class Dataset:
    train_path: str = "datafiles/hdf/surface_basal_train.hdf5"
    test_path: str = "datafiles/hdf/surface_basal_test.hdf5"
    
    real_size: tuple[float, ...] = (25.0, 25.0, 16.0)
    split: list[float] = field(default_factory=lambda: [0.0, 4.0, 8.0, 12.0, 16.0])


@dataclass
class Setting:
    batch_size: int = 8
    epoch: int = 50
    log_every: int = 100
    max_save: int = 3
    num_workers: int = 4
    pin_memory: bool = True
    device: str = "cuda"
    outdir: str = "outputs/"
    debug: bool = False
    
    
@dataclass
class Optimizer:
    lr: float = 1e-3
    betas: tuple[float, float] = (0.9, 0.999)
    weight_decay: float = 1e-5
    clip_grad: float = 1.0


@dataclass
class cVAEConfig:
    setting: Setting = field(default_factory=Setting)
    dataset: Dataset = field(default_factory=Dataset)
    model: Model = field(default_factory=Model)
    optimizer: Optimizer = field(default_factory=Optimizer)
    scheduler: Scheduler = field(default_factory=Scheduler)
