from dataclasses import dataclass, field, make_dataclass

__all__ = ["CycleGANConfig"]

@dataclass
class SchedulerParams:
    step_size: int = 10
    gamma: float = 0.1

@dataclass
class Scheduler:
    name: str = "step"
    params: SchedulerParams = field(default_factory=SchedulerParams)

@dataclass
class ModelParams:
    in_size: tuple[int, int, int] = (10, 100, 100)
    channels: int = 1
    out_conv_blocks: int = 1
    model_channels: int = 16
    num_res_blocks: tuple[int, int] = (1, 1)
    attention_resolutions: list[int] = field(default_factory=lambda: [4, 8])
    dropout: float = 0.0
    gen_channel_mult: list[int] = field(default_factory=lambda: [1, 2, 2, 4])
    disc_channel_mult: list[int] = field(default_factory=lambda: [4, 4, 8])
    out_mult: int = 1
    gen_z_down: list[int] = field(default_factory=lambda: [2, 4, 8])
    disc_z_down: list[int] = field(default_factory=lambda: [])
    conv_resample: bool = True
    num_heads: int = 8
    activation: str = "silu"

@dataclass
class Model:
    name: str = "CycleGAN"
    checkpoint: str = ""
    params: ModelParams = field(default_factory=ModelParams)

@dataclass
class Setting:
    epoch: int = 30
    batch_size: int = 4
    num_workers: int = 6
    pin_memory: bool = True
    log_every: int = 100
    max_save: int = 30
    max_iters: int = 1600
    device: str = "cuda"
    outdir: str = "outputs/"
    debug: bool = False

@dataclass
class Optimizer:
    lr: float = 1.0e-4
    weight_decay: float = 5.0e-3
    clip_grad: float = 5.0
    

@dataclass
class Dataset:
    source_path: str = "datafiles/20240923-bulk-Hup-train/afm"
    target_path: str = "datafiles/20240923-crop-afm"
    
    num_images: list[int] = field(default_factory=lambda: [4, 3, 3])
    image_size: tuple[int, int] = (100, 100)
    image_split: list[int] = field(default_factory=lambda: [8, 16])
    real_size: tuple[float, ...] = (25.0, 25.0, 3.0)
    ion_type: list[str] = field(default_factory=lambda: ['O', 'H'])
    split: list[float] = field(default_factory=lambda: [0.0, 3.0])
    nms: bool = True

@dataclass
class CycleGANConfig:
    model: Model = field(default_factory=Model)
    setting: Setting = field(default_factory=Setting)
    optimizer: Optimizer = field(default_factory=Optimizer)
    scheduler: Scheduler = field(default_factory=Scheduler)
    dataset: Dataset = field(default_factory=Dataset)

