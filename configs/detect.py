from dataclasses import dataclass, field

__all__ = ["DetectConfig"]

@dataclass
class SchedulerParams:
    step_size: int = 3
    gamma: float = 0.3

@dataclass
class Scheduler:
    name: str = "step"
    params: SchedulerParams = field(default_factory=SchedulerParams)

@dataclass
class ModelParams:
    # Model params
    in_size: tuple[int, int, int] = (10, 100, 100)
    in_channels: int = 1
    out_size: tuple[int, int, int] = (4, 32, 32)
    out_channels: list[int] = field(default_factory=lambda: [8])
    model_channels: int = 32
    embedding_input: int = 0
    embedding_channels: int = 128
    num_res_blocks: tuple[int, int] = (1, 1)
    attention_resolutions: list[int] = field(default_factory=lambda: [4, 8])
    dropout: float = 0.1
    channel_mult: list[int] = field(default_factory=lambda: [1, 2, 4, 8])
    out_conv_blocks: int = 2
    out_mult: int = 1
    z_down: list[int] = field(default_factory=lambda: [1, 2, 4])
    conv_resample: bool = True
    num_heads: int = 8
    activation: str = "silu"
    use_gated_conv: bool = False
    gated_conv_heads: int | None = None
    # Loss params
    cls_weight: float = 1.0
    xy_weight: float = 0.5
    z_weight: float = 0.5
    pos_weight: list[float] = field(default_factory=lambda: [5.0, 5.0])

@dataclass
class Model:
    name: str = "UNetND"
    checkpoint: str = ""
    params: ModelParams = field(default_factory=ModelParams)


@dataclass
class TuneModelParams:
    in_size: tuple[int, int, int] = (10, 100, 100)
    channels: int = 1
    out_conv_blocks: int = 1
    model_channels: int = 16
    num_res_blocks: list[int] = field(default_factory=lambda: [1, 1])
    attention_resolutions: list[int] = field(default_factory=lambda: [4, 8])
    dropout: float = 0.0
    gen_channel_mult: list[int] = field(default_factory=lambda: [1, 2, 2, 4])
    disc_channel_mult: list[int] = field(default_factory=lambda: [4, 8, 8])
    out_mult: int = 1
    gen_z_down: list[int] = field(default_factory=lambda: [2, 4, 8])
    disc_z_down: list[int] = field(default_factory=lambda: [])
    conv_resample: bool = True
    num_heads: int = 8
    activation: str = "silu"

@dataclass
class TuneModel:
    name: str               = "CycleGAN"
    checkpoint: str         = ""
    params: TuneModelParams = field(default_factory=TuneModelParams)


@dataclass
class Setting:
    epoch: int          = 30
    batch_size: int     = 8
    num_workers: int    = 6
    pin_memory: bool    = True
    log_every: int      = 100
    max_save: int       = 5
    device: str         = "cuda"
    outdir: str         = "outputs/"
    debug: bool         = False


@dataclass
class Optimizer:
    lr: float           = 1.0e-4
    weight_decay: float = 5.0e-3
    clip_grad: float    = 5.0


@dataclass
class Dataset:
    train_path: str                         = "dataset/.detect-train"
    test_path: str                          = "dataset/.detect-test"
    num_images: list[int]                   = field(default_factory=lambda: [4, 3, 3])
    image_size: tuple[int, int]             = (100, 100)
    image_split: list[int]                  = field(default_factory=lambda: [10, 18])
    real_size: tuple[float, float, float]   = (25.0, 25.0, 3.0)
    box_size: tuple[int, int, int]          = (32, 32, 4)
    ion_type: list[str]                     = field(default_factory=lambda: ['O', 'H'])
    split: list[float]                      = field(default_factory=lambda: [0.0, 1.5, 3.0])
    nms: bool                               = True


@dataclass
class DetectConfig:
    model: Model            = field(default_factory=Model)
    tune_model: TuneModel   = field(default_factory=TuneModel)
    setting: Setting        = field(default_factory=Setting)
    optimizer: Optimizer    = field(default_factory=Optimizer)
    scheduler: Scheduler    = field(default_factory=Scheduler)
    dataset: Dataset        = field(default_factory=Dataset)
