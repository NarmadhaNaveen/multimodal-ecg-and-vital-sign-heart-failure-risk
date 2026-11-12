import os
import yaml
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any


@dataclass
class ExperimentConfig:
    name: str = "multimodal_hf_risk"
    seed: int = 42
    device: str = "cuda"
    output_dir: str = "outputs"
    checkpoint_dir: str = "checkpoints"


@dataclass
class DataConfig:
    dataset_name: str = "synthetic"
    raw_dir: str = "data/raw"
    processed_dir: str = "data/processed"
    synthetic_dir: str = "data/synthetic"
    context_window_sec: int = 600
    prediction_horizon_sec: int = 14400
    ecg_sampling_rate: int = 250
    vital_sampling_rate: int = 1
    ecg_channels: List[str] = field(default_factory=lambda: ["II", "V"])
    vital_channels: List[str] = field(default_factory=lambda: ["HR", "SBP", "DBP", "MAP", "SPO2", "RESP"])
    train_split: float = 0.70
    val_split: float = 0.15
    test_split: float = 0.15
    label_definition: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ECGEncoderConfig:
    in_channels: int = 2
    conv_channels: List[int] = field(default_factory=lambda: [32, 64, 128, 256])
    kernel_sizes: List[int] = field(default_factory=lambda: [7, 7, 5, 3])
    stride_downsample: List[int] = field(default_factory=lambda: [4, 4, 4, 4])
    dropout: float = 0.2
    proj_dim: int = 128


@dataclass
class VitalEncoderConfig:
    in_channels: int = 6
    num_channels: List[int] = field(default_factory=lambda: [32, 64, 128])
    kernel_size: int = 3
    dilation_schedule: List[int] = field(default_factory=lambda: [1, 2, 4, 8])
    dropout: float = 0.2
    proj_dim: int = 128
    use_missingness_indicator: bool = True


@dataclass
class TransformerConfig:
    d_model: int = 128
    nhead: int = 4
    num_layers: int = 3
    dim_feedforward: int = 256
    dropout: float = 0.1
    max_seq_len: int = 1024


@dataclass
class CrossAttentionConfig:
    embed_dim: int = 128
    num_heads: int = 4
    dropout: float = 0.1
    fusion_gate: bool = True


@dataclass
class FoundationConfig:
    use_foundation_init: bool = False
    foundation_model_name: str = "chronos-t5-base"
    pretrained_weights_path: Optional[str] = None
    freeze_foundation: bool = False


@dataclass
class ModelConfig:
    architecture: str = "multimodal_transformer_cross_attn"
    num_classes: int = 1
    ecg_encoder: ECGEncoderConfig = field(default_factory=ECGEncoderConfig)
    vital_encoder: VitalEncoderConfig = field(default_factory=VitalEncoderConfig)
    transformer: TransformerConfig = field(default_factory=TransformerConfig)
    cross_attention: CrossAttentionConfig = field(default_factory=CrossAttentionConfig)
    foundation: FoundationConfig = field(default_factory=FoundationConfig)


@dataclass
class TrainingConfig:
    batch_size: int = 8
    gradient_accumulation_steps: int = 2
    learning_rate: float = 0.0003
    weight_decay: float = 0.01
    max_epochs: int = 20
    patience: int = 5
    grad_clip_norm: float = 1.0
    mixed_precision: bool = True
    num_workers: int = 0
    pos_weight: float = 2.5
    smoke_test_samples: int = 32


@dataclass
class TargetMetricsConfig:
    auroc: float = 0.958
    auprc: float = 0.931
    accuracy: float = 0.956
    f1_score: float = 0.936
    ece_before: Optional[float] = None
    ece_after: Optional[float] = None


@dataclass
class EvaluationConfig:
    batch_size: int = 16
    bootstrap_ci_samples: int = 500
    confidence_level: float = 0.95
    ece_bins: int = 10
    temperature_scaling: bool = True
    target_metrics: TargetMetricsConfig = field(default_factory=TargetMetricsConfig)


@dataclass
class ProjectConfig:
    experiment: ExperimentConfig = field(default_factory=ExperimentConfig)
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    evaluation: EvaluationConfig = field(default_factory=EvaluationConfig)

    @classmethod
    def from_yaml(cls, yaml_path: str) -> "ProjectConfig":
        if not os.path.exists(yaml_path):
            raise FileNotFoundError(f"Configuration file not found: {yaml_path}")
        with open(yaml_path, "r", encoding="utf-8") as f:
            raw = yaml.safe_load(f) or {}

        exp = ExperimentConfig(**raw.get("experiment", {}))
        data_cfg = DataConfig(**raw.get("data", {}))

        model_raw = raw.get("model", {})
        ecg_enc = ECGEncoderConfig(**model_raw.get("ecg_encoder", {}))
        vital_enc = VitalEncoderConfig(**model_raw.get("vital_encoder", {}))
        transformer_cfg = TransformerConfig(**model_raw.get("transformer", {}))
        cross_attn_cfg = CrossAttentionConfig(**model_raw.get("cross_attention", {}))
        foundation_cfg = FoundationConfig(**model_raw.get("foundation", {}))
        model_cfg = ModelConfig(
            architecture=model_raw.get("architecture", "multimodal_transformer_cross_attn"),
            num_classes=model_raw.get("num_classes", 1),
            ecg_encoder=ecg_enc,
            vital_encoder=vital_enc,
            transformer=transformer_cfg,
            cross_attention=cross_attn_cfg,
            foundation=foundation_cfg,
        )

        train_cfg = TrainingConfig(**raw.get("training", {}))

        eval_raw = raw.get("evaluation", {})
        target_metrics = TargetMetricsConfig(**eval_raw.get("target_metrics", {}))
        eval_cfg = EvaluationConfig(
            batch_size=eval_raw.get("batch_size", 16),
            bootstrap_ci_samples=eval_raw.get("bootstrap_ci_samples", 500),
            confidence_level=eval_raw.get("confidence_level", 0.95),
            ece_bins=eval_raw.get("ece_bins", 10),
            temperature_scaling=eval_raw.get("temperature_scaling", True),
            target_metrics=target_metrics,
        )

        return cls(
            experiment=exp,
            data=data_cfg,
            model=model_cfg,
            training=train_cfg,
            evaluation=eval_cfg,
        )
