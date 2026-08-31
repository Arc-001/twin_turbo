"""Dataset-level configuration for the C-MAPSS phase-1 pipeline."""

from dataclasses import dataclass, field

OP_SETTING_COLS = ["os1", "os2", "os3"]
SENSOR_COLS = [f"s{i}" for i in range(1, 22)]
INDEX_COLS = ["unit", "cycle"]
ALL_COLS = INDEX_COLS + OP_SETTING_COLS + SENSOR_COLS

DEFAULT_WINDOW_SIZE = 30
DEFAULT_RUL_CLIP = 125
CONSTANT_SENSOR_STD_THRESHOLD = 1e-5


@dataclass(frozen=True)
class DatasetVariant:
    name: str
    n_regimes: int
    n_fault_modes: int


VARIANTS: dict[str, DatasetVariant] = {
    "FD001": DatasetVariant("FD001", n_regimes=1, n_fault_modes=1),
    "FD002": DatasetVariant("FD002", n_regimes=6, n_fault_modes=1),
    "FD003": DatasetVariant("FD003", n_regimes=1, n_fault_modes=2),
    "FD004": DatasetVariant("FD004", n_regimes=6, n_fault_modes=2),
}


def get_variant(name: str) -> DatasetVariant:
    key = name.upper()
    if key not in VARIANTS:
        raise ValueError(f"Unknown dataset variant: {name!r}. Expected one of {list(VARIANTS)}")
    return VARIANTS[key]
