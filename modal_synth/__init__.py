"""Modal synthesis percussion: struck plates and bars from declared parameters."""

from .physics import Material, Mode, plate_modes, free_free_bar_modes
from .resonator import ModalOscillator, ModalVoice
from .synth import (
    Mallet,
    HitEvent,
    HitReport,
    ModalInstrument,
    ModalEngine,
    BodyState,
    ExcitedMode,
)
from .models import build, BUILDERS, InstrumentSpec
from .audio import to_wav_bytes

__all__ = [
    "Material",
    "Mode",
    "plate_modes",
    "free_free_bar_modes",
    "ModalOscillator",
    "ModalVoice",
    "Mallet",
    "HitEvent",
    "HitReport",
    "ModalInstrument",
    "ModalEngine",
    "BodyState",
    "ExcitedMode",
    "build",
    "BUILDERS",
    "InstrumentSpec",
    "to_wav_bytes",
]
