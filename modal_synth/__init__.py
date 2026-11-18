"""Modal synthesis percussion: struck plates and bars from declared parameters."""

from .physics import Material, Mode, plate_modes, free_free_bar_modes
from .resonator import ModalOscillator
from .synth import Mallet, ModalInstrument, StrikeReport
from .models import build, BUILDERS, InstrumentSpec
from .audio import to_wav_bytes

__all__ = [
    "Material",
    "Mode",
    "plate_modes",
    "free_free_bar_modes",
    "ModalOscillator",
    "Mallet",
    "ModalInstrument",
    "StrikeReport",
    "build",
    "BUILDERS",
    "InstrumentSpec",
    "to_wav_bytes",
]
