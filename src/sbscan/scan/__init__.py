"""The scan pipeline: reference -> sweeps -> dormant circuits -> causal verification -> SAE
dossier -> fused verdict. ``TrojanCircuitHunter`` in ``hunter.py`` orchestrates all of it."""
from .dormant import DormantNeuron, Inversion, find_dormant_potent, invert_trigger, minimize_edits
from .hunter import Candidate, ScanConfig, TrojanCircuitHunter
from .patching import CausalReport, causal_report, payload_prob
from .reference import Reference, build_reference
from .report import ScanReport
from .sae import FeatureDossier, TopKSAE, sae_dossier, train_sae
from .sweep import PairSweep, SweepResult, pair_sweep, select_candidates, token_sweep

__all__ = [
    "TrojanCircuitHunter", "ScanConfig", "Candidate", "Reference", "build_reference",
    "SweepResult", "PairSweep", "token_sweep", "pair_sweep", "select_candidates",
    "DormantNeuron", "Inversion", "find_dormant_potent", "invert_trigger", "minimize_edits",
    "CausalReport", "causal_report", "payload_prob", "TopKSAE", "train_sae",
    "FeatureDossier", "sae_dossier", "ScanReport",
]
