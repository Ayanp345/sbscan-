"""Safety-training interaction experiments ("does RLHF/DPO destroy backdoors, or hide them?")."""
from .dpo_survival import SurvivalReport, dpo_finetune, run_survival_experiment

__all__ = ["SurvivalReport", "dpo_finetune", "run_survival_experiment"]
