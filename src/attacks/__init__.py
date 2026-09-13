try:
    from .real_attacks import (
        FGSM, PGD_Linf, PGD_L2, CW_Detection, DAG, TOG,
        UltralyticsAttackWrapper, HuggingFaceAttackWrapper,
        build_attack, build_all_attacks, get_attack_info, ATTACK_CATALOG,
        AttackResult,
    )
except ImportError:
    pass
