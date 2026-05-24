from .state import FEATURE_NAMES_8D, PreferenceState, init_preference_state
from .synthetic_user import SyntheticUser, make_random_synthetic_user
from .loop import SessionResult, StepRecord, run_many_sessions_v0, run_personalization_session_v0
from .control_basis import (
    CONTROL_NAMES_4D,
    CONTROL_NAMES_6D,
    CONTROL_BASIS_4D_TO_8D,
    CONTROL_BASIS_6D_TO_8D,
    CONTROL_DISPLAY_NAMES_4D_RU,
    CONTROL_DISPLAY_NAMES_6D_RU,
    FEATURE_DISPLAY_NAMES_8D_RU,
    get_control_basis,
    get_control_display_names_ru,
)
from .synthetic_dataset import (
    TARGET_MODES,
    USER_ARCHETYPES_8D,
    NORMAL_ARCHETYPES,
    EXTREME_ARCHETYPES,
    INTENSITY_RANGES,
    INTENSITY_PRIORS,
    generate_synthetic_users_dataset,
    load_synthetic_users_dataset,
    row_to_importance,
    row_to_synthetic_user,
    row_to_target,
)
from .batch_eval import DEFAULT_STRATEGIES, run_batch_on_dataset, summarize_by_strategy, win_rates_vs_baseline
from .analysis import (
    DEFAULT_STRATEGY_ORDER,
    STRATEGY_DISPLAY_NAMES_RU,
    display_strategy_name,
    display_group_name,
    merge_sessions_with_user_metadata,
    summarize_by_group,
    winners_by_group,
    pivot_group_metric,
    win_rates_vs_baseline_by_group,
    compare_two_strategies_by_group,
    win_rate_between_strategies_by_group,
    plot_strategy_improvement_bars,
    plot_group_metric_bars,
    plot_win_rate_bars,
)
from .preference_model import LogisticDistancePreferenceModel, cosine_similarity
from .preference_model_eval import (
    PreferenceModelSessionResult,
    PreferenceModelStepRecord,
    run_preference_model_learning_session_v4a,
    run_preference_model_batch_v4a,
    summarize_v4a_by_target_mode,
    save_v4a_outputs,
)
from .preference_model_heldout import (
    HeldoutPairRecord,
    evaluate_model_on_heldout_pairs,
    run_preference_model_heldout_session_v4a1,
    run_preference_model_heldout_batch_v4a1,
    summarize_v4a1_by_target_mode,
    summarize_v4a1_heldout_by_source,
    save_v4a1_outputs,
)
from .preference_model_calibration import (
    CALIBRATION_DISPLAY_NAMES,
    build_calibrated_vectors,
    run_preference_model_calibration_session_v4a2,
    run_preference_model_calibration_batch_v4a2,
    summarize_calibration_by_target_mode,
    summarize_calibration_heldout_by_source,
    save_v4a2_outputs,
)
from .model_guided_pair_generation import (
    MODEL_GUIDED_STRATEGY_DISPLAY_NAMES,
    PopulationPreferencePrior,
    build_population_preference_prior,
    run_model_guided_pair_session_v4b,
    run_model_guided_pair_batch_v4b,
    summarize_model_guided_sessions,
    save_v4b_outputs,
)
