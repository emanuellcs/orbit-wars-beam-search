#pragma once

namespace orbit {

/// @brief Tunable coefficients for the heuristic state evaluator.
///
/// @note The defaults reproduce the prior hand-tuned values exactly so the
///       engine behavior is bit-for-bit identical when no custom weights are
///       injected. All members are plain doubles to keep the rollout hot path
///       allocation-free and cache-friendly.
struct EvalWeights {
    ///< Multiplier on the (own_ships - opponent_ships) term.
    double ship = 1.0;
    ///< Multiplier on the (own_production - opponent_production) term.
    double production = 25.0;
    ///< Multiplier on the owned-planet centrality bonus.
    double territory_own = 1.0;
    ///< Multiplier on the opponent-planet centrality penalty (0.65 historically).
    double territory_opp = 0.65;
    ///< Multiplier on the incoming-threat penalty.
    double threat = 1.0;
    ///< Multiplier on the owned-comet value contribution (18.0 + 0.35*ships historically).
    double comet_owned = 1.0;
    ///< Multiplier on the enemy-comet value penalty (12.0 + 0.2*ships historically).
    double comet_enemy = 1.0;
    ///< Multiplier on the neutral-comet proximity value (max(0, 10 - 0.1*ships) historically).
    double comet_neutral = 1.0;
    ///< Multiplier on the projected-garrison timeline term (ships + production * 24).
    double timeline = 0.02;
};

/// @brief Tunable coefficients for the atomic-launch scoring prior.
///
/// @note Defaults reproduce the prior hand-tuned constants exactly so the
///       search frontier is unchanged when no custom weights are injected.
///       All members are plain doubles to keep the hot path allocation-free.
struct CandidateWeights {
    ///< Bonus when the target is owned by an opponent player (historical 42.0).
    double owner_enemy = 42.0;
    ///< Bonus when the target is neutral (historical 18.0).
    double owner_neutral = 18.0;
    ///< Penalty when the target is already owned by the controlled player (historical -40.0).
    double owner_self = -40.0;
    ///< Flat bonus when the target is a comet (historical 14.0).
    double comet_bonus = 14.0;
    ///< Per-production point contribution to the score (historical 24.0).
    double prod_per_unit = 24.0;
    ///< Tactical prior for CaptureExact packets (historical 10.0).
    double kind_exact = 10.0;
    ///< Tactical prior for CaptureOver packets (historical 16.0).
    double kind_over = 16.0;
    ///< Tactical prior for AllSafe packets (historical 8.0).
    double kind_all_safe = 8.0;
    ///< Tactical prior for Harass packets (historical 2.0).
    double kind_harass = 2.0;
    ///< Tactical prior for Reinforce packets (historical 6.0).
    double kind_reinforce = 6.0;
    ///< Penalty per tick of expected arrival time (historical 0.8).
    double eta_discount = 0.8;
    ///< Penalty per ship committed to the launch (historical 0.08).
    double ship_cost = 0.08;
};

/// @brief Tunable native search limits and scoring bundles.
///
/// @note ``beam_width``, ``search_depth``, and ``rollout_horizon`` are clamped
///       internally so a hostile caller cannot blow the fixed-buffer contract.
///       ``eval_weights`` and ``candidate_weights`` are passed by const
///       reference into the hot path to keep rollouts allocation-free.
struct SearchConfig {
    ///< Maximum root macro-actions to evaluate.
    int beam_width = 384;
    ///< Deterministic tactical prefix depth after the root action.
    int search_depth = 8;
    ///< Additional rollout ticks after the tactical prefix.
    int rollout_horizon = 64;
    ///< Native hard deadline in milliseconds.
    int hard_stop_ms = 900;
    ///< Tunable heuristic state evaluator coefficients.
    EvalWeights eval_weights{};
    ///< Tunable atomic-launch scoring weights.
    CandidateWeights candidate_weights{};
};

}  // namespace orbit
