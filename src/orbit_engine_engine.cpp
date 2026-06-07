/**
 * @file orbit_engine_engine.cpp
 * @brief High-level engine facade used by the pybind11 wrapper.
 *
 * This layer keeps Python-facing operations small: load observations, step
 * explicit actions, run native search, and expose the heuristic evaluator. The
 * heavy logic remains in simulator, candidate, search, and evaluation modules.
 */
#include "orbit_engine.hpp"

#include "eval.hpp"
#include "search.hpp"

#include <algorithm>

namespace orbit {

/**
 * @brief Construct an engine and reset its simulator state.
 */
Engine::Engine() {
    sim.reset();
}

/**
 * @brief Construct an engine for a specific controlled player.
 * @param player Player id assigned by the Kaggle observation.
 */
Engine::Engine(int player) : Engine() {
    sim.state.player = player;
}

/**
 * @brief Replace simulator state from an observation.
 * @param obs Fixed-buffer observation input.
 */
void Engine::update_observation(const ObservationInput& obs) {
    sim.load_from_observation(obs);
}

/**
 * @brief Advance the internal simulator with explicit launch actions.
 * @param launches Launch list to apply.
 */
void Engine::step_actions(const LaunchList& launches) {
    sim.step(launches);
}

/**
 * @brief Choose actions for the current state using the persisted config.
 * @param time_budget_ms Per-turn budget in milliseconds.
 * @param seed Deterministic seed for search branch mixing.
 * @return Selected launch list.
 */
LaunchList Engine::choose_actions(int time_budget_ms, uint64_t seed) {
    return beam_search_action(sim.state, config, time_budget_ms, seed);
}

/**
 * @brief Evaluate the current state for debugging or tuning.
 * @param player Player id, or -1 to evaluate sim.state.player.
 * @return Heuristic state score.
 */
double Engine::debug_evaluate(int player) const {
    const int eval_player = (player >= 0 && player < MAX_PLAYERS) ? player : sim.state.player;
    return evaluate_state(sim.state, eval_player, config.eval_weights);
}

/**
 * @brief Clamp integer search limits to their inclusive valid ranges.
 * @param cfg Configuration to clamp in place.
 * @note Keeps the configuration object safe regardless of who set its fields.
 */
void clamp_search_config(SearchConfig& cfg) {
    cfg.beam_width = std::min(std::max(1, cfg.beam_width), MAX_BEAM_WIDTH);
    cfg.search_depth = std::min(std::max(1, cfg.search_depth), 10);
    cfg.rollout_horizon = std::min(std::max(1, cfg.rollout_horizon), 96);
    if (cfg.hard_stop_ms < 1) {
        cfg.hard_stop_ms = 1;
    } else if (cfg.hard_stop_ms > 900) {
        cfg.hard_stop_ms = 900;
    }
}

/**
 * @brief Replace the persisted search configuration and clamp it.
 * @param cfg New configuration; numeric limits are clamped to safe ranges.
 */
void Engine::set_search_config(const SearchConfig& cfg) {
    config = cfg;
    clamp_search_config(config);
}

/**
 * @brief Update only the evaluator weights inside the persisted config.
 * @param weights New evaluator coefficient bundle.
 */
void Engine::set_eval_weights(const EvalWeights& weights) {
    config.eval_weights = weights;
}

/**
 * @brief Update only the candidate weights inside the persisted config.
 * @param weights New atomic-launch scoring coefficients.
 */
void Engine::set_candidate_weights(const CandidateWeights& weights) {
    config.candidate_weights = weights;
}

}  // namespace orbit
