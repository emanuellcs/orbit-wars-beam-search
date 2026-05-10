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
 * @brief Choose actions for the current state with default search settings.
 * @param time_budget_ms Per-turn budget in milliseconds.
 * @param seed Deterministic seed for search branch mixing.
 * @return Selected launch list.
 */
LaunchList Engine::choose_actions(int time_budget_ms, uint64_t seed) {
    SearchConfig config{};
    return beam_search_action(sim.state, config, time_budget_ms, seed);
}

/**
 * @brief Evaluate the current state for debugging or tuning.
 * @param player Player id, or -1 to evaluate sim.state.player.
 * @return Heuristic state score.
 */
double Engine::debug_evaluate(int player) const {
    const int eval_player = (player >= 0 && player < MAX_PLAYERS) ? player : sim.state.player;
    return evaluate_state(sim.state, eval_player);
}

}  // namespace orbit
