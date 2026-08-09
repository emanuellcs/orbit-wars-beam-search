#pragma once

#include <cstdint>

#include "board.hpp"
#include "config.hpp"
#include "constants.hpp"
#include "observation.hpp"

namespace orbit {

/// @brief Single-step simulator implementing Orbit Wars turn order.
class OrbitSim {
public:
    ///< Mutable state owned by this simulator instance.
    GameState state{};

    /// @brief Reset the simulator state.
    void reset();
    /// @brief Load a fresh observation into the simulator state.
    /// @param obs Fixed-buffer observation input.
    void load_from_observation(const ObservationInput& obs);
    /// @brief Advance exactly one turn with a joint launch list.
    /// @param launches Launches from one or more owners.
    /// @note Launch validation, production, movement, sweeps, and combat are ordered here.
    void step(const LaunchList& launches);
};

/// @brief High-level engine facade exposed to Python.
struct Engine {
    ///< Simulator used both for live state and debug stepping.
    OrbitSim sim{};
    ///< Persisted search/evaluator/candidate configuration applied to every call.
    SearchConfig config{};

    /// @brief Construct an engine with default player id 0.
    Engine();
    /// @brief Construct an engine for a specific player.
    /// @param player Controlled player id.
    explicit Engine(int player);
    /// @brief Replace engine state from a new observation.
    /// @param obs Fixed-buffer observation input.
    void update_observation(const ObservationInput& obs);
    /// @brief Advance internal simulator with explicit launches.
    /// @param launches Launch list to apply.
    void step_actions(const LaunchList& launches);
    /// @brief Choose a launch list using fixed-buffer beam search.
    /// @param time_budget_ms Caller-provided time budget in milliseconds.
    /// @param seed Deterministic seed mixed into worker evaluation.
    /// @return Selected legal launch list.
    /// @note Search clamps the budget to the native hard stop.
    LaunchList choose_actions(int time_budget_ms, uint64_t seed);
    /// @brief Evaluate the current state for debugging/tuning.
    /// @param player Player id to evaluate, or -1 for the engine player.
    /// @return Heuristic state score.
    double debug_evaluate(int player = -1) const;
    /// @brief Replace the persisted search configuration.
    /// @param cfg New configuration; fields are clamped to fixed ranges internally.
    void set_search_config(const SearchConfig& cfg);
    /// @brief Update only the evaluator weight bundle, keeping the rest of the config.
    /// @param weights New evaluator coefficient bundle.
    void set_eval_weights(const EvalWeights& weights);
    /// @brief Update only the candidate weight bundle, keeping the rest of the config.
    /// @param weights New atomic-launch scoring coefficients.
    void set_candidate_weights(const CandidateWeights& weights);
};

}  // namespace orbit
