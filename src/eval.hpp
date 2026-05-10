/**
 * @file eval.hpp
 * @brief Heuristic state evaluator used by rollout search and debugging.
 *
 * Evaluation condenses material, production, territory, incoming threat, and
 * comet opportunity into one deterministic score. It is deliberately cheap so
 * many fixed-buffer rollouts can finish before the action deadline.
 */
#pragma once

#include "orbit_engine.hpp"

namespace orbit {

/// @brief Score a game state from a player's perspective.
/// @param state Current or rolled-out game state.
/// @param player Player id to evaluate, or negative to use state.player.
/// @return Higher-is-better heuristic score.
double evaluate_state(const GameState& state, int player);

}  // namespace orbit
