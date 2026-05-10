#pragma once

#include "orbit_engine.hpp"

namespace orbit {

struct SearchConfig {
    int beam_width = 384;
    int search_depth = 8;
    int rollout_horizon = 64;
    int hard_stop_ms = 900;
};

LaunchList beam_search_action(const GameState& state, const SearchConfig& config,
                              int time_budget_ms, uint64_t seed);

}  // namespace orbit
