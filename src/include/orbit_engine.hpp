#pragma once

/**
 * @brief Public umbrella header for the Orbit Wars native engine.
 *
 * Re-exports the themed headers that together form the engine contract:
 * compile-time limits and Vec2, observation PODs, board/state types, tunable
 * config bundles, and the OrbitSim/Engine facades.
 */

#include "board.hpp"
#include "config.hpp"
#include "constants.hpp"
#include "engine.hpp"
#include "observation.hpp"
