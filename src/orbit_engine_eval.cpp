/**
 * @file orbit_engine_eval.cpp
 * @brief Deterministic heuristic evaluation for search rollouts.
 *
 * The evaluator is intentionally dense and cheap. It rewards material,
 * production, central territory, and comet opportunity while penalizing
 * near-term incoming threats against owned planets.
 */
#include "eval.hpp"

#include "geometry.hpp"
#include "orbit_engine_internal.hpp"

#include <algorithm>
#include <cmath>

namespace orbit {
namespace {

/**
 * @brief Compute a production-weighted centrality bonus.
 * @param p Planet position.
 * @param production Planet production value.
 * @return Territorial value contribution.
 * @note Central planets project more tactical reach, but the value is capped by
 *       board radius to keep edge planets from dominating negatively.
 */
double centrality_value(Vec2 p, int production) {
    const double d = std::hypot(p.x - CENTER_X, p.y - CENTER_Y);
    return static_cast<double>(production) * (50.0 - std::min(50.0, d)) * 0.08;
}

/**
 * @brief Estimate enemy fleet pressure against owned planets.
 * @param state Current game state.
 * @param player Player id whose planets are threatened.
 * @return Threat penalty in ships-equivalent units.
 * @note Uses a short forward segment rather than full path simulation so the
 *       evaluator stays cheap inside every rollout leaf.
 */
double incoming_threat(const GameState& state, int player) {
    double threat = 0.0;
    for (int f = 0; f < state.fleets.count; ++f) {
        if (state.fleets.alive[static_cast<size_t>(f)] == 0 ||
            state.fleets.owner[static_cast<size_t>(f)] == player) {
            continue;
        }
        const Vec2 start{state.fleets.x[static_cast<size_t>(f)], state.fleets.y[static_cast<size_t>(f)]};
        const double travel = state.fleets.speed[static_cast<size_t>(f)] * 18.0;
        const Vec2 end = point_on_heading(start, state.fleets.angle[static_cast<size_t>(f)], travel);
        for (int p = 0; p < state.planets.count; ++p) {
            if (state.planets.alive[static_cast<size_t>(p)] == 0 ||
                state.planets.owner[static_cast<size_t>(p)] != player) {
                continue;
            }
            double t = 0.0;
            const Vec2 center{state.planets.x[static_cast<size_t>(p)],
                              state.planets.y[static_cast<size_t>(p)]};
            if (segment_circle_hit(start, end, center, state.planets.radius[static_cast<size_t>(p)], t)) {
                threat += static_cast<double>(state.fleets.ships[static_cast<size_t>(f)]) * (1.0 - 0.5 * t);
            }
        }
    }
    return threat;
}

}  // namespace

/**
 * @brief Evaluate a state for a player.
 * @param state Current or rolled-out game state.
 * @param player Player id, or negative to use state.player.
 * @return Higher-is-better heuristic score.
 * @note The score blends ships, production, territory, incoming threat, and
 *       comet value to guide root macro-action selection.
 */
double evaluate_state(const GameState& state, int player) {
    const int eval_player = (player >= 0 && player < MAX_PLAYERS) ? player : state.player;
    double own_ships = 0.0;
    double opp_ships = 0.0;
    double own_prod = 0.0;
    double opp_prod = 0.0;
    double territory = 0.0;
    double comet_value = 0.0;

    for (int p = 0; p < state.planets.count; ++p) {
        if (state.planets.alive[static_cast<size_t>(p)] == 0) {
            continue;
        }
        const int owner = state.planets.owner[static_cast<size_t>(p)];
        const double ships = static_cast<double>(state.planets.ships[static_cast<size_t>(p)]);
        const double prod = static_cast<double>(state.planets.production[static_cast<size_t>(p)]);
        const Vec2 pos{state.planets.x[static_cast<size_t>(p)], state.planets.y[static_cast<size_t>(p)]};
        if (owner == eval_player) {
            own_ships += ships;
            own_prod += prod;
            territory += centrality_value(pos, state.planets.production[static_cast<size_t>(p)]);
            if (state.planets.is_comet[static_cast<size_t>(p)] != 0) {
                comet_value += 18.0 + ships * 0.35;
            }
        } else if (owner >= 0) {
            opp_ships += ships;
            opp_prod += prod;
            territory -= centrality_value(pos, state.planets.production[static_cast<size_t>(p)]) * 0.65;
            if (state.planets.is_comet[static_cast<size_t>(p)] != 0) {
                comet_value -= 12.0 + ships * 0.2;
            }
        } else if (state.planets.is_comet[static_cast<size_t>(p)] != 0) {
            comet_value += std::max(0.0, 10.0 - ships * 0.1);
        }
    }

    for (int f = 0; f < state.fleets.count; ++f) {
        if (state.fleets.alive[static_cast<size_t>(f)] == 0) {
            continue;
        }
        if (state.fleets.owner[static_cast<size_t>(f)] == eval_player) {
            own_ships += static_cast<double>(state.fleets.ships[static_cast<size_t>(f)]);
        } else if (state.fleets.owner[static_cast<size_t>(f)] >= 0) {
            opp_ships += static_cast<double>(state.fleets.ships[static_cast<size_t>(f)]);
        }
    }

    return (own_ships - opp_ships) +
           25.0 * (own_prod - opp_prod) +
           territory -
           incoming_threat(state, eval_player) +
           comet_value;
}

}  // namespace orbit
