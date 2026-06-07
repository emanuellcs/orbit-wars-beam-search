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

/// @brief Historical owned-comet value curve: ``18.0 + 0.35 * ships``.
/// @details Factored out as a constant so the weight-aware evaluator keeps the
///          same numerical answer at the default EvalWeights setting.
constexpr double COMET_OWNED_BASE = 18.0;
constexpr double COMET_OWNED_PER_SHIP = 0.35;
/// @brief Historical enemy-comet penalty curve: ``12.0 + 0.2 * ships``.
constexpr double COMET_ENEMY_BASE = 12.0;
constexpr double COMET_ENEMY_PER_SHIP = 0.2;
/// @brief Historical neutral-comet proximity curve: ``max(0, 10 - 0.1 * ships)``.
constexpr double COMET_NEUTRAL_BASE = 10.0;
constexpr double COMET_NEUTRAL_PER_SHIP = 0.1;

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

/**
 * @brief Score a state using the provided weight bundle.
 * @param state Current or rolled-out game state.
 * @param player Player id to evaluate, or negative to use state.player.
 * @param weights Tunable coefficient bundle.
 * @return Higher-is-better heuristic score.
 * @note The math is identical to the historical formula at default weights so
 *       existing behavior is preserved when no custom weights are injected.
 */
double evaluate_state_impl(const GameState& state, int player, const EvalWeights& weights) {
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
            territory += weights.territory_own * centrality_value(pos, state.planets.production[static_cast<size_t>(p)]);
            if (state.planets.is_comet[static_cast<size_t>(p)] != 0) {
                comet_value += weights.comet_owned *
                               (COMET_OWNED_BASE + ships * COMET_OWNED_PER_SHIP);
            }
        } else if (owner >= 0) {
            opp_ships += ships;
            opp_prod += prod;
            territory -= weights.territory_opp *
                         centrality_value(pos, state.planets.production[static_cast<size_t>(p)]);
            if (state.planets.is_comet[static_cast<size_t>(p)] != 0) {
                comet_value -= weights.comet_enemy *
                               (COMET_ENEMY_BASE + ships * COMET_ENEMY_PER_SHIP);
            }
        } else if (state.planets.is_comet[static_cast<size_t>(p)] != 0) {
            comet_value += weights.comet_neutral *
                           std::max(0.0, COMET_NEUTRAL_BASE - ships * COMET_NEUTRAL_PER_SHIP);
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

    return weights.ship * (own_ships - opp_ships) +
           weights.production * (own_prod - opp_prod) +
           territory -
           weights.threat * incoming_threat(state, eval_player) +
           comet_value;
}

}  // namespace

/**
 * @brief Convenience overload with default weights.
 * @param state Current or rolled-out game state.
 * @param player Player id to evaluate, or negative to use state.player.
 * @return Heuristic score using EvalWeights{}.
 */
double evaluate_state(const GameState& state, int player) {
    return evaluate_state_impl(state, player, EvalWeights{});
}

/**
 * @brief Weight-aware evaluator entry point.
 * @param state Current or rolled-out game state.
 * @param player Player id to evaluate, or negative to use state.player.
 * @param weights Tunable coefficient bundle.
 * @return Heuristic score.
 */
double evaluate_state(const GameState& state, int player, const EvalWeights& weights) {
    return evaluate_state_impl(state, player, weights);
}

}  // namespace orbit
