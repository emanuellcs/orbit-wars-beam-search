/**
 * @file orbit_engine_geometry.cpp
 * @brief Geometry implementation for fleet motion, swept collisions, and intercepts.
 *
 * The search relies on analytic geometry instead of sampled headings. These
 * routines compute continuous segment hits and bounded moving-target
 * time-to-intercept values while remaining allocation-free on the hot path.
 */
#include "geometry.hpp"

#include "orbit_engine_internal.hpp"

#include <algorithm>
#include <cmath>

namespace orbit {
namespace {

/**
 * @brief Test whether an angle lies on a directed one-tick orbital sweep.
 * @param start Starting angular position in radians.
 * @param end Ending angular position in radians.
 * @param value Candidate angle to test.
 * @param velocity Signed angular velocity; its sign defines sweep direction.
 * @return true when value is on the directed arc from start to end.
 * @note Handles wrap-around by normalizing offsets into [0, 2*pi).
 */
bool angle_in_sweep(double start, double end, double value, double velocity) {
    start = detail::normalize_angle(start);
    end = detail::normalize_angle(end);
    value = detail::normalize_angle(value);
    if (std::abs(velocity) <= detail::EPS) {
        return std::abs(detail::angle_delta(start, value)) <= 1.0e-7;
    }
    if (velocity > 0.0) {
        const double span = detail::normalize_angle(end - start);
        const double offset = detail::normalize_angle(value - start);
        return offset <= span + 1.0e-7;
    }
    const double span = detail::normalize_angle(start - end);
    const double offset = detail::normalize_angle(start - value);
    return offset <= span + 1.0e-7;
}

/**
 * @brief Residual for the moving-target intercept equation.
 * @param state Current game state.
 * @param source_index Source planet SoA index.
 * @param target_index Target planet SoA index.
 * @param ships Packet size determining fleet speed.
 * @param tau Candidate time-to-intercept in ticks.
 * @return distance(source, target(tau)) - speed(ships) * tau.
 * @note A root at zero means the fleet and target arrive at the same point.
 */
double intercept_residual(const GameState& state, int source_index, int target_index,
                          int ships, double tau) {
    const Vec2 source{state.planets.x[static_cast<size_t>(source_index)],
                      state.planets.y[static_cast<size_t>(source_index)]};
    const Vec2 target = state.planet_position_after(target_index, tau);
    return detail::distance(source, target) - speed_for_ships(ships) * tau;
}

}  // namespace

/**
 * @brief Compute the Orbit Wars logarithmic fleet speed.
 * @param ships Fleet size.
 * @return Speed in board units per tick, clamped to [1, DEFAULT_SHIP_SPEED].
 * @note The power curve makes packet size part of tactical timing: larger
 *       launches travel faster, but speed saturates at the rule maximum.
 */
double speed_for_ships(int ships) {
    if (ships <= 1) {
        return 1.0;
    }
    const double ratio = std::log(static_cast<double>(std::max(1, ships))) / std::log(1000.0);
    const double curved = std::pow(std::max(0.0, ratio), 1.5);
    const double speed = 1.0 + (DEFAULT_SHIP_SPEED - 1.0) * curved;
    return std::min(DEFAULT_SHIP_SPEED, std::max(1.0, speed));
}

/**
 * @brief Intersect a segment against a circle using the quadratic equation.
 * @param a Segment start point.
 * @param b Segment end point.
 * @param center Circle center.
 * @param radius Circle radius.
 * @param t_hit First hit fraction along the segment when an intersection exists.
 * @return true when the closed segment starts inside or intersects the circle.
 * @note Continuous collision is required because fleets can cross thin targets
 *       between integer ticks.
 */
bool segment_circle_hit(Vec2 a, Vec2 b, Vec2 center, double radius, double& t_hit) {
    const Vec2 d{b.x - a.x, b.y - a.y};
    const Vec2 f{a.x - center.x, a.y - center.y};
    const double aa = d.x * d.x + d.y * d.y;
    const double bb = 2.0 * (f.x * d.x + f.y * d.y);
    const double cc = f.x * f.x + f.y * f.y - radius * radius;
    if (cc <= 0.0) {
        t_hit = 0.0;
        return true;
    }
    if (aa <= detail::EPS) {
        return false;
    }
    const double disc = bb * bb - 4.0 * aa * cc;
    if (disc < 0.0) {
        return false;
    }
    // Earliest valid root wins because combat should occur at the first body hit
    // on this tick, not at the endpoint.
    const double root = std::sqrt(std::max(0.0, disc));
    const double inv = 1.0 / (2.0 * aa);
    const double t0 = (-bb - root) * inv;
    const double t1 = (-bb + root) * inv;
    if (t0 >= -1.0e-9 && t0 <= 1.0 + 1.0e-9) {
        t_hit = detail::clamp(t0, 0.0, 1.0);
        return true;
    }
    if (t1 >= -1.0e-9 && t1 <= 1.0 + 1.0e-9) {
        t_hit = detail::clamp(t1, 0.0, 1.0);
        return true;
    }
    return false;
}

/**
 * @brief Determine whether a segment exits the board during a tick.
 * @param a Segment start point.
 * @param b Segment end point.
 * @param t_exit Approximate boundary crossing fraction.
 * @return true when the segment endpoint is outside the board.
 * @note Only endpoint-outside paths matter here because fleets already inside
 *       the convex square cannot leave and re-enter within one straight segment.
 */
bool segment_exits_board(Vec2 a, Vec2 b, double& t_exit) {
    if (detail::inside_board(b)) {
        return false;
    }
    t_exit = 1.0;
    const double dx = b.x - a.x;
    const double dy = b.y - a.y;
    if (dx < -detail::EPS) {
        t_exit = std::min(t_exit, (0.0 - a.x) / dx);
    } else if (dx > detail::EPS) {
        t_exit = std::min(t_exit, (BOARD_SIZE - a.x) / dx);
    }
    if (dy < -detail::EPS) {
        t_exit = std::min(t_exit, (0.0 - a.y) / dy);
    } else if (dy > detail::EPS) {
        t_exit = std::min(t_exit, (BOARD_SIZE - a.y) / dy);
    }
    t_exit = detail::clamp(t_exit, 0.0, 1.0);
    return true;
}

/**
 * @brief Test a point against a circle swept along a straight segment.
 * @param point Stationary point to test.
 * @param a Segment start center of the moving body.
 * @param b Segment end center of the moving body.
 * @param radius Moving body radius.
 * @return true when the closest point on the swept segment is within radius.
 * @note This is used for comets sweeping stationary fleet positions after fleet
 *       movement, matching the environment's continuous-body intent.
 */
bool swept_point_by_segment(Vec2 point, Vec2 a, Vec2 b, double radius) {
    const Vec2 d{b.x - a.x, b.y - a.y};
    const double len2 = d.x * d.x + d.y * d.y;
    double t = 0.0;
    if (len2 > detail::EPS) {
        t = ((point.x - a.x) * d.x + (point.y - a.y) * d.y) / len2;
        t = detail::clamp(t, 0.0, 1.0);
    }
    const Vec2 closest{a.x + d.x * t, a.y + d.y * t};
    return detail::distance(point, closest) <= radius + 1.0e-9;
}

/**
 * @brief Test a point against a circular body swept along an orbital arc.
 * @param point Stationary point to test.
 * @param old_center Body center before orbit motion.
 * @param next_center Body center after orbit motion.
 * @param orbit_radius Radius of the body's circular path.
 * @param radius Body collision radius.
 * @param angular_velocity Signed angular velocity for this body.
 * @return true when the point falls in the swept angular interval and radial band.
 * @note The radial-band check is the arc analogue of segment closest distance.
 */
bool swept_point_by_orbit_arc(Vec2 point, Vec2 old_center, Vec2 next_center,
                              double orbit_radius, double radius, double angular_velocity) {
    const Vec2 c{CENTER_X, CENTER_Y};
    if (detail::distance(point, old_center) <= radius + 1.0e-9 ||
        detail::distance(point, next_center) <= radius + 1.0e-9) {
        return true;
    }
    if (orbit_radius <= detail::EPS) {
        return false;
    }
    const double point_radius = detail::distance(point, c);
    const double point_angle = std::atan2(point.y - CENTER_Y, point.x - CENTER_X);
    const double old_angle = std::atan2(old_center.y - CENTER_Y, old_center.x - CENTER_X);
    const double next_angle = std::atan2(next_center.y - CENTER_Y, next_center.x - CENTER_X);
    if (angle_in_sweep(old_angle, next_angle, point_angle, angular_velocity)) {
        return std::abs(point_radius - orbit_radius) <= radius + 1.0e-9;
    }
    return false;
}

/**
 * @brief Compute an atan2 heading between two points.
 * @param from Start point.
 * @param to Target point.
 * @return Heading angle in radians.
 */
double heading_to(Vec2 from, Vec2 to) {
    return std::atan2(to.y - from.y, to.x - from.x);
}

/**
 * @brief Advance a point along a heading.
 * @param origin Start point.
 * @param angle Heading in radians.
 * @param distance Distance in board units.
 * @return Translated point.
 */
Vec2 point_on_heading(Vec2 origin, double angle, double distance) {
    return Vec2{origin.x + std::cos(angle) * distance, origin.y + std::sin(angle) * distance};
}

/**
 * @brief Solve launch time and angle for a static, orbiting, or comet target.
 * @param state Current game state.
 * @param source_index Source planet SoA index.
 * @param target_index Target planet SoA index.
 * @param ships Packet size, which defines fleet speed.
 * @param tau Output time-to-intercept in ticks.
 * @param angle Output heading in radians.
 * @return true when the target is reachable within the bounded horizon.
 * @note Moving targets solve f(tau)=0 with secant proposals guarded by
 *       bisection. The bisection fallback ensures convergence even when comet
 *       interpolation or circular motion makes the residual non-linear.
 */
bool solve_intercept(const GameState& state, int source_index, int target_index,
                     int ships, double& tau, double& angle) {
    if (source_index < 0 || target_index < 0 || ships <= 0) {
        return false;
    }
    const Vec2 source{state.planets.x[static_cast<size_t>(source_index)],
                      state.planets.y[static_cast<size_t>(source_index)]};
    const double speed = speed_for_ships(ships);
    if (speed <= 0.0) {
        return false;
    }

    if (state.planets.is_orbiting[static_cast<size_t>(target_index)] == 0 &&
        state.planets.is_comet[static_cast<size_t>(target_index)] == 0) {
        const Vec2 target{state.planets.x[static_cast<size_t>(target_index)],
                          state.planets.y[static_cast<size_t>(target_index)]};
        tau = std::max(1.0, detail::distance(source, target) / speed);
        Vec2 spawn = source;
        double aim = heading_to(spawn, target);
        // Launches spawn just outside the source radius, so aiming from the
        // center would be slightly biased for close planets. Two refinement
        // passes are enough because the spawn offset is tiny.
        for (int i = 0; i < 2; ++i) {
            spawn = point_on_heading(source, aim, state.planets.radius[static_cast<size_t>(source_index)] + 1.0e-3);
            aim = heading_to(spawn, target);
        }
        angle = aim;
        return true;
    }

    double lo = 1.0;
    double hi = 120.0;
    double flo = intercept_residual(state, source_index, target_index, ships, lo);
    double fhi = intercept_residual(state, source_index, target_index, ships, hi);
    if (flo <= 0.0) {
        tau = lo;
    } else if (fhi > 0.0) {
        double best_tau = lo;
        double best_abs = std::abs(flo);
        // If no sign change exists in the normal horizon, keep the best
        // near-miss only when it is plausibly capturable. This avoids wasting
        // macro slots on targets that fast comets have already outrun.
        for (int i = 2; i <= 120; ++i) {
            const double t = static_cast<double>(i);
            const double cur = std::abs(intercept_residual(state, source_index, target_index, ships, t));
            if (cur < best_abs) {
                best_abs = cur;
                best_tau = t;
            }
        }
        if (best_abs > speed * 2.0) {
            return false;
        }
        tau = best_tau;
    } else {
        double a = lo;
        double b = hi;
        double fa = flo;
        double fb = fhi;
        for (int iter = 0; iter < 28; ++iter) {
            double mid = 0.5 * (a + b);
            if (std::abs(fb - fa) > 1.0e-8) {
                const double secant = b - fb * (b - a) / (fb - fa);
                if (secant > a && secant < b) {
                    mid = secant;
                }
            }
            const double fm = intercept_residual(state, source_index, target_index, ships, mid);
            if (std::abs(fm) < 1.0e-6) {
                a = mid;
                b = mid;
                break;
            }
            if (fm > 0.0) {
                a = mid;
                fa = fm;
            } else {
                b = mid;
                fb = fm;
            }
        }
        tau = 0.5 * (a + b);
    }

    const Vec2 target = state.planet_position_after(target_index, tau);
    double aim = heading_to(source, target);
    Vec2 spawn = source;
    // Re-aim from the actual spawn point after tau is known. Three fixed passes
    // keep the correction deterministic and cheaper than another root solve.
    for (int i = 0; i < 3; ++i) {
        spawn = point_on_heading(source, aim, state.planets.radius[static_cast<size_t>(source_index)] + 1.0e-3);
        aim = heading_to(spawn, target);
    }
    angle = aim;
    return tau >= 1.0 && tau <= 120.0;
}

}  // namespace orbit
