/**
 * @file geometry.hpp
 * @brief Continuous-motion geometry primitives for Orbit Wars tactics.
 *
 * The simulator and candidate generator share these helpers for fleet speed,
 * segment-circle collision, swept moving-planet checks, and moving-target
 * interception. Keeping this math native avoids Python overhead and preserves
 * deterministic behavior across rollouts.
 */
#pragma once

#include "orbit_engine.hpp"

namespace orbit {

/// @brief Compute rule-defined fleet speed from launched ship count.
/// @param ships Positive ship count; non-positive values are clamped to probe speed.
/// @return Speed in board units per tick.
double speed_for_ships(int ships);
/// @brief Intersect a closed segment with a circle.
/// @param a Segment start point.
/// @param b Segment end point.
/// @param center Circle center.
/// @param radius Circle radius.
/// @param t_hit Output param receiving first hit fraction in [0,1].
/// @return true when the segment touches or starts inside the circle.
/// @note Solves the quadratic closest approach exactly for continuous collision.
bool segment_circle_hit(Vec2 a, Vec2 b, Vec2 center, double radius, double& t_hit);
/// @brief Detect whether a fleet segment leaves the square board.
/// @param a Segment start point.
/// @param b Segment end point.
/// @param t_exit Output param receiving approximate exit fraction in [0,1].
/// @return true when the segment endpoint is outside the board.
bool segment_exits_board(Vec2 a, Vec2 b, double& t_exit);
/// @brief Test whether a moving circular body following a segment sweeps a point.
/// @param point Stationary fleet point.
/// @param a Swept body start center.
/// @param b Swept body end center.
/// @param radius Swept body radius.
/// @return true when the point is within radius of the swept segment.
bool swept_point_by_segment(Vec2 point, Vec2 a, Vec2 b, double radius);
/// @brief Test whether an orbiting circular body sweeps a point over one tick.
/// @param point Stationary fleet point.
/// @param old_center Body center before motion.
/// @param next_center Body center after motion.
/// @param orbit_radius Radius of the circular orbit around the sun.
/// @param radius Body collision radius.
/// @param angular_velocity Signed angular velocity in radians per tick.
/// @return true when the point lies inside the angular sweep and radial band.
/// @note Avoids sampling the arc, which keeps moving-planet collision deterministic.
bool swept_point_by_orbit_arc(Vec2 point, Vec2 old_center, Vec2 next_center,
                              double orbit_radius, double radius, double angular_velocity);
/// @brief Compute a heading from one point to another.
/// @param from Start point.
/// @param to Target point.
/// @return atan2 heading in radians.
double heading_to(Vec2 from, Vec2 to);
/// @brief Move from an origin along a heading by a distance.
/// @param origin Start point.
/// @param angle Heading in radians.
/// @param distance Distance in board units.
/// @return Resulting point.
Vec2 point_on_heading(Vec2 origin, double angle, double distance);
/// @brief Solve a launch heading and time-to-intercept for a target planet.
/// @param state Current game state.
/// @param source_index SoA index of the source planet.
/// @param target_index SoA index of the target planet or comet.
/// @param ships Packet size, which determines fleet speed.
/// @param tau Output time-to-intercept in ticks.
/// @param angle Output launch heading in radians.
/// @return true when a bounded intercept is available.
/// @note Moving targets use secant-assisted bisection over tau in [1,120].
bool solve_intercept(const GameState& state, int source_index, int target_index,
                     int ships, double& tau, double& angle);

}  // namespace orbit
