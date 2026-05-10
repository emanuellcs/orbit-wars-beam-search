#pragma once

#include "orbit_engine.hpp"

namespace orbit {

double speed_for_ships(int ships);
bool segment_circle_hit(Vec2 a, Vec2 b, Vec2 center, double radius, double& t_hit);
bool segment_exits_board(Vec2 a, Vec2 b, double& t_exit);
bool swept_point_by_segment(Vec2 point, Vec2 a, Vec2 b, double radius);
bool swept_point_by_orbit_arc(Vec2 point, Vec2 old_center, Vec2 next_center,
                              double orbit_radius, double radius, double angular_velocity);
double heading_to(Vec2 from, Vec2 to);
Vec2 point_on_heading(Vec2 origin, double angle, double distance);
bool solve_intercept(const GameState& state, int source_index, int target_index,
                     int ships, double& tau, double& angle);

}  // namespace orbit
