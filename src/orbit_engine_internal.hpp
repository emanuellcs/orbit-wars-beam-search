#pragma once

#include "orbit_engine.hpp"

namespace orbit::detail {

constexpr double PI = 3.141592653589793238462643383279502884;
constexpr double TWO_PI = 2.0 * PI;
constexpr double EPS = 1.0e-9;

double clamp(double value, double lo, double hi);
double sqr(double value);
double distance(Vec2 a, Vec2 b);
double normalize_angle(double angle);
double angle_delta(double from, double to);
bool inside_board(Vec2 p);
bool comet_id_in_observation(const ObservationInput& obs, int planet_id);
int comet_slot_flat(int group, int slot);
int comet_path_flat(int group, int slot, int point);
uint64_t mix64(uint64_t value);

}  // namespace orbit::detail
