#include "orbit_engine_internal.hpp"

#include <cmath>

namespace orbit::detail {

double clamp(double value, double lo, double hi) {
    if (value < lo) {
        return lo;
    }
    if (value > hi) {
        return hi;
    }
    return value;
}

double sqr(double value) {
    return value * value;
}

double distance(Vec2 a, Vec2 b) {
    return std::hypot(a.x - b.x, a.y - b.y);
}

double normalize_angle(double angle) {
    double out = std::fmod(angle, TWO_PI);
    if (out < 0.0) {
        out += TWO_PI;
    }
    return out;
}

double angle_delta(double from, double to) {
    double delta = normalize_angle(to) - normalize_angle(from);
    if (delta > PI) {
        delta -= TWO_PI;
    } else if (delta < -PI) {
        delta += TWO_PI;
    }
    return delta;
}

bool inside_board(Vec2 p) {
    return p.x >= 0.0 && p.x <= BOARD_SIZE && p.y >= 0.0 && p.y <= BOARD_SIZE;
}

bool comet_id_in_observation(const ObservationInput& obs, int planet_id) {
    for (int i = 0; i < obs.comet_planet_id_count; ++i) {
        if (obs.comet_planet_ids[static_cast<size_t>(i)] == planet_id) {
            return true;
        }
    }
    return false;
}

int comet_slot_flat(int group, int slot) {
    return group * MAX_COMETS_PER_GROUP + slot;
}

int comet_path_flat(int group, int slot, int point) {
    return (group * MAX_COMETS_PER_GROUP + slot) * MAX_COMET_PATH_POINTS + point;
}

uint64_t mix64(uint64_t value) {
    value += 0x9e3779b97f4a7c15ULL;
    value = (value ^ (value >> 30U)) * 0xbf58476d1ce4e5b9ULL;
    value = (value ^ (value >> 27U)) * 0x94d049bb133111ebULL;
    return value ^ (value >> 31U);
}

}  // namespace orbit::detail
