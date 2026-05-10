#pragma once

#include "orbit_engine.hpp"

namespace orbit {

enum class PacketKind : uint8_t {
    CaptureExact = 0,
    CaptureOver,
    Harass,
    AllSafe,
};

struct AtomicLaunch {
    int from_planet_id = -1;
    int source_index = -1;
    int target_index = -1;
    int ships = 0;
    double angle = 0.0;
    double eta = 0.0;
    double score = 0.0;
    PacketKind kind = PacketKind::Harass;
};

struct AtomicLaunchList {
    int count = 0;
    std::array<AtomicLaunch, MAX_ATOMIC_LAUNCHES> items{};

    void clear();
    void insert_sorted(const AtomicLaunch& launch);
};

struct MacroAction {
    LaunchList launches{};
    double score = 0.0;
};

struct MacroActionList {
    int count = 0;
    std::array<MacroAction, MAX_MACRO_ACTIONS> items{};

    void clear();
    void insert_sorted(const MacroAction& action);
};

int defensive_reserve(const GameState& state, int source_index, int player);
void generate_atomic_launches(const GameState& state, int player, AtomicLaunchList& out);
void pack_macro_actions(const GameState& state, int player, const AtomicLaunchList& atoms,
                        MacroActionList& out);
void deterministic_launches_for_owner(const GameState& state, int owner, LaunchList& out);

}  // namespace orbit
