/**
 * @file candidate.hpp
 * @brief Tactical launch candidate and macro-action packing interfaces.
 *
 * Candidate generation converts the continuous game state into a fixed set of
 * analytically aimed packets. Macro packing then combines those packets while
 * preserving per-source spend legality, giving search a bounded action frontier.
 */
#pragma once

#include "orbit_engine.hpp"

namespace orbit {

/// @brief Tactical category used to bias atomic launch ranking.
enum class PacketKind : uint8_t {
    ///< Send exactly garrison + 1 ships to attempt a capture.
    CaptureExact = 0,
    ///< Send capture ships plus production-based slack.
    CaptureOver,
    ///< Low-commitment pressure or scouting packet.
    Harass,
    ///< Send all ships above a defensive reserve.
    AllSafe,
};

/// @brief One analytically aimed launch packet before macro packing.
struct AtomicLaunch {
    ///< Source planet id for eventual LaunchList output.
    int from_planet_id = -1;
    ///< Source planet SoA index.
    int source_index = -1;
    ///< Target planet SoA index.
    int target_index = -1;
    ///< Ships assigned to this packet.
    int ships = 0;
    ///< Launch heading in radians.
    double angle = 0.0;
    ///< Time-to-intercept estimate, consistently called tau in geometry.
    double eta = 0.0;
    ///< Tactical prior used for sorted insertion.
    double score = 0.0;
    ///< Packet category that produced this launch.
    PacketKind kind = PacketKind::Harass;
};

/// @brief Score-sorted fixed-capacity buffer of atomic launch packets.
struct AtomicLaunchList {
    ///< Number of valid items.
    int count = 0;
    ///< Bounded candidate storage; kept sorted by descending score.
    std::array<AtomicLaunch, MAX_ATOMIC_LAUNCHES> items{};

    /// @brief Clear the logical list without touching storage.
    /// @note O(1); stale entries beyond count are ignored.
    void clear();
    /// @brief Insert a launch while preserving descending score order.
    /// @param launch Candidate packet to insert.
    /// @note O(MAX_ATOMIC_LAUNCHES) worst case but fixed-bounded and allocation-free.
    void insert_sorted(const AtomicLaunch& launch);
};

/// @brief Packed multi-launch action evaluated by search.
struct MacroAction {
    ///< Launches to apply at the root tick.
    LaunchList launches{};
    ///< Sum/prior score used to order candidates.
    double score = 0.0;
};

/// @brief Score-sorted fixed-capacity buffer of macro-actions.
struct MacroActionList {
    ///< Number of valid macro-actions.
    int count = 0;
    ///< Bounded macro-action storage sorted by descending score.
    std::array<MacroAction, MAX_MACRO_ACTIONS> items{};

    /// @brief Clear the logical macro-action list.
    /// @note O(1); no allocation or deallocation occurs.
    void clear();
    /// @brief Insert a macro-action while preserving descending score order.
    /// @param action Macro-action to insert.
    /// @note Keeps only the highest-priority MAX_MACRO_ACTIONS candidates.
    void insert_sorted(const MacroAction& action);
};

/// @brief Estimate ships that should remain on a source for near-term defense.
/// @param state Current game state.
/// @param source_index Source planet SoA index.
/// @param player Player id that owns the source.
/// @return Reserve ship count that candidate generation should not spend.
int defensive_reserve(const GameState& state, int source_index, int player);
/// @brief Generate ranked atomic launches from owned sources to non-owned targets.
/// @param state Current game state.
/// @param player Controlled player id.
/// @param out Output atomic launch buffer.
/// @note Uses solve_intercept so every packet has an analytic angle and tau.
void generate_atomic_launches(const GameState& state, int player, AtomicLaunchList& out);
/// @brief Pack sorted atomic launches into legal macro-actions.
/// @param state Current game state.
/// @param player Controlled player id.
/// @param atoms Sorted atomic launch candidates.
/// @param out Output macro-action buffer.
/// @note Per-source spend arrays prevent overspending without dynamic maps.
void pack_macro_actions(const GameState& state, int player, const AtomicLaunchList& atoms,
                        MacroActionList& out);
/// @brief Deterministic policy used for rollout opponents and fallback behavior.
/// @param state Current game state.
/// @param owner Owner id to act for.
/// @param out Launch buffer to append into.
/// @note This intentionally shares candidate logic with search for repeatability.
void deterministic_launches_for_owner(const GameState& state, int owner, LaunchList& out);

}  // namespace orbit
