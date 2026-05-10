/**
 * @file orbit_engine.hpp
 * @brief Public fixed-buffer API for the Orbit Wars native engine.
 *
 * This header defines the C++ data contract shared by the simulator, search
 * engine, and pybind11 bridge. All game state is represented with bounded
 * arrays so root rollouts can copy and mutate state without heap allocation in
 * the 900 ms Kaggle action window.
 */
#pragma once

#include <array>
#include <cstdint>

namespace orbit {

/// @brief Maximum number of owners supported by the Kaggle rule set.
constexpr int MAX_PLAYERS = 4;
/// @brief Fixed planet capacity, including transient comet planets.
constexpr int MAX_PLANETS = 96;
/// @brief Fixed active fleet capacity used by simulation and rollouts.
constexpr int MAX_FLEETS = 4096;
/// @brief Maximum number of comet groups retained from observations.
constexpr int MAX_COMET_GROUPS = 8;
/// @brief Mirrored comet slots per group, one per quadrant.
constexpr int MAX_COMETS_PER_GROUP = 4;
/// @brief Maximum sampled path points stored for each comet.
constexpr int MAX_COMET_PATH_POINTS = 512;
/// @brief Capacity for ranked one-source tactical launch candidates.
constexpr int MAX_ATOMIC_LAUNCHES = 1536;
/// @brief Capacity for packed multi-launch macro-actions.
constexpr int MAX_MACRO_ACTIONS = 512;
/// @brief Upper bound on root candidates evaluated by search workers.
constexpr int MAX_BEAM_WIDTH = 512;
/// @brief Maximum launches emitted in a single action list.
constexpr int MAX_LAUNCHES = 128;
/// @brief Hard cap on parallel root-evaluation workers.
constexpr int MAX_SEARCH_THREADS = 20;

/// @brief Width and height of the continuous square board.
constexpr double BOARD_SIZE = 100.0;
/// @brief X coordinate of the sun and orbital center.
constexpr double CENTER_X = 50.0;
/// @brief Y coordinate of the sun and orbital center.
constexpr double CENTER_Y = 50.0;
/// @brief Radius of the central collision hazard.
constexpr double SUN_RADIUS = 10.0;
/// @brief Inner planets rotate only when orbit radius plus body radius fits.
constexpr double ROTATION_RADIUS_LIMIT = 50.0;
/// @brief Asymptotic fleet speed reached by large launches.
constexpr double DEFAULT_SHIP_SPEED = 6.0;
/// @brief Rule-defined radius for comet planets.
constexpr double COMET_RADIUS = 1.0;
/// @brief Maximum simulated turns in one Orbit Wars episode.
constexpr int EPISODE_STEPS = 500;

/// @brief Lightweight 2D point/vector used by geometry hot paths.
struct Vec2 {
    ///< X coordinate in board units.
    double x = 0.0;
    ///< Y coordinate in board units.
    double y = 0.0;
};

/// @brief Raw planet row decoded from a Python observation.
struct PlanetObservation {
    ///< Stable planet identifier from the Kaggle environment.
    int id = -1;
    ///< Owner id in [0, MAX_PLAYERS) or -1 for neutral.
    int owner = -1;
    ///< Current X coordinate.
    double x = 0.0;
    ///< Current Y coordinate.
    double y = 0.0;
    ///< Collision radius.
    double radius = 0.0;
    ///< Ships currently stationed on the planet.
    int ships = 0;
    ///< Ships produced per turn while owned.
    int production = 0;
};

/// @brief Raw fleet row decoded from a Python observation.
struct FleetObservation {
    ///< Stable fleet identifier from the Kaggle environment.
    int id = -1;
    ///< Owner id in [0, MAX_PLAYERS).
    int owner = -1;
    ///< Current X coordinate.
    double x = 0.0;
    ///< Current Y coordinate.
    double y = 0.0;
    ///< Heading in radians, where 0 points right and pi/2 points down.
    double angle = 0.0;
    ///< Planet id that launched the fleet, or -1 when unavailable.
    int from_planet_id = -1;
    ///< Ships carried by the fleet.
    int ships = 0;
};

/// @brief Fixed-buffer observation payload for one mirrored comet group.
struct CometGroupObservation {
    ///< Number of valid comet slots in this group.
    int planet_count = 0;
    ///< Current path sample index supplied by the environment.
    int path_index = 0;
    ///< Planet ids belonging to each comet slot.
    std::array<int, MAX_COMETS_PER_GROUP> planet_ids{};
    ///< Valid path sample count per comet slot.
    std::array<int, MAX_COMETS_PER_GROUP> path_len{};
    ///< Flattened path X coordinates, indexed by slot then point.
    std::array<double, MAX_COMETS_PER_GROUP * MAX_COMET_PATH_POINTS> path_x{};
    ///< Flattened path Y coordinates, indexed by slot then point.
    std::array<double, MAX_COMETS_PER_GROUP * MAX_COMET_PATH_POINTS> path_y{};
};

/// @brief Full observation snapshot after Python-to-C++ conversion.
struct ObservationInput {
    ///< Controlled player id for this engine instance.
    int player = 0;
    ///< Current environment step.
    int step = 0;
    ///< Shared angular velocity used by orbiting planets.
    double angular_velocity = 0.0;
    ///< Kaggle overage time, preserved for diagnostics and future tuning.
    double remaining_overage_time = 0.0;
    ///< Number of valid entries in planets.
    int planet_count = 0;
    ///< Number of valid entries in fleets.
    int fleet_count = 0;
    ///< Number of valid entries in initial_planets.
    int initial_planet_count = 0;
    ///< Number of valid entries in comet_planet_ids.
    int comet_planet_id_count = 0;
    ///< Number of valid entries in comet_groups.
    int comet_group_count = 0;
    ///< Current observed planets, including active comets.
    std::array<PlanetObservation, MAX_PLANETS> planets{};
    ///< Current observed fleets.
    std::array<FleetObservation, MAX_FLEETS> fleets{};
    ///< Initial planet positions used to reconstruct orbital metadata.
    std::array<PlanetObservation, MAX_PLANETS> initial_planets{};
    ///< Planet ids that should be treated as comets.
    std::array<int, MAX_PLANETS> comet_planet_ids{};
    ///< Active comet path observations.
    std::array<CometGroupObservation, MAX_COMET_GROUPS> comet_groups{};
};

/// @brief One legal launch order returned to the environment.
struct Launch {
    ///< Source planet id, not SoA index.
    int from_planet_id = -1;
    ///< Launch heading in radians.
    double angle = 0.0;
    ///< Ships to send from the source.
    int ships = 0;
};

/// @brief Fixed-capacity output buffer for launch orders.
struct LaunchList {
    ///< Number of valid launch rows.
    int count = 0;
    ///< Fixed launch storage; entries beyond count are ignored.
    std::array<Launch, MAX_LAUNCHES> launches{};

    /// @brief Reset the launch count and clear stale rows.
    /// @note O(MAX_LAUNCHES); used outside the deepest collision loops.
    void clear();
    /// @brief Append a launch if capacity and ship-count constraints allow it.
    /// @param from_planet_id Source planet id.
    /// @param angle Heading in radians.
    /// @param ships Positive ship count to launch.
    /// @return true when the row was appended, otherwise false.
    /// @note Does not validate ownership; simulator/search enforce that later.
    bool add(int from_planet_id, double angle, int ships);
};

/// @brief Structure-of-Arrays planet state for cache-friendly simulation.
struct PlanetSoA {
    ///< Number of allocated SoA slots, live or tombstoned.
    int count = 0;
    ///< Slot liveness flags; dead slots are skipped by hot loops.
    std::array<uint8_t, MAX_PLANETS> alive{};
    ///< Planet ids keyed by SoA slot.
    std::array<int, MAX_PLANETS> id{};
    ///< Stationed ship counts keyed by SoA slot.
    std::array<int, MAX_PLANETS> ships{};
    ///< Owner ids keyed by SoA slot.
    std::array<int, MAX_PLANETS> owner{};
    ///< Per-turn production keyed by SoA slot.
    std::array<int, MAX_PLANETS> production{};
    ///< Current X coordinates keyed by SoA slot.
    std::array<double, MAX_PLANETS> x{};
    ///< Current Y coordinates keyed by SoA slot.
    std::array<double, MAX_PLANETS> y{};
    ///< Collision radii keyed by SoA slot.
    std::array<double, MAX_PLANETS> radius{};
    ///< True when the planet follows circular motion around the sun.
    std::array<uint8_t, MAX_PLANETS> is_orbiting{};
    ///< Per-slot angular velocity; zero for static and comet slots.
    std::array<double, MAX_PLANETS> angular_velocity{};
    ///< Initial polar angle around the sun, used to reconstruct orbits.
    std::array<double, MAX_PLANETS> initial_angle{};
    ///< Initial distance from the sun center.
    std::array<double, MAX_PLANETS> orbit_radius{};
    ///< True when the slot represents a comet planet.
    std::array<uint8_t, MAX_PLANETS> is_comet{};
    ///< Comet group index for comet slots, otherwise -1.
    std::array<int, MAX_PLANETS> comet_group{};
    ///< Comet slot within its group, otherwise -1.
    std::array<int, MAX_PLANETS> comet_slot{};

    /// @brief Reset all SoA arrays to neutral defaults.
    /// @note O(MAX_PLANETS); called when replacing an observation snapshot.
    void clear();
};

/// @brief Structure-of-Arrays fleet state with tombstone removal.
struct FleetSoA {
    ///< Number of allocated fleet slots, live or tombstoned.
    int count = 0;
    ///< Next generated id for simulated launches.
    int next_id = 1;
    ///< Slot liveness flags.
    std::array<uint8_t, MAX_FLEETS> alive{};
    ///< Fleet ids keyed by SoA slot.
    std::array<int, MAX_FLEETS> id{};
    ///< Owner ids keyed by SoA slot.
    std::array<int, MAX_FLEETS> owner{};
    ///< Ships carried by each fleet.
    std::array<int, MAX_FLEETS> ships{};
    ///< Source planet id retained for debugging and output parity.
    std::array<int, MAX_FLEETS> from_planet_id{};
    ///< Current X coordinates keyed by SoA slot.
    std::array<double, MAX_FLEETS> x{};
    ///< Current Y coordinates keyed by SoA slot.
    std::array<double, MAX_FLEETS> y{};
    ///< Heading in radians keyed by SoA slot.
    std::array<double, MAX_FLEETS> angle{};
    ///< Cached speed derived from ship count.
    std::array<double, MAX_FLEETS> speed{};

    /// @brief Reset all fleet arrays and id generation.
    /// @note O(MAX_FLEETS); keeps state copies allocation-free.
    void clear();
    /// @brief Insert or reuse a fleet slot.
    /// @param fleet_id Existing observation id, or negative to auto-generate.
    /// @param fleet_owner Owner id.
    /// @param fleet_x X coordinate.
    /// @param fleet_y Y coordinate.
    /// @param fleet_angle Heading in radians.
    /// @param from_id Source planet id.
    /// @param fleet_ships Ships carried by the fleet.
    /// @return SoA slot index, or -1 when MAX_FLEETS is exhausted.
    /// @note Reuses tombstoned slots to preserve the fixed-buffer contract.
    int add(int fleet_id, int fleet_owner, double fleet_x, double fleet_y,
            double fleet_angle, int from_id, int fleet_ships);
    /// @brief Tombstone a fleet slot.
    /// @param index SoA slot index to mark dead.
    /// @note Removal is O(1); compaction is avoided to keep indices stable.
    void remove(int index);
};

/// @brief Flattened comet path storage used for deterministic prediction.
struct CometPathStore {
    ///< Number of valid comet groups.
    int group_count = 0;
    ///< Valid comet slots per group.
    std::array<int, MAX_COMET_GROUPS> planet_count{};
    ///< Current path index per group.
    std::array<int, MAX_COMET_GROUPS> path_index{};
    ///< Flattened comet planet ids by group and slot.
    std::array<int, MAX_COMET_GROUPS * MAX_COMETS_PER_GROUP> planet_id{};
    ///< Flattened path lengths by group and slot.
    std::array<int, MAX_COMET_GROUPS * MAX_COMETS_PER_GROUP> path_len{};
    ///< Flattened path X coordinates by group, slot, and point.
    std::array<double, MAX_COMET_GROUPS * MAX_COMETS_PER_GROUP * MAX_COMET_PATH_POINTS> path_x{};
    ///< Flattened path Y coordinates by group, slot, and point.
    std::array<double, MAX_COMET_GROUPS * MAX_COMETS_PER_GROUP * MAX_COMET_PATH_POINTS> path_y{};

    /// @brief Reset comet path metadata and samples.
    /// @note O(MAX_COMET_GROUPS * MAX_COMETS_PER_GROUP * MAX_COMET_PATH_POINTS).
    void clear();
    /// @brief Flatten a comet group/slot pair.
    /// @param group Comet group index.
    /// @param slot Slot within the group.
    /// @return Flat index into per-slot comet arrays.
    int slot_index(int group, int slot) const;
    /// @brief Flatten a comet group/slot/path point triple.
    /// @param group Comet group index.
    /// @param slot Slot within the group.
    /// @param point Path sample index.
    /// @return Flat index into path_x/path_y arrays.
    int path_index_flat(int group, int slot, int point) const;
};

/// @brief Fixed combat accumulator indexed by planet slot and owner.
struct CombatQueue {
    ///< Flattened arriving ships by planet_index * MAX_PLAYERS + owner.
    std::array<int, MAX_PLANETS * MAX_PLAYERS> ships{};

    /// @brief Clear all queued arrivals.
    /// @note O(MAX_PLANETS * MAX_PLAYERS).
    void clear();
    /// @brief Queue ships arriving at a planet for a specific owner.
    /// @param planet_index SoA planet slot.
    /// @param owner Attacking or reinforcing owner id.
    /// @param ship_count Positive arriving ship count.
    /// @note Invalid owners, planets, and non-positive counts are ignored.
    void add(int planet_index, int owner, int ship_count);
    /// @brief Read queued ships for a planet/owner pair.
    /// @param planet_index SoA planet slot.
    /// @param owner Owner id.
    /// @return Queued ship count, or 0 for invalid coordinates.
    int at(int planet_index, int owner) const;
};

/// @brief Complete native game state snapshot used by simulator and search.
struct GameState {
    ///< Controlled player id for evaluation/search.
    int player = 0;
    ///< Current turn index.
    int step = 0;
    ///< Environment angular velocity for orbiting planets.
    double angular_velocity = 0.0;
    ///< Remaining Kaggle overage time, carried for diagnostics.
    double remaining_overage_time = 0.0;
    ///< Terminal-state flag.
    bool done = false;
    ///< Winning owner id, or -1 for none/tie.
    int winner = -1;
    ///< Planet SoA state.
    PlanetSoA planets{};
    ///< Fleet SoA state.
    FleetSoA fleets{};
    ///< Comet path prediction state.
    CometPathStore comets{};

    /// @brief Reset the entire state to an empty non-terminal snapshot.
    /// @note O(total fixed capacities); no heap storage is touched.
    void reset();
    /// @brief Replace the state from a converted Python observation.
    /// @param obs Fixed-buffer observation input.
    /// @note Reconstructs orbital/comet metadata needed for future ticks.
    void load_from_observation(const ObservationInput& obs);
    /// @brief Locate a live planet slot by environment planet id.
    /// @param planet_id Planet id from observation or launch action.
    /// @return SoA index, or -1 when absent/dead.
    /// @note O(planets.count), which is small and fixed-bounded.
    int planet_index_by_id(int planet_id) const;
    /// @brief Predict planet center after a number of ticks.
    /// @param planet_index SoA planet slot.
    /// @param ticks Future offset in simulation ticks.
    /// @return Predicted center position, or {0,0} for invalid slots.
    /// @note Uses circular motion for orbiting planets and interpolation for comets.
    Vec2 planet_position_after(int planet_index, double ticks) const;
    /// @brief Test whether an owner still has any live planets or fleets.
    /// @param owner Owner id to inspect.
    /// @return true if the owner remains active.
    bool is_active_player(int owner) const;
};

/// @brief Single-step simulator implementing Orbit Wars turn order.
class OrbitSim {
public:
    ///< Mutable state owned by this simulator instance.
    GameState state{};

    /// @brief Reset the simulator state.
    void reset();
    /// @brief Load a fresh observation into the simulator state.
    /// @param obs Fixed-buffer observation input.
    void load_from_observation(const ObservationInput& obs);
    /// @brief Advance exactly one turn with a joint launch list.
    /// @param launches Launches from one or more owners.
    /// @note Launch validation, production, movement, sweeps, and combat are ordered here.
    void step(const LaunchList& launches);
};

/// @brief High-level engine facade exposed to Python.
struct Engine {
    ///< Simulator used both for live state and debug stepping.
    OrbitSim sim{};

    /// @brief Construct an engine with default player id 0.
    Engine();
    /// @brief Construct an engine for a specific player.
    /// @param player Controlled player id.
    explicit Engine(int player);
    /// @brief Replace engine state from a new observation.
    /// @param obs Fixed-buffer observation input.
    void update_observation(const ObservationInput& obs);
    /// @brief Advance internal simulator with explicit launches.
    /// @param launches Launch list to apply.
    void step_actions(const LaunchList& launches);
    /// @brief Choose a launch list using fixed-buffer beam search.
    /// @param time_budget_ms Caller-provided time budget in milliseconds.
    /// @param seed Deterministic seed mixed into worker evaluation.
    /// @return Selected legal launch list.
    /// @note Search clamps the budget to the native hard stop.
    LaunchList choose_actions(int time_budget_ms, uint64_t seed);
    /// @brief Evaluate the current state for debugging/tuning.
    /// @param player Player id to evaluate, or -1 for the engine player.
    /// @return Heuristic state score.
    double debug_evaluate(int player = -1) const;
};

}  // namespace orbit
