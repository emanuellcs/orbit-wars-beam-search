#pragma once

#include <array>
#include <cstdint>

namespace orbit {

constexpr int MAX_PLAYERS = 4;
constexpr int MAX_PLANETS = 96;
constexpr int MAX_FLEETS = 4096;
constexpr int MAX_COMET_GROUPS = 8;
constexpr int MAX_COMETS_PER_GROUP = 4;
constexpr int MAX_COMET_PATH_POINTS = 512;
constexpr int MAX_ATOMIC_LAUNCHES = 1536;
constexpr int MAX_MACRO_ACTIONS = 512;
constexpr int MAX_BEAM_WIDTH = 512;
constexpr int MAX_LAUNCHES = 128;
constexpr int MAX_SEARCH_THREADS = 20;

constexpr double BOARD_SIZE = 100.0;
constexpr double CENTER_X = 50.0;
constexpr double CENTER_Y = 50.0;
constexpr double SUN_RADIUS = 10.0;
constexpr double ROTATION_RADIUS_LIMIT = 50.0;
constexpr double DEFAULT_SHIP_SPEED = 6.0;
constexpr double COMET_RADIUS = 1.0;
constexpr int EPISODE_STEPS = 500;

struct Vec2 {
    double x = 0.0;
    double y = 0.0;
};

struct PlanetObservation {
    int id = -1;
    int owner = -1;
    double x = 0.0;
    double y = 0.0;
    double radius = 0.0;
    int ships = 0;
    int production = 0;
};

struct FleetObservation {
    int id = -1;
    int owner = -1;
    double x = 0.0;
    double y = 0.0;
    double angle = 0.0;
    int from_planet_id = -1;
    int ships = 0;
};

struct CometGroupObservation {
    int planet_count = 0;
    int path_index = 0;
    std::array<int, MAX_COMETS_PER_GROUP> planet_ids{};
    std::array<int, MAX_COMETS_PER_GROUP> path_len{};
    std::array<double, MAX_COMETS_PER_GROUP * MAX_COMET_PATH_POINTS> path_x{};
    std::array<double, MAX_COMETS_PER_GROUP * MAX_COMET_PATH_POINTS> path_y{};
};

struct ObservationInput {
    int player = 0;
    int step = 0;
    double angular_velocity = 0.0;
    double remaining_overage_time = 0.0;
    int planet_count = 0;
    int fleet_count = 0;
    int initial_planet_count = 0;
    int comet_planet_id_count = 0;
    int comet_group_count = 0;
    std::array<PlanetObservation, MAX_PLANETS> planets{};
    std::array<FleetObservation, MAX_FLEETS> fleets{};
    std::array<PlanetObservation, MAX_PLANETS> initial_planets{};
    std::array<int, MAX_PLANETS> comet_planet_ids{};
    std::array<CometGroupObservation, MAX_COMET_GROUPS> comet_groups{};
};

struct Launch {
    int from_planet_id = -1;
    double angle = 0.0;
    int ships = 0;
};

struct LaunchList {
    int count = 0;
    std::array<Launch, MAX_LAUNCHES> launches{};

    void clear();
    bool add(int from_planet_id, double angle, int ships);
};

struct PlanetSoA {
    int count = 0;
    std::array<uint8_t, MAX_PLANETS> alive{};
    std::array<int, MAX_PLANETS> id{};
    std::array<int, MAX_PLANETS> ships{};
    std::array<int, MAX_PLANETS> owner{};
    std::array<int, MAX_PLANETS> production{};
    std::array<double, MAX_PLANETS> x{};
    std::array<double, MAX_PLANETS> y{};
    std::array<double, MAX_PLANETS> radius{};
    std::array<uint8_t, MAX_PLANETS> is_orbiting{};
    std::array<double, MAX_PLANETS> angular_velocity{};
    std::array<double, MAX_PLANETS> initial_angle{};
    std::array<double, MAX_PLANETS> orbit_radius{};
    std::array<uint8_t, MAX_PLANETS> is_comet{};
    std::array<int, MAX_PLANETS> comet_group{};
    std::array<int, MAX_PLANETS> comet_slot{};

    void clear();
};

struct FleetSoA {
    int count = 0;
    int next_id = 1;
    std::array<uint8_t, MAX_FLEETS> alive{};
    std::array<int, MAX_FLEETS> id{};
    std::array<int, MAX_FLEETS> owner{};
    std::array<int, MAX_FLEETS> ships{};
    std::array<int, MAX_FLEETS> from_planet_id{};
    std::array<double, MAX_FLEETS> x{};
    std::array<double, MAX_FLEETS> y{};
    std::array<double, MAX_FLEETS> angle{};
    std::array<double, MAX_FLEETS> speed{};

    void clear();
    int add(int fleet_id, int fleet_owner, double fleet_x, double fleet_y,
            double fleet_angle, int from_id, int fleet_ships);
    void remove(int index);
};

struct CometPathStore {
    int group_count = 0;
    std::array<int, MAX_COMET_GROUPS> planet_count{};
    std::array<int, MAX_COMET_GROUPS> path_index{};
    std::array<int, MAX_COMET_GROUPS * MAX_COMETS_PER_GROUP> planet_id{};
    std::array<int, MAX_COMET_GROUPS * MAX_COMETS_PER_GROUP> path_len{};
    std::array<double, MAX_COMET_GROUPS * MAX_COMETS_PER_GROUP * MAX_COMET_PATH_POINTS> path_x{};
    std::array<double, MAX_COMET_GROUPS * MAX_COMETS_PER_GROUP * MAX_COMET_PATH_POINTS> path_y{};

    void clear();
    int slot_index(int group, int slot) const;
    int path_index_flat(int group, int slot, int point) const;
};

struct CombatQueue {
    std::array<int, MAX_PLANETS * MAX_PLAYERS> ships{};

    void clear();
    void add(int planet_index, int owner, int ship_count);
    int at(int planet_index, int owner) const;
};

struct GameState {
    int player = 0;
    int step = 0;
    double angular_velocity = 0.0;
    double remaining_overage_time = 0.0;
    bool done = false;
    int winner = -1;
    PlanetSoA planets{};
    FleetSoA fleets{};
    CometPathStore comets{};

    void reset();
    void load_from_observation(const ObservationInput& obs);
    int planet_index_by_id(int planet_id) const;
    Vec2 planet_position_after(int planet_index, double ticks) const;
    bool is_active_player(int owner) const;
};

class OrbitSim {
public:
    GameState state{};

    void reset();
    void load_from_observation(const ObservationInput& obs);
    void step(const LaunchList& launches);
};

struct Engine {
    OrbitSim sim{};

    Engine();
    explicit Engine(int player);
    void update_observation(const ObservationInput& obs);
    void step_actions(const LaunchList& launches);
    LaunchList choose_actions(int time_budget_ms, uint64_t seed);
    double debug_evaluate(int player = -1) const;
};

}  // namespace orbit
