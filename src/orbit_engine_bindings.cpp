/**
 * @file orbit_engine_bindings.cpp
 * @brief pybind11 bridge between Kaggle Python observations and the native engine.
 *
 * The bridge accepts dictionary-style and attribute-style observations, copies
 * rows into fixed-capacity native buffers, and returns plain Python launch lists.
 * All capacity checks happen here before data reaches the zero-allocation C++
 * simulator/search hot path.
 */
#include "orbit_engine.hpp"

#include "candidate.hpp"
#include "eval.hpp"
#include "geometry.hpp"
#include "search.hpp"

#include <pybind11/pybind11.h>

#include <algorithm>
#include <cstdint>
#include <stdexcept>
#include <string>

namespace py = pybind11;

namespace {

/**
 * @brief Read a named field from a Python dict or attribute object.
 * @param obj Python observation-like object.
 * @param name Field name to read.
 * @return Borrowed field value, or None when absent.
 * @note Kaggle can provide dict-like observations while tests use namespaces.
 */
py::object get_field(const py::handle& obj, const char* name) {
    if (py::isinstance<py::dict>(obj)) {
        py::dict dict = py::reinterpret_borrow<py::dict>(obj);
        py::str key(name);
        if (dict.contains(key)) {
            return py::reinterpret_borrow<py::object>(dict[key]);
        }
    }
    if (py::hasattr(obj, name)) {
        return py::reinterpret_borrow<py::object>(obj.attr(name));
    }
    return py::none();
}

/**
 * @brief Read an integer field with fallback.
 * @param obj Python object to inspect.
 * @param name Field name.
 * @param fallback Value to return when the field is absent.
 * @return Cast integer value or fallback.
 */
int get_int_field(const py::handle& obj, const char* name, int fallback) {
    py::object value = get_field(obj, name);
    if (value.is_none()) {
        return fallback;
    }
    return py::cast<int>(value);
}

/**
 * @brief Read a floating-point field with fallback.
 * @param obj Python object to inspect.
 * @param name Field name.
 * @param fallback Value to return when the field is absent.
 * @return Cast double value or fallback.
 */
double get_double_field(const py::handle& obj, const char* name, double fallback) {
    py::object value = get_field(obj, name);
    if (value.is_none()) {
        return fallback;
    }
    return py::cast<double>(value);
}

/**
 * @brief Parse one planet row into native observation storage.
 * @param item Python sequence [id, owner, x, y, radius, ships, production].
 * @param out Output planet observation.
 * @return true when the row has the expected minimum length.
 */
bool parse_planet(const py::handle& item, orbit::PlanetObservation& out) {
    py::sequence seq = py::reinterpret_borrow<py::sequence>(item);
    if (py::len(seq) < 7) {
        return false;
    }
    out.id = py::cast<int>(seq[0]);
    out.owner = py::cast<int>(seq[1]);
    out.x = py::cast<double>(seq[2]);
    out.y = py::cast<double>(seq[3]);
    out.radius = py::cast<double>(seq[4]);
    out.ships = py::cast<int>(seq[5]);
    out.production = py::cast<int>(seq[6]);
    return true;
}

/**
 * @brief Parse one fleet row into native observation storage.
 * @param item Python sequence [id, owner, x, y, angle, from_planet_id, ships].
 * @param out Output fleet observation.
 * @return true when the row has the expected minimum length.
 */
bool parse_fleet(const py::handle& item, orbit::FleetObservation& out) {
    py::sequence seq = py::reinterpret_borrow<py::sequence>(item);
    if (py::len(seq) < 7) {
        return false;
    }
    out.id = py::cast<int>(seq[0]);
    out.owner = py::cast<int>(seq[1]);
    out.x = py::cast<double>(seq[2]);
    out.y = py::cast<double>(seq[3]);
    out.angle = py::cast<double>(seq[4]);
    out.from_planet_id = py::cast<int>(seq[5]);
    out.ships = py::cast<int>(seq[6]);
    return true;
}

/**
 * @brief Parse current or initial planet rows into fixed observation arrays.
 * @param obj Python sequence of planet rows, or None.
 * @param obs Observation being populated.
 * @param initial true to populate initial_planets, false for current planets.
 * @note Count is clamped to MAX_PLANETS before entering native state loading.
 */
void parse_planets(const py::object& obj, orbit::ObservationInput& obs, bool initial) {
    if (obj.is_none()) {
        return;
    }
    py::sequence seq = py::reinterpret_borrow<py::sequence>(obj);
    const int n = std::min<int>(static_cast<int>(py::len(seq)), orbit::MAX_PLANETS);
    for (int i = 0; i < n; ++i) {
        orbit::PlanetObservation planet{};
        if (!parse_planet(seq[i], planet)) {
            continue;
        }
        if (initial) {
            obs.initial_planets[static_cast<size_t>(obs.initial_planet_count++)] = planet;
        } else {
            obs.planets[static_cast<size_t>(obs.planet_count++)] = planet;
        }
    }
}

/**
 * @brief Parse fleet rows into the fixed observation fleet array.
 * @param obj Python sequence of fleet rows, or None.
 * @param obs Observation being populated.
 * @note Count is clamped to MAX_FLEETS, matching FleetSoA capacity.
 */
void parse_fleets(const py::object& obj, orbit::ObservationInput& obs) {
    if (obj.is_none()) {
        return;
    }
    py::sequence seq = py::reinterpret_borrow<py::sequence>(obj);
    const int n = std::min<int>(static_cast<int>(py::len(seq)), orbit::MAX_FLEETS);
    for (int i = 0; i < n; ++i) {
        orbit::FleetObservation fleet{};
        if (parse_fleet(seq[i], fleet)) {
            obs.fleets[static_cast<size_t>(obs.fleet_count++)] = fleet;
        }
    }
}

/**
 * @brief Parse comet planet id markers from Python.
 * @param obj Python sequence of planet ids, or None.
 * @param obs Observation being populated.
 * @note These ids later override orbiting metadata and attach path slots.
 */
void parse_comet_ids(const py::object& obj, orbit::ObservationInput& obs) {
    if (obj.is_none()) {
        return;
    }
    py::sequence seq = py::reinterpret_borrow<py::sequence>(obj);
    const int n = std::min<int>(static_cast<int>(py::len(seq)), orbit::MAX_PLANETS);
    for (int i = 0; i < n; ++i) {
        obs.comet_planet_ids[static_cast<size_t>(obs.comet_planet_id_count++)] =
            py::cast<int>(seq[i]);
    }
}

/**
 * @brief Parse active comet group path data from Python.
 * @param obj Python sequence of group objects, or None.
 * @param obs Observation being populated.
 * @note Paths are flattened by slot then point so interpolation uses direct
 *       fixed-array indexing during solve_intercept and simulation.
 */
void parse_comets(const py::object& obj, orbit::ObservationInput& obs) {
    if (obj.is_none()) {
        return;
    }
    py::sequence groups = py::reinterpret_borrow<py::sequence>(obj);
    const int group_count = std::min<int>(static_cast<int>(py::len(groups)), orbit::MAX_COMET_GROUPS);
    for (int g = 0; g < group_count; ++g) {
        py::object group = py::reinterpret_borrow<py::object>(groups[g]);
        orbit::CometGroupObservation& out = obs.comet_groups[static_cast<size_t>(obs.comet_group_count)];
        out.path_index = get_int_field(group, "path_index", 0);
        py::object ids = get_field(group, "planet_ids");
        if (!ids.is_none()) {
            py::sequence id_seq = py::reinterpret_borrow<py::sequence>(ids);
            out.planet_count = std::min<int>(static_cast<int>(py::len(id_seq)), orbit::MAX_COMETS_PER_GROUP);
            for (int s = 0; s < out.planet_count; ++s) {
                out.planet_ids[static_cast<size_t>(s)] = py::cast<int>(id_seq[s]);
            }
        }
        py::object paths = get_field(group, "paths");
        if (!paths.is_none()) {
            py::sequence path_groups = py::reinterpret_borrow<py::sequence>(paths);
            const int slots = std::min<int>(static_cast<int>(py::len(path_groups)), orbit::MAX_COMETS_PER_GROUP);
            out.planet_count = std::max(out.planet_count, slots);
            for (int s = 0; s < slots; ++s) {
                py::sequence path = py::reinterpret_borrow<py::sequence>(path_groups[s]);
                const int len = std::min<int>(static_cast<int>(py::len(path)), orbit::MAX_COMET_PATH_POINTS);
                out.path_len[static_cast<size_t>(s)] = len;
                for (int p = 0; p < len; ++p) {
                    py::sequence point = py::reinterpret_borrow<py::sequence>(path[p]);
                    if (py::len(point) < 2) {
                        continue;
                    }
                    const int index = s * orbit::MAX_COMET_PATH_POINTS + p;
                    out.path_x[static_cast<size_t>(index)] = py::cast<double>(point[0]);
                    out.path_y[static_cast<size_t>(index)] = py::cast<double>(point[1]);
                }
            }
        }
        ++obs.comet_group_count;
    }
}

/**
 * @brief Convert a Python observation into the native fixed-buffer input type.
 * @param raw Python dict or attribute object with Orbit Wars fields.
 * @return Fully populated ObservationInput with safe defaults.
 * @note If initial_planets is absent, current planets are copied so orbital
 *       reconstruction remains well-defined.
 */
orbit::ObservationInput make_observation(const py::object& raw) {
    orbit::ObservationInput obs{};
    obs.player = get_int_field(raw, "player", 0);
    obs.step = get_int_field(raw, "step", 0);
    obs.angular_velocity = get_double_field(raw, "angular_velocity", 0.0);
    obs.remaining_overage_time = get_double_field(raw, "remainingOverageTime", 0.0);
    parse_planets(get_field(raw, "planets"), obs, false);
    parse_fleets(get_field(raw, "fleets"), obs);
    parse_planets(get_field(raw, "initial_planets"), obs, true);
    parse_comet_ids(get_field(raw, "comet_planet_ids"), obs);
    parse_comets(get_field(raw, "comets"), obs);
    if (obs.initial_planet_count == 0) {
        obs.initial_planet_count = obs.planet_count;
        for (int i = 0; i < obs.planet_count; ++i) {
            obs.initial_planets[static_cast<size_t>(i)] = obs.planets[static_cast<size_t>(i)];
        }
    }
    return obs;
}

/**
 * @brief Convert Python launch rows into a native LaunchList.
 * @param raw Python sequence of [from_planet_id, angle, ships], or None.
 * @return Fixed-capacity launch list.
 * @note Invalid short rows are skipped and total count is clamped.
 */
orbit::LaunchList make_launches(const py::object& raw) {
    orbit::LaunchList out{};
    out.clear();
    if (raw.is_none()) {
        return out;
    }
    py::sequence seq = py::reinterpret_borrow<py::sequence>(raw);
    const int n = std::min<int>(static_cast<int>(py::len(seq)), orbit::MAX_LAUNCHES);
    for (int i = 0; i < n; ++i) {
        py::sequence launch = py::reinterpret_borrow<py::sequence>(seq[i]);
        if (py::len(launch) < 3) {
            continue;
        }
        out.add(py::cast<int>(launch[0]), py::cast<double>(launch[1]), py::cast<int>(launch[2]));
    }
    return out;
}

/**
 * @brief Convert a native LaunchList into Kaggle action rows.
 * @param launches Native fixed-buffer launch list.
 * @return Python list [[from_planet_id, angle, ships], ...].
 */
py::list launch_list_to_python(const orbit::LaunchList& launches) {
    py::list out;
    for (int i = 0; i < launches.count; ++i) {
        const orbit::Launch& launch = launches.launches[static_cast<size_t>(i)];
        py::list row;
        row.append(launch.from_planet_id);
        row.append(launch.angle);
        row.append(launch.ships);
        out.append(row);
    }
    return out;
}

/**
 * @brief Convert native state into a Python debug dictionary.
 * @param state Native game state.
 * @return Python dictionary exposing live planets and fleets.
 * @note This is for tests/debugging only; search does not call back into Python.
 */
py::dict debug_state_to_python(const orbit::GameState& state) {
    py::dict out;
    out["player"] = state.player;
    out["step"] = state.step;
    out["done"] = state.done;
    out["winner"] = state.winner;
    out["planet_count"] = state.planets.count;
    out["fleet_count"] = state.fleets.count;
    py::list planets;
    for (int i = 0; i < state.planets.count; ++i) {
        if (state.planets.alive[static_cast<size_t>(i)] == 0) {
            continue;
        }
        py::dict p;
        p["id"] = state.planets.id[static_cast<size_t>(i)];
        p["owner"] = state.planets.owner[static_cast<size_t>(i)];
        p["x"] = state.planets.x[static_cast<size_t>(i)];
        p["y"] = state.planets.y[static_cast<size_t>(i)];
        p["radius"] = state.planets.radius[static_cast<size_t>(i)];
        p["ships"] = state.planets.ships[static_cast<size_t>(i)];
        p["production"] = state.planets.production[static_cast<size_t>(i)];
        p["is_orbiting"] = state.planets.is_orbiting[static_cast<size_t>(i)] != 0;
        p["is_comet"] = state.planets.is_comet[static_cast<size_t>(i)] != 0;
        planets.append(p);
    }
    py::list fleets;
    for (int i = 0; i < state.fleets.count; ++i) {
        if (state.fleets.alive[static_cast<size_t>(i)] == 0) {
            continue;
        }
        py::dict f;
        f["id"] = state.fleets.id[static_cast<size_t>(i)];
        f["owner"] = state.fleets.owner[static_cast<size_t>(i)];
        f["x"] = state.fleets.x[static_cast<size_t>(i)];
        f["y"] = state.fleets.y[static_cast<size_t>(i)];
        f["angle"] = state.fleets.angle[static_cast<size_t>(i)];
        f["speed"] = state.fleets.speed[static_cast<size_t>(i)];
        f["ships"] = state.fleets.ships[static_cast<size_t>(i)];
        f["from_planet_id"] = state.fleets.from_planet_id[static_cast<size_t>(i)];
        fleets.append(f);
    }
    out["planets"] = planets;
    out["fleets"] = fleets;
    return out;
}

/**
 * @brief Set a single field on an EvalWeights struct, preserving others.
 * @param target EvalWeights to mutate.
 * @param key Field name.
 * @param value New double value.
 * @return true when ``key`` matched a known field.
 */
bool assign_eval_weight(orbit::EvalWeights& target, const std::string& key, double value) {
    if (key == "ship")            { target.ship = value; return true; }
    if (key == "production")      { target.production = value; return true; }
    if (key == "territory_own")   { target.territory_own = value; return true; }
    if (key == "territory_opp")   { target.territory_opp = value; return true; }
    if (key == "threat")          { target.threat = value; return true; }
    if (key == "comet_owned")     { target.comet_owned = value; return true; }
    if (key == "comet_enemy")     { target.comet_enemy = value; return true; }
    if (key == "comet_neutral")   { target.comet_neutral = value; return true; }
    return false;
}

/**
 * @brief Set a single field on a CandidateWeights struct, preserving others.
 * @param target CandidateWeights to mutate.
 * @param key Field name.
 * @param value New double value.
 * @return true when ``key`` matched a known field.
 */
bool assign_candidate_weight(orbit::CandidateWeights& target, const std::string& key, double value) {
    if (key == "owner_enemy")     { target.owner_enemy = value; return true; }
    if (key == "owner_neutral")   { target.owner_neutral = value; return true; }
    if (key == "owner_self")      { target.owner_self = value; return true; }
    if (key == "comet_bonus")     { target.comet_bonus = value; return true; }
    if (key == "prod_per_unit")   { target.prod_per_unit = value; return true; }
    if (key == "kind_exact")      { target.kind_exact = value; return true; }
    if (key == "kind_over")       { target.kind_over = value; return true; }
    if (key == "kind_all_safe")   { target.kind_all_safe = value; return true; }
    if (key == "kind_harass")     { target.kind_harass = value; return true; }
    if (key == "eta_discount")    { target.eta_discount = value; return true; }
    if (key == "ship_cost")       { target.ship_cost = value; return true; }
    return false;
}

/// @brief Thin Python-owned wrapper around orbit::Engine.
class PyEngine {
public:
    /**
     * @brief Construct a Python engine wrapper.
     * @param player Controlled player id.
     */
    explicit PyEngine(int player = 0) : engine(player) {}

    /**
     * @brief Replace native state from a Python observation.
     * @param obs Python dict or attribute-style observation.
     */
    void update_observation(const py::object& obs) {
        engine.update_observation(make_observation(obs));
    }

    /**
     * @brief Run native search and return Kaggle action rows.
     * @param time_budget_ms Per-turn time budget in milliseconds.
     * @param seed Deterministic seed for search.
     * @return Python list of launch rows.
     */
    py::list choose_actions(int time_budget_ms = 900, uint64_t seed = 0) {
        return launch_list_to_python(engine.choose_actions(time_budget_ms, seed));
    }

    /**
     * @brief Step the native simulator with explicit Python launch rows.
     * @param launches Python sequence of launch rows.
     */
    void step(const py::object& launches) {
        engine.step_actions(make_launches(launches));
    }

    /**
     * @brief Return a Python representation of live native state.
     * @return Debug dictionary for tests and local inspection.
     */
    py::dict debug_state() const {
        return debug_state_to_python(engine.sim.state);
    }

    /**
     * @brief Score the current native state.
     * @param player Player id, or -1 for the engine player.
     * @return Heuristic evaluation score.
     */
    double debug_evaluate(int player = -1) const {
        return engine.debug_evaluate(player);
    }

    /**
     * @brief Replace the persisted search/evaluator/candidate configuration in one call.
     * @param kwargs Optional keyword arguments overriding the defaults:
     *   - ``beam_width`` (int)
     *   - ``search_depth`` (int)
     *   - ``rollout_horizon`` (int)
     *   - ``hard_stop_ms`` (int)
     *   - ``ship`` / ``production`` / ``territory_own`` / ``territory_opp`` /
     *     ``threat`` / ``comet_owned`` / ``comet_enemy`` / ``comet_neutral`` (float)
     *   - ``owner_enemy`` / ``owner_neutral`` / ``owner_self`` /
     *     ``comet_bonus`` / ``prod_per_unit`` / ``kind_exact`` / ``kind_over`` /
     *     ``kind_all_safe`` / ``kind_harass`` / ``eta_discount`` / ``ship_cost`` (float)
     * @note Missing keys keep their existing values, so the method is fully
     *       additive and safe to call repeatedly during tuning.
     */
    void set_hyperparameters(const py::kwargs& kwargs) {
        orbit::SearchConfig cfg = engine.config;
        for (auto item : kwargs) {
            const std::string key = py::cast<std::string>(item.first);
            const py::handle value = item.second;
            if (key == "beam_width") {
                cfg.beam_width = py::cast<int>(value);
            } else if (key == "search_depth") {
                cfg.search_depth = py::cast<int>(value);
            } else if (key == "rollout_horizon") {
                cfg.rollout_horizon = py::cast<int>(value);
            } else if (key == "hard_stop_ms") {
                cfg.hard_stop_ms = py::cast<int>(value);
            } else {
                const double dvalue = py::cast<double>(value);
                if (assign_eval_weight(cfg.eval_weights, key, dvalue)) {
                    continue;
                }
                if (assign_candidate_weight(cfg.candidate_weights, key, dvalue)) {
                    continue;
                }
                throw std::invalid_argument("orbit_engine.set_hyperparameters: unknown key '" + key + "'");
            }
        }
        engine.set_search_config(cfg);
    }

    /**
     * @brief Return a shallow copy of the persisted search configuration.
     * @return Python dict mirroring the native SearchConfig.
     */
    py::dict get_hyperparameters() const {
        const orbit::SearchConfig& cfg = engine.config;
        py::dict out;
        out["beam_width"] = cfg.beam_width;
        out["search_depth"] = cfg.search_depth;
        out["rollout_horizon"] = cfg.rollout_horizon;
        out["hard_stop_ms"] = cfg.hard_stop_ms;
        out["weight_ship"] = cfg.eval_weights.ship;
        out["weight_production"] = cfg.eval_weights.production;
        out["weight_territory_own"] = cfg.eval_weights.territory_own;
        out["weight_territory_opp"] = cfg.eval_weights.territory_opp;
        out["weight_threat"] = cfg.eval_weights.threat;
        out["weight_comet_owned"] = cfg.eval_weights.comet_owned;
        out["weight_comet_enemy"] = cfg.eval_weights.comet_enemy;
        out["weight_comet_neutral"] = cfg.eval_weights.comet_neutral;
        out["weight_owner_enemy"] = cfg.candidate_weights.owner_enemy;
        out["weight_owner_neutral"] = cfg.candidate_weights.owner_neutral;
        out["weight_owner_self"] = cfg.candidate_weights.owner_self;
        out["weight_comet_bonus"] = cfg.candidate_weights.comet_bonus;
        out["weight_prod_per_unit"] = cfg.candidate_weights.prod_per_unit;
        out["weight_kind_exact"] = cfg.candidate_weights.kind_exact;
        out["weight_kind_over"] = cfg.candidate_weights.kind_over;
        out["weight_kind_all_safe"] = cfg.candidate_weights.kind_all_safe;
        out["weight_kind_harass"] = cfg.candidate_weights.kind_harass;
        out["weight_eta_discount"] = cfg.candidate_weights.eta_discount;
        out["weight_ship_cost"] = cfg.candidate_weights.ship_cost;
        return out;
    }

    /**
     * @brief Set the maximum number of search worker threads for this process.
     * @param n Requested thread count; clamped to [1, MAX_SEARCH_THREADS].
     * @note tune.py forces this to 1 when ``n_jobs`` Optuna trials run in parallel
     *       to prevent host oversubscription. The Kaggle runtime never calls it.
     */
    void set_search_thread_limit(int n) {
        orbit::set_search_thread_limit(n);
    }

    /**
     * @brief Read the current search worker thread cap.
     * @return Effective thread limit applied to beam_search_action.
     */
    int get_search_thread_limit() const {
        return orbit::search_thread_limit();
    }

private:
    orbit::Engine engine;
};

}  // namespace

/**
 * @brief Define the Python orbit_engine extension module.
 * @param m pybind11 module object.
 * @note The exposed API intentionally stays small to keep Kaggle entrypoint
 *       overhead below the native search budget.
 */
PYBIND11_MODULE(orbit_engine, m) {
    m.doc() = "Fixed-buffer C++ engine for Kaggle Orbit Wars.";
    m.attr("__version__") = "0.2.0";

    py::class_<PyEngine>(m, "Engine")
        .def(py::init<int>(), py::arg("player") = 0)
        .def("update_observation", &PyEngine::update_observation, py::arg("obs"),
             "Load a dict or attribute-style Orbit Wars observation into native buffers.")
        .def("choose_actions", &PyEngine::choose_actions,
             py::arg("time_budget_ms") = 900, py::arg("seed") = 0,
             "Run fixed-buffer native search and return Kaggle launch rows.")
        .def("step", &PyEngine::step, py::arg("launches"),
             "Advance the native simulator using Python launch rows.")
        .def("debug_state", &PyEngine::debug_state,
             "Return live planets and fleets from the native simulator.")
        .def("debug_evaluate", &PyEngine::debug_evaluate, py::arg("player") = -1,
             "Evaluate the native state for a player.")
        .def("set_hyperparameters", &PyEngine::set_hyperparameters,
             "Inject search/evaluator/candidate hyperparameters as keyword arguments.")
        .def("get_hyperparameters", &PyEngine::get_hyperparameters,
             "Return a Python dict snapshot of the active configuration.")
        .def("set_search_thread_limit", &PyEngine::set_search_thread_limit, py::arg("n"),
             "Cap the number of worker threads the C++ search can spawn per call.")
        .def("get_search_thread_limit", &PyEngine::get_search_thread_limit,
             "Return the current C++ search worker thread cap.");

    m.def("speed_for_ships", &orbit::speed_for_ships, py::arg("ships"),
          "Return the Orbit Wars rule speed for a fleet size.");
    m.def("set_search_thread_limit", &orbit::set_search_thread_limit, py::arg("n"),
          "Process-wide cap on C++ search worker threads.");
    m.def("get_search_thread_limit", &orbit::search_thread_limit,
          "Process-wide cap on C++ search worker threads.");
}
