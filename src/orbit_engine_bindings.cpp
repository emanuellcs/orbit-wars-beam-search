#include "orbit_engine.hpp"

#include "geometry.hpp"

#include <pybind11/pybind11.h>

#include <algorithm>
#include <cstdint>
#include <string>

namespace py = pybind11;

namespace {

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

int get_int_field(const py::handle& obj, const char* name, int fallback) {
    py::object value = get_field(obj, name);
    if (value.is_none()) {
        return fallback;
    }
    return py::cast<int>(value);
}

double get_double_field(const py::handle& obj, const char* name, double fallback) {
    py::object value = get_field(obj, name);
    if (value.is_none()) {
        return fallback;
    }
    return py::cast<double>(value);
}

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

class PyEngine {
public:
    explicit PyEngine(int player = 0) : engine(player) {}

    void update_observation(const py::object& obs) {
        engine.update_observation(make_observation(obs));
    }

    py::list choose_actions(int time_budget_ms = 900, uint64_t seed = 0) {
        return launch_list_to_python(engine.choose_actions(time_budget_ms, seed));
    }

    void step(const py::object& launches) {
        engine.step_actions(make_launches(launches));
    }

    py::dict debug_state() const {
        return debug_state_to_python(engine.sim.state);
    }

    double debug_evaluate(int player = -1) const {
        return engine.debug_evaluate(player);
    }

private:
    orbit::Engine engine;
};

}  // namespace

PYBIND11_MODULE(orbit_engine, m) {
    m.doc() = "Fixed-buffer C++ engine for Kaggle Orbit Wars.";
    m.attr("__version__") = "0.1.0";

    py::class_<PyEngine>(m, "Engine")
        .def(py::init<int>(), py::arg("player") = 0)
        .def("update_observation", &PyEngine::update_observation, py::arg("obs"))
        .def("choose_actions", &PyEngine::choose_actions,
             py::arg("time_budget_ms") = 900, py::arg("seed") = 0)
        .def("step", &PyEngine::step, py::arg("launches"))
        .def("debug_state", &PyEngine::debug_state)
        .def("debug_evaluate", &PyEngine::debug_evaluate, py::arg("player") = -1);

    m.def("speed_for_ships", &orbit::speed_for_ships, py::arg("ships"));
}
