#include "orbit_engine.hpp"

#include "eval.hpp"
#include "search.hpp"

namespace orbit {

Engine::Engine() {
    sim.reset();
}

Engine::Engine(int player) : Engine() {
    sim.state.player = player;
}

void Engine::update_observation(const ObservationInput& obs) {
    sim.load_from_observation(obs);
}

void Engine::step_actions(const LaunchList& launches) {
    sim.step(launches);
}

LaunchList Engine::choose_actions(int time_budget_ms, uint64_t seed) {
    SearchConfig config{};
    return beam_search_action(sim.state, config, time_budget_ms, seed);
}

double Engine::debug_evaluate(int player) const {
    const int eval_player = (player >= 0 && player < MAX_PLAYERS) ? player : sim.state.player;
    return evaluate_state(sim.state, eval_player);
}

}  // namespace orbit
