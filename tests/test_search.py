"""Search behavior tests: interception format and macro-packer spend safety."""

from __future__ import annotations

from _fixtures import make_engine, obs, total_launched_from


def test_static_interception_action_format_and_angle():
    """Return a valid launch toward a static target with an expected heading."""

    engine = make_engine(
        obs(
            [
                [0, 0, 10.0, 10.0, 2.0, 30, 2],
                [1, -1, 25.0, 10.0, 2.0, 5, 3],
            ]
        )
    )
    actions = engine.choose_actions(50, seed=1)
    assert actions
    first = actions[0]
    assert isinstance(first[0], int)
    assert isinstance(first[1], float)
    assert isinstance(first[2], int)
    assert first[0] == 0
    assert abs(first[1]) < 0.05
    assert 1 <= total_launched_from(actions, 0) <= 30


def test_macro_packer_does_not_overspend_source():
    """Ensure packed macro-actions respect per-source ship availability."""

    engine = make_engine(
        obs(
            [
                [0, 0, 10.0, 10.0, 2.0, 8, 1],
                [1, -1, 25.0, 10.0, 2.0, 2, 3],
                [2, -1, 10.0, 25.0, 2.0, 2, 3],
            ]
        )
    )
    actions = engine.choose_actions(100, seed=2)
    assert total_launched_from(actions, 0) <= 8
