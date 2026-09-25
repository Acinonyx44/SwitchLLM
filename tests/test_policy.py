import pytest

from switchllm import Policy, PolicyError


def minimal(**overrides):
    data = {
        "defaults": {"baseline_model": "m1", "default_role": "general"},
        "models": [{"id": "m1", "provider": "mock", "tier": 1, "input_per_mtok": 1, "output_per_mtok": 1}],
        "roles": {"general": {}},
    }
    for key, value in overrides.items():
        data[key] = value
    return data


def test_example_policy_loads():
    p = Policy.example()
    assert p.baseline.tier == 3
    assert p.rule_for("legal").min_tier == 3
    assert p.rule_for("unknown").min_tier == 1


def test_minimal_policy():
    p = Policy.from_dict(minimal())
    assert p.roles["general"].max_tier == 3


@pytest.mark.parametrize(
    "overrides, message",
    [
        ({"models": []}, "no models"),
        ({"defaults": {"baseline_model": "nope", "default_role": "general"}}, "unknown model"),
        ({"defaults": {"baseline_model": "m1", "default_role": "ghost"}}, "default_role"),
        ({"directory": {"group_roles": {"g": "ghost"}}}, "undefined role"),
        ({"models": [{"id": "m1", "provider": "mock", "tier": 7, "input_per_mtok": 1, "output_per_mtok": 1}]},
         "tier"),
    ],
)
def test_validation(overrides, message):
    with pytest.raises(PolicyError, match=message):
        Policy.from_dict(minimal(**overrides))


def test_load_from_file(tmp_path):
    path = tmp_path / "p.toml"
    path.write_text(
        '[defaults]\nbaseline_model = "m1"\n'
        '[[models]]\nid = "m1"\nprovider = "mock"\ntier = 1\ninput_per_mtok = 1\noutput_per_mtok = 1\n'
        "[roles.general]\n"
    )
    assert Policy.load(path).baseline_model == "m1"
