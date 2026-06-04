"""Tests for the skill library reader."""

from __future__ import annotations

from src.continuous.library import SkillLibrary, parse_skill_file


def _write_skill(skills_dir, name, *, frontmatter=True, description="does a thing", body="The rule."):
    d = skills_dir / name
    d.mkdir(parents=True)
    if frontmatter:
        text = f"---\nname: {name}\ndescription: {description}\n---\n\n{body}"
    else:
        text = body
    (d / "SKILL.md").write_text(text)
    return d / "SKILL.md"


class TestParseSkillFile:
    def test_parses_frontmatter(self, tmp_path):
        p = _write_skill(tmp_path, "my-skill", description="preserve units")
        skill = parse_skill_file(p)
        assert skill.name == "my-skill"
        assert skill.description == "preserve units"
        assert skill.body == "The rule."
        assert "my-skill: preserve units" == skill.text

    def test_no_frontmatter_uses_dir_name(self, tmp_path):
        p = _write_skill(tmp_path, "bare", frontmatter=False, body="just body text")
        skill = parse_skill_file(p)
        assert skill.name == "bare"
        assert skill.description == ""
        assert skill.body == "just body text"

    def test_bad_yaml_tolerated(self, tmp_path):
        d = tmp_path / "broken"
        d.mkdir()
        (d / "SKILL.md").write_text("---\nname: [unclosed\n---\nbody")
        skill = parse_skill_file(d / "SKILL.md")
        assert skill.name == "broken"  # falls back to dir name

    def test_text_strips_trailing_colon_when_no_description(self, tmp_path):
        p = _write_skill(tmp_path, "nodesc", description="")
        skill = parse_skill_file(p)
        assert skill.text == "nodesc"


class TestSkillLibrary:
    def test_list_and_names(self, tmp_path):
        sd = tmp_path / "skills"
        _write_skill(sd, "alpha")
        _write_skill(sd, "beta")
        lib = SkillLibrary(sd)
        assert lib.names() == ["alpha", "beta"]
        assert len(lib.list()) == 2

    def test_get_by_name_and_dir(self, tmp_path):
        sd = tmp_path / "skills"
        _write_skill(sd, "gamma")
        lib = SkillLibrary(sd)
        assert lib.get("gamma") is not None
        assert lib.get("missing") is None

    def test_ignores_dirs_without_skill_md(self, tmp_path):
        sd = tmp_path / "skills"
        _write_skill(sd, "real")
        (sd / "empty").mkdir()
        (sd / "loose.txt").write_text("x")
        assert SkillLibrary(sd).names() == ["real"]

    def test_missing_dir(self, tmp_path):
        assert SkillLibrary(tmp_path / "nope").list() == []
