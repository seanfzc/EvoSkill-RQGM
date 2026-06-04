"""evoskill library — inspect and curate the skill library.

Surfaces the Phase 2 lifecycle operations:
  evoskill library                     list skills + usage/credit stats
  evoskill library --duplicates        find near-duplicate skills
  evoskill library --select "<task>"   preview which skills a task would load
  evoskill library --deprecated        list archived skills
"""

from __future__ import annotations

from pathlib import Path

import click
from rich.console import Console
from rich.table import Table

console = Console()


def _similarity_backend(cfg):
    """Build the configured similarity backend, resolving the embedding API key."""
    from src.continuous import make_similarity_backend
    from src.harness.provider_auth import resolve_provider_api_key

    lc = cfg.continuous.lifecycle
    api_key, _ = resolve_provider_api_key(lc.embedding_provider)
    return make_similarity_backend(
        backend=lc.similarity_backend,
        provider=lc.embedding_provider,
        model=lc.embedding_model,
        api_key=api_key,
        cache_path=str(cfg.continuous_embeddings_cache_path),
        log=lambda m: console.print(f"  [dim]{m}[/dim]"),
    )


@click.command("library")
@click.option("--duplicates", is_flag=True, default=False, help="Find near-duplicate skills.")
@click.option("--select", "select_task", default=None, help="Preview top-k skills for a task.")
@click.option("--deprecated", is_flag=True, default=False, help="List archived/deprecated skills.")
@click.option("--config", "config_path", type=click.Path(dir_okay=False, path_type=Path),
              default=None, help="Load a specific config TOML file.")
def library_cmd(duplicates, select_task, deprecated, config_path):
    """Inspect and curate the live skill library."""
    from src.cli.config import load_config
    from src.continuous import (
        SkillLibrary,
        SkillStatsStore,
        find_duplicates,
        select_skills,
    )

    cfg = load_config(config_path=config_path)
    library = SkillLibrary(cfg.skills_dir)
    skills = library.list()

    if deprecated:
        _list_deprecated(cfg)
        return

    if not skills:
        console.print("  No skills yet. Run [bold]evoskill run[/bold] or graduate candidates first.")
        return

    if select_task:
        backend = _similarity_backend(cfg)
        matches = select_skills(skills, select_task, backend, k=cfg.continuous.lifecycle.retrieval_top_k)
        table = Table(box=None, pad_edge=False, show_header=True, header_style="bold",
                      title=f"Top skills for: {select_task[:50]}")
        table.add_column("Score", width=7)
        table.add_column("Skill")
        table.add_column("Description")
        for m in matches:
            table.add_row(f"{m.score:.3f}", m.skill.name, (m.skill.description or "")[:60])
        console.print(table)
        return

    if duplicates:
        backend = _similarity_backend(cfg)
        groups = find_duplicates(skills, backend, threshold=cfg.continuous.lifecycle.dedupe_similarity)
        if not groups:
            console.print(f"  No near-duplicates above {cfg.continuous.lifecycle.dedupe_similarity}.")
            return
        for g in groups:
            console.print(f"\n  [bold]similar ({g.max_similarity:.3f})[/bold]: "
                          f"{', '.join(g.names)}")
        console.print("\n  [dim]Merge proposals/graduation come with the gate (Phase 3).[/dim]\n")
        return

    # Default: list skills with stats.
    stats_store = SkillStatsStore(cfg.continuous_skill_stats_path)
    table = Table(box=None, pad_edge=False, show_header=True, header_style="bold")
    table.add_column("Skill", min_width=24)
    table.add_column("Used", width=6)
    table.add_column("Wins", width=6)
    table.add_column("Contrib", width=8)
    table.add_column("Strikes", width=8)
    table.add_column("Description")
    for s in skills:
        st = stats_store.get(s.name) if stats_store.has(s.name) else None
        table.add_row(
            s.name,
            str(st.episodes_active) if st else "0",
            str(st.success_when_active) if st else "0",
            f"{st.contribution:.2f}" if st else "0.00",
            str(st.deprecation_strikes) if st else "0",
            (s.description or "")[:50],
        )
    console.print(table)
    console.print(f"\n  {len(skills)} skill(s) in {cfg.skills_dir}\n")


def _list_deprecated(cfg) -> None:
    archive = cfg.continuous_deprecated_dir
    if not archive.is_dir() or not any(archive.iterdir()):
        console.print("  No deprecated skills.")
        return
    table = Table(box=None, pad_edge=False, show_header=True, header_style="bold")
    table.add_column("Archived skill")
    for child in sorted(archive.iterdir()):
        if child.is_dir():
            table.add_row(child.name)
    console.print(table)
