# Claude project memory

This directory is the Git-backed, portable copy of durable project knowledge
collected while developing and evaluating the xLLM multi-model branch.

## Index

- [Project overview](project_overview.md)
- [Architecture map](xllm_arch_map.md)
- [Multi-model mechanism](multimodel_mechanism.md)
- [XTensor memory](xtensor_memory.md)
- [Conventions](xllm_conventions.md)
- [Build and run](build_and_run.md)
- [Fork master quirks](fork_master_quirks.md)
- [Long-running task detachment](long_running_tasks_detach.md)
- [Container PID isolation](container_pid_isolation.md)
- [Replica dispatch symmetry wall](dispatch_ab_symmetry_wall.md)
- [Map/unmap overhead experiment](overhead_map_unmap_experiment.md)
- [Chinese retrospective notes](retro-notes-in-chinese.md)
- [Multi-model retrospective map](retro/00-multimodel-retro-moc.md)

## Portability

These notes are snapshots of knowledge, not executable configuration. Paths,
line numbers, device IDs, ports, branch names, and environment assumptions may
be stale or machine-specific. Verify them against the current checkout and
runtime before acting on them.

On another machine, clone the repository and start Claude Code from the
repository root. `CLAUDE.md` and `.claude/skills/` provide the project entry
points. If an external Claude project-memory directory is used, keep only a
small local index there that points back to this directory; do not rely on an
absolute symlink copied from another machine.

Personal career/interview notes and machine-local Claude settings are
intentionally excluded from this repository copy.
