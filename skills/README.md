# Skills

`paper/` holds the Skills of the paper.

| Folder | Skill |
|---|---|
| `paper/S0_seed` | the seed Skill, which states the input, the output, and the SDK |
| `paper/S3`, `paper/S6`, `paper/S9` | the three Skills accepted by the validation gate in the main run |
| `paper/human_curated` | the human-curated Skill of the ablation |

A Skill is a directory with `SKILL.md` (the procedure), `ACTIONS.md` (the SDK reference), and for the
improved Skills `TOOLS.md` and `tools/` (helper scripts written by the Skill Evolver).

The folders `v_minimal`, `v_io`, `v_io_sdk`, and `v_beginner` are small Skills that the unit tests load.
