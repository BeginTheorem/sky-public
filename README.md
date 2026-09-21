# SkyNet

**WARNING:** this is a very raw alpha version. If it does not run for you, that is
expected — the current version is not meant to work out of the box. The author is
not a professional programmer. This repository is a cleaned-up public copy that
lags behind the local version in functionality. Bug fixes and feature suggestions
are welcome, but acceptance is not guaranteed.

**About the project.** When OpenAI announced the solution to a millennium problem,
it seemed to me that the time of AGI had arrived. In my understanding, that means,
above all, the ability to make decisions independently. Testing that ability is
precisely the goal of this experiment. More precisely, I tried to build a digital
analogue of intelligent life as I understand it.

## Design in one screen

- An Agent Loop consists of a typical ReAct Loop followed by a Memory Loop, which
  compresses the ReAct history into long-term memory, plus a task scheduler.
- After the Agent Loop ends, the Heartbeat Cycle automatically starts a new one,
  using the task scheduler to build a fresh initial prompt.
- The system can run autonomously, without human involvement. There is still a
  soft channel for talking to it through a Telegram bot — but note that it is a
  soft channel: the system may ignore you if it deems that necessary.
- The system is capable of self-improvement: it can change its own code. To keep
  it from breaking itself, there is automatic rollback to the last working version
  (via git), along with type checking, tests, and softly forbidden paths.
- This is an engineering experiment, not a scientific one. We deliberately did not
  create a benchmark, in order to avoid degenerating toward it.

## Layout

| Path | What |
| --- | --- |
| `skynet/` | runtime: reactor, ReAct loop, memory, planner, providers, tools, MCP, store |
| `tests/` | test suite (ruff + pyright + pytest are all blocking) |
| `deploy/` | systemd units and SSH hardening |
| `scripts/` | deploy, rollback, alert, and test entry points |
| `config/` | `skynet.env` and `deploy.env` templates — empty, fill them in locally |
| `SOUL.md` | the agent's identity (the only authoritative identity document) |
| `AGENTS.md` | runbook for a coding agent working on this repository |

## How do I run it?

The project was conceived primarily for personal use. I strongly recommend asking
your coding agent to study (and fix) and then deploy it. Making it work out of the
box for third-party users is not a priority right now.

It runs only on Linux. For the best experience it needs sudo privileges. Run it
only on an isolated server that you do not mind losing.

Requires Python 3.13+. Set up a virtual environment:

```bash
python3 -m venv .venv
.venv/bin/pip install -e '.[dev]'
```

Next, install the MCP servers you want and fill in the keys (cloud providers,
Telegram bot, GitHub MCP) in `config/skynet.env`.

## Tests

```bash
./scripts/test.sh          # ruff + pyright + pytest
```
## Known bags:

In the current public version, the task scheduler is faulty. I have already fixed it; the fix will be published in the next version (in the coming days).

## License

Released under the [MIT License](LICENSE). You may freely study, use, and
redistribute it, including for machine learning and for creating similar
projects.

## Acknowledgments

Many projects are similar to this one to varying degrees. The following were
studied as sources of ideas — concepts and patterns only, with no code copied:

- [OpenCode](https://github.com/sst/opencode) — used as the coding agent while
  building this project; the `bash` and `webfetch` tools and the MCP client were
  inspired by it. (MIT)
- [Hermes Agent](https://github.com/NousResearch/hermes-agent) — the idea of the
  Heartbeat Cycle. (MIT)
- [Darwin Gödel Machine](https://github.com/jennyzzt/dgm) — a kindred
  self-improving project; its experience was taken into account, though this
  project pursues a different goal and implements many things differently.
  (Apache-2.0)
- [Anima](https://github.com/huodebing-alt/anima) — the sleep/consolidation,
  forgetting, and identity patterns. (MIT)
- [Kiri](https://github.com/T-80BVVD/kiri) — bounded tool-calling and ordinary
  message semantics. (AGPL-3.0)
- [alive](https://github.com/marchantdev/alive) — the minimal, transparent
  wake loop. (MIT)
- [Letta Code](https://github.com/letta-ai/letta-code) — persistent memory and
  cognitive state. (Apache-2.0)
- [Alkaline](https://github.com/davccavalcante/alkaline) — durable execution:
  replay, retries, and recovery. (Apache-2.0)
- [AgentOS](https://github.com/Roxmix/agentos-mcp) — memory, goals, and reflection
  as separate data, plus event patterns. (MIT)

During development we also distilled many arXiv papers, to confirm or refute our
hypotheses and to find new ideas. Among them:

- reward hacking and self-improvement safety —
  [STOP 2310.02304](https://arxiv.org/abs/2310.02304),
  [RLVR reward hacking 2604.15149](https://arxiv.org/abs/2604.15149),
  [1-bit danger signals 2604.23210](https://arxiv.org/abs/2604.23210);
- open-endedness and quality-diversity —
  [Voyager 2305.16291](https://arxiv.org/abs/2305.16291),
  [POET 1901.01753](https://arxiv.org/abs/1901.01753),
  [OMNI 2306.01711](https://arxiv.org/abs/2306.01711);
- the limits of a single self-improvement objective —
  [Darwin Gödel Machine 2505.22954](https://arxiv.org/abs/2505.22954).

No inspiration project is a dependency of SkyNet, and none of their code was
copied — only ideas and patterns were reused.
