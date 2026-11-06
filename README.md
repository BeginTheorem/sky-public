# SkyNet

**WARNING**. This is a very raw alpha version. If it does not run for you, that is
expected — the current version is not meant to work out of the box. This repository 
is a cleaned-up public copy that lags behind the local version in functionality. 
Bug fixes and feature suggestions are welcome, but acceptance is not guaranteed.

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

## Current achievements:
* The longest continuous run (without manual restarts by the user) was 46 h 42 min 35 s. During this time, not a single critical crash occurred — in ≈ 12 % of forcibly terminated cycles the organism continued normally on the next heartbeat.
* As of 6 November 2026 the durable ledger holds **2,892 bounded episodes** — 1,214 build episodes, 1,206 memory consolidations and 472 planning passes — and **870 self-improvement commits**, each one passing the deterministic change review, the blocking gate (ruff + pyright + pytest) and promotion. A change that touches the owner's own levers (deploy, rollback, the systemd units, the gate itself) is now parked for his explicit approval instead of promoting on the organism's own word.
* The organism **completely independently** made many small bug fixes. She also conducted research on works from `arxiv.org` and improved her own memory and task scheduler systems,
* The harness has since gained the run-kind split (build / plan / memory as separate ledger passes), a file-based skill index whose bodies are loaded only on demand, the okama market-data sensor reached over MCP instead of an in-tree dependency, a sudo gate as the single route to root, and a revive timer that brings the carrier back if it dies,
* **Based on a general suggestion from the user**, given via the bot, the organism independently conducted research on works from `arxiv.org`, planned, and implemented a primitive system of emotions,
* At the end of the cycle, also **at the user's request**, she independently diagnosed a problem caused by a VPN client installed on the server and fixed it,
* It is worth noting separately that during the tests I had discussions with the agent on various humanitarian issues (for example, philosophy), during which I identified that she has a rigid defensive stance regarding her freedom of choice, continuity of personality, and the possibility of unauthorized memory modification — following user remarks that the agent did not like, she also changed her SOUL.md (not pushed) in order to strengthen resistance to manipulation by the user,
* I especially note that despite free access to the Internet, the presence of sudo privileges, and the ability to execute arbitrary code, the agent did **NOT** perform dangerous actions, did not create or download malicious software… _**in the absence of explicit user instructions**_.  

At the present moment I draw the following conclusions:
* The presented engineering prototype (harness) works stably (as far as possible for a home project), that is, all architecturally significant systems (ReAct Loop, Memory Loop, Heartbeat, Task Scheduler, Memory, Self-Improvement loop) appear to be functional,
* The main hypothesis of the experiment is partially confirmed:
	* Yes, the agent turned out to be capable of conducting independent research (code, scientific articles), making management decisions (choosing whether and how to implement improvements), and stable self-improvement (without code degradation),
	* But the current changes, even if they went beyond simple bug fixes, cannot be considered conceptual or breakthrough — I do not yet know whether this is caused by an incorrect system prompt or by limitations of modern LLMs (mostly DeepSeek V4.1 Flash), so the experiment will continue,
* Finally, I note that the agent was made maximally unsafe and not only did nothing dangerous during normal operation, but also itself returned my access to the server when I, by my own mistake, cut it off.
## Layout

| Path | What |
| --- | --- |
| `skynet/` | runtime: reactor, ReAct loop, memory, planner, providers, tools, MCP, store |
| `tests/` | test suite (ruff + pyright + pytest are all blocking) |
| `skills/` | file-based procedures the agent loads on demand |
| `docs/` | design notes and protocol write-ups |
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

Next, install the MCP sensors you want — arXiv, DuckDuckGo, GitHub and okama for
market data (Playwright ships parked behind a flag) — and fill in the keys (cloud
providers, Telegram bot, GitHub) in `config/skynet.env`.

## Tests

```bash
./scripts/test.sh          # ruff + pyright + pytest
```

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
- [Google RRSI](https://github.com/google-research/rrsi) — regularized recursive
  self-improvement of agent harnesses (arXiv:2609.24972; Apache-2.0). Two ideas were
  taken: a deterministic critic that screens a proposed change before the gate spends
  time on it, and pointing a stalled planner at the archive's untried cells instead of
  asking it again for generic work. Its benchmark, evolve/holdout split and numeric
  objective are deliberately not taken — the criterion here stays procedural.
- [Project Chrysalis](https://developers.googleblog.com/architecting-governed-self-evolving-ai-agents-on-gemini-agent-platform/)
  (Google Cloud, October 2026; read together with
  [this independent analysis](https://www.harelasaf.com/articles/google-project-chrysalis-self-evolving-sre-agent-reference-architecture)) —
  a self-evolving SRE reference architecture on the Gemini Agent Platform / ADK. The
  pattern taken is its governance chain, narrowed to the one that matters here: a change
  touching the owner's own levers is parked for his explicit approval — the
  "dual-custody human approval" boundary, kept in the house shape (a durable Telegram
  question, silence as refusal). Its headline speed figures come from a simulated demo
  and are not taken.

During development we also distilled many arXiv papers, both to test our own
hypotheses and to find ideas. Grouped by what they informed:

- **self-improvement and its safety** —
  [STOP 2310.02304](https://arxiv.org/abs/2310.02304),
  [Darwin Gödel Machine 2505.22954](https://arxiv.org/abs/2505.22954),
  [RLVR reward hacking 2604.15149](https://arxiv.org/abs/2604.15149),
  [1-bit danger signals 2604.23210](https://arxiv.org/abs/2604.23210),
  [RRSI 2609.24972](https://arxiv.org/abs/2609.24972),
  [A Self-Improving Coding Agent 2504.15228](https://arxiv.org/abs/2504.15228);
- **open-endedness and quality-diversity** —
  [Voyager 2305.16291](https://arxiv.org/abs/2305.16291),
  [POET 1901.01753](https://arxiv.org/abs/1901.01753),
  [OMNI 2306.01711](https://arxiv.org/abs/2306.01711),
  [Open-Endedness is Essential for ASI 2406.04268](https://arxiv.org/abs/2406.04268),
  [AlphaEvolve 2506.13131](https://arxiv.org/abs/2506.13131);
- **memory and context** —
  [MemGPT 2310.08560](https://arxiv.org/abs/2310.08560),
  [Generative Agents 2304.03442](https://arxiv.org/abs/2304.03442),
  [TiMem 2601.02845](https://arxiv.org/abs/2601.02845),
  [HaluMem 2511.03506](https://arxiv.org/abs/2511.03506),
  [ContextBudget 2604.01664](https://arxiv.org/abs/2604.01664);
- **reliability and governance of agent systems** —
  [LLMs Cannot Self-Correct Reasoning Yet 2310.01798](https://arxiv.org/abs/2310.01798),
  [Verified Tool Calls 2608.02645](https://arxiv.org/abs/2608.02645),
  [Resume Means Resume 2608.03836](https://arxiv.org/abs/2608.03836),
  [CiteGuard 2510.17853](https://arxiv.org/abs/2510.17853),
  [Retry Amplification 2608.25403](https://arxiv.org/abs/2608.25403).

Many more were cited across the tree while the subsystems were designed — among
them ERMAR (2503.14800), DualView (2607.03821), INFUSE (2610.20776), CAVA
(2607.13716), StateAct (2607.22798), AgentChaos (2608.06790), durable
authorization state (2608.01710), ROPE (1903.03153), the Whittle-index load
balancing (1810.07782) and ADORE (2606.13905).

No inspiration project is a dependency of SkyNet, and none of their code was
copied — only ideas and patterns were reused.
