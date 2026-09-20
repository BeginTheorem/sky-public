# Project SkyNet — runbook для coding-агента

Этот файл читает opencode как инструкции к проекту. Он описывает **только то, что
проверяемо по коду**. Если файл и код расходятся — прав код, а расхождение в этом
файле является багом.

## Что это

SkyNet — research-harness: проверяет, может ли обычная LLM работать как
непрерывно существующая автономная система за счёт harness. LLM дискретна
(`stimulus -> reasoning -> tool calls -> result`), непрерывность обеспечивает
внешний Reactor; пользователь в lifecycle не участвует, а свой код модель может
менять только через gated self-improvement.

**Критерий процедурный, а не числовой, и это осознанное решение.** Внешнего
бенчмарка у проекта нет намеренно: рукотворная числовая функция деградирует
систему — reward hacking ([2310.02304](https://arxiv.org/abs/2310.02304),
[2604.15149](https://arxiv.org/abs/2604.15149),
[2604.23210](https://arxiv.org/abs/2604.23210)), а DGM сам называет ограничением
«self-improvement being tied to a single objective»
([2505.22954](https://arxiv.org/abs/2505.22954)). Вместо бенчмарка harness требует
**проверяемости**: успешный tool result (`verified_progress`), прохождение
gate-набора и цитату для `research`. Это компромиссный подход, взятый из
литературы ([Voyager 2305.16291](https://arxiv.org/abs/2305.16291),
[POET 1901.01753](https://arxiv.org/abs/1901.01753),
[OMNI 2306.01711](https://arxiv.org/abs/2306.01711)) — не «любые критерии» и не
«наш бенчмарк». Отсутствие бенчмарка — гипотеза эксперимента, а не долг; не
«чинить» его.

## Инварианты, которые нельзя ломать

- **Reactor — единственный владелец lifecycle.** Только он владеет
  wake/sleep/scheduling, checkpointing и восстановлением. Memory Loop, MCP,
  reflection и фоновые сервисы не запускают конкурирующих daemon-loop'ов.
- **Костяк цикла**:
  `SYSTEM -> Start -> ReAct -> Finish -> Memory Loop -> evaluation -> checkpoint -> schedule`.
- **Transcript ограничен на каждый outer cycle**, не накапливается бесконечно;
  история prun'ится после checkpoint'а (`store.prune_run_history`, вызывается из
  `reactor.py`).
- **Единственный kind критерия — `verified_progress`** (kind —
  `store.evaluate_criteria`; навешивается на прогон `reactor._success_criteria`;
  COMPLETED без пройденных критериев даунгрейдится в BLOCKED в `reactor.py`).
  Отчёт модели без успешного tool result не считается выполнением.
- **Память — управляемый ресурс.** `pinned` инжектится всегда; автопоиск
  ограничен `SKYNET_MEMORY_INJECT_LIMIT` (default 3,
  `reactor.ReactorConfig.memory_inject_limit`, читается в `cli.py`); глубже — по
  запросу tool'ом `memory`; неверная память superseded с evidence
  (`store.supersede_memory`).
- **Daily maintenance не гейтится метриками**: retention и decay идут раз в
  UTC-день по sidecar-маркеру (`reactor._maybe_run_daily_maintenance`;
  `store.prune_event_log` — retention, `store.decay_memory_confidence` — decay),
  не под `SKYNET_METRICS_SNAPSHOT`.
- **Никаких дублирующих контуров**: один владелец lifecycle, один planner
  выбора работы, один stale-run путь, один criteria-kind. Мета-контур
  (дескрипторы, политика выбора работы, метрики) вне досягаемости
  self-improvement — судимый не редактирует судью.
- `SOUL.md` — единственный авторитетный документ про identity; не переписывать
  его вслепую.

## Открытость

- Архив идей хранит behavioural descriptor: `subsystem × change_type ×
  evidence_source` = 144 ячейки, в каждой остаётся только лучшая идея
  (`skynet/idea_archive.py`; MAP-Elites-правило — `store.archive_idea`,
  сэмплирование родителей — `store.sample_archive_parents`, материализация —
  `store.materialize_idea`).
- `research`-предложение без цитаты отклоняется (learnability gate,
  `idea_archive.learnability_defect`).
- **Архив кормится реальным трафиком, а не только идеями планировщика.**
  `_archive_run_self_improvements` (`reactor.py`) сканирует `tool_result`
  прогона и архивирует каждое успешное `propose_self_improvement`, с качеством,
  равным измеренному value прогона. Раньше архивировались только идеи, созданные
  `_run_autonomous_planning`, поэтому правки организма не попадали в архив.
- **Клапан `external_seek` — каденция, а не исчерпание.**
  `_seek_external_evidence` вызывается при
  `generation % SKYNET_EXTERNAL_SEEK_EVERY == 0` (+ durable cooldown), если есть
  живой орган чувств, независимо от того, пуст ли портфель (`reactor.py`).
  Раньше он стоял после `_create_planner_fallback`, который всегда что-то
  создавал, поэтому клапан мог не открыться ни разу.
- **Сообщение владельца — уведомление, а не приказ.** `_inbox_notifications`
  (`reactor.py`) отдаёт каждое pending-сообщение как observation
  `inbox_notification` с `event_id`; задача **не** создаётся, обычную работу оно
  не вытесняет. Решение принимает организм: действовать своими tools, создать
  задачу или закрыть уведомление через `acknowledge_inbox(event_id, decision)`
  (`dialogue.py`). Пока не закрыто — приходит снова. Пустой планировщик
  заполняется **фиксированным пулом**: `_ensure_roadmap_seed` при полностью
  пустой таблице целей re-seed'ит `ROADMAP_TASKS` (`handoff.seed`), затем идёт
  bounded bootstrap-задача (`_create_planner_fallback`). Блокированная (не
  пустая) цель сид не трогает — иначе маскируется блок.
- **Scratch-каталог согласован.** `SKYNET_SCRATCH_DIR` (default
  `/tmp/skynet-scratch`) входит в `ExecutionPolicy.allowed_roots()`, поэтому
  `read`, `grep` и `bash` видят одну границу (`policy.py`, `tools.py`).

## Тесты

- Полный прогон: `./scripts/test.sh` — **ruff + pyright + pytest**; тот же набор
  выполняет self-improvement gate. Или `.venv/bin/python -m pytest -q`.
- **ruff — блокирующий этап гейта** (`_run_ruff_stage`, `self_improvement.py`) и
  шаг `test.sh`. Набор правил — в `[tool.ruff.lint]` `pyproject.toml`
  (`E,W,F,I,B,C4,UP,SIM,RUF,PERF,PIE,RET,S,TRY` + курируемые `ignore`); дерево
  ruff-clean, baseline'а нет. Этап пропускается, только если ruff не установлен
  или в worktree нет `pyproject.toml` (минимальная тестовая фикстура).
- **pyright — блокирующий этап гейта** (`_run_pyright_stage`) и шаг `test.sh`.
  Дерево pyright-clean (0 ошибок в `skynet/` и `tests/`), baseline'а нет: любой
  диагностик — регрессия. Версия запинена (`pyright==1.1.413`) в
  `[project.optional-dependencies] dev`; в proposal worktree без `.venv`
  запускается с `--pythonpath <root>/.venv/bin/python`.
- **mypy удалён**: pyright — единственный type-авторитет, второй
  чекер означал второй baseline. `pyflakes` тоже убран — его покрывает `F`.
- Размер набора постоянно меняется — проверяй фактическое число,
  а не верь этому файлу.
- С Python работать только через `.venv/bin/python`; системный интерпретатор не
  имеет pytest.

## Provider chain

- **Paid-режим: активен только openrouter.** Остальные реализации
  (`ollama`, `nvidia*`, `nemotron`, `openai`) **архивированы, не удалены**:
  выключены через env (`*_ENABLED=false`, `SKYNET_PROVIDER_CHAIN=openrouter`) и
  помечены комментарием в `providers/__init__.py`. Вернуть — флип флагов без
  правки кода.
- OpenRouter попадает в активную цепочку только при включённом money-boost
  (`active_chain_names`, `providers/__init__.py`; env `SKYNET_MONEY_BOOST` или
  `state/money-boost.json`); имя читается заново, без рестарта процесса.
- **Таймаут openrouter — connect + idle, не общий дедлайн.** `OPENROUTER_TIMEOUT_SECONDS`
  (30) — connect; `OPENROUTER_CHUNK_TIMEOUT_SECONDS` (900) — тишина между чанками,
  во время активного стрима не горит; `OPENROUTER_STREAM_DEADLINE_SECONDS` (0 = off) —
  опциональный общий потолок. Ступень лестницы `FallbackProvider` трактуется как
  idle-бюджет, а не как дедлайн потока (`openrouter.py`).
- Единственный владелец ретраев — `FallbackProvider`; не дублировать ретраи в
  других слоях.
- **Credentials — только в `config/skynet.env`.** В этом репозитории файл
  поставляется пустым шаблоном (только имена ключей), чтобы было видно, какие
  ключи нужны, но реальные значения в него не попадали. Никогда не коммить
  секреты и не держать бэкапы env внутри дерева репозитория.

## Бюджет и контекст (paid-режим)

- `SKYNET_MAX_SECONDS=3600` — один прогон = час модельного времени; watchdog
  `SKYNET_WATCHDOG_SECONDS=3900` должен его превышать.
- `SKYNET_MAX_STEPS=400` (сообщения/шаги ReAct), `SKYNET_REACT_INPUT_TOKENS=500000`
  (окно ReAct), `SKYNET_OUTPUT_TOKENS=16384`, `SKYNET_MEMORY_INPUT_TOKENS=100000`.
- `SKYNET_CONTEXT_FINISH_RESERVE=50000` и `SKYNET_TOOL_RESULT_MAX_CHARS=16000`
  читаются в `reactor.py` при сборке `ReActConfig`.

## Deploy и restart

- Деплой и рестарт сервисов выполняются **только по явной команде владельца**.
  Агент не запускает их сам и не коммитит.
- `scripts/deploy.sh` — деплой; `scripts/rollback.sh <commit|tag>` — откат.
- Адрес/доступы сервера живут в `config/deploy.env` (gitignored), не в скриптах
  и не в этом файле.
- **Пакет ставится editable с dev-extra** (`pip install -e '.[dev]'`,
  `deploy.sh`), и это не косметика: self-improvement правит рабочее дерево, а
  editable гарантирует, что живой процесс импортирует актуальный код, а не
  замороженную копию в `site-packages` (обычная установка молча ломала бы
  promote). Юниты запускают `python -m skynet` из корня
  (`deploy/skynet.service:13,17`), так что импорт работал бы и без установки;
  editable дополнительно даёт console-скрипты `skynet`/`skynet-telegram`
  (`pyproject.toml:14-16`) и установку зависимостей. `[dev]` обязателен, потому
  что блокирующий этап гейта — `pyright`. Побочный продукт сборки —
  `skynet.egg-info/`, он в `.gitignore`.

## Логи

- Один вход: `skynet logs --tier system|mandatory|advanced|verbose` (default
  `advanced`, `cli.py:60`).
- **system** — `state/system-probe.jsonl`, кольцо 250 записей (`SystemProbeLog`);
  cadence — `SKYNET_SYSTEM_PROBE_EVERY` (default 20, `heartbeat.py:23`).
- **mandatory** — пары `SCHEDULER`+`REPORT` по завершённым run'ам из event log
  (`skynet/reporting.py`); 5 по умолчанию, максимум 100 (`cli.py`).
- **advanced** — `state/runtime.jsonl`, 25 MB × 3 бэкапа
  (`runtime_log.RuntimeLog._rotate_locked`; конструируется в `store.py`); по
  умолчанию отдаёт последние 500 записей.
- **verbose** — полные дампы запросов/ответов провайдера в
  `state/verbose.jsonl` (кольцо 50 МБ, mode 600, секреты редактируются), по
  умолчанию выключен; `skynet verbose on|off|status` пишет `state/verbose.json`
  (hot-reload без рестарта, `cli.py:172`), плюс env-переключатель
  `SKYNET_VERBOSE_PROVIDER` (`runtime_log.verbose_enabled`).
- Для system-яруса обязателен `psutil` (в `dependencies` `pyproject.toml`);
  сервисному аккаунту на сервере доступен `sudo` без пароля, поэтому probe
  вызывает `sudo -n journalctl` (`system_probe.py`).

## Tools

- Встроенный набор `default_tools()` — **`bash`, `webfetch`, `read`, `grep`,
  `db`** (`skynet/tools.py`).
- `Reactor.__init__` всегда добавляет `ask_user`, `send_message_to_user`,
  `acknowledge_inbox` (закрыть pending-уведомление владельца), `memory`
  (действия `search/remember/forget/pin/unpin/correct`,
  `skynet/memory_tool.py`) (`reactor.Reactor.__init__`); при наличии
  `<root>/.git` — `propose_self_improvement`, `promote_self_improvement`,
  `request_rollback` (`reactor.Reactor.__init__`).
- Telegram владельца — обычный чат: текст без `/` идёт как observation, команды
  начинаются с `/` (`telegram_bot.py`); блокирует только
  `ask_user(wait_seconds>0)` (`dialogue.AskUserTool`).
- Self-improvement gate отклоняет правки `tests/`, `conftest.py`,
  `pyproject.toml`, `pytest.ini`, `tox.ini`, `setup.cfg` и мета-контура
  (`skynet/self_improvement.py`, `idea_archive.py`, `planner.py`,
  `autonomous_planner.py`, `metrics.py` — «судимый не редактирует судью»);
  нечитаемые changed paths — тоже отказ
  (`self_improvement.GATE_PROTECTED_PATHS`).
- Изменение кода durable **только** через `propose_self_improvement` /
  `promote_self_improvement`; правка main worktree напрямую не засчитывается.
- MCP-органы чувств подключаются через `SKYNET_MCP_<NAME>_COMMAND`
  (`cli._mcp_config`). В `config/skynet.env` перечислены четыре:
  DuckDuckGo, Playwright, arXiv, GitHub (`--read-only`).
- На практике реально используются **arxiv** и **github**; `ddg`, `playwright`,
  `webfetch` — заметно реже. Не судить о пользе по факту подключения.

## Документы

- `SOUL.md` — identity, инжектится в system-промпт ReAct и Memory. Авторитетен.
- `AGENTS.md` (этот файл) — runbook для coding-агента.
- **Источник истины — код.** Никакой документ не нормативен, кроме `SOUL.md`.

## Ссылки

Проекты, изученные как источники идей (только концепции и паттерны, без
копирования кода; ни один не является зависимостью):

- [OpenCode](https://github.com/sst/opencode) — coding-агент, на котором
  строился проект; идеи для `bash`/`webfetch` и MCP-клиента. MIT.
- [Hermes Agent](https://github.com/NousResearch/hermes-agent) — идея Heartbeat
  Cycle. MIT.
- [Darwin Gödel Machine](https://github.com/jennyzzt/dgm) — родственный
  self-improving проект; его опыт учтён, но цель и реализация иные. Apache-2.0.
- [Anima](https://github.com/huodebing-alt/anima) — sleep/consolidation,
  forgetting, identity. MIT.
- [Kiri](https://github.com/T-80BVVD/kiri) — bounded tool-calling и обычные
  message semantics. AGPL-3.0.
- [alive](https://github.com/marchantdev/alive) — минимальный прозрачный
  wake-loop. MIT.
- [Letta Code](https://github.com/letta-ai/letta-code) — persistent memory и
  cognitive state. Apache-2.0.
- [Alkaline](https://github.com/davccavalcante/alkaline) — durable execution:
  replay, retries, recovery. Apache-2.0.
- [AgentOS](https://github.com/Roxmix/agentos-mcp) — memory/goal/reflection как
  отдельные данные и event patterns. MIT.

Статьи arXiv, на которые опирается дизайн (см. «Что это»):

- reward hacking и безопасность self-improvement:
  [STOP 2310.02304](https://arxiv.org/abs/2310.02304),
  [RLVR reward hacking 2604.15149](https://arxiv.org/abs/2604.15149),
  [1-bit danger signals 2604.23210](https://arxiv.org/abs/2604.23210);
- open-endedness и quality-diversity:
  [Voyager 2305.16291](https://arxiv.org/abs/2305.16291),
  [POET 1901.01753](https://arxiv.org/abs/1901.01753),
  [OMNI 2306.01711](https://arxiv.org/abs/2306.01711);
- границы единой цели self-improvement:
  [Darwin Gödel Machine 2505.22954](https://arxiv.org/abs/2505.22954).
