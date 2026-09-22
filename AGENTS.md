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
систему — reward hacking (`2310.02304`, `2604.15149`, `2604.23210`), а DGM сам
называет ограничением «self-improvement being tied to a single objective»
(`2505.22954`). Вместо бенчмарка harness требует **проверяемости**: успешный
tool result (`verified_progress`), прохождение gate-набора и цитату для
`research`. Это middle-ground из литературы (Voyager `2305.16291`, POET
`1901.01753`, OMNI `2306.01711`) — не «любые критерии» и не «наш бенчмарк».
Отсутствие бенчмарка — гипотеза эксперимента, а не долг; не «чинить» его.

## Инварианты, которые нельзя ломать

- **Reactor — единственный владелец lifecycle.** Только он владеет
  wake/sleep/scheduling, checkpointing и восстановлением. Memory Loop, MCP,
  reflection и фоновые сервисы не запускают конкурирующих daemon-loop'ов.
- **Костяк цикла**:
  `SYSTEM -> Start -> ReAct -> Finish -> Memory Loop -> evaluation -> checkpoint -> schedule`.
- **Transcript ограничен на каждый outer cycle**, не накапливается бесконечно;
  история prun'ится после checkpoint'а (`store.prune_run_history`).
- **Единственный kind критерия — `verified_progress`** (kind —
  `store.py:1711-1724`; навешивается на прогон `reactor.py:1396`; COMPLETED без
  пройденных критериев даунгрейдится в BLOCKED, `reactor.py:325-331`). Отчёт
  модели без успешного tool result не считается выполнением.
- **Память — управляемый ресурс.** `pinned` инжектится всегда; автопоиск
  ограничен `SKYNET_MEMORY_INJECT_LIMIT` (default 3, `reactor.py:83`,
  `cli.py:606`); глубже — по запросу tool'ом `memory`; неверная память
  superseded с evidence (`store.py:1606-1642`).
- **Daily maintenance не гейтится метриками**: retention и decay идут раз в
  UTC-день по sidecar-маркеру (`reactor.py:1033`; `store.py:933` retention,
  `store.py:1888` decay), не под `SKYNET_METRICS_SNAPSHOT`.
- **Никаких дублирующих контуров**: один владелец lifecycle, один planner
  выбора работы, один stale-run путь, один criteria-kind. Мета-контур
  (дескрипторы, политика выбора работы, метрики) больше не вне досягаемости
  self-improvement: **owner decision — мета-контур открыт**, первая
  отправка такого предложения даёт warning, идентичная повторная проходит
  полный gate.
- **Судья открыт, но предупреждается.** `self_improvement.py`, `planner.py`,
  `autonomous_planner.py`, `idea_archive.py`, `metrics.py` входят в **warn-set**
  (`GATE_PROTECTED_PATHS`): первая отправка правки в них не применяется и
  возвращает модели warning, идентичный (тот же `change_fingerprint`)
  повтор проходит полный gate. Ни одна из этих правок не удерживается и не
  отклоняется. Инструмент, которым планировщик разговаривает с моделью, живёт
  в `skynet/planner_contract.py` — системный промпт
  (`planner_contract.planner_system_prompt`), payload-инструкция
  (`PLANNER_INSTRUCTION`), repair-инструкция (`PLANNER_RETRY_INSTRUCTION`) и
  декодер ответа (`planner_contract.parse_planner_reply`). Граница принципиальна:
  `parse_planner_reply` только превращает текст в JSON-значение и нормализует
  top-level список в `{"proposals": [...]}`; что считается валидным
  предложением и что становится работой, решает warn-set
  `autonomous_planner` (`validate_shape`, `_validate`,
  `_fingerprints_occupied`, `learnability_defect`,
  `record_planner_proposal`, `add_task`, `_apply_goal_proposals`).
- **Деградация судьи не может быть молчаливой.** `judge_health.check_judge_health`
  (незащищённый `skynet/judge_health.py`, вызывается Reactor'ом раз в цикл через
  `reactor._check_judge_health`) читает durable `planner_attempts` за последние
  `SKYNET_JUDGE_HEALTH_WINDOW` (default 12) завершённых попыток и при доле
  успеха ниже `SKYNET_JUDGE_HEALTH_THRESHOLD` (default 0.5) на выборке не
  меньше `SKYNET_JUDGE_HEALTH_MIN_SAMPLES` (default 5) эскалирует через
  существующий `store.raise_alert` (durable dedup-cooldown
  `SKYNET_JUDGE_HEALTH_COOLDOWN_SECONDS`, default 21600) в owner-outbox.
  Read-only по состояния судьи, работу не выбирает, lifecycle не владеет и
  daily maintenance не гейтит — retention/decay идут независимо.
- `SOUL.md` — единственный авторитетный документ про identity; не переписывать
  его вслепую.

## Открытость

- Архив идей хранит behavioural descriptor: `subsystem × change_type ×
  evidence_source` = 144 ячейки, в каждой остаётся только лучшая идея
  (`skynet/idea_archive.py`; MAP-Elites-правило — `store.py:1235`,
  сэмплирование родителей — `store.py:1293`, материализация — `store.py:1343`).
- `research`-предложение без цитаты отклоняется (learnability gate,
  `idea_archive.py:83-104`).
- **Архив кормится реальным трафиком, а не только идеями планировщика.**
  `_archive_run_self_improvements` (`reactor.py`) сканирует `tool_result`
  прогона и архивирует каждое успешное `propose_self_improvement`, с качеством =
  измеренный value прогона. До этого архивировал только
  `_run_autonomous_planning`, поэтому 9 promoted-правок организма не дали ни
  одной строки.
- **Клапан `external_seek` — каденция, а не исчерпание.**
  `_seek_external_evidence` вызывается при
  `generation % SKYNET_EXTERNAL_SEEK_EVERY == 0` (+ durable cooldown), если есть
  живой орган чувств, независимо от того, пуст ли портфель (`reactor.py`).
  Раньше он стоял после `_create_planner_fallback`, который всегда что-то
  создавал, поэтому за 181 поколение клапан не открылся ни разу.
- **Safety net не выбирается ранжированием.** `_create_planner_fallback` и
  `_seek_external_evidence` создают задачи со **стабильными** fingerprints
  (`reactor._FALLBACK_FINGERPRINTS`, `reactor._EXTERNAL_SEEK_FINGERPRINTS`), а не
  с fingerprint, меняющимся от generation. После завершения такой задачи её
  fingerprint читается `PortfolioPlanner.rank()` как `terminal` на время TTL
  гипотезы, поэтому выбираются они **напрямую**, а не через ранжирование
  портфеля. Это намеренно: обе задачи — аварийная сетка (safety net), и связку не
  следует принимать за баг.
- **Повтор bootstrap-diagnostic ограничен окном поколений.**
  `reactor._fallback_repeat_blocked` читает generation последнего
  `planner_fallback_created` из durable event log, поэтому кап
  (`SKYNET_FALLBACK_REPEAT_GENERATIONS`, default 4) переживает рестарт. Если
  событие вычищено retention'ом, проверка fail-open: safety net снова разрешён.
- **Retention ledger'а эффектов pointer-aware.** `store.prune_capability_effects`
  (`SKYNET_EFFECT_RETENTION_DAYS`, default 30) не удаляет строку, на которую
  ссылается retained `tool_result` event **или** retained `react_history`
  transcript row; pin снимается, когда `store.prune_run_history` уносит
  транскрипт.
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
- **mypy снесён**: pyright — единственный type-авторитет, второй
  чекер означал второй baseline. `pyflakes` тоже убран — его покрывает `F`.
- **Структурный контракт — единственное жёсткое правило о «беспорядке».**
  `tests/test_dependency_contracts.py` (AST, без зависимостей) фиксирует слои:
  `providers` не импортируют `store`/`reactor`/`cli`/`memory`; `memory` не
  импортирует `store` (поэтому и не может писать durable-события); `store` не
  импортирует `reactor`; `judge_health` — чистый читатель без внутренних
  импортов. Это про корректность, поэтому падает билд.
- Размер набора на момент написания: **784 теста**. Число живое — проверяй,
  а не верь этому файлу.
- Python-работать только через `.venv/bin/python`; системный интерпретатор не
  имеет pytest.

## Provider chain

- **Paid-режим: активен один платный провайдер — openrouter.** Остальные
  реализации (`ollama`, `nvidia*`, `nemotron`, `openai`) **архивированы, не
  удалены**: выключены через env (`*_ENABLED=false`,
  `SKYNET_PROVIDER_CHAIN=openrouter`) и помечены комментарием в
  `providers/__init__.py`. Вернуть — флип флагов без правки кода.
- OpenRouter попадает в активную цепочку только при включённом money-boost
  (`active_chain_names`, `providers/__init__.py`; env `SKYNET_MONEY_BOOST` или
  `state/money-boost.json`); имя читается заново, без рестарта процесса.
- **Таймаут openrouter — connect + idle, не общий дедлайн.**
  `OPENROUTER_TIMEOUT_SECONDS` (30) — connect; `OPENROUTER_CHUNK_TIMEOUT_SECONDS`
  (900) — тишина между чанками, во время активного стрима не горит;
  `OPENROUTER_STREAM_DEADLINE_SECONDS` (0 = off) — опциональный общий потолок.
  Rung лестницы из `FallbackProvider` трактуется как idle-бюджет, а не как
  дедлайн потока (`openrouter.py`).
- Единственный владелец ретраев — `FallbackProvider`; не дублировать ретраи в
  других слоях.
- **Credentials — только в `config/skynet.env`**, файл gitignored
  (`.gitignore:10-11`, mode 600). Никогда не коммитить секреты и не держать
  бэкапы env внутри дерева репозитория.

## Бюджет и контекст (paid-режим)

- `SKYNET_MAX_SECONDS=7200` — один прогон = два часа модельного времени; watchdog
  `SKYNET_WATCHDOG_SECONDS=7500` должен его превышать (systemd stop timeout —
  1200s, `deploy/skynet.service.in`).
- `SKYNET_MAX_STEPS=800` (сообщения/шаги ReAct), `SKYNET_REACT_INPUT_TOKENS=500000`
  (окно ReAct), `SKYNET_OUTPUT_TOKENS=16384`, `SKYNET_MEMORY_INPUT_TOKENS=100000`.
- `SKYNET_PLANNER_OUTPUT_TOKENS=16384` — отдельный output-бюджет планировщика
  (`planner_output_tokens`, читается `cli.py`); раньше его резал `min(4096, ...)`.
  `SKYNET_MEMORY_LOOP_TIMEOUT=300` — таймаут memory loop'а
  (`memory_loop_timeout_seconds`).
- `SKYNET_CONTEXT_FINISH_RESERVE=50000` и `SKYNET_TOOL_RESULT_MAX_CHARS=16000`
  читаются в `reactor.py` при сборке `ReActConfig`.
- Деградация memory loop'а durable: `memory_degraded` (retention-protected,
  deviation-kind) плюс `memory_loop_finished` с `degraded=true`
  (`memory.py`, `reactor.py`). Провайдерные сбои идут в durable
  `provider_error_classified`/`provider_failure`, а классификация остановки —
  в `provider_failures` (`metrics.py`).

## Deploy и restart

- Деплой и рестарт сервисов выполняются **только по явной команде владельца**.
  Агент не запускает их сам и не коммитит.
- `scripts/deploy.sh` — деплой; `scripts/rollback.sh <commit|tag>` — откат.
- Адрес/доступы деплой-хоста живут в `config/deploy.env` (gitignored), не в скриптах
  и не в этом файле.
- **Пакет ставится editable с dev-extra** (`pip install -e '.[dev]'`,
  `deploy.sh:85`), и это не косметика: self-improvement правит рабочее дерево, а
  editable гарантирует, что живой процесс импортирует актуальный код, а не
  замороженную копию в `site-packages` (обычная установка молча ломала бы
  promote). Юниты запускают `python -m skynet` из корня
  (`deploy/skynet.service.in:13,17`), так что импорт работал бы и без установки;
  editable дополнительно даёт console-скрипты `skynet`/`skynet-telegram`
  (`pyproject.toml:12-14`) и установку зависимостей. `[dev]` обязателен, потому
  что блокирующий этап гейта — `pyright`. Побочный продукт сборки —
  `skynet.egg-info/`, он в `.gitignore`.

## Логи

- Один вход: `skynet logs --tier system|mandatory|advanced|verbose` (default
  `advanced`, `cli.py:60`).
- **system** — `state/system-probe.jsonl`, кольцо 250 записей (`SystemProbeLog`);
  cadence — `SKYNET_SYSTEM_PROBE_EVERY` (default 20, `heartbeat.py:23`).
- **mandatory** — пары `SCHEDULER`+`REPORT` по завершённым run'ам из event log
  (`skynet/reporting.py`); 5 по умолчанию, максимум 100 (`cli.py:227`).
- **advanced** — `state/runtime.jsonl`, 25 MB × 3 бэкапа (`runtime_log.py:125`;
  конструируется в `store.py:505`); по умолчанию отдаёт последние 500 записей.
- **verbose** — полные дампы запросов/ответов провайдера в
  `state/verbose.jsonl` (кольцо 50 МБ, mode 600, секреты редактируются), по
  умолчанию выключен; `skynet verbose on|off|status` пишет `state/verbose.json`
  (hot-reload без рестарта, `cli.py:172`), плюс env-переключатель
  `SKYNET_VERBOSE_PROVIDER` (`runtime_log.py:51`).
- Для system-яруса обязателен `psutil` (в `dependencies` `pyproject.toml`);
  probe вызывает `sudo -n journalctl` (`system_probe.py:212`).

## Tools

- Встроенный набор `default_tools()` — **`bash`, `webfetch`, `read`, `grep`,
  `db`, `structure`** (`skynet/tools.py:900`).
- **`structure` — read-only зеркало собственного кода** (`skynet/structure.py`,
  pyscn, MIT): сложность, клоны, мёртвый код, связность, архитектура. Это
  **наблюдаемость, а не критерий промоушена** — «меньше беспорядка» как цель
  заставило бы дробить функции и переименовывать клоны ради цифры. pyscn стоит
  в `[dev]`, запускается в системном temp-каталоге (чтобы не сорить в
  worktree), и при отсутствии бинаря честно возвращает «не установлен», а не
  роняет прогон.
- `Reactor.__init__` всегда добавляет `ask_user`, `send_message_to_user`,
  `acknowledge_inbox` (закрыть pending-уведомление владельца), `memory`
  (действия `search/remember/forget/pin/unpin/correct`,
  `skynet/memory_tool.py:26`) (`reactor.py:127-130`); при наличии
  `<root>/.git` — `propose_self_improvement`, `request_rollback`
  (`reactor.py:120-124`).
- Telegram владельца — обычный чат: текст без `/` идёт как observation, команды
  начинаются с `/` (`telegram_bot.py:491-497`); блокирует только
  `ask_user(wait_seconds>0)` (`dialogue.py:55`).
- Self-improvement gate — **warn once, allow on identical resubmission**.
  (owner decision, отменяет прежний `/approve`-механизм и инвариант
  «судимый не редактирует судью».) Правки `tests/`, `conftest.py`,
  `pyproject.toml`, `pytest.ini`, `tox.ini`, `setup.cfg` и мета-контура
  (`skynet/self_improvement.py`, `idea_archive.py`, `planner.py`,
  `autonomous_planner.py`, `metrics.py`) в main worktree не применяются, а
  записываются как `warned_protected`
  (`self_improvement._warn_protected_proposal`) и возвращают модели
  предупреждение (`ok:false, warned:true, awaiting_resubmission:true`).
  Организм должен перепредложить *идентичный* дифф: fingerprint включает
  files/changes/hypothesis, поэтому переформулировка warning не подхватит.
  При совпадении `change_fingerprint` warning подтверждается
  (`self_improvement._resume_warned`, kind `gate_protected_warning_acknowledged`)
  и предложение проходит полный gate (ruff/pyright/pytest) из своего worktree.
  Никакого owner-одобрения, TTL-сайдкара и `/approve`/`/reject` больше нет.
  Жёсткими остаются только секреты, resurrection-guard, `stale_base` и
  ignored-paths; нечитаемые changed paths — отказ.
- **Warning не может заклинить самоулучшение.** Предупреждённый worktree нужен
  лишь до идентичной переподачи, поэтому при достижении квоты
  (`SKYNET_MAX_WORKTREES`, default 8) `_enforce_worktree_quota` освобождает
  самые старые warning-worktree'ы, а не отказывает в новой работе; TTL
  предупреждения — `SKYNET_WARNED_TTL_HOURS` (default 6), sweep идёт и на
  каждом `propose()`, не только при старте. Освобождённое предупреждение при
  повторной подаче выдаётся заново, а не отклоняется.
- **`uncaptured_changes` — только по проверяемому сигналу.** Отчёт, называющий
  path-like изменение без proposal'а, возвращает задачу в очередь лишь когда
  main worktree реально грязный (`git status`); проза-отрицание и чистый
  worktree задачу не переоткрывают.
- Изменение кода durable **только** через `propose_self_improvement` (он же
  гейтит и промоутит); правка main worktree напрямую не засчитывается.
- MCP-органы чувств подключаются через `SKYNET_MCP_<NAME>_COMMAND`
  (`skynet/cli.py:187`). В `config/skynet.env` прописаны четыре:
  DuckDuckGo, Playwright, arXiv, GitHub.
- GitHub запускается `--read-only --toolsets=repos,git,issues`: `--read-only`
  выкидывает все write-инструменты из списка **флагом самого сервера**, а не
  нашим блок-листом; `pull_requests`/`users`/`orgs` дали ноль вызовов.
  Резать тоньше можно тем же сервером — `--tools` (белый список) или
  `--exclude-tools` (чёрный), но это микрооптимизация.
- arXiv внутренней фильтрации инструментов **не имеет** (ни 0.6.3, ни 0.7.2:
  список из 14 инструментов зашит в `server.py`, из CLI только `--storage-path`;
  README прямо относит гейтинг к клиенту). Поэтому arxiv оставлен целиком:
  «нельзя, значит не трогаем».
- MCP измерен на длинном прогоне: вызывались **arxiv** (106),
  **ddg** (41), **github** (9, все read) и `webfetch`; **playwright** — 0 вызовов
  за всё время. Ноль вызовов — это «не было задач», а не «мусор»: playwright
  оставлен по решению владельца. Не судить о пользе по факту подключения.
- Системный промпт перечисляет органы чувств **динамически**: `Reactor.__init__`
  дописывает блок `SENSES` из реально обнаруженных MCP-инструментов и блок
  `WORKSPACE` с реальным корнем (`skynet/reactor.py`, `mcp_senses_line`). Явное
  перечисление сохранено намеренно — без него модель игнорирует инструменты, —
  но генерируется из живого инвентаря, а не второй зашитой копией, которая
  разъезжается при смене toolsets.

## Документы

- `SOUL.md` — identity, инжектится в system-промпт ReAct и Memory. Авторитетен.
- `AGENTS.md` (этот файл) — runbook для coding-агента.
- **Источник истины — код.** `Architecture.md` уничтожен и не восстанавливается:
  это был черновик плана, а не контракт, и `handoff` на него больше не ссылается.
  Никакой документ не нормативен, кроме `SOUL.md`.
