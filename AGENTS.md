# Project SkyNet — runbook для coding-агента

Этот файл читает opencode как инструкции к проекту. Он описывает **только то, что
проверяемо по коду**. Если файл и код расходятся — прав код, а расхождение в этом
файле является багом. Ссылки даны как `файл:символ` (а не `файл:номер строки`):
номера гниют на каждом коммите, имена — нет.

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
- **Единственный kind критерия — `verified_progress`** (считается в
  `store.evaluate_criteria`; навешивается на прогон и COMPLETED без пройденных
  критериев даунгрейдится в BLOCKED в `reactor.Reactor.tick` через
  `store.update_run_result_status`). Отчёт модели без успешного tool result не
  считается выполнением.
- **Память — управляемый ресурс.** `pinned` инжектится всегда; автопоиск
  ограничен `SKYNET_MEMORY_INJECT_LIMIT` (default 3, читается в `cli`); глубже —
  по запросу tool'ом `memory`; неверная память superseded с evidence
  (`store.supersede_memory`).
- **Daily maintenance не гейтится метриками**: retention и decay идут раз в
  UTC-день по sidecar-маркеру (`reactor._run_per_cycle_housekeeping`;
  `store.prune_event_log`/`store.prune_run_history` — retention,
  `store.decay_memory_confidence` — decay), не под `SKYNET_METRICS_SNAPSHOT`.
  Сбой любого maintenance-прохода durable, а не строка в journald:
  `reactor._record_maintenance_failure` пишет `maintenance_failed`, kind входит
  в `store.PROTECTED_EVENT_KINDS`.
- **Никаких дублирующих контуров**: один владелец lifecycle, один planner
  выбора работы, один stale-run путь, один criteria-kind. Мета-контур
  (дескрипторы, политика выбора работы, метрики) больше не вне досягаемости
  self-improvement: **design decision 2026-09-21 — мета-контур открыт**, первая
  отправка такого предложения даёт warning, идентичная повторная проходит
  полный gate.
- **Судья открыт, но предупреждается.** `self_improvement.py`, `planner.py`,
  `autonomous_planner.py`, `idea_archive.py`, `metrics.py` входят в **warn-set**
  (`self_improvement.GATE_PROTECTED_PATHS`): первая отправка правки в них не
  применяется и возвращает модели warning, идентичный (тот же
  `change_fingerprint`) повтор проходит полный gate. Ни одна из этих правок не
  удерживается и не отклоняется. Инструмент, которым планировщик разговаривает с
  моделью, живёт в `skynet/planner_contract.py` — системный промпт
  (`planner_contract.planner_system_prompt`, `PLANNER_SYSTEM_PROMPT`),
  payload-инструкция (`PLANNER_INSTRUCTION`), repair-инструкция
  (`PLANNER_RETRY_INSTRUCTION`) и декодер ответа
  (`planner_contract.parse_planner_reply`). Граница принципиальна:
  `parse_planner_reply` только превращает текст в JSON-значение и нормализует
  top-level список в `{"proposals": [...]}`; что считается валидным
  предложением и что становится работой, решает warn-set
  `autonomous_planner` (`validate_shape`, `_validate`,
  `_fingerprints_occupied`, `learnability_defect`,
  `record_planner_proposal`, `add_task`, `_apply_goal_proposals`).
- **Деградация судьи не может быть молчаливой.** `judge_health.check_judge_health`
  (незащищённый `skynet/judge_health.py`, чистый читатель без внутренних
  импортов; вызывается Reactor'ом раз в цикл через `reactor._check_judge_health`)
  читает durable `planner_attempts` за последние `SKYNET_JUDGE_HEALTH_WINDOW`
  (default 12) завершённых попыток и при доле успеха ниже
  `SKYNET_JUDGE_HEALTH_THRESHOLD` (default 0.5) на выборке не меньше
  `SKYNET_JUDGE_HEALTH_MIN_SAMPLES` (default 5) эскалирует через
  существующий `store.raise_alert` (durable dedup-cooldown
  `SKYNET_JUDGE_HEALTH_COOLDOWN_SECONDS`, default 21600) в owner-outbox.
  Дефолты живут константами в `judge_health` (`DEFAULT_WINDOW`,
  `DEFAULT_MIN_SAMPLES`, `DEFAULT_THRESHOLD`, `DEFAULT_COOLDOWN_SECONDS`), env —
  читается в `reactor._check_judge_health`. Read-only по состояния судьи, работу
  не выбирает, lifecycle не владеет и daily maintenance не гейтит —
  retention/decay идут независимо.
- **Housekeeping и stall-сторож достижимы и на пути «нет работы».**
  `reactor._run_per_cycle_housekeeping` (метрики под флагом, daily maintenance,
  судья) вызывается и в ветке `selected_work is None`, до
  `reactor._sleep_without_work`; раньше эти три стояли после раннего `return`, и
  столл 2026-09-23 на 7 часов не оставил ни одного judge-события.
  `reactor._check_progress_stall` поднимает `raise_alert("no_progress_stall")`,
  если с последнего завершённого прогона (или с restart-boundary) прошло больше
  `SKYNET_STALL_ALERT_SECONDS` — это единственный наблюдатель класса «процесс
  жив, но прогоны не стартуют», который `StaleRunWatchdog` структурно не видит
  (он смотрит только активный прогон).
- **Схема версионируется и мигрируется.** `store.SCHEMA_VERSION` (сейчас 12) и
  `store.MIGRATION_NAMES`; миграция 12 — `run-progress` (`runs.heartbeat_at` /
  `runs.last_progress_at`). Бамп версии без имени миграции — баг (см. инцидент
  2026-09-19 ниже).
- `SOUL.md` — единственный авторитетный документ про identity; не переписывать
  его вслепую.

## Открытость

- Архив идей хранит behavioural descriptor: `subsystem × change_type ×
  evidence_source` = 144 ячейки, в каждой остаётся только лучшая идея
  (`skynet/idea_archive.py`: `cell_key`, `classify`; MAP-Elites-правило и
  материализация — `store.record_idea_outcome`/`store.materialize_idea`).
- `research`-предложение без цитаты отклоняется (learnability gate,
  `idea_archive.learnability_defect`).
- **Архив кормится реальным трафиком, а не только идеями планировщика.**
  `reactor._archive_run_self_improvements` сканирует `tool_result` прогона и
  архивирует каждое успешное `propose_self_improvement`, с качеством = измеренный
  value прогона. До этого архивировал только `_run_autonomous_planning`, поэтому
  promoted-правки организма не дали ни одной строки.
- **Клапан `external_seek` — каденция, а не исчерпание.**
  `reactor._seek_external_evidence` вызывается при
  `generation % SKYNET_EXTERNAL_SEEK_EVERY == 0` (+ durable cooldown), если есть
  живой орган чувств, независимо от того, пуст ли портфель. Раньше он стоял
  после `_create_planner_fallback`, который всегда что-то создавал, поэтому за
  многие поколения клапан не открылся ни разу.
- **Safety net не выбирается ранжированием.** `reactor._create_planner_fallback`
  и `reactor._seek_external_evidence` создают задачи со **стабильными**
  fingerprints (`reactor._FALLBACK_FINGERPRINTS`,
  `reactor._EXTERNAL_SEEK_FINGERPRINTS`), а не с fingerprint, меняющимся от
  generation. После завершения такой задачи её fingerprint читается
  `PortfolioPlanner.rank()` как `terminal` на время TTL гипотезы, поэтому
  выбираются они **напрямую**, а не через ранжирование портфеля. Это намеренно:
  обе задачи — аварийная сетка (safety net), и связку не следует принимать за
  баг.
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
- **Сообщение владельца — уведомление, а не приказ.** `reactor._inbox_notifications`
  отдаёт каждое pending-сообщение как observation `inbox_notification` с
  `event_id`; задача **не** создаётся, обычную работу оно не вытесняет. Решение
  принимает организм: действовать своими tools, создать задачу или закрыть
  уведомление через `acknowledge_inbox(event_id, decision)` (`dialogue.py`).
  Пока не закрыто — приходит снова. Пустой планировщик заполняется
  **фиксированным пулом**: `reactor._ensure_roadmap_seed` при полностью пустой
  таблице целей re-seed'ит `ROADMAP_TASKS` (`handoff.py`), затем идёт bounded
  bootstrap-задача (`reactor._create_planner_fallback`). Блокированная (не
  пустая) цель сид не трогает — иначе маскируется блок.
- **Scratch-каталог согласован.** `SKYNET_SCRATCH_DIR` (default
  `/tmp/skynet-scratch`) входит в `policy.ExecutionPolicy.allowed_roots()`,
  поэтому `read`, `grep` и `bash` видят одну границу.

## Память и аудит

- **Memory audit — append-only, metadata-only.** `skynet/memory_audit.py` пишет
  JSONL-цепочку: каждая запись несёт `previous` = SHA-256 предыдущей строки, а по
  чекпойнтам фиксируются `rows`/`active_rows`/`attributed_rows`, байтовый размер
  и цепочный SHA-256 (контент хешируется, но не хранится). `verify_chain`
  называет строку, где цепочка порвалась. Чекпойнт вызывается из Reactor на
  конце эпизода (`reactor` → `memory_audit.checkpoint_memory_audit`).
- **Якорь против split view.** `memory-audit-anchors.jsonl` — обязательство,
  которое сам писатель не переписывает: одна строка на чекпойнт с sequence и
  хешем строки лога. Это повышает стоимость согласованной перезаписи лога и
  чекпойнта, но честно названо не криптозащитой от хоста; root of trust —
  обязательство оператора не менять лог.
- **Ничего не решается и не чинится автоматически**: audit лишь делает изменение
  видимым (факт + провенанс `source_run`, включая `unattributed_rows`).
  Тесты — `tests/test_memory_audit.py`.

## Тесты

- Полный прогон: `./scripts/test.sh` — **ruff + pyright + pytest**; тот же набор
  выполняет self-improvement gate. Или `.venv/bin/python -m pytest -q`.
- **ruff — блокирующий этап гейта** (`self_improvement._run_ruff_stage`) и шаг
  `test.sh`. Набор правил — в `[tool.ruff.lint]` `pyproject.toml`
  (`E,W,F,I,B,C4,UP,SIM,RUF,PERF,PIE,RET,S,TRY` + курируемые `ignore`); дерево
  ruff-clean, baseline'а нет. Этап пропускается, только если ruff не установлен
  или в worktree нет `pyproject.toml` (минимальная тестовая фикстура).
- **pyright — блокирующий этап гейта** (`self_improvement._run_pyright_stage`) и
  шаг `test.sh`. Дерево pyright-clean (0 ошибок в `skynet/` и `tests/`),
  baseline'а нет: любой диагностик — регрессия. Версия запинена
  (`pyright==1.1.413`) в `[project.optional-dependencies] dev`; в proposal
  worktree без `.venv` запускается с `--pythonpath <root>/.venv/bin/python`.
- **Gate-набор не наследует provider-флаги сервиса.** `self_improvement.GATE_SUITE_ENV_EXCLUSIONS`
  снимает `OLLAMA_ENABLED`/`NVIDIA_ENABLED`/`DEEPSEEK_ENABLED`/`OPENROUTER_ENABLED`
  и money-boost, чтобы вердикт выносился по коду, а не по окружению.
- **mypy снесён** (2026-09-19): pyright — единственный type-авторитет, второй
  чекер означал второй baseline. `pyflakes` тоже убран — его покрывает `F`.
- **Структурный контракт — единственное жёсткое правило о «беспорядке».**
  `tests/test_dependency_contracts.py` (AST, без зависимостей) фиксирует слои:
  `providers` не импортируют `store`/`reactor`/`cli`/`memory`; `memory` не
  импортирует `store` (поэтому и не может писать durable-события); `store` не
  импортирует `reactor`; `judge_health` — чистый читатель без внутренних
  импортов. Это про корректность, поэтому падает билд.
- Размер набора на момент ревизии: **около 900 тестов**. Число живое — проверяй,
  а не верь этому файлу. Отдельные модули для новых подсистем:
  `tests/test_tor_control.py`, `tests/test_memory_audit.py`,
  `tests/test_change_digest.py`, `tests/test_recall_ablation.py`,
  `tests/test_recall_scorecard_verdict.py`, `tests/test_retention_proof.py`.
- Python-работать только через `.venv/bin/python`; системный интерпретатор не
  имеет pytest.

## Provider chain

- **Активный провайдер выбирается env, не кодом.** `providers.active_chain_names()`
  читает цепочку заново, без рестарта. `providers.DEFAULT_CHAIN` перечисляет все
  реализации; лишние **не удалены, а выключены** флагами (`providers._ENABLE_FLAGS`;
  канонический флаг — первый в кортеже, легаси-алиасы приняты), так что вернуть
  их можно флипом env без правки кода.
- **Платные реализации гейтятся money-boost.** `providers.active_chain_names`,
  `providers.PAID_PROVIDER_NAMES`; env `SKYNET_MONEY_BOOST` или sidecar
  `state/money-boost.json`; имя читается заново, без рестарта процесса.
- **Пустой ответ при `finish_reason='length'` — ошибка провайдера, не ответ.**
  `openai_compatible._parse` и openrouter поднимают retryable
  `ProviderError(response_quality)`, если видимого текста нет: иначе обрезанный
  на потолке ответ считался успехом и лестница не делала ни повтора, ни перехода.
  `length` с непустым text — валидный длинный ответ.
- **Потолок вывода — наш, а не сервера.** `NEMOTRON_MAX_OUTPUT_TOKENS` (default
  8192) клампится в `openai_compatible.py`; сервер может принимать больше (до
  16384), поэтому потолок задаётся осознанно рядом с
  `SKYNET_PLANNER_OUTPUT_TOKENS`.
- **HTTP 403 через Tor = грязный выход, не отказ ключа.** Провайдер с заданным
  `proxy_url` при 403 один раз просит у Tor новую цепь
  (`providers/tor_control.py`, control-сокет + cookie, `SIGNAL NEWNYM`,
  `tor_control.newnym()`) и повторяет запрос; повторный 403 снова фатален.
  Требует прав на контрольный сокет (`/run/tor/control`; root или группа
  `debian-tor`), иначе тихо пропускается; отключается
  `SKYNET_TOR_CONTROL_ENABLED=false`.
- **Таймаут openrouter — connect + idle, не общий дедлайн.** `OPENROUTER_TIMEOUT_SECONDS`
  (30) — connect; `OPENROUTER_CHUNK_TIMEOUT_SECONDS` (900) — тишина между чанками,
  во время активного стрима не горит; `OPENROUTER_STREAM_DEADLINE_SECONDS` (0 = off) —
  опциональный общий потолок. Rung лестницы из `FallbackProvider` трактуется как
  idle-бюджет, а не как дедлайн потока.
- **Устойчивость цепочки**: `SKYNET_PROVIDER_TIMEOUT_LADDER` (900),
  `SKYNET_PROVIDER_STRIKE_DELAY`, `SKYNET_PROVIDER_LADDER_DEADLINE`,
  `SKYNET_PROVIDER_LOCKOUT_THRESHOLD`/`_SECONDS`; успех после раннего провала
  фиксируется durable-событием `fallback_retry_recovered`
  (`providers/fallback.py`).
- Единственный владелец ретраев — `FallbackProvider`; не дублировать ретраи в
  других слоях.
- **Credentials — только в `config/skynet.env`**, файл gitignored, mode 600.
  Никогда не коммитить секреты и не держать бэкапы env внутри дерева
  репозитория.

## Бюджет и контекст

- `SKYNET_MAX_SECONDS=7200` — один прогон = два часа модельного времени; watchdog
  `SKYNET_WATCHDOG_SECONDS=7500` должен его превышать (systemd stop timeout —
  1200s, `deploy/skynet.service.in`).
- `SKYNET_MAX_STEPS=800` (сообщения/шаги ReAct), `SKYNET_REACT_INPUT_TOKENS=500000`
  (окно ReAct), `SKYNET_OUTPUT_TOKENS=16384`, `SKYNET_MEMORY_INPUT_TOKENS=100000`.
- `SKYNET_PLANNER_OUTPUT_TOKENS=16384` — отдельный output-бюджет планировщика
  (читается в `autonomous_planner`); раньше его резал `min(4096, ...)`.
  `SKYNET_MEMORY_LOOP_TIMEOUT=300` — таймаут memory loop'а (в `cli`).
- `SKYNET_CONTEXT_FINISH_RESERVE=50000` и `SKYNET_TOOL_RESULT_MAX_CHARS=16000`
  читаются в `reactor` при сборке `ReActConfig`.
- `SKYNET_RUN_PROGRESS_SECONDS=3600` (читается в `cli`) — одна короткая заметка
  владельцу за этот интервал модельного времени, чтобы двухчасовой прогон не
  молчал до Finish; 0 выключает.
- Тайминги устойчивости на пути «нет работы»: `SKYNET_STALL_ALERT_SECONDS=3600`
  (порог `no_progress_stall`, 0 выключает), `SKYNET_EXTERNAL_SEEK_MIN_SECONDS=3600`
  (wall-clock клапан внешнего поиска сверх generation-каденции),
  `SKYNET_FALLBACK_REPEAT_SECONDS=900` (wall-clock потолок повтора fallback).
- **Retention-защищённый след столла.** `provider_output_truncated`,
  `planner_backoff`, `planner_fallback_created` входят в
  `store.PROTECTED_EVENT_KINDS`, а `provider_output_truncated` — ещё и в
  `store.DURABLE_PROVIDER_EVENT_KINDS`, чтобы посмертие потолка вывода переживало
  ротацию `runtime.jsonl`. Ежедневный снапшот несёт `work_mix` (микс принятых
  предложений по kind, `metrics.work_mix`) — измеритель «метаболизм vs рост».
- Деградация memory loop'а durable: `memory_degraded` (retention-protected,
  deviation-kind) плюс `memory_loop_finished` с `degraded=true`
  (`memory.py`, `reactor.py`). Провайдерные сбои идут в durable
  `provider_error_classified`/`provider_failure`, а классификация остановки —
  в `provider_failures` (`metrics.py`).

## Deploy и restart

- Деплой и рестарт сервисов выполняются **только по явной команде оператора**.
  Агент не запускает их сам и не коммитит.
- `scripts/deploy.sh` — деплой; `scripts/rollback.sh <commit|tag>` — откат.
- Адрес/доступы deployment-хоста живут в `config/deploy.env` (gitignored), не в
  скриптах и не в этом файле.
- **Пакет ставится editable с dev-extra** (`pip install -e '.[dev]'`,
  `scripts/deploy.sh`), и это не косметика: self-improvement правит рабочее
  дерево, а editable гарантирует, что живой процесс импортирует актуальный код,
  а не замороженную копию в `site-packages` (обычная установка молча ломала бы
  promote). Юниты запускают `python -m skynet` из корня
  (`deploy/skynet.service.in`, `ExecStart`), так что импорт работал бы и без
  установки; editable дополнительно даёт console-скрипты `skynet`/`skynet-telegram`
  (`pyproject.toml:[project.scripts]`) и установку зависимостей. `[dev]`
  обязателен, потому что блокирующий этап гейта — `pyright`. Побочный продукт
  сборки — `skynet.egg-info/`, он в `.gitignore`.

## Логи

- Один вход: `skynet logs --tier system|mandatory|advanced|verbose` (default
  `advanced`, `cli`).
- **system** — `state/system-probe.jsonl`, кольцо 250 записей (`SystemProbeLog`);
  cadence — `SKYNET_SYSTEM_PROBE_EVERY` (default 20, `heartbeat.py`).
- **mandatory** — пары `SCHEDULER`+`REPORT` по завершённым run'ам из event log
  (`skynet/reporting.py`); 5 по умолчанию, максимум 100 (`cli._logs_limit`).
- **advanced** — `state/runtime.jsonl`, 25 MB × 3 бэкапа (`runtime_log.RuntimeLog`);
  по умолчанию отдаёт последние 500 записей.
- **verbose** — полные дампы запросов/ответов провайдера в
  `state/verbose.jsonl` (кольцо 50 МБ, mode 600, секреты редактируются), по
  умолчанию выключен; `skynet verbose on|off|status` пишет `state/verbose.json`
  (hot-reload без рестарта, `cli._verbose_command`), плюс env-переключатель
  `SKYNET_VERBOSE_PROVIDER` (`runtime_log.py`).
- Для system-яруса обязателен `psutil` (в `dependencies` `pyproject.toml`);
  probe вызывает `sudo -n journalctl`, поэтому на хосте `journalctl` должен быть
  доступен сервисному аккаунту без пароля (`system_probe.py`), иначе слой честно
  деградирует.

## Tools

- Встроенный набор `default_tools()` — **`bash`, `webfetch`, `read`, `grep`,
  `db`, `structure`** (`skynet/tools.py`).
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
  `memory_tool.MemoryTool.ACTIONS`); при наличии `<root>/.git` —
  `propose_self_improvement`, `request_rollback`.
- Telegram владельца — обычный чат: текст без `/` идёт как observation
  (`telegram_bot`), команды начинаются с `/`; блокирует только
  `ask_user(wait_seconds>0)` (`dialogue.AskUserTool`).
- Self-improvement gate — **warn once, allow on identical resubmission**.
  (design decision 2026-09-21, отменяет прежний `/approve`-механизм и инвариант
  «судимый не редактирует судью».) Правки `tests/`, `conftest.py`,
  `pyproject.toml`, `pytest.ini`, `tox.ini`, `setup.cfg` и warn-set
  (`self_improvement.GATE_PROTECTED_PATHS`; см. «Инварианты») в main worktree не
  применяются, а записываются как `warned_protected`
  (`self_improvement._warn_protected_proposal`) и возвращают модели
  предупреждение (`ok:false, warned:true, awaiting_resubmission:true`).
  Организм должен перепредложить *идентичный* дифф: fingerprint включает
  files/changes/hypothesis, поэтому переформулировка warning не подхватит.
  При совпадении `change_fingerprint` warning подтверждается
  (`self_improvement._resume_warned`, kind `gate_protected_warning_acknowledged`)
  и предложение проходит полный gate (ruff/pyright/pytest) из своего worktree.
  Никакого owner-одобрения, TTL-сайдкара и `/approve`/`/reject` больше нет.
  Жёсткими остаются только секреты, resurrection-guard, `stale_base` и
  ignored-paths; нечитаемые changed paths — отказ. Контентный отпечаток
  правки несёт `change_digest` (`self_improvement.change_content_digest`,
  `tests/test_change_digest.py`).
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
  (`cli._mcp_config`). В `config/skynet.env` прописаны четыре:
  DuckDuckGo, Playwright, arXiv, GitHub.
- GitHub запускается `--read-only --toolsets=repos,git,issues`: `--read-only`
  выкидывает все write-инструменты из списка **флагом самого сервера**, а не
  нашим блок-листом. Резать тоньше можно тем же сервером — `--tools` (белый
  список) или `--exclude-tools` (чёрный), но это микрооптимизация.
- arXiv внутренней фильтрации инструментов **не имеет** (ни 0.6.3, ни 0.7.2:
  список из 14 инструментов зашит в `server.py`, из CLI только `--storage-path`;
  README прямо относит гейтинг к клиенту). Поэтому arxiv оставлен целиком:
  «нельзя, значит не трогаем».
- Практическое наблюдение из эксплуатации: чаще всего вызываются **arxiv**,
  **ddg**, **github** и `webfetch`, тогда как **playwright** может не вызываться
  вовсе. Ноль вызовов — это «не было задач», а не «мусор»: не судить о пользе
  инструмента по факту его подключения.
- Системный промпт перечисляет органы чувств **динамически**: `Reactor.__init__`
  дописывает блок `SENSES` из реально обнаруженных MCP-инструментов и блок
  `WORKSPACE` с реальным корнем (`reactor.mcp_senses_line`). Явное
  перечисление сохранено намеренно — без него модель игнорирует инструменты, —
  но генерируется из живого инвентаря, а не второй зашитой копией, которая
  разъезжается при смене toolsets.

## Исторические наблюдения

Раздел сохранён как история; конкретные счётчики здесь — снимок на дату и могут
быть неверны. Актуальный хеш смотри `git log -1`, числа — в БД и `pytest`.

- **Организм самоулучшается автономно.** За сутки он делал несколько
  organism-коммитов подряд — каждый через `propose_self_improvement` → gate →
  promote; ни один не тронул gate-protected пути.
- **Gate реально кусается**: предложения отклонялись за `protected_path`
  (включая `tests/test_providers.py`) и за regression; отклонённые затем
  переподавались без правки тестов и проходили.
- **Первые правки — багфиксы harness, и это by design**: seed (`handoff.py`,
  `CANONICAL_MEMORIES` + `ROADMAP_TASKS`, «scaffolding, not an order») прямо
  называет трение — «query it read-only with the db tool instead of guessing
  table names», «write prototypes to /tmp/skynet-scratch». Организм чинит то,
  что назвал сид; не судить об «оригинальности» по первым часам.
- **Схема разъезжалась (2026-09-19)**: коммит добавил `memories.decayed_at` без
  бампа `SCHEMA_VERSION` → на живом хосте оказалось два разных v10. Исправлено
  (миграция `memory-decay-clock`). Урок закреплён инвариантом выше: бамп
  `SCHEMA_VERSION` без имени миграции — баг.
- **Inbox: сообщение владельца — уведомление, а не задача.** Автосоздание
  inbox-задачи (`_select_inbox_work`) убрано; сообщение приходит как
  observation `inbox_notification`, организм сам решает и закрывает его через
  `acknowledge_inbox`.
- **Отказы прогонов — это провайдерная цепочка, а не бюджет времени.**
  Разбор отказов `needs_recovery`: большинство — исчерпание провайдеров в хвосте
  цепочки, меньшинство — time/output budget. Исправлено: таймаут openrouter стал
  connect+idle (общий дедлайн off), `ModelTurn.model_seconds` исключает бэкофф,
  `provider_failures` даёт вердикт `degraded`.

## Инцидент планировщика (2026-09-20)

- **Судья сломался молча.** Декодер `autonomous_planner._parse` срезал текст
  модели от первой `{` до последней `}` и уничтожал легитимный top-level
  JSON-массив раньше, чем его видел парсер: большинство попыток планировщика
  провалились с `structured model response must be a JSON object`.
- **Провал маскировался под «нет новой работы».** `generate()` возвращал `[]`,
  что неотличимо от «портфель пуст», поэтому Reactor молча падал в
  фиксированный пул: generic fallback-задачи создавались в большинстве прогонов.
- **Организм не мог починить это сам**: `_parse` жил в gate-protected
  `autonomous_planner.py`. Отсюда разделение — инструмент
  (`skynet/planner_contract.py`) открыт, решения судьи закрыты, а
  `judge_health.check_judge_health` делает деградацию видимой владельцу.
- **Де-маскировка.** Провал парсинга планировщика никогда не записывается как
  «нет новой работы»: это отдельные причины `planner_invalid_response` и
  `planner_provider_error` (эмитятся `autonomous_planner` / `reactor`), и оба
  kind'а защищены от retention как post-mortem.

## Документы

- `SOUL.md` — identity, инжектится в system-промпт ReAct и Memory. Авторитетен.
- `AGENTS.md` (этот файл) — runbook для coding-агента.
- **Источник истины — код.** Никакой документ не нормативен, кроме `SOUL.md`.