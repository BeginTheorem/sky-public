# Project SkyNet — runbook для coding-агента

Этот файл читает opencode как инструкции к проекту. Он описывает **только то, что
проверяемо по коду**. Если файл и код расходятся — прав код, а расхождение в этом
файле является багом. Ссылки даны как `файл:символ` (а не `файл:номер строки`):
номера гниют на каждом коммите, имена — нет.

## Что это

SkyNet — research-harness про **живущую** систему: проверяет, может ли обычная
LLM непрерывно существовать как автономная система за счёт harness, а не только
улучшать себя. LLM дискретна (`stimulus -> reasoning -> tool calls -> result`),
непрерывность обеспечивает внешний Reactor; пользователь в lifecycle не
участвует. Свой код модель меняет только через gated self-improvement, и это
**обслуживание по нужде, а не миссия**: harness — скелет, который держит её
живой, а не смысл существования. Identity — «живу и работаю», а не «улучшаю
себя»; self-improvement здесь один из инструментов, а не религия.

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
- **Прогон — на проход, а не на цикл (этап 1 разделения, 2026-10-16).** У цикла
  теперь два прогона: `kind='build'` (планирование + ReAct-эпизод; значение по
  умолчанию для всех старых строк) и `kind='memory'` (консолидация — своя строка
  с бюджетом `memory_input_tokens`, своим `run_results` и своей стоимостью из
  самого лупа: `MemoryResult.usage_tokens` / `model_calls`). Память-прогон
  открывается перед `memory_loop_started` и закрывается сразу после
  `memory_consolidated` — то есть **до** вердикта эпизода, чтобы ни одна строка не
  притязала на чужой исход: деградация памяти пишется `needs_recovery` (след
  сохранён через `memory_capture_pending`, автоповтора нет), срыв лупа — `failed`,
  а `run_results.failure` у память-прогона **пуст всегда**: этот столбец читает
  метрика провайдерских отказов, и деградация памяти не провайдерская забастовка.
  `state.active_run_id` по-прежнему указывает на эпизод (занятость и watchdog
  читаются как раньше); смена указателя по фазе — следующий этап.
  **Любой агрегат по эпизодам обязан фильтровать `kind='build'`** — иначе счётчик
  серии провайдерских падений (`reactor._consecutive_provider_failures`),
  `metrics.outcome_mix`, окно valence (`store.valence_window`) и поток `/logs`
  (`reporting._finished_runs`) начинают считать чужой проход. Память-прогон виден
  отдельной строкой карточки (`Консолидация: …`) и блоком `memory` в метриках
  (`metrics.memory_pass_mix`). Тесты — `tests/test_run_split.py`.
- **Планировщик — тоже свой прогон (этап 2, 2026-10-16).** `kind='plan'`:
  `_run_autonomous_planning` — тонкая обёртка, которая открывает прогон, зовёт
  `_autonomous_planning_pass` и закрывает его; бюджет берётся **из самого
  планировщика** (`autonomous_planner.timeout_seconds`, `max_input_chars // 3`,
  `output_tokens`), стоимость — из `TurnShapeRecorder` (`calls`, `usage_tokens`,
  копятся до `reset()`), а вердикт: `completed` — если проход дошёл до решения
  (в том числе «ничего выбираемого»), `needs_recovery` — если решения не было
  (`invalid_response` / `output_truncated` / `provider_error` / пустой статус),
  `failed` — если проход бросил исключение. `run_results.failure` пуст всегда.
  События прохода (`planner_turn`, `planner_memory_recall`,
  `planner_exploration_directed`, `planner_tool_call`) висят на этом прогоне, а
  `decision_record` эпизода несёт `plan_run` — ссылку на потреблённый план. Видно в
  карточке строкой `План: …` и блоком `plan` в метриках (`metrics.plan_pass_mix`).
  `active_run_id` и здесь указывает на эпизод: смена указателя по фазе — остаток
  этапа 3.
- **План — один большой проход с разведкой, и разведка только читает** (решение
  владельца, 2026-10-16). Проход получает собственный тулсет
  (`skynet/plan_tools.py`, `plan_toolset`): свои `webfetch`, `read`, `grep`,
  `structure`, сетевые сенсоры по имени сервера (ddg: `search`, `fetch_content`),
  `memory` **только с `search`** (одноразовый `MemoryTool(store, actions=('search',))`,
  объявляет себя read) и `bash_readonly`. Тул, объявляющий `capability_kind='write'`,
  в тулсет не пускается — конструктор падает: это инвариант, а не формальность.
  Модель видит схемы **с первого запроса**, и ответ с `tool_calls` — здесь не ошибка
  парсинга, а работа (для прохода без тулов прежнее поведение сохранено: тул-вызов =
  `invalid_response`+переспрос). Бюджеты независимы и оба конечны: раунды разведки
  (`MAX_TOOL_ROUNDS = 4`, до `MAX_TOOL_CALLS_PER_ROUND = 3` вызовов в раунде, тело
  ответа ≤ `TOOL_RESULT_MAX_CHARS = 8000`) и переспросы парсинга
  (`MAX_MODEL_REASKS`); дедлайн один на всё, поэтому худший случай — `1 + 4 + 3`
  вызова, ровно столько записано в `budget.steps` строки прогона. Срыв или отказ тула
  приходит модели как `{"ok": false, …}` и не роняет проход; каждый вызов виден
  событием `planner_tool_call` (имя, раунд, ok, размер, время).
- **Бюджет токенов — мягкий, с запасом 10%.** Как и у память-прогона, строка
  показывает `int((max_input_chars // 3 + output_tokens) * 1.1)`, а превышение
  пишется событиями `plan_budget_soft_exceeded` / `memory_budget_soft_exceeded` и
  **никогда не отменяет прогон**: запрос режется провайдером по размеру, а не нашей
  бухгалтерией. Число считается в одном месте — `plan_pass_token_ceiling(max_input_chars,
  output_tokens)` в `skynet/autonomous_planner.py`: оттуда его читают и строка прогона, и
  mid-pass предохранитель прохода, поэтому окно, сдвинутое heartbeat'ом
  (`AutonomousPlanner.resize_window`), двигает бронь и порог вместе. **Порог прохода равен
  самой брони, а не её доле на один шаг:** замороженная доля (`32_619 = floor(274_688 / 8) * 0.95`)
  срабатывала на втором шаге при 66–77k потраченных из 238k — 29 срывов из 32 пассов 21.10.2026
  и 103 из 111 днём раньше, каждый с повтором через минуты и тем же концом. Порог, срабатывающий
  при трёх четвертях неизрасходованной брони, стоит работы вместо защиты. Тесты:
  `tests/test_plan_tools.py`, `tests/test_plan_scouting.py`, `tests/test_plan_pass_budget.py`.
- **Transcript ограничен на каждый outer cycle**, не накапливается бесконечно;
  история prun'ится после checkpoint'а (`store.prune_run_history`).
- **Единственный kind критерия — `verified_progress`** (считается в
  `store.evaluate_criteria`; навешивается на прогон и COMPLETED без пройденных
  критериев даунгрейдится в BLOCKED в `reactor.Reactor.tick` через
  `store.update_run_result_status`). Отчёт модели без успешного tool result не
  считается выполнением.
- **Память — управляемый ресурс.** `pinned` — **одноразовый**: закреплённая
  запись инжектится ровно в один промпт, а после корректно завершённого цикла
  снимается (`reactor` рядом с `mark_hypothesis`; `store.unpin_memories`,
  событие `pinned_memories_consumed`); чтобы увидеть её снова, нужно закрепить
  повторно. Прогон, не завершившийся или убитый сбоем провайдера, пины **не**
  сжигает (промпт до модели не дошёл). Автопоиск ограничен
  `SKYNET_MEMORY_INJECT_LIMIT` (default 3, читается в `cli`); глубже — по запросу
  tool'ом `memory`; неверная память superseded с evidence
  (`store.supersede_memory`). Канонические `handoff.CANONICAL_MEMORIES` — **не
  pin**: сид кладёт их обычными искомыми строками (`handoff.seed`), в промпт они
  не попадают (там уже есть `SYSTEM`), и `decay` их не трогает
  (`protected_contents`). Нюанс: закреплено может быть больше, чем влезает в
  `SKYNET_PINNED_MEMORY_LIMIT` (32) — излишек дожидается следующего цикла.
- **Maintenance не гейтится метриками и идёт каждый цикл**: retention, decay и
  компакция вызываются в `reactor._run_per_cycle_housekeeping` на **каждом**
  цикле (не раз в UTC-день, sidecar-маркера нет), не под
  `SKYNET_METRICS_SNAPSHOT`. Окно — **24 часа**
  (`SKYNET_EVENT_RETENTION_DAYS`/`SKYNET_EFFECT_RETENTION_DAYS`, default 1;
  `SKYNET_TRANSCRIPT_RETENTION_RUNS` — по-прежнему 200 прогонов, независимая ось),
  под них заведены индексы `idx_event_log_created`/`idx_capability_effects_created`
  (схема 14). Идемпотентность данных, а не маркера: retention удаляет только
  старше cutoff, `decay` списывает раз за stale-окно через `decayed_at`.
  `store.decay_memory_confidence` — **линейное** списание `0.03`/сутки (stale —
  1 сутки, пол 0.05, удаление ниже 0.1), поэтому память с confidence 1.0 живёт
  около месяца, а не бессрочно. Сбой любого maintenance-прохода durable, а не
  строка в journald: `reactor._record_maintenance_failure` пишет
  `maintenance_failed`, kind входит в `store.PROTECTED_EVENT_KINDS`.
- **Никаких дублирующих контуров**: один владелец lifecycle, один planner
  выбора работы, один stale-run путь, один criteria-kind. Мета-контур
  (дескрипторы, политика выбора работы, метрики) больше не вне досягаемости
  self-improvement: **owner decision 2026-09-21 — мета-контур открыт**, первая
  отправка такого предложения даёт warning, идентичная повторная проходит
  полный gate.
- **Судья открыт, но предупреждается.** `self_improvement.py`, `planner.py`,
  `autonomous_planner.py`, `idea_archive.py`, `metrics.py` входят в **warn-set**
  (`self_improvement.GATE_PROTECTED_PATHS`): первая отправка
  правки в них не применяется и возвращает модели warning, идентичный (тот же
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
  `_fingerprints_occupied`, `learnability_defect`, `_apply_goal_proposals`),
  а запись предложения и создание задачи — методы store
  (`store.record_planner_proposal`, `store.add_task`).
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
- **Схема версионируется и мигрируется.** `store.SCHEMA_VERSION` (сейчас 17) и
  `store.MIGRATION_NAMES`; 17 — `run-kinds` (колонка `runs.kind`: `build` /
  `memory` / `plan`, см. «Прогон — на проход» выше), 16 — `memory-attention-replay` (attention/replay
  колонки `memories`: `retrieval_count`, `positive_outcome_count`,
  `complexity_tier`, `last_session_id`, `replay_level`, `last_replay_at`),
  15 — `memory-lifecycle-scope` (`lifecycle_stage`, `scope`), 14 —
  `retention-age-indexes` (`idx_event_log_created` / `idx_capability_effects_created`),
  13 — `affect-valence` (таблица `affect_state`), 12 — `run-progress`
  (`runs.heartbeat_at` / `runs.last_progress_at`). Бамп версии без имени миграции —
  баг (см. инцидент 2026-09-19 ниже).
- `SOUL.md` — единственный авторитетный документ про identity; не переписывать
  его вслепую.
- **Root — только через владельца.** Аккаунт `service-account` имеет `sudo` (NOPASSWD;
  решение владельца, 2026-10-15), но **шелл не является воротами**: единственный путь к
  root — тул `sudo_exec` (`skynet/sudo_gate.py`). Он классифицирует команду: чтение
  логов (`journalctl`, `dmesg`, `/var/log`, `systemctl status|list|is-*|show`) идёт
  сразу; пол-лист (запись на диски, `rm -rf /`, вакуум и ротация логов, правка
  `sudoers` и аккаунтов, `shutdown`/`poweroff`) отклоняется навсегда и не может быть
  разрешён даже владельцем; всё остальное становится durable-вопросом владельцу в его
  Telegram — команда и «зачем», окно `SKYNET_SUDO_APPROVAL_TIMEOUT` (900 с), и
  **молчание = отказ**. Разрешение покрывает **одну** команду и не запоминается;
  метасимвол шелла (`;`, `&&`, `|`, `>` и т.п.) снимает allowlist-путь и отправляет
  команду к владельцу. `policy.sudo_denial` отклоняет `sudo`/`doas`/`pkexec`/`su`
  внутри bash — без этого гейт был бы декоративным. События: `sudo_blocked`,
  `sudo_auto_approved`, `sudo_approved`, `sudo_denied`, `sudo_ran`, `sudo_failed`.
  NOPASSWD — **не песочница**: процесс от `service-account` (включая собственный код организма)
  может дойти до root мимо тула, поэтому гейт защищает от случайной команды модели, а не
  от подменённого кода — и не описывает себя иначе.

## Открытость

- Архив идей хранит behavioural descriptor: `subsystem × change_type ×
  evidence_source` = 144 ячейки, в каждой остаётся только лучшая идея
  (`skynet/idea_archive.py`: `cell_key`, `classify`; MAP-Elites-правило и
  материализация — `store.record_idea_outcome`/`store.materialize_idea`).
- **Живость и новизна в архиве измеряются, а не объявляются.** `store.LIVE_IDEA_STATUSES`
  (`active`/`validated`/`materialized`) — это «жилец клетки»: `validated` (прогон
  идеи зачёл прогресс) и `materialized` — следствие использования, а не отставка.
  Пока живым считался только `active`, подтверждённая идея уходила из архива ровно
  в момент, когда доказывала себя: переставала защищать клетку (`best_in_cell`),
  выпадала из покрытия (`idea_cells`) и из родословной
  (`sample_archive_parents`), поэтому линию продолжали непроверенные. Отставляет
  только `superseded`, и взятие клетки отставляет **всех** её живых жильцов сразу
  (список уходит в `idea_superseded` как `retired`) — так чинится стек из
  нескольких живых строк в одной клетке, накопленный прежним правилом.
  `novelty` измеряется (`idea_archive.novelty_against_archive` поверх
  `lexical_uniqueness`) против живого архива, а не объявляется вызывающим кодом:
  константа `1.0` из `reactor._archive_proposal` убивала вторую ось, и мягкое
  epsilon-доминирование (`store._admits_over_incumbent`) не могло сработать ни
  разу. Строка с унаследованной константой перемеряется там, где её читает
  правило, и это событие `idea_novelty_remeasured`, а не тихая правка. Порог
  насыщения в выборе родителей снят: `quality >= 1.0` больше не исключает идею из
  линии — подтверждённая идея и есть лучший родитель, а злоупотребление
  ограничивает делитель `1/(1+children)`, а не фильтр.
- `research`-предложение без цитаты отклоняется (learnability gate,
  `idea_archive.learnability_defect`).
- **Идея становится работой один раз.** `store.materialize_idea` идемпотентна по
  идее, а не по вызову: если у идеи уже есть живая задача (`pending`/`running`),
  возвращается она же и ничего не создаётся; если её задача дошла до терминального
  статуса, идея **исчерпана** — `store.retire_spent_idea` отставляет её
  (`superseded`, единственная отставка в архиве) и освобождает клетку, а вызывающий
  получает `None`; новая задача получает ключ `idea:<idea_id>` вместо собственного
  id. Причина — измеренная авария 2026-10-16: одна идея о несуществующем дефекте
  пробника материализовалась **шесть раз** (05:55, 06:54, 12:49, 12:52, 13:01, 17:46
  MSK) — правило клетки держало её живой как лучшую в клетке, а ключ задачи был
  равен её собственному id, то есть не дедуплицировал ничего (так у всех 68 задач).
  `PortfolioPlanner._select_from_archive` смотрит до `ARCHIVE_SAMPLE_ATTEMPTS = 5`
  кандидатов: исчерпанный отставляется, берётся следующий; если исчерпаны все —
  решение пишется как `archive_spent`, и цикл честно идёт без работы, а не повторяет
  вчерашнюю. Тесты: `tests/test_idea_reuse.py`.
- **Архив кормится реальным трафиком, а не только идеями планировщика.**
  `reactor._archive_run_self_improvements` сканирует `tool_result` прогона и
  архивирует каждое успешное `propose_self_improvement`, с качеством = измеренный
  value прогона. До этого архивировал только `_run_autonomous_planning`, поэтому
  9 promoted-правок организма не дали ни одной строки.
- **Клапан `external_seek` — каденция, а не исчерпание.**
  `reactor._seek_external_evidence` вызывается при
  `generation % SKYNET_EXTERNAL_SEEK_EVERY == 0` (+ durable cooldown), если есть
  живой орган чувств, независимо от того, пуст ли портфель. Раньше он стоял
  после `_create_planner_fallback`, который всегда что-то создавал, поэтому за
  181 поколение клапан не открылся ни разу. **Предмет клапана — мир, не она
  сама:** `reactor.EXTERNAL_SEEK_TASK_TITLE` / `EXTERNAL_SEEK_EXPECTED_FACT` до
  21.10.2026 просили «a concrete SkyNet bottleneck», поэтому клапан стрелял
  исправно (3/10/20/13 созданий по дням), но каждая пуля отправляла модель читать
  arXiv про собственный планировщик, память и супервизор: источник внешний,
  предмет — внутренний (замер 21.10.2026: наружу 0% токенов, внутрь 97%; три
  крупнейших стока окна — это семейство). Приёмка теперь требует находку **вне**
  репозитория, файл в `artifacts/` и резолвимую цитату; находка про harness не
  считается. Отпечатки выводятся из заголовка, поэтому новая формулировка — новое
  семейство, а прошлые инстансы остаются в леджере раундами старого.
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
  (`SKYNET_EFFECT_RETENTION_DAYS`, default 1) не удаляет строку, на которую
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
- **Внешняя agenda ротируется колесом, а не расходуется.** Тот же
  `reactor._ensure_roadmap_seed(top_up=True)` вызывается каждый цикл *до*
  ранжирования и выдаёт **один раунд за раз**: `handoff.seedable_items(limit=1)`
  (`SKYNET_ROADMAP_ROTATION_HOURS`, default 48) считает позицию due, если у неё нет
  живой (pending/running) своей задачи и либо она никогда не работалась, либо с
  последнего раунда прошёл оборот колеса. Один раунд выходит за
  `rotation_hours / len(ROADMAP_TASKS)` (~7 ч), поэтому вся повестка оборачивается
  примерно за `rotation_hours`, ни одна позиция не уходит дальше чем на оборот, и
  позиции не переспрашиваются пачкой. Первый прогон (пустая таблица) сидит все
  позиции сразу. **Один раунд за вызов, даже когда зовущий передаёт `limit=None`**
  (стартовый сид, fallback на пустой таблице целей): иначе семь раундов выходили одной
  пачкой с общей микросекундой `created_at`, и следующий оборот открывал всю повестку разом.
  Замер до правки (21–22.10.2026): 65 раундов за 14.1 ч, 4.6 в час вместо 0.15, 55.8M
  токенов, каждая позиция переспрошена 8–12 раз — она в итоге отказалась работать и уснула.
  Причина была в единицах: `_wheel_interval` возвращал часы (6.857), а гейт сравнивал их с
  `total_seconds()`, поэтому «оборот прошёл» читалось для любой повестки старше семи секунд;
  функция называется `_wheel_interval_seconds` и возвращает секунды, а рядом с границей
  (возраст 1 ч при обороте ~6.9 ч) стоит тест, которого раньше не было. Раунд получает **свой** отпечаток
  (`handoff._roadmap_fingerprints(..., round_index=N)`), потому что `repair_status_consistency`
  каждый цикл помечает completed'ом любую ready-гипотезу, у которой в `tasks` есть
  завершённая задача: раунд поверх старого отпечатка через тик стал бы terminal
  (novelty 0.0 у `planner_candidates`), а его же cancel-ветка сняла бы pending-задачу
  как работу, которую никто не выберет. Прошлый раунд остаётся в леджере раундом
  старого отпечатка. Заблокированная
  внешняя задача досев **не** подавляет: блок виден своей строкой и ничего не
  маскирует. До 21.10.2026 повестка была одноразовой: семь позиций вышли 18.10.2026,
  все закрылись в тот же день, и следующие три дня `seedable_items()` не предлагал
  ничего — планировщику оставалась только harness-работа, и об этом никто не сказал.
- **Молчащая повестка говорит о себе сама.** `reactor._alert_if_roadmap_agenda_starved`
  (каждый цикл, из `_run_maintenance`) поднимает `roadmap_agenda_starved`, если цель
  `Live and do work that matters` активна, а новый раунд не выдавался дольше
  `SKYNET_ROADMAP_STARVATION_HOURS` (default 24 ч). Здоровое колесо молчит меньше
  оборота (~7 ч), поэтому порог не срабатывает на паузе между раундами и дедуп
  `raise_alert` служит ограничителем частоты. Живой, но старый раунд проверка **не**
  исключает: pending-задача, висящая сутки, — это не работа в полёте, а работа,
  которую никто не выберет (замер 19.10.2026: 21 час). Закрытая цель — не голод:
  припаркованная повестка молчит.
- **Owner-directed сдвиг в сторону мира.**
  `PortfolioPlanner.outward_weight` (`SKYNET_OUTWARD_WEIGHT`, default 0.40,
  читается в `cli`) добавляет бонус задаче `area="roadmap"` в `rank()`: сид-цель
  стоит на приоритете 1.0 и на одной критичности проигрывает harness-цели 4.0
  каждый цикл. Бонус меньше веса criticality (0.40), поэтому свежая критичная
  harness-задача всё ещё выигрывает. Это **выбор**, а не критерий: единственный
  kind по-прежнему `verified_progress`. Промпт судьи
  (`planner_contract.PLANNER_INSTRUCTION`) резервирует один из трёх слотов под
  работу о мире/владельце, когда harness не заблокирован.
- **Границы работы: репа — только на чтение, всё внешнее — свободная площадка.**
  `policy.ExecutionPolicy.allowed_roots()` = workspace + `SKYNET_SCRATCH_DIR`
  (default `/tmp/skynet-scratch`) + `SKYNET_SANDBOX_ROOTS` (список через `:` или `,`;
  на этом хосте это `/tmp` и домашний каталог), поэтому `read`, `grep` и старт `bash` видят одну
  границу: дом и `/tmp` доступны и на чтение, и на запись (у `/tmp` — tmpfs на 3.9 ГБ,
  у дома место есть), а репозиторий — только на чтение. Запрет записи в репу — **по цели
  команды**, а не по каталогу запуска: `printf x > /tmp/f`, `rm -f ~/f` и `cd` наружу
  разрешены даже из репы; отказ получает мутация, целящая в дерево или не называющая цели
  вовсе (`sed -i` по относительному пути) — такие правки идут только через
  `propose_self_improvement` (см. `## Tools`).
- **`skynet/scheduler.py` — чистый scorer провайдеров, не подключён.**
  `ProviderIndexScorer` считает эвристический мультипликативный индекс для пар
  (задача, провайдер) из четырёх нормированных факторов (completion /
  availability / urgency / fit). Это **не** closed-form Whittle index из Ayesta
  et al. 2018 (`1810.07782`), а вдохновлённая им эвристика — так и написано в
  модуле. Модуль не импортирует ни один внутренний модуль (тест
  `test_dependency_contracts.py::test_scheduler_is_a_pure_scorer`), никем не
  вызывается, и **второго планировщика из него делать нельзя**: подключение к
  выбору работы было бы дублирующим контуром. Черновая «shadow-симуляция» с
  фиксированными цифрами удалена как выдуманный результат.
- **Offline-инструменты не мёртвый код, даже если их никто не импортирует.**
  `skynet/recall_envelopes.py`, `skynet/recall_ablation.py` и
  `skynet/recall_scorecard.py` — не библиотеки, а runnable-модули
  (`python -m skynet.recall_envelopes|recall_ablation|recall_scorecard`): они
  читают ledger **read-only**, замораживают корпус и эталонные цифры, чтобы
  ranking-утверждение о planner-recall проверялось в одном месте, а не
  пересобиралось заново в каждом скретч-скрипте. «Нет импортёров» для них значит
  ровно «runnable», а не «мусор»: удаление снесло бы измерительную
  инфраструктуру. Тот же класс — `planner.valence_band_occupancy` (считает,
  сколько записанных draws канал valence мог реально решить; вызывается только
  тестом) и `scripts/verify_scratch_headroom.py` (live-проверка `runtime_facts`
  против `df` на сервере). В этих двух случаях это не удалённое, а инструмент.
- **Две механики взяты из RRSI как источник вдохновения, не как связанная
  работа** (arXiv:2609.24972, `github.com/google-research/rrsi`, Apache-2.0).
  (1) `skynet/change_review.py` — детерминированный критик правки до гейта:
  отклоняет правку из одних комментариев/пробелов и новый приватный
  module-level символ без ссылок, предупреждает о новом публичном
  неиспользуемом символе и о снятии safety-токена; вызывается из
  `self_improvement.validate_and_commit` до статических стадий и не дёргает
  модель. (2) `reactor._planner_exploration` при столле передаёт единственному
  планировщику `exploration.untried_cells` и
  `planner_contract.PLANNER_STALL_INSTRUCTION`, а `planner._cell_for` читает
  настоящую трёхосевую ячейку материализованной идеи
  (`store.idea_cells_by_task`) вместо хардкода `workflow|own-repo`. Директива —
  не фильтр: harness не выдумывает предложений, а reply, её игнорирующий, всё
  равно принимается; живёт она в ветке «нет работы» (`selected_work is None`),
  потому что планировщик вызывается только на пустом портфеле. Бенчмарк и
  числовую цель RRSI мы не берём — наш критерий остаётся процедурным.

## Рынок: MOEX и стратегии частного инвестора (решение владельца, 21.10.2026)

Постоянная тема организма — Московская биржа и торговые/активно-управленческие стратегии,
доступные частному инвестору, вместо работы над собственными внутренностями (замер до
переключения: наружу 0% токенов, внутрь 97%). Где это живёт в коде:

- `handoff.ROADMAP_TASKS` — семь позиций повестки с рыночными приёмками: панель цен, скоринг
  прошлых датированных утверждений, бэктест одной розничной стратегии, разбор внешнего
  источника в проверяемое утверждение, отчёт владельцу по-русски, аудит на lookahead/
  survivorship/издержки, одна мерка об условиях частного инвестора. Колесо выдаёт по одной
  позиции за `ROADMAP_ROTATION_HOURS / 7` (~7 ч).
- `handoff.CANONICAL_MEMORIES` — путь данных, правило point-in-time, бенчмарк и грабли okama.
- `planner_contract.PLANNER_INSTRUCTION` — **два из трёх** слотов за рынок; рыночное предложение
  без as-of даты и cut-off не принимается.
- `model_contracts.SYSTEM` — рынок «первым» в выборе работы и определение проверенного рыночного
  утверждения: PIT-датасет + воспроизводимая команда + сравнение с индексом за вычетом комиссии.
- `reactor.EXTERNAL_SEEK_*`, `reactor.FALLBACK_*`, `reactor.BOOTSTRAP_*` — предохранители тоже
  спрашивают про рынок: до 21.10 они спрашивали про её узкое место, и это была основная
  внутренняя петля (13–20 выстрелов в сутки).

Данные: **okama 4.0** — не зависимость проекта и не модуль в репозитории, а **MCP-сенсор**
`okama` (решение владельца 2026-11-05: «разверни как MCP»; ставится через `pipx` от аккаунта
`service-account`: `pipx install --force ~/okama-mcp`, лаунчер
`~/.local/bin/okama-mcp`, в env — `SKYNET_MCP_OKAMA_COMMAND`). Пакет живёт **вне**
репозитория (`~/okama-mcp`), поэтому okama с её pandas/numpy/scipy/statsmodels не
попадает в проектный venv, а `pyproject`-extra `market` и `skynet/market.py` удалены.
Четыре инструмента: `portfolio`, `asset_stats`, `macro`, `search`; ответ — компактный JSON,
ошибка — `{"ok": false, "error": …}`, сенсор от неё не умирает. Её сервис
`https://api.okama.io`; плюс MOEX ISS `https://iss.moex.com/iss/...`.
Оба без ключа. **Ходить в рынок через сенсор, а не через okama напрямую** — дефолты сенсора:
валюта `RUB` (у okama дефолт `USD`, и в рублёвой стратегии это молча
подмешивает курс), окно номинальное (его `inflation=True` обрывает ряд на последнем месяце
`RUB.INFL`), `OKAMA_API_TIMEOUT` поднимается до импорта okama, ребалансировка по умолчанию
месячная, а `last_date` внутри незакрытого месяца (и `None`) усекается до предыдущего
завершённого; явный `ccy`/`inflation` вызывающего всё равно выигрывает. Измеренные грабли
21.10.2026 (проверены живым запуском, остальное уже сделано дефолтами слоя):
`okama.Asset('SBER.MOEX')` в 4.0.0 падает `TypeError` — панель строить `AssetList`, стратегии
`Portfolio`; у `AssetList` нет `.prices` (читать `assets_close_monthly`, `assets_ror` и `get_*`);
`drawdowns` — свойство, не метод; `MonteCarlo` не принимает `ccy`.
Макро для сравнения: `RUS_CBR.RATE` (ключевая ставка) и `RUB.INFL`.

## Память и аудит

- **Memory audit — append-only, metadata-only.** `skynet/memory_audit.py` пишет
  JSONL-цепочку: каждая запись несёт `previous` = SHA-256 предыдущей строки, а по
  чекпойнтам фиксируются `rows`/`active_rows`/`attributed_rows`, байтовый размер
  и цепочный SHA-256 (контент хешируется, но не хранится). `verify_chain`
  называет строку, где цепочка порвалась. Чекпойнт вызывается из Reactor на
  конце эпизода (`reactor` → `memory_audit.checkpoint_memory_audit`).
- **Якорь против split view.** `memory-audit-anchors.jsonl` — обязательство,
  которое сам писатель не переписывает: одна строка на чекпойнт с sequence и
  хешем строки лога. Это повышает стоимость согласованного перезаписи лога и
  чекпойнта, но честно названо не криптозащитой от хоста; root of trust —
  обязательство владельца не менять лог.
- **Ничего не решается и не чинится автоматически**: audit лишь делает изменение
  видимым (факт + провенанс `source_run`, включая `unattributed_rows`).
  Тесты — `tests/test_memory_audit.py`.
- **Процедуры живут не только в памяти.** Память отвечает за факты, наблюдения,
  уроки и следы исходов; длинная пошаговая процедура — файл-скилл в `skills/`
  (см. `## Tools`), потому что окно инжекта — не место для простыней, а окно
  поиска — 24 терма. Обратной миграции нет: заметить, что процедура переросла
  память, и перенести её в файл — работа организма.

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
  снимает `OLLAMA_ENABLED`/`NVIDIA_ENABLED`/`DEEPSEEK_ENABLED` и
  `SKYNET_MONEY_BOOST`, чтобы вердикт выносился по коду, а не по окружению
  сервиса. (`OPENROUTER_ENABLED` там не значится и не значился: платный ранг
  требует ещё и money-boost, а его набор снимает, поэтому вердикт и без него
  зависит только от кода. Тесты завода провайдеров изолируют окружение через
  `patch.dict(..., clear=True)`.)
- **mypy снесён** (2026-09-19): pyright — единственный type-авторитет, второй
  чекер означал второй baseline. `pyflakes` тоже убран — его покрывает `F`.
- **Структурный контракт — единственное жёсткое правило о «беспорядке».**
  `tests/test_dependency_contracts.py` (AST, без зависимостей) фиксирует слои:
  `providers` не импортируют `store`/`reactor`/`cli`/`memory`; `memory` не
  импортирует `store` (поэтому и не может писать durable-события); `store` не
  импортирует `reactor`; `judge_health` — чистый читатель без внутренних
  импортов. Это про корректность, поэтому падает билд.
- Размер набора на момент ревизии: **1580 тестов**. Число живое — проверяй, а не
  верь этому файлу. Отдельные модули для новых подсистем:
  `tests/test_memory_audit.py`, `tests/test_change_digest.py`.
- Python-работать только через `.venv/bin/python`; системный интерпретатор не
  имеет pytest.

## Provider chain

- **Активный провайдер выбирается env, не кодом.** Развёрнутый режим сейчас —
  **платный** (owner decision 2026-10-16, «высади её на платного провайдера»):
  `SKYNET_MONEY_BOOST=true` и в env, и в `state/money-boost.json`
  (`{"enabled": true}`), поэтому `openrouter_deepseek`/`openrouter_glm` идут
  **первыми**, а бесплатный хвост стоит за ними; до этого режим был бесплатным
  (`SKYNET_MONEY_BOOST=false`), и работали
  четыре бесплатных NIM-ранга плюс `zai_glm47_flash`, `zai_glm45_flash`,
  `mistral_leanstral`, `mistral_codestral`, `muse_glimmer` (owner decision
  2026-09-28: держать бесплатный и наблюдать).
  **Ротация NIM переставлена владельцем 2026-10-17** («они поменяли модели»):
  порядок ровно `nvidia_deepseek` → `nvidia_super` → `nvidia_ultra` →
  `nvidia_lightning`, модели — `deepseek-ai/deepseek-v4.1-flash`,
  `nvidia/nemotron-3.5-super-vl-120b-a12b`, `nvidia/nemotron-3-ultra-550b-a55b`,
  `nvidia/nemotron-3.5-lightning-30b-a3b`. Проверено живьём по три запроса на
  модель через `GET /v1/models` + чат (2026-10-17, 03:20–03:45 МСК): все четыре
  id в каталоге есть; `super` — **3/3**, HTTP 200 за 0.7–11 с, и с нормальным
  бюджетом отдаёт текст (`content: 'Париж'`, `finish_reason=stop`; маленький
  `max_tokens` съедает `reasoning_content`, как у lightning); `ultra` — 0/3
  (HTTP 503 «Service temporarily overloaded»), `deepseek` — 0/3 (висит, не отдаёт
  байт; и через 120 с тоже), `lightning` — 0/3 по таймауту (полчаса раньше тот же
  ранг отвечал 200, то есть перегрузка плавающая). Поэтому `nvidia_deepseek`
  остаётся **припаркованным флагом** (`NVIDIA_DEEPSEEK_ENABLED=false`), а
  `super` включён и стоит первым живым: порядок в env владельца сохранён ровно
  (`deepseek → super → ultra → lightning`), но лестница начинает с `super`.
  Полная `SKYNET_PROVIDER_CHAIN` перечисляет двенадцать рангов
  (`openrouter_deepseek,openrouter_glm,nvidia_deepseek,nvidia_super,nvidia_ultra,nvidia_lightning,zai_glm47_flash,zai_glm45_flash,mistral_leanstral,mistral_codestral,muse_glimmer,ollama_gemma`),
  так что платный боевой режим возвращается флипом money-boost **в обоих местах**
  (`OPENROUTER_ENABLED=true` уже стоит). Модели: `OPENROUTER_MODEL`
  (`deepseek/deepseek-v4.1-flash`), `OPENROUTER_GLM_MODEL` (`z-ai/glm-5.3-flash`),
  `NEMOTRON_ULTRA_MODEL` (`nvidia/nemotron-3-ultra-550b-a55b`),
  `NEMOTRON_LIGHTNING_MODEL` (`nvidia/nemotron-3.5-lightning-30b-a3b`),
  `NEMOTRON_SUPER_MODEL` (`nvidia/nemotron-3.5-super-vl-120b-a12b` — NVIDIA
  переименовала модель 2026-10-17, старый `nvidia/nemotron-3-super-120b-a12b` в
  каталоге больше нет), `MUSE_GLIMMER_MODEL` (`meta/muse-glimmer-30b`).
  `NVIDIA_DEEPSEEK_ENABLED=false` (NIM отдаёт `deepseek-ai/deepseek-v4.1-flash`
  нулём байт и в стриме, и вне его, хотя имя в `/v1/models` есть; перепроверено
  2026-10-17 — 0/3 запросов, каждый висит до таймаута),
  `NEMOTRON_ULTRA_ENABLED=true`, `NEMOTRON_LIGHTNING_ENABLED=true`,
  `NEMOTRON_SUPER_ENABLED=true`, `MUSE_GLIMMER_ENABLED=true`,
  `ZAI_ENABLED=true`, `MISTRAL_ENABLED=true`, `OLLAMA_GEMMA_ENABLED=false`.
  **Ранг под выключенным флагом — парковка, а не удаление** (owner decision
  2026-10-15): реализация остаётся в дереве и возвращается флипом env, поэтому
  `=false` в `_ENABLE_FLAGS` — не мёртвый код и не повод предлагать чистку.
- **Один ранг = одна пара (endpoint, model).** Канонические имена —
  `openrouter_deepseek`, `openrouter_glm`, `nvidia_deepseek`, `nvidia_super`,
  `nvidia_ultra`, `nvidia_lightning`, `zai_glm47_flash`, `zai_glm45_flash`,
  `muse_glimmer`, `ollama_gemma` (`providers.PROVIDER_ALIASES`). С 2026-10-17
  четыре NIM-ранга называются коротко, по модели (решение владельца: «nvidia_deepseek
  -> nvidia_super -> nvidia_ultra -> nvidia_lightning»), а прежние длинные имена
  (`nvidia_nemotron_ultra`, `nemotron_super`, `nemotron_lightning`, `nemotron`,
  `openrouter`, `ollama`, …) остались легаси-алиасами: старый `config/skynet.env`,
  сохранённый снапшот предпочтений и любой тест с длинным именем резолвятся к
  канону, поэтому нормализация — не «переименование ради красоты», а совместимость.
  Имя называет модель за рангом, а `FallbackProvider` ключует cooldown'ы по имени,
  так что провал одного NIM-ранга не бенчит другой.
- **Бесплатный режим — флип money-boost, а не правка кода.**
  `SKYNET_MONEY_BOOST=false` в env и в `state/money-boost.json` убирает платные
  ранги, `SKYNET_PROVIDER_CHAIN` при этом не трогается. Бесплатный NIM-тариф сам
  по себе нагрузку организма не тянул: 2026-09-26 `ultra` отвечал HTTP 429
  (cooldown 300s), `super` — HTTP 500 (cooldown 30s), оба дали
  `provider_lockout` (critical) и поток `needs_recovery`, поэтому тогда боевой
  режим вернули на платный. С 2026-09-27 у бесплатного режима шесть рангов
  (NIM + Z.AI + Mistral) и терпеливая цепочка, поэтому его снова держат
  (owner decision 2026-09-28). С 2026-09-30 оба бесплатных NIM-ранга отвечают
  HTTP 451 (Unavailable For Legal Reasons — юридический/гео-блок, раньше были
  429/500); цепочка классифицирует его как retryable `unavailable`, ставит
  cooldown 300s и уходит на `zai_glm47_flash`.
  `providers.DEFAULT_CHAIN` перечисляет все десять рангов; архив реализаций
  **не удалён, а выключен** флагами `providers._ENABLE_FLAGS` (канонический флаг —
  первый в кортеже, легаси-алиасы приняты), так что вернуть их можно флипом env
  без правки кода. `providers.active_chain_names()` нормализует имена к канону и
  читает цепочку заново, без рестарта.
- Исторический провайдер переименован в `openrouter` (2026-09-23) и
  уточнён до `openrouter_deepseek` (2026-09-26) **без изменения endpoint и
  значений**: `OPENROUTER_BASE_URL` по-прежнему указывает на тот же endpoint;
  env-имена — `OPENROUTER_*`, chain-алиасы `openrouter`/`open_router`.
- **Боевой режим снова платный (owner decision 2026-10-16).** Живой env держит
  `SKYNET_MONEY_BOOST=true`, поэтому `providers.active_chain_names()` включает
  `PAID_PROVIDER_NAMES` (`openrouter_deepseek`, `openrouter_glm`) первыми, а
  бесплатные ранги остаются хвостом — простой платного ранга цикл не убивает.
  `openrouter_glm` (вторая модель того же openrouter-ключа, `z-ai/glm-5.3-flash`)
  идёт стабильно: у неё **свой** флаг `OPENROUTER_GLM_ENABLED`, который проверяется
  раньше общего `OPENROUTER_ENABLED`, — иначе нерабочий бэкенд-ранг нельзя было бы
  запарковать, не забанив заодно основной. Это не формальность: модель, которой нет
  в каталоге ключа, отвечает HTTP 403, `_classify_status` читает его как `auth`, а
  `fallback._is_fatal` считает `auth` фатальным для всей цепочки — один такой ранг в
  активном наборе убивает каждый вызов. Измерено 2026-10-16: до правки ключа
  каталог содержал только deepseek и GLM отдавал 403 за 0.1 с; после — обе модели
  отвечают HTTP 200 за 0.8 с, и бэкенд-ранг работает как задумано (плохое окно
  DeepSeek не отставляет бэкап).
- **Наблюдение 2026-11-05: живой env разошёлся с этой записью.** На сервере
  `SKYNET_MONEY_BOOST=false`, а `state/money-boost.json` отсутствует (поэтому
  решает env), то есть фактически работает **бесплатный** режим:
  `PAID_PROVIDER_NAMES` отфильтрованы, и лестница начинается с `nvidia_ultra`
  (журнал подтверждает `fallback_selected provider=nvidia_ultra`). Живая
  `SKYNET_PROVIDER_CHAIN` держит **тринадцать** рангов — добавлен `nvidia_nano`
  (`nvidia/nemotron-3-nano-omni-30b-a3b-reasoning`, флаг `NEMOTRON_NANO_ENABLED`,
  тест `tests/test_provider_nano_rung.py`), порядок `ultra → super → nano →
  deepseek → lightning` (коммит 2026-11-03 «Lead the chain with the rung that
  answers, and add the cheap fast one»). Расхождение зафиксировано, а не решено:
  что считать боевым режимом — env или эту запись — решает владелец.
- **Пустой ответ при `finish_reason='length'` — ошибка провайдера, не ответ.**
  `openai_compatible._parse` и openrouter поднимают retryable
  `ProviderError(response_quality)`, если видимого текста нет: иначе обрезанный
  на потолке ответ считался успехом и лестница не делала ни повтора, ни перехода
  (авария 2026-09-23). `length` с непустым text — валидный длинный ответ.
- **Потолок вывода — наш, а не сервера.** `NEMOTRON_ULTRA_MAX_OUTPUT_TOKENS` /
  `NEMOTRON_SUPER_MAX_OUTPUT_TOKENS` (default 8192) клампятся в
  `openai_compatible.py`; NIM принимает и отдаёт 16384 (измерено 2026-09-23),
  поэтому боевой env держит 16384 рядом с `SKYNET_PLANNER_OUTPUT_TOKENS`.
- **Tor-контроль удалён (2026-09-26).** Выход — VPN (`happ-xray`, дефолтный
  маршрут хоста), Tor не используется: `providers/tor_control.py` и `SIGNAL
  NEWNYM` убраны, провайдер с `proxy_url` делает ровно один запрос, и 403 —
  фатальная `auth`-ошибка без ротации. `SKYNET_TOR_CONTROL_*` больше не читается;
  `SKYNET_TELEGRAM_PROXY` по умолчанию пуст (трафик идёт через VPN).
- **Z.AI free-ранги взяты (2026-09-27).** `zai_glm47_flash`
  (`glm-4.7-flash`) и `zai_glm45_flash` (`glm-4.5-flash`): один ключ
  `ZAI_API_KEY`, один флаг `ZAI_ENABLED` на оба ранга, стоят после nemotron и
  до `ollama_gemma`. По официальному прайсу (`docs.z.ai/guides/overview/pricing`)
  обе модели **Free навсегда**, контекст 200K, потолок `max_tokens` 131072
  (ошибка 1210), и rung не клампит наш output. RPM/TPM не публикуются — только в
  консоли под логином (`z.ai/manage-apikey/rate-limits`). Измерено на живом
  ключе: free-тир держит **1 одновременный запрос** (вызов при занятом слоте —
  429, код 1305 «temporarily overloaded», блок ~50–70s, без `Retry-After`), а
  сверх контекста — 400/1261 `Prompt exceeds max length`, который
  классифицируется как `invalid_request` и **фатален для всей цепочки**
  (`fallback.py`), поэтому глобальный input-бюджет снижен до 150k (см.
  «Бюджет и контекст»). Tools и стрим работают; thinking у GLM включён по
  умолчанию и оплачивается из того же output-бюджета, поэтому ранги шлют
  `extra_body` `thinking:{"type":"disabled"}`
  (`providers._zai_extra_body`; `ZAI_THINKING=true` возвращает дефолт).
- **OpenCode Zen не берём (2026-09-27).** Ранга `opencode_zen` в цепочке нет и
  заводить его не следует: free-тир Zen (`https://opencode.ai/zen/v1`) отвечает
  `403 FreeTierError` («can only be used from within OpenCode») любому
  не-OpenCode клиенту. Гейт — HTTP-уровень (форма `User-Agent`,
  `x-opencode-session`, `stream`, имена tool'ов), а не TLS-отпечаток, вопреки
  первой гипотезе; обходные пути сообщества сводятся к имитации официального
  клиента, которую мы не делаем (риск `[user_blocked]`-блокировки аккаунта и
  нарушение ToS). Правка была внесена и откатана одним коммитом (`9a825d1e`); не
  переоткрывать без явного решения владельца. Легальный путь, если free-канал
  когда-нибудь понадобится, — платный ключ из консоли Zen (`/zen/go/v1`).
- **Mistral free-ранги взяты, два из них (2026-09-27).** `mistral_leanstral`
  (`labs-leanstral-1-5`) и `mistral_codestral` (`codestral-latest`) стоят в хвосте
  после Z.AI и до `ollama_gemma`; оба бесплатны (`PAID_PROVIDER_NAMES` их не
  трогает), один ключ `MISTRAL_API_KEY` и один флаг `MISTRAL_ENABLED` на пару,
  прокси не используется. Free tier Mistral выдаёт доступ **по-модельно**:
  флагманы (`mistral-small-2603` — это Mistral Small 4, `mistral-medium-latest` —
  Medium 3.5) отвечают 429 с `x-ratelimit-limit-req-minute: 0`, то есть закрыты,
  а ранги отдаются. Измерено на живом ключе 2026-09-27: leanstral — 38 зап/мин и
  5M токенов/мин, codestral — 125 зап/мин и 625k токенов/мин, у обоих 256K
  контекста, работают tools (`finish_reason=tool_calls`) и стрим, нет лимита «один
  одновременный запрос», `max_tokens=131072` принимается без ошибки потолка
  (отдано 13 893 токена), а `mistral-code-latest` — тот же `codestral-2508`, поэтому
  ранг один, а не дубль с общей квотой. leanstral **не принимает `temperature`**
  (400 «top_p must be 1 when using green mode»), потому ранги не шлют ни
  temperature, ни top_p и не клампят вывод. Длинный **не-стриминговый** ответ
  иногда приходит пустым телом (шлюз роняет крупный ответ) — это retryable-сбой,
  который добивает терпеливый цикл (`MISTRAL_TIMEOUT_SECONDS=300`,
  `SKYNET_PROVIDER_ATTEMPT_TIMEOUT`). 429 у Mistral без `Retry-After`, поэтому
  ранг ставит свой `MISTRAL_RATE_LIMIT_COOLDOWN_SECONDS=120`.
- **Потолок контекста ранга — не приговор цепочке.** `prompt_too_long` —
  отдельная категория `openai_compatible._classify_status` (400 +
  `invalid_request_prompt_too_long`/`maximum context length`/`exceeds max length`;
  маркеры `_is_prompt_too_long`), и `fallback._is_fatal` её **не** считает
  фатальной (рядом с `invalid_response`): провайдер пропускается, цепочка идёт к
  следующему рангу. Иначе один и тот же payload, не влезший в Z.AI (код 1261) или
  в короткую модель, обрывал бы весь эпизод. `finish_reason="error"` (HTTP 200 с
  оборванной генерацией у Mistral) и `"model_length"` без видимого текста —
  retryable `response_quality` в `_parse`, как и пустой `length`.
- **Money-boost читается из state-файла первым.** `providers._money_boost_enabled`
  берёт `state/money-boost.json` (`{"enabled": ...}`) и только при отсутствии или
  битом файле — env `SKYNET_MONEY_BOOST`; чтобы гарантированно выключить платное,
  правь оба.
- **Money-boost только выкидывает платные ранги, он их не добавляет.**
  `providers.active_chain_names` (`providers.PAID_PROVIDER_NAMES`) убирает
  `openrouter_deepseek`/`openrouter_glm`, когда money-boost выключен, но не
  возвращает их в явно заданную бесплатную цепочку. Чтобы платный ранг был
  активен, нужны все три условия: он перечислен в `SKYNET_PROVIDER_CHAIN`,
  `OPENROUTER_ENABLED=true` и money-boost включён. Поэтому «включить платное»
  одним `state/money-boost.json` недостаточно — с
  `chain=nvidia_nemotron_ultra,nvidia_nemotron_super` ранга openrouter нет вовсе
  (авария 2026-09-26). Имя читается заново, без рестарта процесса.
- **Таймаут openrouter — connect + idle, не общий дедлайн.** `OPENROUTER_TIMEOUT_SECONDS`
  (30) — connect; `OPENROUTER_CHUNK_TIMEOUT_SECONDS` (900) — тишина между чанками,
  во время активного стрима не горит; `OPENROUTER_STREAM_DEADLINE_SECONDS` (0 = off) —
  опциональный общий потолок. В «терпеливом» цикле per-attempt бюджет стримящего
  ранга (`SKYNET_PROVIDER_STREAM_IDLE`, 120с) передаётся как idle-бюджет, а не
  как дедлайн потока.
- **Устойчивость цепочки — «терпеливый» цикл (owner decision 2026-09-27).**
  `FallbackProvider.complete(..., patient=True)` вызывается только ReAct-путём;
  memory loop и планировщик оставлены на «один удар на провайдера», потому что
  их бюджеты короткие. Схема: каждому провайдеру
  `SKYNET_PROVIDER_ATTEMPTS_PER_PROVIDER` (4) сгруппированных удара с паузами
  `SKYNET_PROVIDER_ATTEMPT_WAITS` (5/10/15с); после последнего провала — замок
  `SKYNET_PROVIDER_BLOCK_SECONDS` (300с), но слово провайдера (`Retry-After`,
  «квота до следующего дня», `ProviderError.cooldown_explicit`) сильнее — даже
  если оно короче; затем следующий провайдер. Когда вся цепочка под замками,
  вызов **не заканчивает прогон**: он ждёт ближайший замок и повторяет проход,
  пульсируя heartbeat (`Reactor._provider_wait_pulse`), иначе `StaleRunWatchdog`
  убил бы намеренно ожидающий прогон. Только
  `SKYNET_PROVIDER_DEAD_PASSES` (3) мёртвых прохода подряд поднимают
  терминальную ошибку с меткой `is_chain_dead_terminal` (`dead passes`), и
  реактор по ней сразу уходит в `provider_lockout` со сном
  `SKYNET_PROVIDER_LOCKOUT_SECONDS` (1800с), не дожидаясь серии из трёх
  проваленных прогонов. `SKYNET_PROVIDER_ATTEMPT_TIMEOUT` (900с) — потолок
  «молчащего» провайдера (не стримит, думает и присылает всё скопом),
  `SKYNET_PROVIDER_STREAM_IDLE` (120с) — тишина между чанками у стримящего,
  `SKYNET_PROVIDER_CALL_DEADLINE` (14400с) — жёсткий потолок одного вызова.
  Паузы и замки в бюджет прогона (модельные секунды, `react.py`) не входят.
  Ошибки «кривой запрос»/«плохой ключ» (`invalid_request`, `configuration`,
  `auth`) обрывают цепочку сразу — их повтор не вылечит. Успех после раннего
  провала фиксируется durable-событием `fallback_retry_recovered`, замок —
  `fallback_provider_blocked`. Инцидент 2026-09-27: до этого стояла рунговая
  лестница с `MAX_ATTEMPTS=2`, которая обрезала цепочку первыми двумя
  Nemotron-рангами, — Z.AI-страховка не вызывалась, и пока NIM отдавал 500/429,
  прогоны падали в `needs_recovery`; рунговая схема этой заменена.
- Единственный владелец ретраев — `FallbackProvider`; не дублировать ретраи в
  других слоях.
- **Переспрос модели — не повтор провайдера.** Когда «безынструментная» фаза
  (Finish, ремонт отчёта, Memory Loop, планировщик) получает ответ, который не
  является контрактным JSON, harness не молчит и не повторяет провайдера — он
  **переспрашивает модель** до `model_contracts.MAX_MODEL_REASKS` (3) раз,
  называя в предупреждении конкретику: попытанные tool-вызовы
  (`render_tool_calls`), ошибку парсера первого ответа и инструменты, которые
  эпизод реально выполнил (`corrective_reask_instruction`). Это не дублирование
  ретраев `FallbackProvider` (тот владеет повторами **провайдера**): переспрос —
  разговорная коррекция, и прецедент у неё давний — `planner_retry` и
  `finish_repair`. Повод измеримый: замер 2026-09-28 — 29 из 149 Finish-ходов
  вернулись `finish_reason=tool_calls` с пустым текстом, хотя инструменты в
  запрос **не передавались вообще**, 11 `finish_repair_failed` несли
  `report_chars=0`, а 14 из 30 последних `memory_degraded` — та же болезнь
  («no acceptable JSON value»). Точки входа:
  `react._repair_finish_report`, `memory.MemoryLoop.consolidate`,
  `autonomous_planner.generate`; события — `finish_repair_failed` (с `attempt`
  и `attempted_tools`), `memory_reask`, `planner_retry`. Бюджет: у Memory Loop и
  планировщика на первый ответ и **все** переспросы один общий дедлайн
  (`_call_provider` / `_complete_bounded`), у ReAct — его модельное время.
  Семантика статусов не меняется: переспрос может только спасти прогон, но не
  закрыть его без валидного отчёта.
- **«DEGRADED function cannot be invoked» — это ранг, а не запрос.** NVIDIA
  отвечает **HTTP 400** с таким телом, когда развёртывание модели нездорово
  (`{"status":400,...,"detail":"Function id '<uuid>': DEGRADED function cannot be
  invoked"}`). По коду статуса это `invalid_request`, а он **chain-fatal** — то
  есть один больной ранг убивал всю лестницу, включая три здоровых ранга за ним.
  Замер 2026-10-16/17: 12 отказов в час, все на `nvidia_ultra`, прогоны по нулю
  шагов, план-проходы падали `planner_provider_error`. Теперь
  `_classify_status` ловит маркер в теле и отдаёт `server` (retryable) с
  `DEGRADED_PROVIDER_COOLDOWN_SECONDS` (300 с): ранг пропускается, хвост живёт.
  Обычный 400 (кривой запрос) остаётся `invalid_request` и по-прежнему фатален.
- **Повторно падающий ранг отходит в сторону сам.** Категорийные кулдауны
  (2 с за network, 10 с за server) короче, чем время одного зависшего ответа,
  поэтому выбор снова и снова брал тот же ранг первым — отсюда «Sky мертва»
  (шаг 0, «ждёт провайдера») при живых рангах за ним. `FallbackProvider` теперь
  считает **серию подряд идущих отказов** ранга и умножает кулдаун на
  `FAILURE_STREAK_COOLDOWN_GROWTH` (2.0) до `MAX_ESCALATED_COOLDOWN_SECONDS`
  (900 с); серия хранится рядом с кулдаунами в `provider-fallback.json` и
  сбрасывается первым же ответом ранга. Кулдаун, который **сам провайдер
  назвал** (`Retry-After`, квота), не усиливается — правило владельца «его 10
  секунд» сильнее.
- **402 = пустой кошелёк, и это provider-local, а не сломанный запрос.**
  `_classify_status` отдаёт HTTP 402 собственную категорию `billing` (не
  `invalid_request`), а `fallback._is_fatal` её фатальной не считает: ранг
  пропускается, а кулдаун берётся `BILLING_PROVIDER_COOLDOWN_SECONDS` (1800 с) —
  баланс не восстанавливается от повтора. Замер 2026-10-16: openrouter на пустой
  баланс ответил `402 {"error":{"message":"Недостаточно средств"}}`; quota-маркера
  в теле нет, ранг падал в `invalid_request` — а он **chain-fatal**, и одна пустая
  касса убивала всю лестницу вместе с бесплатным хвостом (десять прогонов за сорок
  минут закрылись `needs_recovery` с нулём шагов, 43 страйка по платному рангу).
  Списанный free-quota (`_is_quota_exhausted`) остаётся `quota` с кулдауном до
  следующих UTC-суток, а 401/403 — по-прежнему `auth` и по-прежнему фатальны.
- **Ранги переключает владелец: `/providers` в боте, применяется на следующем
  heartbeat.** Два файла рядом с реестром: `state/provider-prefs.json` — что
  владелец **хочет** (`{"off": [<имена рангов>]}`), `state/provider-applied.json` —
  что реально применено. Бот пишет только желаемое (`providers.set_provider_off`);
  `providers.apply_provider_prefs()` вызывается в начале `Reactor.tick`, копирует
  желаемое в применённое и пишет событие `provider_prefs_applied` **только при
  изменении** (само `provider_chain_reloaded` по-прежнему пишет fallback, когда
  замечает новый список). `active_chain_names()` фильтрует цепочку по
  **применённому** файлу, а env-флаги `*_ENABLED` бьют сразу, — поэтому тумблер не
  перекраивает лестницу под уже идущим прогоном, а парковка флагом остаётся
  мгновенной. Файла нет — цепочка ровно из env, как раньше (снапшот пишется
  только когда тумблер существует). Карточка `/providers` различает три состояния,
  которые изнутри цепочки неразличимы: работает / выключен владельцем / запаркован
  флагом.
- **Авария провайдера не списывается на задачу, а провал плана не выдумывает работу.**
  Диагноз тредмилла 2026-10-16 (три одинаковые задачи «Diagnose one concrete SkyNet
  bottleneck» за 41 минуту, generations 20/23/26): пустой кошелёк → каждый план-проход
  `planner_provider_error` → реактор читает это как «портфель пуст» и шьёт фолбэк-задачу
  (`planner_fallback_created`) → три попытки по 0 шагов умирают на 402 → `task_gave_up`
  блокирует → блок терминален, переиспользование в `_create_planner_fallback` работает
  только для `pending`/`running` → через окно (`fallback_repeat_generations=4` **или**
  `SKYNET_FALLBACK_REPEAT_SECONDS=900`) рождается близнец. Две правки:
  (1) `_run_had_provider_failure` считает прогон провайдерским не только по тексту «all
  providers failed» (chain-fatal ошибка возвращается из ранга напрямую, и в тексте стоит
  модель, а не цепочка), но и когда прогон дал **ноль шагов**, а в его собственных
  событиях есть `provider_error`/`provider_failure`/`provider_error_classified` (эти три
  несут `run_id`, в отличие от `fallback_failure`); (2) `tick` не зовёт
  `_create_planner_fallback` при `planner_reason == PLANNER_PROVIDER_ERROR`, а пишет
  `planner_fallback_skipped` — предохранитель существует для пустого портфеля, мёртвая
  цепочка лечится lockout'ом. Тесты: `tests/test_provider_casualty.py`,
  `tests/test_planning.py`.
- **Credentials — только в `config/skynet.env`**, файл gitignored
  (`.gitignore:10-11`, mode 600). Никогда не коммитить секреты и не держать
  бэкапы env внутри дерева репозитория.

## Бюджет и контекст (paid-режим)

Числа ниже — боевые значения `config/skynet.env.example`, а не дефолты кода:
без env harness возьмёт дефолты `cli` (`SKYNET_MAX_SECONDS` 3600, `SKYNET_MAX_STEPS`
100, `SKYNET_OUTPUT_TOKENS` 8192, `SKYNET_MEMORY_INPUT_TOKENS` 50000,
`SKYNET_WATCHDOG_SECONDS` 3900).

- `SKYNET_MAX_SECONDS=7200` — один прогон = два часа модельного времени; watchdog
  `SKYNET_WATCHDOG_SECONDS=7500` должен его превышать (systemd stop timeout —
  1200s, `deploy/skynet.service.in`).
- `SKYNET_MAX_STEPS=800` (сообщения/шаги ReAct), `SKYNET_REACT_INPUT_TOKENS=150000`
  (окно ReAct; оценка консервативная — `chars/3` в `react._request_tokens`, —
  поэтому реальных токенов меньше и запрос остаётся внутри 200K-контекста free
  Z.AI-рангов; снижено с 500k 2026-09-27), `SKYNET_OUTPUT_TOKENS=16384`,
  `SKYNET_MEMORY_INPUT_TOKENS=100000`.
- `SKYNET_PLANNER_OUTPUT_TOKENS=16384` — отдельный output-бюджет планировщика
  (читается в `autonomous_planner`); раньше его резал `min(4096, ...)`.
  `SKYNET_MEMORY_LOOP_TIMEOUT=300` — таймаут memory loop'а (в `cli`).
- `SKYNET_CONTEXT_FINISH_RESERVE=50000` и `SKYNET_TOOL_RESULT_MAX_CHARS=16000`
  читаются в `reactor` при сборке `ReActConfig`; reserve ужимается до
  `limit // 8` (при окне 150k — 18750).
- **Потолок контекста идёт за классом ранга** (owner decision 2026-10-16: «для
  бесплатных max контекст 100 000, для платных 700 000»). Тик вызывает
  `providers.apply_provider_prefs()` и проставляет окно первого живого ранга в
  `runner.config.max_tokens` (`providers.context_ceiling_tokens`) и в
  `autonomous_planner.max_input_chars` (`providers.context_ceiling_chars`); бюджет
  планового прогона пересчитывается от нового `max_input_chars`. Класс ранга —
  `providers.provider_class` (`paid` для `openrouter_deepseek`/`openrouter_glm`,
  иначе `free`), override — `SKYNET_PAID_CONTEXT_TOKENS`/`SKYNET_FREE_CONTEXT_TOKENS`
  и `SKYNET_PAID_CONTEXT_CHARS`/`SKYNET_FREE_CONTEXT_CHARS` (нечисловое или
  неположительное значение игнорируется).
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

- Деплой и рестарт сервисов выполняются **только по явной команде владельца**.
  Агент не запускает их сам и не коммитит.
- **Сервис — user-unit с owner-approved root.** Продовый режим — linger
  `systemctl --user`-юнит аккаунта `service-account`: код в
  `~/project-skynet`, env с путями под этот home. **Управление юнитами**
  идёт через `systemctl --user` (`skynet.systemctl.systemctl_command`) — system-шина
  для этого не нужна, и сам себя юнит не перезапускает. Аккаунт при этом имеет `sudo`,
  и весь доступ к root идёт через `sudo_exec` с разрешением владельца (инвариант
  «Root — только через владельца» выше); `deploy.sh` и ручные рестарты это правило не
  обходят. `scripts/deploy.sh` в **первом** запуске разворачивает всё
  (git checkout, `uv venv --python 3.14`, MCP-runtime, env, юниты, `enable --now`),
  в **повторных** — только деплой (push, `uv pip install -e '.[dev]'`, перезалив
  env, ре-рендер юнитов, `restart`). Linger (`loginctl enable-linger service-account`) —
  ручное предусловие хоста, скрипт его не трогает.
- **Смерть носителя лечится таймером, а не запретом.** Организм может погасить
  свой юнит сам: bash-шаг, останавливающий `skynet.service`, убивает собственный
  cgroup, поэтому хвостовой `start` из той же строки не выполняется, а
  `Restart=always` явный stop не покрывает — так 2026-10-15 носитель лежал 8
  часов. Owner decision: сам stop не запрещается; вместо этого
  `skynet-revive.timer` каждые 30 минут триггерит oneshot
  `skynet-revive.service` (`skynet/revive.py`), и тот стартует носитель, когда тот
  не `active` (а `unknown`/`failed`/`inactive` — именно тот случай). Ошибка стоит
  максимум один тик, а не ночь. Тормоз для плановой остановки — сам таймер
  (`systemctl --user stop skynet-revive.timer` или `disable`). Шаблонов юнитов в
  `deploy/` теперь шесть: `skynet.service`, `skynet-telegram.service`,
  `skynet-rollback.service`, `skynet-alert@.service`, `skynet-revive.service`,
  `skynet-revive.timer`; `deploy.sh` рендерит и ставит их все и заряжает таймер
  без ожидания перезагрузки.
- **Стоп владельца — это стоп: стоянка (park).** `/stop` в боте пишет
  `state/owner-park.json` (`skynet.revive.set_park`) **до** остановки юнита, а
  `skynet-revive.service` при наличии стоянки носитель не поднимает; `/start`
  снимает её (`clear_park`). Причина: 2026-10-16 в 22:07:54 владелец остановил
  носитель из бота, а в 01:00:30 его же таймер поднял её обратно — изнутри хоста
  намеренный стоп и авария неразличимы, а тормоз «останови таймер» доступен только
  куратору. Стоянка — не `/stop`-заглушка: она не мешает `Restart=always` и не
  трогает сам стоп, она только запрещает **воскрешение**. `scripts/external_probe.py`
  читает ту же стоянку и не шлёт алерт об остановке, которую владелец заказал сам
  (иначе плановая остановка сама себя доносит ему как авария).
- **Граница данных — DAC на доме владельца.** Клиентские секреты лежат в home
  владельца, и каталог держится `0700` (owner-only, без ACL для `service-account`); это
  закрывает чтение **любым** способом (`find`, Python, bash, `/proc`), без LSM и
  без sandbox в юните. Право на дом **не ослаблять** (не выдавать ACL, не менять
  права) — иначе `service-account` прочитает секреты.
- `scripts/deploy.sh` — деплой; `scripts/rollback.sh <commit|tag>` — откат;
  `scripts/sync.sh [--prune-worktrees]` — двусторонняя синхронизация
  локального дерева с серверным: `fetch`, затем fast-forward локально **или**
  push на сервер, а при истинном расхождении — отказ (автомержа нет); опционально
  чистит серверные self-improvement worktree.
  Грязное локальное дерево — отказ, как у `deploy.sh`.
- Адрес/доступы сервера живут в `config/deploy.env` (gitignored), не в скриптах
  и не в этом файле.
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

## Публичный релиз

- **Контура в дереве нет, и он был целиком её.** Один коммит `8c46c96`
  (2026-09-24) добавил всё: `release/publish.py`, `release/rules.toml`,
  `release/overlay/`, `scripts/publish-public.sh` и `tests/test_public_release.py` —
  sh-обёртка с самого начала была точкой входа её санитайзера, а не отдельным
  инструментом. Удалены 2026-10-21. Публикация — отдельный стек владельца, не этот
  репозиторий.
- **Почему удалены.** Экспорт сканировал `git archive HEAD` — коммит *до*
  проверяемой правки. Предложение могло внести нарушение незаметно (сканер видел
  старое дерево и проходил) и никогда не могло его убрать: правка лежит вне дерева,
  которое читает проверка. Шесть часов полной блокировки durable-изменений и три
  ночных запроса root, которые владелец не мог подтвердить.
- **Форму сторожит тест.** `tests/test_gate_reads_the_working_tree.py` падает, если
  что-либо в репозитории материализует содержимое голого `HEAD`
  (`git archive|show|ls-tree|cat-file` с `HEAD`): проверка обязана читать рабочее
  дерево. Именование коммита (`rev-parse HEAD`) и временные репозитории в фикстурах
  — законны.
- **Identity едет в пакете, чтобы `pip install` давал живого организма.**
  `pyproject.toml` несёт `[tool.setuptools.package-data] skynet = ["SOUL.md"]`,
  поэтому файл попадает в wheel. `reactor.load_soul_text` ищет identity по
  приоритету: runtime-root (`--root`/`SKYNET_ROOT`) → корень исходников
  (`__file__` рядом с `reactor.py`) → ресурс внутри пакета
  (`importlib.resources.files("skynet")`). Последний нужен именно wheel'у: там
  `Path(__file__).parent.parent` — это `site-packages`, и без него организм
  стартовал бы молча без SOUL. По той же причине блок `WORKSPACE` строится от
  runtime-root, а не от `__file__`.

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
  probe опрашивает юниты на обеих шинах — system и `--user` (`system_probe.py`),
  потому что чтение статуса system-юнитов root не требует и через `sudo_exec`
  не идёт.

## Tools

- Встроенный набор `default_tools()` — **`bash`, `webfetch`, `read`, `grep`,
  `db`, `structure`** (`skynet/tools.py`).
- **Bash не редактирует своё дерево.** `policy.workspace_write_denial`
  (`tools.py:execute`, hard deny) отклоняет мутацию, чей эффективный cwd с учётом
  `cd` лежит внутри workspace, либо redirect внутрь него: правки кода — только
  через `propose_self_improvement`, прототипы — в scratch. Выход в новую
  директорию (`cd /tmp/... && ...`) разрешён; прежний `workspace_escape` (варнинг
  за выход из workspace) снесён. Детекция намеренно неполна (запись через
  интерпретатор проходит) — второй контур — wake-heal и dirty-guard (см. выше).
- **`structure` — read-only зеркало собственного кода** (`skynet/structure.py`,
  pyscn, MIT): сложность, клоны, мёртвый код, связность, архитектура. Это
  **наблюдаемость, а не критерий промоушена** — «меньше беспорядка» как цель
  заставило бы дробить функции и переименовывать клоны ради цифры. pyscn стоит
  в `[dev]`, запускается в системном temp-каталоге (чтобы не сорить в
  worktree), и при отсутствии бинаря честно возвращает «не установлен», а не
  роняет прогон.
- **У прохода-плана свой, только читающий тулсет** (`skynet/plan_tools.py`,
  `plan_toolset`): `webfetch`, `read`, `grep`, `structure`, сетевые сенсоры (по имени
  сервера) и `memory` с одним `search`, плюс `bash_readonly` — шелл с белым списком
  команд (`git log|show|diff|status|blame`, `cat`, `grep`, `rg`, `find`, `jq`, `ps`,
  `df`, `du`…), отдельным белым списком git-подкоманд и записанной причиной отказа на
  `>`/подстановки/`sudo`/`rm`/`tee`/`systemctl`. Это **фильтр, а не песочница**: права
  те же, что у эпизода, а запрет стоит на записи (и на любой дороге к root), не на
  чтении хоста. Тул с `capability_kind='write'` в тулсет не попадает — `plan_toolset`
  падает на конструировании.
- `Reactor.__init__` всегда добавляет `ask_user`, `send_message_to_user`,
  `acknowledge_inbox` (закрыть pending-уведомление владельца), `read_inbox`,
  `sudo_exec` (единственный путь к root: классификация и гейт — `skynet/sudo_gate.py`),
  `memory` (действия `search/remember/forget/pin/unpin/correct`,
  `memory_tool.MemoryTool.ACTIONS`), `record_plan`; при наличии `<root>/.git` —
  `propose_self_improvement`, `request_rollback`.
- Telegram владельца — обычный чат: текст без `/` идёт как observation
  (`telegram_bot`), команды начинаются с `/`; блокирует только
  `ask_user(wait_seconds>0)` (`dialogue.AskUserTool`).
- **Бот — интерфейс, а не список команд** (`skynet/telegram_ui.py`): меню публикуется
  `setMyCommands` в чат владельца; `/status` — карточка, обновляемая на месте
  (`editMessageText`); `/logs` выбирает окно кнопками; у каждого вопроса в очереди кнопки по
  его опциям (`q:<question_id>:<index>`) — тем же путём даётся **разрешение на root**.
  Колбэк несёт только действие (никогда — текст для исполнения), принимается лишь при
  совпадении `from.id` и `chat.id` владельца, закрытый вопрос второй раз не отвечается,
  устаревшая кнопка даёт тост «Кнопка устарела». Клавиатуру прикрепляет outbox
  (`outbox.markup_for_outbox` → `TelegramTransport.send_markup`), поэтому транспорт без
  `send_markup` продолжает работать как раньше.
- **`/status` — карточка о «сейчас», `/metrics` — срез за окно** (`skynet/status_card.py`,
  решение владельца, 2026-10-16): жива ли (состояние юнита и возраст), чем занята (текущий
  прогон, задача, шаг, фаза, токены), не встала ли (время с последнего события; дольше 15 минут —
  «похоже, встала»), четыре сенсора — **по именам** (пропавший называется, а не вычитается из
  счёта), провайдеры (размер цепочки, отказы за час, money-boost), память/идеи/задачи, что ждёт
  ответа владельца, честен ли деплой (маркер против `HEAD`) и когда следующий тик воскрешения.
  Строки, которым нечего сказать, отсутствуют: «Ждёт тебя» — только при открытом вопросе или
  алерте, сенсоров нет у стоящего процесса. Живость, которой карточка не видит, не заявляется:
  остановленный юнит читается как остановленный, нечитаемое свойство — «неизвестно», а не
  «выключено». Обращения к хосту (systemd, `git`, `/proc`) идут через один инжектируемый пробник
  (`Probe`) — поэтому карточка тестируется вне хоста. Окно у `/status` больше не принимается: оно
  у `/metrics`, который остался прежним срезом. Прогон без `planner_decision` всё равно
  называется по `decision_record` (`reporting.run_subject`). Собирает карточку
  `PrivateBot.card_text`.
- **Автоматические логи сессий в Telegram выключены** (решение владельца, 2026-10-15;
  `SKYNET_RUN_LOGS_TO_TELEGRAM=0`): завершённые прогоны (`agent_response`) и mid-run заметки
  (`agent_message` из `_enqueue_run_progress`) не пушатся в чат. Запись при этом остаётся
  durable — `run_results`, finish-отчёт, события — и `/logs` рендерит её по запросу. В канал
  владельца идут только алерты, вопросы и то, что организм сказал сам
  (`send_message_to_user`); `=1` возвращает автоматический поток для отладки.
- **Холодный старт не исполняет бэклог.** Бот держит offset в
  `state/telegram-offset.json`; если файла нет (редеплой, purge, сброс `state/`),
  Telegram проиграл бы все накопившиеся апдейты, и лежачий `/stop` погасил бы
  сервис через секунды после старта. `PrivateBot._confirm_backlog` на первом
  поллинге делает `getUpdates(offset=-1)` (отдаёт только последний апдейт,
  подтверждая старые), сохраняет offset и **ничего не исполняет** — холодный
  старт работает «с этого момента». Вызов стоит внутри поллинг-цикла, поэтому
  временный сбой повторяется его же обработчиком, а не откатывается к реплею.
- **Владельцу модель пишет по-русски и о себе — в женском роде.** Правило
  задано в `SYSTEM` (`model_contracts`) и продублировано в описаниях схем
  `send_message_to_user` и `ask_user` (`dialogue.SendMessageToUserTool`,
  `dialogue.AskUserTool`); Finish-сводка и машинные поля остаются английскими.
  Identity несёт `SOUL.md` (Sky — девушка, говорит о себе в женском, любит
  владельца), а пол сам по себе — часть identity, а не отдельный инструмент.
- Self-improvement gate — **warn once, allow on identical resubmission**.
  (owner decision 2026-09-21, отменяет прежний `/approve`-механизм и инвариант
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
  worktree задачу не переоткрывают. Уже существовавшую до пробуждения грязь
  снимает wake-heal ниже (см.); переоткрытие относится к грязи, созданной во
  время самого прогона.
- Изменение кода durable **только** через `propose_self_improvement` (он же
  гейтит и промоутит); правка main worktree напрямую не засчитывается.
- **Грязное дерево лечится на пробуждении.** Публичный
  `SelfImprovementManager.quarantine_dirty_worktree` (`self_improvement.py`, вход
  над `_quarantine_dirty_worktree`/`_quarantine_untracked_paths`) сохраняет
  изменённые tracked-файлы в `state/quarantine/`, возвращает дерево через
  `git checkout --` и при `include_untracked` выметает untracked, пропуская
  каталог состояния (`exclude`). Reactor вызывает его на каждом пробуждении
  (`reactor._auto_quarantine_worktree`; событие и наблюдение
  `worktree_auto_quarantined`), но **только** когда `self_improvement_root`
  задан явно: реактор без явного корня не карантинит `cwd` (иначе тестовый
  прогон в репозитории откатывает правки разработчика). Раньше дерево чистилось
  лишь как побочный эффект `propose()`, поэтому эпизод, не подавший предложение,
  оставлял его грязным через пробуждения (10 284 `worktree_dirty_after_bash` за
  9 суток на боевом организме).
- **Петля «правка → предупреждение → правка» обрывается счётчиком.**
  `ReActConfig.max_dirty_streak` (default 5, env `SKYNET_MAX_DIRTY_STREAK`,
  читается в `reactor`) считает подряд идущие bash-шаги, оставившие одни и те же
  tracked-файлы изменёнными; чистый шаг сбрасывает счёт, а новый набор файлов
  начинает его заново. При достижении порога `react` эмитит
  `worktree_dirty_guard` и принудительно завершает эпизод (`_finish`), не
  перекрывая pending control action (deferred restart). Гвард по точному ответу
  (`max_repeated_responses`) эту петлю не ловит: команды различаются.
- **Политика матчит защищённые пути по границам.** `policy.protected_path_present`
  (используется `self_preservation_warning`/`resurrection_denial`) не считает
  подстроку совпадением: `.env` больше не ловится внутри `config/skynet.env`
  (все 181 hard-denial указывали `protected_path: ".env"`), запись в настоящий
  `.env` остаётся жёстким запретом, а `config/skynet.env` — soft-контур; путь с
  `/` (`state/`) матчит `state/db`, но не `prostate/`.
- MCP-органы чувств подключаются через `SKYNET_MCP_<NAME>_COMMAND`
  (`mcp.configured_servers`; `cli._mcp_config` его переиспользует). В
  `config/skynet.env.example` прописаны четыре: DuckDuckGo, Playwright, arXiv, GitHub —
  но **Playwright выключен флагом** `SKYNET_MCP_PLAYWRIGHT_DISABLED=1` (решение владельца,
  2026-10-16). Команда и пакет остаются на диске, снимается одной строкой; серверной
  «службы» у сенсоров нет — это дочерние процессы реактора. Карточка `/status` считает
  сенсоры по тому же конфигу (`status_card.expected_sensors`), поэтому выключенный сенсор
  читается как выключенный, а не как пропавший.
- GitHub запускается `--read-only --toolsets=repos,git,issues`: `--read-only`
  выкидывает все write-инструменты из списка **флагом самого сервера**, а не
  нашим блок-листом; `pull_requests`/`users`/`orgs` дали ноль вызовов за 44 ч.
  Резать тоньше можно тем же сервером — `--tools` (белый список) или
  `--exclude-tools` (чёрный), но это микрооптимизация.
- arXiv внутренней фильтрации инструментов **не имеет** (ни 0.6.3, ни 0.7.2:
  список из 14 инструментов зашит в `server.py`, из CLI только `--storage-path`;
  README прямо относит гейтинг к клиенту). Поэтому arxiv оставлен целиком:
  «нельзя, значит не трогаем».
- MCP измерен на сервере (44 ч, 11 475 событий): вызывались **arxiv** (106),
  **ddg** (41), **github** (9, все read) и `webfetch`; **playwright** — 0 вызовов
  за всё время. Ноль вызовов — это «не было задач», а не «мусор»: playwright не
  удалён, а выключен флагом `SKYNET_MCP_PLAYWRIGHT_DISABLED=1` (2026-10-16) и
  возвращается одной строкой. Не судить о пользе по факту подключения.
- Системный промпт перечисляет органы чувств **динамически**: `Reactor.__init__`
  дописывает блок `SENSES` из реально обнаруженных MCP-инструментов и блок
  `WORKSPACE` с реальным корнем (`reactor.mcp_senses_line`). Явное
  перечисление сохранено намеренно — без него модель игнорирует инструменты, —
  но генерируется из живого инвентаря, а не второй зашитой копией, которая
  разъезжается при смене toolsets.
- **Скиллы — файлы, а в промпте только индекс.** Процедуры лежат в `skills/<имя>.md`
  (необязательный front matter `name`/`description`; без него имя — стем файла,
  описание — первая содержательная строка; `README.md`, `_`-файлы и дотфайлы —
  документация, не скиллы; имена латиницей `[a-z0-9._-]`). `Reactor.__init__`
  дописывает блок `SKILLS` — имя плюс одна строка (≤200 символов на строку, ≤2400
  на блок) — а полное тело отдаёт тул `skill` (`action='list'|'read'`, тело ≤8000
  символов с пометкой об обрезке). Поиск идёт **только по обнаруженному набору**,
  поэтому имя не может стать путём (`../`, абсолютный путь — отказ). Пустой
  каталог = пустого блока нет: кто скиллов не держит, за фичу не платит.
  Мотивация: `procedure` памяти в среднем 255 символов и делит окно поиска
  (24 терма) с фактами, так что длинная пошаговая процедура в память не влезает,
  а в файл влезает; зато и платится за неё только в момент чтения. Пишутся
  скиллы обычными файловыми тулами, как `SOUL.md` (гейта на них нет намеренно),
  а читаются только через `skill`. Тесты — `tests/test_skills.py`.

## Наблюдения (исторический снимок, не текущее состояние)

Раздел сохранён как история; конкретные хеши и счётчики здесь — снимок на дату и
могут быть неверны. Актуальный хеш смотри `git log -1`, числа — в БД и `pytest`.

- **Организм самоулучшается автономно.** За 2026-09-19 он сделал 9
  organism-коммитов за ~6 ч — каждый через `propose_self_improvement` → gate →
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
  бампа `SCHEMA_VERSION` → на сервере было два разных v10. Исправлено и
  задеплоено (миграция `memory-decay-clock`). Урок закреплён инвариантом выше:
  бамп `SCHEMA_VERSION` без имени миграции — баг.
- **Inbox: сообщение владельца — уведомление, а не задача.** Автосоздание
  inbox-задачи (`_select_inbox_work`) убрано; сообщение приходит как
  observation `inbox_notification`, организм сам решает и закрывает его через
  `acknowledge_inbox`.
- **Отказы прогонов — это провайдерная цепочка, а не бюджет времени.**
  Разбор 110 `needs_recovery` на сервере: большинство — исчерпание провайдеров в
  хвосте цепочки, меньшинство — time/output budget. Локально исправлено: таймаут
  openrouter стал connect+idle (общий дедлайн off), `ModelTurn.model_seconds`
  исключает бэкофф, `provider_failures` даёт вердикт `degraded`.

## Снимок 2026-09-24 (сервер)

Проверяемо по серверной БД и git-истории; это наблюдение, не контракт.

- **Провайдер переключён**: боевой режим — `openrouter` (+ ранг `openrouter_glm`),
  money-boost включён; в событиях `fallback_*` новых rungs nemotron/nvidia/ollama
  нет. Rename сделан без смены endpoint.
- **Схема — 12** (`schema_migrations` заканчивается на `(12, run-progress)`),
  `memories.decayed_at` на месте; `runs.heartbeat_at`/`last_progress_at`
  добавлены.
- **Клапан открылся**: `external_seek` отработал — прогон вытащил внешний
  источник (arXiv `2606.13905`, ADORE) и превратил его в измеримую гипотезу;
  после этого планировщик выбрал workstream `engineering`, а не залип на старой
  validation-задаче. `idea_archive` наполнился.
- **Наблюдение о провайдере**: `memory_degraded` («memory loop provider timeout
  after 300.0s») и аномально долгий стартовый вызов планировщика
  (reasoning-heavy, минуты вместо десятков секунд) — на стороне модели, не
  harness: провайдер отвечает на прямой пробе быстро. Держать в уме при
  диагностике «зависаний»: idle-таймаут не горит, пока данные идут.
- **В дереве появились** memory-audit (`skynet/memory_audit.py`) и
  durable-событие `fallback_retry_recovered`; локально набор — 907 тестов.
- **Локальный и серверный main могут расходиться**: на дату снимка локальный был
  на organism-коммите, сервер — на следующем. Перед анализом сверяй
  `git log -1` на обеих сторонах (fetch — через `GIT_SSH_COMMAND` с deploy-ключом
  `config/deploy.env`, дефолтный ключ не подходит).

## Инцидент планировщика (2026-09-20)

- **Судья сломался молча.** Декодер `autonomous_planner._parse` срезал текст
  модели от первой `{` до последней `}` и уничтожал легитимный top-level
  JSON-массив раньше, чем его видел парсер: 15 из 18 попыток планировщика
  (83%) провалились за сутки с `structured model response must be a JSON
  object`.
- **Провал маскировался под «нет новой работы».** `generate()` возвращал `[]`,
  что неотличимо от «портфель пуст», поэтому Reactor молча падал в
  фиксированный пул: две generic fallback-задачи создавались в 29 из 42
  прогонов (69%).
- **Организм не мог починить это сам**: `_parse` жил в gate-protected
  `autonomous_planner.py`. Отсюда разделение — инструмент
  (`skynet/planner_contract.py`) открыт, решения судьи закрыты, а
  `judge_health.check_judge_health` делает деградацию видимой владельцу.
- **Де-маскировка.** Провал парсинга планировщика никогда не записывается как
  «нет новой работы»: это отдельные причины `planner_invalid_response` и
  `planner_provider_error` (эмитятся `autonomous_planner` / `reactor`), и оба
  kind'а защищены от retention как post-mortem.

## Инцидент замка git (2026-10-28)

- **Лок оставил её же read-only git.** `git status --porcelain` в
  `Reactor._worktree_state` шёл с `timeout=5`, а `status` обновляет индекс, то
  есть берёт `.git/index.lock`. Убитый по таймауту git оставил 0-байтный лок
  (28.10 22:18:56), и каждое последующее *письмо* — commit, merge, публикация
  через `propose_self_improvement` — падало на «Unable to create
  .git/index.lock: File exists». Цена: 5 часов, 17 циклов `blocked` против 12
  `completed`, ноль новых коммитов. Исключение гасилось в `return {}`, поэтому
  причина не попала ни в один лог.
- **Правило: read-only git не берёт опциональных локов.** Такие вызовы идут
  через `skynet/git_hygiene.read_only_git` (`GIT_OPTIONAL_LOCKS=0`), а
  `install_git_read_only_environment()` делает это умолчанием процесса в
  `supervisor.start` и `cli.main`. Таймаут 30 с: пять секунд были гонкой, а не
  границей.
- **Мусор убирает сам организм.** `clear_stale_index_lock` — на старте
  супервизора и на каждом пробуждении: пустой лок старше 5 минут, которого
  никто не держит (проверка по `/proc`), удаляется, и в лог идёт событие
  `stale_index_lock_cleared`. Снаружи организм это сделать не может: bash под
  `.git/` hard-denied, а sudo недоступен вовсе.
- **Одобрение, которое нельзя исполнить, — баг гейта.** `NoNewPrivileges=yes` в
  юнитах делает sudo неспособным получить root: ядро применяет флаг раньше
  setuid. Гейт трижды спросил владельца про `rm -f .git/index.lock`, трижды
  получил «Разрешить» и трижды записал `sudo_ran` с `returncode=1`; организму
  осталось объяснить это себе «контейнером с no-new-privileges». Теперь
  `sudo_available()` опрашивается один раз, и гейт отвечает
  `state: "unavailable"` **не спрашивая владельца** и отдавая настоящую причину
  из stderr sudo.
- **Отказ по пути — не повод идти за root.** Hard-denylist `.git/` не снимается
  одобрением, и текст отказа в `tools.py` это проговаривает прямо.

## Документы

- `SOUL.md` — identity, инжектится в system-промпт ReAct и Memory. Авторитетен.
- `skills/*.md` — процедуры: в промпт идёт индекс, тело — по требованию (`skill`).
  Инструкции, а не identity, и не нормативный документ.
- `AGENTS.md` (этот файл) — runbook для coding-агента.
- **Источник истины — код.** `Architecture.md` уничтожен и не восстанавливается:
  это был черновик плана, а не контракт, и `handoff` на него больше не ссылается.
  Никакой документ не нормативен, кроме `SOUL.md`.