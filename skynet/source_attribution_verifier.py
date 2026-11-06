from __future__ import annotations

import re
from typing import Any


class SourceAttributionVerifier:
    """Verify new extraction candidates against their cited sources.

    Adapts the three-stage source-attribution framework of
    arXiv:2605.06635v1 (deterministic citation-claim extraction -> parallel
    cited-source retrieval -> Link Works / Relevant Content / Fact Check
    evaluators) to SkyNet's candidate dict
    (kind/content/confidence/evidence[]/supersedes_memory_id).

    `verify(candidate, source_context)` returns ``(verdict, breakdown)``
    where ``verdict`` is one of:

    - ``"Correct"``            — all retrievable sources support the claim
    - ``"Hallucination"``      — a retrieved source contradicts the claim's key facts
    - ``"RelevanceLow"``       — sources reachable but topically misaligned
    - ``"unverifiable_link"``  — no retrievable source (Link Works failure)

    ``breakdown`` is ``{"verdict", "reason", "sources", "evals", "new_facts"}``
    and is stored on the candidate for audit.
    """

    _DATE = re.compile(r"\b\d{4}-\d{2}-\d{2}\b")
    _NUMBER = re.compile(r"(?<!\d-)\b\d+(?:\.\d+)?(?=[A-Za-z]?\b)(?!-\d)")
    _PROPER = re.compile(r"\b[A-Z][a-zA-Z]+\b")
    _DESCRIPTOR = re.compile(r"^(url|commit|tool|memory|date):\s*(.+)$", re.IGNORECASE)
    _RELEVANCE_THRESHOLD = 0.2

    _GENERIC_ENTITY_SUFFIXES = frozenset(
        {"Corp", "Inc", "Ltd", "LLC", "Co", "Company", "Corporation"}
    )
    _PROPER_STOPWORDS = frozenset(
        {
            "The", "This", "That", "These", "Those", "A", "An", "It", "He", "She",
            "They", "We", "I", "My", "Our", "Your", "His", "Her", "Its",
        }
    )

    # ------------------------------------------------------------------ #
    # Citation extraction (paper Phase 1, candidate-scoped)
    # ------------------------------------------------------------------ #
    def extract_citations(self, evidence: Any) -> list[dict[str, Any]]:
        """Parse candidate ``evidence[]`` into typed source descriptors.

        Recognised prefixes: ``url:``, ``commit:``, ``tool:``, ``memory:``,
        ``date:``. Anything else is treated as a bare, non-fetchable anchor.
        """
        descriptors: list[dict[str, Any]] = []
        if isinstance(evidence, str):
            raw = [evidence]
        elif isinstance(evidence, (list, tuple, set)):
            raw = list(evidence)
        else:
            return descriptors

        seen: set[str] = set()
        for item in raw:
            text = str(item).strip()
            if not text:
                continue
            match = self._DESCRIPTOR.match(text)
            if match:
                dtype = match.group(1).lower()
                value = match.group(2).strip()
                descriptor_id = f"{dtype}:{value}"
            else:
                dtype, value, descriptor_id = "bare", text, f"bare:{text}"
            if descriptor_id in seen:
                continue
            seen.add(descriptor_id)
            descriptors.append(
                {
                    "type": dtype,
                    "value": value,
                    "id": descriptor_id,
                    "raw": text,
                }
            )
        return descriptors

    # ------------------------------------------------------------------ #
    # Key facts (mirrors MemoryTransitionVerifier._key_facts / _contradicts)
    # ------------------------------------------------------------------ #
    def _key_facts(self, text: str) -> set[str]:
        date_tokens = set()
        for d in self._DATE.findall(text):
            date_tokens.update(d.split("-"))
        proper = {
            t
            for t in self._PROPER.findall(text)
            if t not in date_tokens
            and t not in self._GENERIC_ENTITY_SUFFIXES
            and t not in self._PROPER_STOPWORDS
        }
        return set(self._DATE.findall(text)) | set(self._NUMBER.findall(text)) | proper

    def _contradicts(self, old: set[str], new: set[str]) -> bool:
        old_dates = {t for t in old if re.fullmatch(r"\d{4}-\d{2}-\d{2}", t)}
        new_dates = {t for t in new if re.fullmatch(r"\d{4}-\d{2}-\d{2}", t)}
        old_nums = {t for t in old if re.fullmatch(r"\d+(?:\.\d+)?", t) and "-" not in t}
        new_nums = {t for t in new if re.fullmatch(r"\d+(?:\.\d+)?", t) and "-" not in t}
        if old_dates and new_dates and not (old_dates & new_dates):
            return True
        if old_nums and new_nums and not (old_nums & new_nums):
            return True
        old_proper = old - old_dates - old_nums
        new_proper = new - new_dates - new_nums
        return bool(old_proper and new_proper and not (old_proper & new_proper))

    @staticmethod
    def _words(text: str) -> frozenset[str]:
        stop = {
            "the", "a", "an", "and", "or", "but", "if", "then", "is", "are",
            "was", "were", "be", "been", "to", "of", "in", "for", "on", "with",
            "at", "by", "from", "as", "it", "this", "that", "these", "those",
            "note", "per", "prior", "notice", "remains", "authoritative",
            "effective", "completed", "updated", "active",
        }
        return frozenset(
            w.strip(".,:;()[]{}").lower()
            for w in text.split()
            if len(w) > 3 and w.lower() not in stop
        )

    # ------------------------------------------------------------------ #
    # Verification
    # ------------------------------------------------------------------ #
    def verify(
        self, candidate: dict[str, Any], source_context: dict[str, Any] | None
    ) -> tuple[str, dict[str, Any]]:
        """Verify a new candidate against its retrieved sources.

        Returns ``(verdict, breakdown)``.

        When the candidate carries ``target_content``, a target-aware
        faithfulness/coverage/preservation rubric (adapted from the HaluMem
        target-aware memory-updating design, arXiv:2511.03506v3) is applied
        before the source-only check:

        - Faithfulness: candidate must not introduce novel key facts beyond
          source + target.
        - Coverage: all non-date target key fields must be present in the
          candidate.
        - Preservation: key fields removed from the target must not be
          retained in the candidate.

        Source-only contradictions are also overridden when explained by the
        target (legitimate date/number/proper-noun updates authorized by the
        target no longer trigger Hallucination).

        When ``target_content`` is absent, behavior is unchanged (source-only).
        """
        source_context = source_context or {}
        content = (candidate.get("content") or "").strip()
        if not content:
            return "Hallucination", {
                "reason": "empty candidate content",
                "sources": [],
                "evals": {},
                "new_facts": [],
            }

        evidence = candidate.get("evidence")
        descriptors = self.extract_citations(evidence)
        new_facts = self._key_facts(content)

        # Target-aware rubric inputs
        target_content = candidate.get("target_content") or ""
        target_facts = self._key_facts(target_content) if target_content else set()

        # Collect source facts from all accessible sources
        source_entries: list[tuple[str, dict[str, Any], str]] = []
        all_source_facts: set[str] = set()
        for desc in descriptors:
            sid = desc["id"]
            src = source_context.get(sid) or {}
            source_text = src.get("content") or ""
            source_entries.append((sid, desc, source_text))
            if source_text:
                all_source_facts |= self._key_facts(source_text)

        # --- Target-aware rubric (only when target_content is present) ---
        if target_content:
            # 1. Faithfulness: no novel key facts beyond source + target
            allowed = all_source_facts | target_facts
            novel = new_facts - allowed
            if novel:
                return "Hallucination", {
                    "reason": f"candidate introduces unsupported key facts beyond source and target: {sorted(novel)}",
                    "sources": [],
                    "evals": {},
                    "new_facts": sorted(new_facts),
                    "target_facts": sorted(target_facts),
                }

            # 2. Coverage: all non-date target key fields must be present
            missing = target_facts - new_facts
            missing_dates = {f for f in missing if re.fullmatch(r"\d{4}-\d{2}-\d{2}", f)}
            missing_non_dates = missing - missing_dates
            if missing_non_dates:
                return "Hallucination", {
                    "reason": f"candidate omits target key fields: {sorted(missing_non_dates)}",
                    "sources": [],
                    "evals": {},
                    "new_facts": sorted(new_facts),
                    "target_facts": sorted(target_facts),
                }

            # 3. Preservation: fields removed from target must not be retained
            removed_from_target = all_source_facts - target_facts
            retained_removed = new_facts & removed_from_target
            if retained_removed:
                return "Hallucination", {
                    "reason": f"candidate retains fields removed from target: {sorted(retained_removed)}",
                    "sources": [],
                    "evals": {},
                    "new_facts": sorted(new_facts),
                    "target_facts": sorted(target_facts),
                }

        sources: list[dict[str, Any]] = []
        evals: dict[str, dict[str, Any]] = {}
        any_link_works = False
        any_relevant = False
        hallucination = False

        for sid, desc, source_text in source_entries:
            src = source_context.get(sid) or {}
            accessible = bool(src.get("accessible", False))

            link_works = accessible and bool(source_text)
            relevance = self._relevance(content, source_text) if source_text else 0.0
            source_facts = self._key_facts(source_text) if source_text else set()
            fact_check = self._fact_check(
                new_facts,
                source_facts,
                target_facts if target_content else None,
            )

            sources.append(
                {
                    "id": sid,
                    "type": desc["type"],
                    "link_works": link_works,
                    "relevance": round(relevance, 3),
                    "fact_check": fact_check,
                }
            )
            evals[sid] = {
                "link_works": link_works,
                "relevance": relevance,
                "fact_check": fact_check,
            }

            if link_works:
                any_link_works = True
            if relevance >= self._RELEVANCE_THRESHOLD:
                any_relevant = True
            if fact_check is False:
                hallucination = True

        if hallucination:
            verdict, reason = "Hallucination", "candidate contradicts retrieved source key facts"
        elif not any_link_works:
            verdict, reason = (
                "unverifiable_link",
                "no retrievable source; Link Works failed for all citations",
            )
        elif not any_relevant:
            verdict, reason = "RelevanceLow", "sources reachable but topically misaligned with the claim"
        else:
            verdict, reason = "Correct", "claim supported by retrieved sources"

        breakdown = {
            "verdict": verdict,
            "reason": reason,
            "sources": sources,
            "evals": evals,
            "new_facts": sorted(new_facts),
        }
        return verdict, breakdown

    def _relevance(self, claim: str, source_text: str) -> float:
        if not source_text:
            return 0.0
        claim_words = self._words(claim)
        if not claim_words:
            return 0.0
        source_words = self._words(source_text)
        return len(claim_words & source_words) / len(claim_words)

    def _fact_check(
        self,
        new_facts: set[str],
        source_facts: set[str],
        target_facts: set[str] | None = None,
    ) -> bool | None:
        """True if supported, False if contradicted, None if unverifiable.

        When ``target_facts`` is provided, a source↔candidate contradiction is
        treated as authorized (not a contradiction) when the candidate's
        divergent key facts are all present in the target — i.e. a legitimate
        date/number/proper-noun update that the target endorses.
        """
        if not new_facts:
            return None
        if not source_facts:
            return None
        if target_facts:
            explained = self._contradiction_explained_by_target(source_facts, new_facts, target_facts)
            if explained:
                return True
        return not self._contradicts(source_facts, new_facts)

    def _contradiction_explained_by_target(
        self,
        old_facts: set[str],
        new_facts: set[str],
        target_facts: set[str],
    ) -> bool:
        """Return True if every old↔new key-fact contradiction is authorized by target.

        A contradiction in a category (date / number / proper noun) is
        explained when the candidate's divergent facts in that category are all
        present in the target — i.e. the target legitimately replaces the old
        value with the new one.
        """
        old_dates = {t for t in old_facts if re.fullmatch(r"\d{4}-\d{2}-\d{2}", t)}
        new_dates = {t for t in new_facts if re.fullmatch(r"\d{4}-\d{2}-\d{2}", t)}
        target_dates = {t for t in target_facts if re.fullmatch(r"\d{4}-\d{2}-\d{2}", t)}

        old_nums = {
            t for t in old_facts if re.fullmatch(r"\d+(?:\.\d+)?", t) and "-" not in t
        }
        new_nums = {
            t for t in new_facts if re.fullmatch(r"\d+(?:\.\d+)?", t) and "-" not in t
        }
        target_nums = {
            t for t in target_facts if re.fullmatch(r"\d+(?:\.\d+)?", t) and "-" not in t
        }

        old_proper = old_facts - old_dates - old_nums
        new_proper = new_facts - new_dates - new_nums
        target_proper = target_facts - target_dates - target_nums

        date_ok = not (old_dates and new_dates and not (old_dates & new_dates))
        date_ok = date_ok or bool(new_dates and new_dates <= target_dates)
        num_ok = not (old_nums and new_nums and not (old_nums & new_nums))
        num_ok = num_ok or bool(new_nums and new_nums <= target_nums)
        proper_ok = not (old_proper and new_proper and not (old_proper & new_proper))
        proper_ok = proper_ok or bool(new_proper and new_proper <= target_proper)

        return date_ok and num_ok and proper_ok
