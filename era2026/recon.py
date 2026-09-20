"""
Phase 0 data reconnaissance for the 2026 era.

Answers, from real data rather than assumption:
  - which 2026 rounds are loadable
  - what the lap and telemetry schemas look like, versus a 2024 reference
  - whether the DRS channel is gone, present-but-dead, or repurposed
  - whether anything exposes Overtake Mode, energy deployment or active aero
  - what race control says about the new mechanisms

Run this on your own machine. The Claude sandboxes cannot reach
livetiming.formula1.com, so this is the one step that has to be local.

    pip install fastf1
    python era2026/recon.py

Writes era2026/recon_report.md next to this file.
"""

from __future__ import annotations

import json
import sys
import traceback
import warnings
from datetime import datetime, timezone
from pathlib import Path

warnings.filterwarnings("ignore")

HERE = Path(__file__).resolve().parent
CACHE = HERE / ".fastf1_cache"
REPORT = HERE / "recon_report.md"

REFERENCE_YEAR = 2024
REFERENCE_ROUND = 16  # Italian GP, conventional weekend
SAMPLE_2026_ROUND = None  # picked automatically: latest completed conventional round

KEYWORDS = ["override", "overtake mode", "overtake", "boost", "active aero",
            "aero mode", "drs", "energy", "deploy", "mgu", "ers"]

out: list[str] = []
facts: dict = {}


def say(line: str = "") -> None:
    print(line)
    out.append(line)


def section(title: str) -> None:
    say()
    say(f"## {title}")
    say()


def guarded(label: str, fn):
    """Run a check; never let one failure kill the report."""
    try:
        return fn()
    except Exception as exc:
        say(f"- **{label}: FAILED** `{type(exc).__name__}: {exc}`")
        facts.setdefault("failures", []).append({"check": label, "error": repr(exc)})
        return None


def main() -> int:
    import fastf1
    import pandas as pd

    CACHE.mkdir(parents=True, exist_ok=True)
    fastf1.Cache.enable_cache(str(CACHE))

    say("# 2026 data reconnaissance")
    say()
    say(f"- generated: `{datetime.now(timezone.utc).isoformat(timespec='seconds')}`")
    say(f"- fastf1: `{fastf1.__version__}`")
    say(f"- python: `{sys.version.split()[0]}`")
    facts["fastf1_version"] = fastf1.__version__

    # ---------------------------------------------------------------- schedule
    section("1. 2026 schedule and which rounds are complete")

    sched = fastf1.get_event_schedule(2026, include_testing=False)
    now = pd.Timestamp.utcnow().tz_localize(None)
    sched = sched[sched["RoundNumber"] > 0].copy()
    sched["done"] = pd.to_datetime(sched["EventDate"]) < now

    done = sched[sched["done"]]
    say(f"- rounds on calendar: **{len(sched)}**")
    say(f"- rounds complete: **{len(done)}**")
    say()
    say("| R | Event | Date | Format | Complete |")
    say("|---|---|---|---|---|")
    for _, r in sched.iterrows():
        say(f"| {int(r['RoundNumber'])} | {r['EventName']} | "
            f"{pd.to_datetime(r['EventDate']).date()} | {r['EventFormat']} | "
            f"{'yes' if r['done'] else 'no'} |")

    facts["rounds_total"] = int(len(sched))
    facts["rounds_complete"] = int(len(done))

    # pick the sample round: latest completed conventional weekend
    target = SAMPLE_2026_ROUND
    if target is None:
        conv = done[done["EventFormat"] == "conventional"]
        pool = conv if len(conv) else done
        if not len(pool):
            say()
            say("**No completed 2026 rounds. Nothing further to check.**")
            return 1
        target = int(pool.iloc[-1]["RoundNumber"])
    say()
    say(f"Sample round for deep inspection: **R{target}**")
    facts["sample_round"] = target

    # ---------------------------------------------------------------- loading
    section("2. Session loading")

    def load(year, rnd, telemetry=True):
        s = fastf1.get_session(year, rnd, "R")
        s.load(telemetry=telemetry, weather=True, messages=True)
        return s

    s26 = guarded(f"load 2026 R{target}", lambda: load(2026, target))
    s24 = guarded(f"load {REFERENCE_YEAR} R{REFERENCE_ROUND}",
                  lambda: load(REFERENCE_YEAR, REFERENCE_ROUND))

    if s26 is None:
        say()
        say("**2026 session would not load. Everything below is unavailable.**")
        say("If the 2024 reference loaded fine, this is a 2026 data availability")
        say("problem rather than a network or install problem.")
        write_report()
        return 1

    for tag, s in (("2026", s26), (str(REFERENCE_YEAR), s24)):
        if s is None:
            continue
        guarded(f"{tag} basic shape", lambda s=s, tag=tag: say(
            f"- **{tag}**: {len(s.laps)} lap rows, {s.laps['Driver'].nunique()} drivers, "
            f"event `{s.event['EventName']}`"))

    # ---------------------------------------------------------------- schemas
    section("3. Lap schema, 2026 versus 2024")

    def cols(s):
        return list(s.laps.columns) if s is not None else []

    c26, c24 = cols(s26), cols(s24)
    facts["lap_columns_2026"] = c26
    if c24:
        only26 = [c for c in c26 if c not in c24]
        only24 = [c for c in c24 if c not in c26]
        say(f"- columns 2026: **{len(c26)}**, {REFERENCE_YEAR}: **{len(c24)}**")
        say(f"- new in 2026: `{only26 or 'none'}`")
        say(f"- gone from 2026: `{only24 or 'none'}`")
        facts["lap_columns_new_in_2026"] = only26
        facts["lap_columns_missing_in_2026"] = only24
    say()
    say("2026 lap columns:")
    say()
    say("```")
    say(", ".join(c26))
    say("```")

    # ------------------------------------------------------------- telemetry
    section("4. Car telemetry channels, and the fate of DRS")

    def car_channels(s, tag):
        drv = s.drivers[0]
        car = s.car_data[drv]
        chans = list(car.columns)
        say(f"- **{tag}** channels ({len(chans)}): `{', '.join(chans)}`")
        facts[f"car_channels_{tag}"] = chans
        if "DRS" in car.columns:
            vc = car["DRS"].value_counts().sort_index()
            dist = {int(k): int(v) for k, v in vc.items()}
            nonzero = {k: v for k, v in dist.items() if k != 0}
            say(f"  - `DRS` present. value counts: `{dist}`")
            if not nonzero:
                say("  - **DRS channel is present but entirely zero: dead field.**")
            else:
                say(f"  - **DRS channel carries non-zero values: {nonzero}**")
                say("    Investigate before assuming it is dead.")
            facts[f"drs_distribution_{tag}"] = dist
        else:
            say("  - `DRS` column absent entirely.")
            facts[f"drs_distribution_{tag}"] = None
        return chans

    ch26 = guarded("2026 car channels", lambda: car_channels(s26, "2026"))
    ch24 = guarded(f"{REFERENCE_YEAR} car channels",
                   lambda: car_channels(s24, str(REFERENCE_YEAR))) if s24 else None

    if ch26 and ch24:
        say()
        say(f"- channels new in 2026: `{[c for c in ch26 if c not in ch24] or 'none'}`")
        say(f"- channels gone in 2026: `{[c for c in ch24 if c not in ch26] or 'none'}`")

    say()
    say("Position data channels:")
    guarded("2026 position channels", lambda: say(
        f"- `{', '.join(s26.pos_data[s26.drivers[0]].columns)}`"))

    say()
    say("Weather columns:")
    guarded("2026 weather", lambda: say(
        f"- `{', '.join(s26.weather_data.columns)}`"))

    # -------------------------------------------------------- race control
    section("5. Race control messages: any sign of the new mechanisms")

    def rcm(s, tag):
        m = s.race_control_messages
        if m is None or not len(m):
            say(f"- **{tag}**: no race control messages.")
            return
        col = "Message" if "Message" in m.columns else m.columns[-1]
        text = m[col].astype(str)
        say(f"- **{tag}**: {len(m)} messages. Keyword hits:")
        hits = {}
        for kw in KEYWORDS:
            n = int(text.str.lower().str.contains(kw, regex=False).sum())
            if n:
                hits[kw] = n
        say(f"  - `{hits or 'no keyword matches'}`")
        facts[f"rcm_keyword_hits_{tag}"] = hits
        for kw in ("override", "overtake mode", "active aero", "boost"):
            sub = text[text.str.lower().str.contains(kw, regex=False)]
            for msg in sub.head(3):
                say(f"  - _{kw}_: `{msg[:180]}`")

    guarded("2026 race control", lambda: rcm(s26, "2026"))
    if s24:
        guarded(f"{REFERENCE_YEAR} race control", lambda: rcm(s24, str(REFERENCE_YEAR)))

    # ------------------------------------------------- all rounds smoke test
    section("6. Every completed 2026 round, laps only")

    say("| R | Event | Status | Lap rows | Drivers |")
    say("|---|---|---|---|---|")
    totals = {"ok": 0, "rows": 0}
    per_round = []
    for _, r in done.iterrows():
        rnd, name = int(r["RoundNumber"]), r["EventName"]
        try:
            s = fastf1.get_session(2026, rnd, "R")
            s.load(telemetry=False, weather=False, messages=False)
            n, d = len(s.laps), s.laps["Driver"].nunique()
            say(f"| {rnd} | {name} | ok | {n} | {d} |")
            totals["ok"] += 1
            totals["rows"] += n
            per_round.append({"round": rnd, "event": name, "status": "ok",
                              "lap_rows": int(n), "drivers": int(d)})
        except Exception as exc:
            say(f"| {rnd} | {name} | FAILED | - | - |")
            per_round.append({"round": rnd, "event": name, "status": "failed",
                              "error": repr(exc)})
    say()
    say(f"- loaded **{totals['ok']}/{len(done)}** completed rounds, "
        f"**{totals['rows']}** lap rows total")
    facts["per_round"] = per_round

    # -------------------------------------------------------------- verdict
    section("7. What this means")

    drs26 = facts.get("drs_distribution_2026")
    if drs26 is None:
        say("- DRS channel **absent**. Drop all DRS features, no ambiguity.")
    elif not any(k != 0 for k in drs26):
        say("- DRS channel **present but dead**. Drop all DRS features.")
    else:
        say("- DRS channel **carries values**. Needs a closer look before dropping.")

    hits = facts.get("rcm_keyword_hits_2026") or {}
    new_mech = {k: v for k, v in hits.items()
                if k in ("override", "overtake mode", "active aero", "boost")}
    if new_mech:
        say(f"- Race control mentions the new mechanisms: `{new_mech}`. "
            "Worth mining as an event stream even if there is no telemetry channel.")
    else:
        say("- No race control mentions of Overtake Mode or active aero. "
            "Expect no direct feature for the new mechanism; proxy it with "
            "gap-at-detection-point and speed-trap deltas instead.")

    say(f"- {facts.get('rounds_complete')} completed rounds is the in-era sample. "
        "Confirms the small-sample design.")

    write_report()
    return 0


def write_report() -> None:
    REPORT.write_text("\n".join(out) + "\n", encoding="utf-8")
    (HERE / "recon_facts.json").write_text(json.dumps(facts, indent=2, default=str),
                                           encoding="utf-8")
    print()
    print(f"Wrote {REPORT}")
    print(f"Wrote {HERE / 'recon_facts.json'}")


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception:
        traceback.print_exc()
        write_report()
        sys.exit(2)
