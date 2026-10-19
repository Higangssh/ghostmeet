"""Recompute CER in results/*.json from the hypotheses kept in results/raw/.

    python -I bench/rescore.py

For a change to the CER normalisation: no engine is rerun, so timings stay as measured.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from common import RAW, RESULTS, cer_parts, load_fleurs  # noqa: E402
from run_stream import build_session  # noqa: E402


def rate(pairs) -> float:
    e = n = 0
    for ref, hyp in pairs:
        de, dn = cer_parts(ref, hyp)
        e, n = e + de, n + dn
    return round(e / n, 4)


def main() -> None:
    refs = {u.id: u.text for u in load_fleurs()}
    session_ref = {}
    for res in sorted(RESULTS.glob("*.json")):
        out = json.loads(res.read_text())
        stem = res.name.removesuffix(".json")
        old = out["cer"]
        if stem.endswith((".utt", ".coldstart")):
            rows = [json.loads(line) for line in (RAW / f"{stem}.jsonl").read_text().splitlines()]
            out["cer"] = rate((refs[r["id"]], r["hyp"]) for r in rows)
            out["cer_no_digits"] = rate(
                (refs[r["id"]], r["hyp"]) for r in rows if not any(c.isdigit() for c in refs[r["id"]])
            )
        else:
            minutes = 36  # every stream/longform result here used the default session
            if minutes not in session_ref:
                session_ref[minutes] = build_session(minutes)[1]
            if stem.endswith(".longform"):
                segs = json.loads((RAW / f"{stem}.hyp.json").read_text())
                hyp = " ".join(t.strip() for _, _, t in segs if t.strip())
            else:
                hyp = json.loads((RAW / f"{stem.removesuffix('.stream')}.passes.json").read_text())["hyp"]
            out["cer"] = rate([(session_ref[minutes], hyp)])
        res.write_text(json.dumps(out, ensure_ascii=False, indent=1))
        print(f"{res.name:55s} {old:.4f} -> {out['cer']:.4f}")


if __name__ == "__main__":
    main()
