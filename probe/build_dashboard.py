#!/usr/bin/env python3
"""Build results/experiments/dashboard.html from the config-driven sweep. Self-contained (stdlib only, inline
CSS/JS, no deps). TABBED BY ENCODER; within each tab one BIG table: rows = experiments (one method x one token
setting), columns = the 5 families (disease=macro AUROC; size/density/location/radiomics=mean R2, best epoch).
Color-coded (red->green), sortable. Reads results/experiments/exp_*/{config.yaml, {enc}_{fam}.csv}.

  python probing/build_dashboard.py        # -> results/experiments/dashboard.html
"""
import csv, glob, html, os
from pathlib import Path
import yaml

ROOT = Path(__file__).resolve().parent.parent
EXPD = ROOT / "results" / "experiments"
FAMS = ["disease", "size", "density", "location", "radiomics"]
ENCODERS = ["colipri", "ct_clip", "btb3d"]
TOK = {"colipri": {2: 1728, 3: 512, 4: 216}, "ct_clip": {2: 1728, 3: 512, 4: 216},
       "btb3d": {2: 4096, 3: 1331, 4: 512}}
ENC_DIM = {"colipri": 768, "ct_clip": 512, "btb3d": 18}   # per-token channel dim D (channel-preserving methods)

# ---- concat-pack (old probe2 A/shared) baseline: results/experiments/{fam}_full/colipri_{fam}_dout4096.csv ----
# METRIC ALIGNMENT: recompute each family aggregate over the SAME target set the new sweep uses. Only LOCATION
# differs (concat used 4 targets incl. the since-DROPPED diaphragm_asym at index 3 -> keep [0,1,2]); the others
# match (size 5 / density 4 / radiomics 3), disease uses macro. concat is colipri-only, D = C*r^3 (dim-heavy).
CONCAT_ALIGN = {"disease": "macro", "size": [0, 1, 2, 3, 4], "density": [0, 1, 2, 3],
                "location": [0, 1, 2], "radiomics": [0, 1, 2]}
CONCAT_N = {2: 1728, 4: 216}                              # colipri token count after pack r
CONCAT_D = {2: 768 * 8, 4: 768 * 64}                      # concat dim = C*r^3 = 6144 / 49152


def best(csv_path):                                       # (best aggregate score, epoch) or (None, 0)
    if not os.path.exists(csv_path):
        return None, 0
    c = list(csv.DictReader(open(csv_path)))
    agg = "macro" if any(r["target"] == "macro" for r in c) else "mean"
    vals = [(float(r["score"]), int(r["epoch"])) for r in c
            if r["target"] == agg and r["score"] not in ("", "nan")]
    return max(vals) if vals else (None, 0)


def collect():
    rows = []
    for d in sorted(glob.glob(str(EXPD / "exp_*"))):
        cfgp = Path(d) / "config.yaml"
        if not cfgp.exists():
            continue
        cfg = yaml.safe_load(open(cfgp))
        enc = cfg["encoder"]; method = cfg["compression"]["method"]
        params = cfg["compression"].get("params", {})
        r = params.get("r"); budget = cfg["compression"].get("budget")
        tok = budget if budget else TOK.get(enc, {}).get(r, 0)
        Cd = ENC_DIM.get(enc, 768)
        if method == "packcrop":                              # N x D: D = kept channels (capped at C*r^3)
            dim = min(int(params.get("dkeep", Cd)), Cd * (r ** 3))
        elif method == "pack":                                # concat: D = C*r^3 (dim-heavy)
            dim = Cd * (r ** 3)
        else:                                                 # channel-preserving methods
            dim = Cd
        rec = {"exp": os.path.basename(d), "num": os.path.basename(d).split("_")[1],
               "method": method, "enc": enc, "tokens": tok, "dim": dim, "ep": 0}
        for fam in FAMS:
            s, ep = best(Path(d) / f"{enc}_{fam}.csv")
            rec[fam] = s; rec["ep"] = max(rec["ep"], ep)
        rows.append(rec)
    rows += load_concat()
    return rows


def _concat_best(csv_path, pack, spec):
    """Best-epoch aligned aggregate for one concat family at one pack. spec='macro' or a list of target indices."""
    if not os.path.exists(csv_path):
        return None, 0
    c = [r for r in csv.DictReader(open(csv_path)) if r["pack"] == str(pack) and r["score"] not in ("", "nan")]
    by_ep = {}                                            # epoch -> {target: score}
    for r in c:
        by_ep.setdefault(int(r["epoch"]), {})[str(r["target"])] = float(r["score"])
    best = (None, 0)
    for ep, d in by_ep.items():
        if spec == "macro":
            v = d.get("macro")
        else:
            vals = [d[str(i)] for i in spec if str(i) in d]
            v = sum(vals) / len(vals) if len(vals) == len(spec) else None
        if v is not None and (best[0] is None or v > best[0]):
            best = (v, ep)
    return best


def load_concat():
    rows = []
    for pack in (2, 4):
        rec = {"exp": f"concat-pack r{pack}", "num": f"cat·r{pack}", "method": "pack(concat)",
               "enc": "colipri", "tokens": CONCAT_N[pack], "dim": CONCAT_D[pack], "ep": 0}
        any_data = False
        for fam in FAMS:
            csvp = EXPD / f"{fam}_full" / f"colipri_{fam}_dout4096.csv"
            s, ep = _concat_best(csvp, pack, CONCAT_ALIGN[fam])
            rec[fam] = s; rec["ep"] = max(rec["ep"], ep); any_data = any_data or s is not None
        if any_data:
            rows.append(rec)
    return rows


def color(v, lo, hi):                                     # per-COLUMN normalization (lo/hi = that column's min/max)
    if v is None:
        return "#2a2d34", "#666", "-"
    t = 0.5 if hi - lo < 1e-6 else max(0.0, min(1.0, (v - lo) / (hi - lo)))
    rr, gg = int(200 * (1 - t) + 40 * t), int(55 * (1 - t) + 175 * t)
    return f"rgb({rr},{gg},70)", "#fff", f"{v:.3f}"


HEAD = ["exp", "method", "tokens", "dim", "ep"] + FAMS


def _sel(enc, kind, vals):
    opts = "".join(f'<option value="{v}">{v}</option>' for v in sorted(set(vals)))
    return (f'{kind}: <select onchange="filtPane(\'{enc}\')" id="f-{enc}-{kind}">'
            f'<option value="">(all)</option>{opts}</select>')


def table_for(rows, enc):
    rng = {}                                              # per-family min/max over THIS tab's present values
    for fam in FAMS:
        vals = [r[fam] for r in rows if r[fam] is not None]
        rng[fam] = (min(vals), max(vals)) if vals else (0.0, 1.0)
    trs = []
    for d in sorted(rows, key=lambda x: (x["method"], -x["tokens"])):
        cells = (f'<td class=l title="{html.escape(d["exp"])}">{html.escape(d["num"])}</td>'
                 f'<td>{html.escape(d["method"])}</td>'
                 f'<td class=num>{d["tokens"]}</td><td class=num style="color:#89b">{d["dim"]}</td>'
                 f'<td class=num style="color:{"#7d7" if d["ep"]>=10 else "#e90"}">{d["ep"]}</td>')
        for fam in FAMS:
            bg, fg, txt = color(d[fam], *rng[fam])
            cells += f'<td class=num style="background:{bg};color:{fg};font-weight:600">{txt}</td>'
        trs.append(f'<tr data-tokens="{d["tokens"]}" data-dim="{d["dim"]}">{cells}</tr>')
    ths = "".join(f'<th onclick="sortT(this,{i})">{h}</th>' for i, h in enumerate(HEAD))
    bar = (f'<div class=filt>{_sel(enc, "tokens", [r["tokens"] for r in rows])}'
           f'&nbsp;&nbsp;{_sel(enc, "dim", [r["dim"] for r in rows])}'
           f'&nbsp;&nbsp;<span class=hint>(filter rows by token count N / dim D)</span></div>')
    return f'{bar}<table class=t id="t-{enc}"><thead><tr>{ths}</tr></thead><tbody>{"".join(trs)}</tbody></table>'


def build():
    rows = collect()
    tabs, panes = [], []
    for i, enc in enumerate(ENCODERS):
        er = [r for r in rows if r["enc"] == enc]
        act = " active" if i == 0 else ""
        tabs.append(f'<button class="tab{act}" onclick="showTab(\'{enc}\')">{enc} ({len(er)})</button>')
        panes.append(f'<div class="pane{act}" id="pane-{enc}">{table_for(er, enc) if er else "<p>(no runs yet)</p>"}</div>')
    doc = f"""<!doctype html><meta charset=utf-8><title>Compression Sweep Dashboard</title>
<style>
body{{font-family:system-ui,Arial;margin:22px;background:#0f1115;color:#e6e6e6}}
h1{{font-size:20px;margin:0 0 4px}} .sub{{color:#9aa;font-size:13px;margin-bottom:14px}}
.tab{{background:#1a1d24;color:#bcd;border:1px solid #333;padding:7px 16px;cursor:pointer;font-size:14px;border-radius:6px 6px 0 0;margin-right:3px}}
.tab.active{{background:#2b3550;color:#fff;font-weight:600}}
.pane{{display:none;border:1px solid #2b3550;border-radius:0 6px 6px 6px;padding:10px}} .pane.active{{display:block}}
table.t{{border-collapse:collapse;width:100%;font-size:13px}}
th,td{{padding:5px 10px;border-bottom:1px solid #2a2d34;text-align:left}}
td.num{{text-align:right;font-variant-numeric:tabular-nums}} td.l{{font-family:ui-monospace,monospace;color:#bcd}}
th{{position:sticky;top:0;background:#1a1d24;cursor:pointer;user-select:none;white-space:nowrap}}
th:hover{{background:#252a33}}
.filt{{margin:2px 0 12px;font-size:13px;color:#9aa}}
.filt select{{background:#1a1d24;color:#e6e6e6;border:1px solid #333;padding:4px 6px;border-radius:5px;margin:0 2px}}
.hint{{color:#667;font-size:12px}}
</style>
<h1>Compression Sweep Dashboard <span class=sub>— rate&ndash;distortion N&times;D probe, family-shared (A)</span></h1>
<div class=sub>rows = experiments (method &times; token budget N &times; dim D); columns = 5 families. disease = macro AUROC;
size/density/location/radiomics = mean R&sup2; (best epoch). <b>Color is per-column</b> (each family shaded by its own
min&rarr;max within the tab, red&rarr;green). <b>ep</b> orange = still running (&lt;10). Click a header to sort. {len(rows)} runs.</div>
<div>{"".join(tabs)}</div>
{"".join(panes)}
<script>
function showTab(e){{document.querySelectorAll('.tab').forEach(t=>t.classList.toggle('active',t.textContent.startsWith(e)));
 document.querySelectorAll('.pane').forEach(p=>p.classList.toggle('active',p.id=='pane-'+e));}}
function filtPane(e){{let tk=document.getElementById('f-'+e+'-tokens').value,dm=document.getElementById('f-'+e+'-dim').value;
 document.querySelectorAll('#t-'+e+' tbody tr').forEach(r=>{{
  let ok=(!tk||r.dataset.tokens===tk)&&(!dm||r.dataset.dim===dm);r.style.display=ok?'':'none';}});}}
function sortT(th,i){{let tb=th.closest('table').querySelector('tbody'),rs=[...tb.rows];
 let num=!isNaN(parseFloat(rs[0]?.cells[i].innerText));
 rs.sort((a,b)=>{{let x=a.cells[i].innerText,y=b.cells[i].innerText;
  return num?(parseFloat(y)||-9)-(parseFloat(x)||-9):x.localeCompare(y);}});
 rs.forEach(r=>tb.appendChild(r));}}
</script>"""
    out = EXPD / "dashboard.html"
    out.write_text(doc)
    print(f"[dashboard] {len(rows)} runs across {len(ENCODERS)} encoders -> {out}")


if __name__ == "__main__":
    build()
