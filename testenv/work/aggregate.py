#!/usr/bin/env python3
"""Aggregate efficiency stats from work/logs/*.jsonl into a markdown table."""
import glob, json, statistics

rows = []
total_imgs = 0
total_wall = 0.0
for mf in sorted(glob.glob("/home/dreamchaser/selflabel/testenv/work/logs/*_meta.json")):
    meta = json.load(open(mf))
    ds = meta["dataset"]
    recs = [json.loads(l) for l in open(mf.replace("_meta.json", ".jsonl"))]
    ok = [r for r in recs if r.get("ok")]
    n_ok = len(ok)
    parse_rate = n_ok / len(recs) * 100 if recs else 0
    lat = [r["t_total"] for r in ok if "t_total" in r]
    tok = [r["tok_s"] for r in ok if r.get("tok_s")]
    ptok = sum(r.get("prompt_tokens", 0) for r in ok)
    ctok = sum(r.get("completion_tokens", 0) for r in ok)
    lines = sum(r.get("n_lines", 0) for r in ok)
    rows.append({
        "dataset": ds, "type": meta["type"], "n": len(recs),
        "parse_ok": n_ok, "parse_rate": parse_rate,
        "lat_mean": statistics.mean(lat) if lat else 0,
        "lat_med": statistics.median(lat) if lat else 0,
        "lat_max": max(lat) if lat else 0,
        "tok_s_mean": statistics.mean(tok) if tok else 0,
        "img_per_min": meta["img_per_min"], "wall_s": meta["wall_s"],
        "gpu_max": meta.get("gpu_mem_max_mb"),
        "ptok": ptok, "ctok": ctok, "lines": lines,
    })
    total_imgs += len(recs)
    total_wall += meta["wall_s"]

out = []
out.append("| dataset | type | n | parse_ok | parse% | lat_mean_s | lat_med_s | lat_max_s | gen_tok_s | img/min | wall_s | gpu_max_MB | prompt_tok | gen_tok | out_lines |")
out.append("|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|")
for r in rows:
    out.append(f"| {r['dataset']} | {r['type']} | {r['n']} | {r['parse_ok']} | {r['parse_rate']:.0f}% "
               f"| {r['lat_mean']:.1f} | {r['lat_med']:.1f} | {r['lat_max']:.1f} | {r['tok_s_mean']:.1f} "
               f"| {r['img_per_min']:.1f} | {r['wall_s']:.0f} | {r['gpu_max']} | {r['ptok']} | {r['ctok']} | {r['lines']} |")
tot_tok = sum(r["ctok"] for r in rows)
out.append(f"\nTOTAL: {total_imgs} images, {total_wall:.0f}s wall, overall {total_imgs/total_wall*60:.1f} img/min, "
           f"generated {tot_tok} tokens")
print("\n".join(out))
open("/home/dreamchaser/selflabel/testenv/work/logs/efficiency_table.md", "w").write("\n".join(out) + "\n")
