"""Independent data QA for loadlab/data (dataset_deploy / dataset_pods).

    python runner/qa_dataset.py      # -> data/QA_REPORT.md, data/qa_results.json, data/plots/*.png

Per run: coverage gaps, NaN ratios, monotonic timestamps, counter monotonicity, label/window
consistency, and the physical signature expected for the run's fault or scenario class.
Thresholds are deliberately loose and documented next to each check; numbers are in the report.
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
PLOTS = DATA / "plots"
STEP = 5
SLO_P95, SLO_ERR = 0.3, 0.01
REQ_CORES = 0.25

# reference palette (dataviz skill, categorical slots 1-4, light mode) + neutral ink
C1, C2, C3, C4 = "#2a78d6", "#eb6834", "#1baf7a", "#eda100"
INK, INK2, GRID, FAULT = "#0b0b0b", "#52514e", "#e6e5e1", "#d9d7d0"
plt.rcParams.update({"axes.edgecolor": INK2, "axes.labelcolor": INK2, "xtick.color": INK2,
                     "ytick.color": INK2, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6,
                     "axes.spines.top": False, "axes.spines.right": False, "font.size": 8,
                     "lines.linewidth": 1.4, "figure.facecolor": "white", "axes.titlesize": 9,
                     "axes.titlecolor": INK, "legend.frameon": False, "legend.fontsize": 7})

DEP_CORE = ["rps", "inflight", "p50", "p95", "p99", "error_rate", "cpu_cores_used", "cpu_util_vs_request",
            "throttle_ratio", "mem_ws_bytes_total", "mem_ws_max_pod_frac_of_limit", "replicas_desired",
            "replicas_ready", "restarts_total", "oomkills_total", "crashloop_pods", "hpa_desired"]
POD_SIG = ["cpu", "throttle", "mem_ws", "mem_frac_limit", "restarts", "rps", "p95", "ready", "cpu_proc"]


def med(s):
    s = pd.Series(s).dropna()
    return float(s.median()) if len(s) else float("nan")


def win(g, a, b):
    return g[(g.seconds_since_start >= a) & (g.seconds_since_start <= b)]


def worst(*xs):
    order = {"PASS": 0, "WARN": 1, "FAIL": 2}
    return max(xs, key=lambda x: order[x])


def load_faults(run_id):
    p = ROOT / "runs" / run_id / "faults.jsonl"
    return [json.loads(x) for x in p.read_text().splitlines() if x.strip()] if p.exists() else []


# ------------------------------------------------------------------ integrity
def integrity(g, gp, sc):
    ev, st = [], "PASS"
    d = np.diff(g.ts.to_numpy())
    mono = bool((d > 0).all())
    gaps = int((d > STEP).sum())
    dups = int(g.ts.duplicated().sum())
    pdups = int(gp.duplicated(["ts", "pod"]).sum())
    pmono = bool(gp.groupby("pod").ts.apply(lambda s: (np.diff(s.to_numpy()) > 0).all()).all())
    first, last = g.seconds_since_start.min(), g.seconds_since_start.max()
    cover = (last - first) / sc["duration_s"]
    if not (mono and pmono) or dups or pdups:
        st = "FAIL"
    if gaps or cover < 0.97 or first > 15:
        st = worst(st, "WARN")
    ev.append(f"rows {len(g)}, gaps {gaps}, dups {dups}/{pdups}, monotonic {mono and pmono}, "
              f"span {first:.0f}-{last:.0f}s of {sc['duration_s']}s ({cover:.0%})")
    # NaN ratios
    nan_core = {c: float(g[c].isna().mean()) for c in DEP_CORE if g[c].isna().any()}
    if nan_core:
        st = "FAIL"
        ev.append("deploy NaN " + ", ".join(f"{k} {v:.1%}" for k, v in nan_core.items()))
    mg = float(g.mem_growth_bytes_per_s.isna().mean())
    pn = {c: float(gp[c].isna().mean()) for c in POD_SIG}
    pmax = max(pn, key=pn.get)
    if mg > 0.16 or pn[pmax] > 0.10:
        st = worst(st, "WARN")
    ev.append(f"mem_growth NaN {mg:.1%}; worst pod NaN {pmax} {pn[pmax]:.1%}")
    # counters
    for c in ["restarts_total", "oomkills_total"]:
        if (g[c].diff() < 0).any():
            st = "FAIL"
            ev.append(f"{c} decreases")
    # pods per ts == replicas_ready (ready pods)
    rp = gp[gp.ready == 1].groupby("ts").size().reindex(g.ts).fillna(0).to_numpy()
    mism = float((np.abs(rp - g.replicas_ready.to_numpy()) > 0.5).mean())
    if mism > 0.10:
        st = worst(st, "WARN")
    ev.append(f"ready-pod rows vs replicas_ready mismatch {mism:.1%}")
    # bounds
    bad = int(((g.error_rate < 0) | (g.error_rate > 1) | (g.throttle_ratio < 0) | (g.throttle_ratio > 1)
               | (g.p50 > g.p95 + 1e-9) | (g.p95 > g.p99 + 1e-9) | (g.rps < 0)).sum())
    zero_mem = int((gp.mem_ws == 0).sum())
    if bad:
        st = "FAIL"
    if zero_mem:
        st = worst(st, "WARN")
    ev.append(f"out-of-range rows {bad}; pod rows with mem_ws==0 {zero_mem}")
    return st, ev, {"nan_mem_growth": mg, "pod_nan": pn, "gaps": gaps, "ready_mismatch": mism,
                    "zero_mem_pod_rows": zero_mem}


def label_check(g, faults):
    st, ev = "PASS", []
    for f in faults:
        end = f["start_rel_s"] + 30 if f["type"] == "pod_kill" else f["end_rel_s"]
        inwin = win(g, f["start_rel_s"], end)
        lab = inwin.fault_ids.fillna("").str.contains(f["fault_id"], regex=False)
        outside = g[~g.index.isin(inwin.index)].fault_ids.fillna("").str.contains(f["fault_id"], regex=False)
        if lab.mean() < 0.99 or outside.any():
            st = "FAIL"
        ev.append(f"{f['type']}: {int(lab.sum())}/{len(inwin)} rows in window labelled, {int(outside.sum())} outside")
    return st, ev


# ------------------------------------------------------------------ signatures
def sig_capacity(g, sc):
    p = sc["traffic"]["params"]
    n = sc["scaling"]["replicas"]
    steps = []
    for i in range(int(sc["duration_s"] // p["step_s"])):
        a = i * p["step_s"] + 15
        w = win(g, a, a + p["step_s"] - 15)
        if not len(w):
            continue
        off = (p["start_rps"] + i * p["step_rps"]) / n
        steps.append({"offered_per_pod": off, "achieved_per_pod": med(w.rps) / n, "p95": med(w.p95),
                      "cpu_util": med(w.cpu_util_vs_request), "err": med(w.error_rate),
                      "inflight": med(w.inflight)})
    s = pd.DataFrame(steps)
    ok = s[s.p95 < SLO_P95]
    bad = s[s.p95 >= SLO_P95]
    last_ok = float(ok.offered_per_pod.max()) if len(ok) else float("nan")
    first_bad = float(bad.offered_per_pod.min()) if len(bad) else float("nan")
    p95_before = float(ok.p95.max()) if len(ok) else float("nan")
    p95_after = float(s.p95.iloc[-1])
    sat = float(s.achieved_per_pod.max())
    # "clear knee": sub-SLO p95 below the knee, >=10x jump within 2 steps after, throughput plateau
    sharp = (p95_before < 0.15) and (p95_after > 10 * p95_before) and (first_bad - last_ok <= 2.01 * p["step_rps"] / n)
    st = "PASS" if sharp and 15 <= last_ok <= 22 else "WARN"
    ev = [f"last SLO-ok step {last_ok:.1f} RPS/pod (p95 {p95_before*1e3:.0f} ms), first breach {first_bad:.1f} RPS/pod, "
          f"final p95 {p95_after:.1f}s, max achieved {sat:.1f} RPS/pod, final err {s.err.iloc[-1]:.2f}"]
    return st, ev, s


def sig_hpa(g, sc):
    rmax = g.replicas_ready.max()
    viol = (g.p95 > SLO_P95) | (g.error_rate > SLO_ERR)
    # longest consecutive violation streak
    streak = int(viol.groupby((~viol).cumsum()).sum().max()) * STEP
    corr = float(np.corrcoef(g.rps, g.hpa_desired)[0, 1]) if g.hpa_desired.std() > 0 else float("nan")
    tail = g.iloc[-12:]
    scaled_in = bool((g.replicas_desired.diff() < 0).any())
    st = "PASS" if rmax > 1 and streak <= 120 and med(tail.p95) < SLO_P95 else "WARN"
    ev = [f"replicas 1->{rmax:.0f}, scale-in seen {scaled_in}, viol rows {int(viol.sum())}, longest breach {streak}s, "
          f"corr(rps,hpa_desired) {corr:.2f}, end p95 {med(tail.p95)*1e3:.0f} ms"]
    return st, ev


def sig_under(g, sc):
    hot = g[g.rps >= g.rps.quantile(0.8)]
    p95h, cpuh = med(hot.p95), med(hot.cpu_util_vs_request)
    viol = float(((g.p95 > SLO_P95) | (g.error_rate > SLO_ERR)).mean())
    st = "PASS" if p95h > SLO_P95 and cpuh > 1.4 else "WARN"
    return st, [f"top-20% load rows: p95 {p95h:.2f}s, cpu/request {cpuh:.2f}, throttle {med(hot.throttle_ratio):.2f}; "
                f"SLO-violating rows {viol:.0%}"]


def sig_over(g, sc):
    w = win(g, 30, 1e9)
    cpu, p95 = med(w.cpu_util_vs_request), med(w.p95)
    st = "PASS" if cpu < 0.35 and p95 < 0.1 else "WARN"
    return st, [f"cpu/request median {cpu:.2f} (HPA target 0.70), p95 {p95*1e3:.0f} ms, "
                f"rps/pod {med(w.rps)/sc['scaling']['replicas']:.1f} (cap 18)"]


def pre_win(g, f, lead=120):
    return win(g, max(20, f["start_rel_s"] - lead), f["start_rel_s"] - 10)


def sig_fault(g, gp, f, sc):
    t = f["type"]
    a, b = f["start_rel_s"], f["end_rel_s"]
    pre, dur = pre_win(g, f), win(g, a + 30, b)
    ev = []
    if t == "memory_leak":
        pod = f["pods"][0]
        pp = gp[gp.pod == pod]
        pdur = win(pp, a, b)
        rate = f["params"]["leak_kb_per_s"] * 1024
        # slope of the target pod's working set during the first leak segment (before any restart)
        seg = pdur[pdur.restarts == pdur.restarts.min()].dropna(subset=["mem_ws"])
        slope = float(np.polyfit(seg.seconds_since_start, seg.mem_ws, 1)[0]) if len(seg) > 5 else float("nan")
        m0, m1 = med(win(pp, a - 60, a).mem_frac_limit), float(pdur.mem_frac_limit.max())
        oom = int(g.oomkills_total.max())
        rst = float(pp.restarts.max())
        headroom = (1 - m0) * 256 * 2**20
        expect_oom = rate * (b - a) > headroom
        t_oom = None
        if oom:
            t_oom = float(g[g.oomkills_total > 0].seconds_since_start.min())
        p95_ratio = med(dur.p95) / med(pre.p95)
        ok = slope > 0.6 * rate and m1 > m0 + 0.1 and (oom >= 1) == expect_oom
        st = "PASS" if ok else "FAIL"
        ev.append(f"target pod slope {slope/1024:.0f} KB/s (injected {rate/1024:.0f}), mem frac {m0:.2f}->{m1:.2f}, "
                  f"OOM expected {expect_oom} / seen {oom} (at {t_oom if t_oom is None else round(t_oom)}s), "
                  f"pod restarts {rst:.0f}, p95 ratio {p95_ratio:.2f}")
        return st, ev
    if t == "dep_latency":
        if sc["traffic"]["shape"] == "spike":  # isolate the dep-only part before the traffic spike
            dur = win(g, a + 30, sc["traffic"]["params"]["spike_at_s"])
        r95 = med(dur.p95) / med(pre.p95)
        rcpu = med(dur.cpu_util_vs_request) / med(pre.cpu_util_vs_request)
        cpu = med(dur.cpu_util_vs_request)
        rep = (pre.replicas_ready.median(), dur.replicas_ready.max())
        st = "PASS" if r95 > 3 and med(dur.p95) > SLO_P95 and rcpu < 1.3 and cpu < 1.0 else "FAIL"
        ev.append(f"p95 {med(pre.p95)*1e3:.0f}->{med(dur.p95)*1e3:.0f} ms (x{r95:.1f}), cpu/request {med(pre.cpu_util_vs_request):.2f}"
                  f"->{cpu:.2f} (x{rcpu:.2f}), inflight {med(pre.inflight):.0f}->{med(dur.inflight):.0f}, "
                  f"replicas {rep[0]:.0f}->{rep[1]:.0f}, err {med(dur.error_rate):.3f}")
        return st, ev
    if t == "error_rate":
        e = med(dur.error_rate)
        target = f["params"]["error_rate"]
        r95 = med(dur.p95) / med(pre.p95)
        rcpu = med(dur.cpu_util_vs_request) / med(pre.cpu_util_vs_request)
        st = "PASS" if abs(e - target) < 0.05 and med(pre.error_rate) < SLO_ERR and r95 < 1.5 else "FAIL"
        ev.append(f"error_rate {med(pre.error_rate):.3f}->{e:.3f} (injected {target}), p95 x{r95:.2f}, cpu x{rcpu:.2f}")
        return st, ev
    if t == "cpu_hog":
        pod = f["pods"][0]
        pdur = win(gp, a + 45, b)
        tgt, oth = pdur[pdur.pod == pod], pdur[pdur.pod != pod]
        ppre = win(gp, a - 120, a - 10)
        thr_t, thr_o = med(tgt.throttle), med(oth.throttle)
        cpu_t, cpu_o = med(tgt.cpu), med(oth.cpu)
        st = "PASS" if thr_t > 0.2 and thr_t > 3 * max(thr_o, 0.02) and cpu_t > 1.5 * cpu_o else "FAIL"
        ev.append(f"target pod throttle {med(ppre[ppre.pod == pod].throttle):.2f}->{thr_t:.2f} vs others {thr_o:.2f}; "
                  f"cpu {cpu_t:.2f} vs others {cpu_o:.2f} cores; deploy cpu/request {med(pre.cpu_util_vs_request):.2f}"
                  f"->{med(dur.cpu_util_vs_request):.2f}, replicas {pre.replicas_ready.median():.0f}->{dur.replicas_ready.max():.0f}, "
                  f"p95 {med(dur.p95)*1e3:.0f} ms")
        return st, ev
    if t == "bad_deploy":
        late = win(g, a + 60, b)
        cpr_pre = med(pre.cpu_cores_used / pre.rps) * 1e3
        cpr = med(late.cpu_cores_used / late.rps) * 1e3
        ratio = cpr / cpr_pre
        exp = f["params"]["cpu_ms_per_req"] / sc["app"]["cpu_ms_per_req"]
        post = win(g, b + 60, 1e9)
        st = "PASS" if ratio > 0.6 * exp and med(post.p95) < SLO_P95 else "FAIL"
        ev.append(f"cpu ms/request {cpr_pre:.1f}->{cpr:.1f} (x{ratio:.2f}, injected x{exp:.1f}; process CPU understates under "
                  f"throttle), throttle {med(pre.throttle_ratio):.2f}->{med(late.throttle_ratio):.2f}, p95 {med(late.p95):.2f}s, "
                  f"replicas {pre.replicas_ready.median():.0f}->{late.replicas_ready.max():.0f}, post-rollback p95 {med(post.p95)*1e3:.0f} ms")
        return st, ev
    if t == "crashloop":
        w = win(g, a, b)
        clp = int(w.crashloop_pods.max())
        r0, r1 = g.restarts_total[g.seconds_since_start < a].max(), w.restarts_total.max()
        short = float((w.replicas_ready < w.replicas_desired).mean())
        unready_new = int(((gp.ready == 0) & (gp.pod_is_fault_target == 1)).sum())
        pnr = int(w.pods_not_ready.max()) if "pods_not_ready" in w else -1
        st = "PASS" if clp > 0 and r1 > r0 else "FAIL"
        # replicas_ready never drops (maxSurge: old RS keeps serving); accept pods_not_ready instead
        if st == "PASS" and w.replicas_ready.min() >= pre.replicas_ready.median() and pnr < 1:
            st = "WARN"
        ev.append(f"crashloop_pods max {clp}, restarts {r0:.0f}->{r1:.0f}, pods_not_ready max {pnr} "
                  f"(none rows max {int(g[g.active_fault == 'none'].pods_not_ready.max()) if pnr >= 0 else -1}), "
                  f"unready new-RS pod rows {unready_new}, "
                  f"replicas_ready min {w.replicas_ready.min():.0f} (pre {pre.replicas_ready.median():.0f}), "
                  f"rows ready<desired {short:.0%}, p95 {med(w.p95)*1e3:.0f} ms")
        return st, ev
    if t == "pod_kill":
        w = win(g, a, a + 30)
        dip = w.replicas_ready.min() < pre.replicas_ready.median()
        rec = win(g, a + 30, a + 60).replicas_ready.max() >= pre.replicas_ready.median()
        st = "PASS" if dip and rec else "FAIL"
        ev.append(f"kill @{a:.0f}s: replicas_ready {pre.replicas_ready.median():.0f}->{w.replicas_ready.min():.0f}, "
                  f"recovered within 60s {rec}, p95 max {w.p95.max()*1e3:.0f} ms")
        return st, ev
    return "WARN", [f"no check for {t}"]


def baseline_check(g, faults):
    """Fault runs: the clean stretch before the first fault must be healthy."""
    if not faults:
        return "PASS", []
    a = min(f["start_rel_s"] for f in faults)
    pre = win(g, 30, a - 10)
    viol = int(((pre.p95 > SLO_P95) | (pre.error_rate > SLO_ERR)).sum())
    st = "PASS" if viol == 0 else "WARN"
    return st, [f"pre-fault baseline: p95 {med(pre.p95)*1e3:.0f} ms, cpu/request {med(pre.cpu_util_vs_request):.2f}, "
                f"SLO-violating rows {viol}"]


def unlabelled_anomalies(D):
    """Rows labelled none with p95 > SLO in runs whose traffic is steady and below capacity."""
    steady = D[(D["shape"] == "steady") & (D.active_fault == "none") & (D.seconds_since_start > 30)]
    out = []
    for rid, g in steady.groupby("run_id"):
        faults = load_faults(rid)
        ends = [f["start_rel_s"] + 30 if f["type"] == "pod_kill" else f["end_rel_s"] for f in faults]
        for _, r in g[g.p95 > SLO_P95].iterrows():
            near = any(0 <= r.seconds_since_start - e <= 40 for e in ends)
            out.append({"run_id": rid, "t": r.seconds_since_start, "ts": int(r.ts), "p95": r.p95,
                        "kind": "post-fault tail (<=40 s)" if near else "UNEXPLAINED"})
    return pd.DataFrame(out)


# ------------------------------------------------------------------ plots
def shade(ax, faults):
    for f in faults:
        end = f["start_rel_s"] + 30 if f["type"] == "pod_kill" else f["end_rel_s"]
        ax.axvspan(f["start_rel_s"], end, color=FAULT, alpha=0.7, lw=0, zorder=0)


def plot_run(rid, g, gp, faults, status):
    fig, ax = plt.subplots(5, 1, figsize=(9, 9), sharex=True)
    x = g.seconds_since_start
    ax[0].plot(x, g.rps, color=C1, label="rps")
    ax[0].set_ylabel("req/s")
    ax[0].set_title(f"{rid}  [{status}]  (grey = fault label window)", loc="left")
    a2 = ax[1]
    a2.plot(x, g.replicas_ready, color=C1, label="ready")
    a2.plot(x, g.replicas_desired, color=C2, ls="--", label="desired")
    a2.set_ylabel("replicas")
    a2.legend(loc="upper left", ncol=2)
    ax[2].semilogy(x, g.p95.clip(lower=0.01), color=C1, label="p95")
    ax[2].axhline(SLO_P95, color=INK2, lw=0.8, ls=":")
    ax[2].set_ylabel("p95 s (log)")
    ax[3].plot(x, g.cpu_util_vs_request, color=C1, label="cpu / request")
    ax[3].plot(x, g.throttle_ratio, color=C2, label="throttle ratio")
    ax[3].plot(x, g.error_rate, color=C3, label="5xx error rate")
    ax[3].axhline(0.7, color=INK2, lw=0.8, ls=":")
    ax[3].set_ylabel("ratio")
    ax[3].legend(loc="upper left", ncol=3)
    for i, (pod, pg) in enumerate(gp.groupby("pod")):
        tgt = pg.pod_is_fault_target.max() == 1 and any(f.get("pods") and pod in f["pods"] for f in faults
                                                          if isinstance(f.get("pods"), list))
        ax[4].plot(pg.seconds_since_start, pg.mem_frac_limit, color=C2 if tgt else C1,
                   lw=1.6 if tgt else 0.8, alpha=1 if tgt else 0.6)
    rs = g[g.restarts_total.diff() > 0].seconds_since_start
    for t in rs:
        ax[4].axvline(t, color=INK, lw=0.8, ls="--")
    ax[4].set_ylabel("pod mem / limit")
    ax[4].set_xlabel("seconds since start  (dashed = container restart; orange = leak target pod)")
    for a in ax:
        shade(a, faults)
    fig.tight_layout()
    fig.savefig(PLOTS / f"run_{g.scenario.iloc[0]}.png", dpi=110)
    plt.close(fig)


def plot_knee(caps):
    fig, ax = plt.subplots(1, 2, figsize=(10, 3.8))
    colors = [C1, C2, C3]
    for (name, s), c in zip(caps.items(), colors):
        ax[0].semilogy(s.offered_per_pod, s.p95, marker="o", ms=4, color=c, label=name)
        ax[1].plot(s.offered_per_pod, s.achieved_per_pod, marker="o", ms=4, color=c, label=name)
    ax[0].axhline(SLO_P95, color=INK2, ls=":", lw=0.8)
    ax[0].text(6.2, SLO_P95 * 1.15, "SLO 300 ms", color=INK2, fontsize=7)
    ax[0].set_xlabel("offered RPS per pod")
    ax[0].set_ylabel("p95 s (log)")
    ax[0].set_title("Capacity knee: p95 vs offered load", loc="left")
    ax[1].plot([6, 24], [6, 24], color=INK2, ls=":", lw=0.8)
    ax[1].set_xlabel("offered RPS per pod")
    ax[1].set_ylabel("served RPS per pod")
    ax[1].set_title("Throughput plateau (dotted = served = offered)", loc="left")
    for a in ax:
        a.legend()
    fig.tight_layout()
    fig.savefig(PLOTS / "capacity_knee.png", dpi=110)
    plt.close(fig)


def plot_nan(D):
    cols = [c for c in D.columns if D[c].dtype != object and c not in ("ts", "base_rps", "seconds_since_start")]
    m = D.groupby("scenario", sort=False)[cols].apply(lambda g: g.isna().mean())
    fig, ax = plt.subplots(figsize=(10, 6))
    im = ax.imshow(m.to_numpy(), aspect="auto", cmap="Blues", vmin=0, vmax=0.25)
    ax.set_xticks(range(len(cols)), cols, rotation=60, ha="right")
    ax.set_yticks(range(len(m)), m.index)
    ax.grid(False)
    for i in range(m.shape[0]):
        for j in range(m.shape[1]):
            v = m.iat[i, j]
            if v > 0:
                ax.text(j, i, f"{v:.0%}", ha="center", va="center", fontsize=6, color=INK if v < 0.15 else "white")
    fig.colorbar(im, ax=ax, label="NaN share (deploy rows)")
    ax.set_title("NaN share per run and column (dataset_deploy)", loc="left")
    fig.tight_layout()
    fig.savefig(PLOTS / "nan_heatmap.png", dpi=110)
    plt.close(fig)


def plot_signatures(D):
    """Fault rows vs none rows in the D group on the two axes that separate scale-vs-fix."""
    d = D[D.group.isin(["C", "D", "E"])].copy()
    d["cpr_ms"] = d.cpu_cores_used / d.rps * 1e3
    fig, ax = plt.subplots(1, 2, figsize=(10, 4))
    groups = [("none", INK2, 0.25), ("dep_latency", C1, 0.9), ("bad_deploy", C2, 0.9), ("cpu_hog", C3, 0.9),
              ("error_rate", C4, 0.9)]
    for name, c, al in groups:
        s = d[d.active_fault == name]
        ax[0].scatter(s.cpu_util_vs_request, s.p95, s=8, color=c, alpha=al, label=name, lw=0)
        ax[1].scatter(s.cpr_ms, s.error_rate, s=8, color=c, alpha=al, label=name, lw=0)
    ax[0].set_yscale("log")
    ax[0].axhline(SLO_P95, color=INK2, ls=":", lw=0.8)
    ax[0].set_xlabel("cpu / request")
    ax[0].set_ylabel("p95 s (log)")
    ax[0].set_title("High p95 at low CPU = dependency, not capacity (groups C-E)", loc="left")
    ax[1].set_xlabel("cpu ms per request")
    ax[1].set_ylabel("5xx error rate")
    ax[1].set_xlim(0, 120)
    ax[1].set_title("Bad deploy shifts cost/request; error burst shifts 5xx", loc="left")
    for a in ax:
        a.legend(markerscale=2)
    fig.tight_layout()
    fig.savefig(PLOTS / "fault_signatures.png", dpi=110)
    plt.close(fig)


# ------------------------------------------------------------------ main
def main():
    PLOTS.mkdir(parents=True, exist_ok=True)
    D = pd.read_csv(DATA / "dataset_deploy.csv.gz")
    P = pd.read_csv(DATA / "dataset_pods.csv.gz")
    results, caps = [], {}
    for rid, g in D.groupby("run_id", sort=False):
        g = g.sort_values("ts").reset_index(drop=True)
        gp = P[P.run_id == rid].sort_values(["pod", "ts"])
        sc = json.loads((ROOT / "runs" / rid / "scenario.json").read_text())
        faults = load_faults(rid)
        name = sc["id"]
        st_i, ev_i, extra = integrity(g, gp, sc)
        st_l, ev_l = label_check(g, faults)
        st_b, ev_b = baseline_check(g, faults)
        sig_st, sig_ev = "PASS", []
        if name.startswith("a-cap"):
            sig_st, sig_ev, s = sig_capacity(g, sc)
            caps[name] = s
        elif name.startswith("b-hpa"):
            sig_st, sig_ev = sig_hpa(g, sc)
        elif name.startswith("c-under"):
            sig_st, sig_ev = sig_under(g, sc)
        elif name.startswith("c-over"):
            sig_st, sig_ev = sig_over(g, sc)
        for f in faults:
            s_, e_ = sig_fault(g, gp, f, sc)
            sig_st, sig_ev = worst(sig_st, s_), sig_ev + e_
        if name.startswith("e-spike"):
            spk = sc["traffic"]["params"]
            w = win(g, spk["spike_at_s"] + 30, spk["spike_at_s"] + spk["spike_s"])
            sig_ev.append(f"spike+dep phase: cpu/request {med(w.cpu_util_vs_request):.2f}, p95 {med(w.p95):.2f}s, "
                          f"replicas {w.replicas_ready.max():.0f}")
        overall = worst(st_i, st_l, sig_st, st_b)
        results.append({"scenario": name, "run_id": rid, "group": sc.get("group"), "overall": overall,
                        "integrity": st_i, "labels": st_l, "baseline": st_b, "signature": sig_st,
                        "integrity_ev": ev_i, "label_ev": ev_l, "baseline_ev": ev_b, "signature_ev": sig_ev,
                        **{k: v for k, v in extra.items() if k != "pod_nan"}})
        plot_run(rid, g, gp, faults, overall)
    plot_knee(caps)
    plot_nan(D)
    plot_signatures(D)
    anom = unlabelled_anomalies(D)
    (DATA / "qa_results.json").write_text(json.dumps({"runs": results,
                                                      "unlabelled_anomalies": anom.to_dict("records")},
                                                     indent=1, default=float))
    for r in results:
        print(f"{r['overall']:4s} {r['scenario']:24s} I:{r['integrity']} L:{r['labels']} B:{r['baseline']} "
              f"S:{r['signature']}")
        for e in r["integrity_ev"] + r["label_ev"] + r["baseline_ev"] + r["signature_ev"]:
            print("      ", e)
    print(anom.to_string())
    for name, s in caps.items():
        print(name)
        print(s.round(3).to_string())


if __name__ == "__main__":
    main()
