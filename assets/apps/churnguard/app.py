import math
import sys
from pathlib import Path

import streamlit as st

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))
import core  # noqa: E402

INKS = {
    "light": {"paper": "#eeeade", "paper2": "#e5e0d1", "paper3": "#d9d3c2", "ink": "#17150f", "ink2": "#47433a",
              "hair": "#b5ae9c", "ox": "#7b1e2b", "blue": "#1b365d"},
    "dark": {"paper": "#14120e", "paper2": "#1d1a15", "paper3": "#29251e", "ink": "#ebe5d6", "ink2": "#beb7a5",
             "hair": "#4a453a", "ox": "#e08a7e", "blue": "#9db9df"},
}
FONT = "EB Garamond, Garamond, Georgia, serif"
MONO = "Courier Prime, Courier New, monospace"
DISPLAY = "Bodoni Moda, Bodoni 72, Didot, serif"


@st.cache_resource
def data():
    return core.load(HERE.parent.parent / "models" / "churn.json"), core.load(HERE / "population.json")


st.set_page_config(page_title="ChurnGuard", layout="wide")
M, POP = data()
F, SAMPLES, META = M["features"], M["samples"], M["meta"]
C = INKS["dark" if st.get_option("theme.base") == "dark" else "light"]
NAMES = [f"Customer {chr(65 + i)}" for i in range(len(SAMPLES))] + ["Median customer"]
MEDIAN = len(SAMPLES)
DEFAULT = next(i for i, s in enumerate(SAMPLES) if s["churn"] == 1 and s["p"] >= 0.3)


def axis_config():
    return {"font": FONT, "background": None, "view": {"stroke": None},
            "axis": {"labelColor": C["ink2"], "titleColor": C["ink2"], "domainColor": C["hair"], "tickColor": C["hair"],
                     "gridColor": C["hair"], "labelFont": MONO, "titleFont": FONT, "labelFontSize": 11, "titleFontSize": 13},
            "legend": {"labelColor": C["ink"], "labelFont": FONT, "labelFontSize": 13, "titleColor": C["ink2"]}}


def pick(k):
    x = SAMPLES[k]["x"] if k < MEDIAN else M["baseline"]
    for j, v in enumerate(x):
        st.session_state[f"x{j}"] = float(v)
    st.session_state.scored = [float(v) for v in x]


def on_pick():
    pick(st.session_state.pick)


def wrap(s, width=15):
    lines, cur = [], ""
    for w in s.split():
        if cur and len(cur) + 1 + len(w) > width:
            lines.append(cur)
            cur = w
        else:
            cur = f"{cur} {w}".strip()
    return lines + [cur]


def radar(x):
    n = len(F)
    ang = [math.pi / 2 - 2 * math.pi * k / n for k in range(n)]
    pt = lambda r, k: {"x": r * math.cos(ang[k]), "y": r * math.sin(ang[k])}
    rings = [{"ring": r, "o": i, **pt(r, i % n)} for r in (0.25, 0.5, 0.75, 1) for i in range(n + 1)]
    spokes = [{"x": 0, "y": 0, "x2": pt(1, k)["x"], "y2": pt(1, k)["y"]} for k in range(n)]
    ticks = [{"t": f"{int(r * 100)}th", **pt(r, 0)} for r in (0.25, 0.5, 0.75)]
    series = [("This customer", x), ("Median churned customer", [s["churned"]["median"] for s in POP["features"]]),
              ("Median retained customer", [s["retained"]["median"] for s in POP["features"]])]
    poly = []
    for name, vals in series:
        for i in range(n + 1):
            k = i % n
            q = core.percentile(vals[k], POP["features"][k]["percentiles"])
            poly.append({"s": name, "o": i, "f": F[k]["label"], "pct": round(q), "v": core.value_text(F[k]["key"], vals[k]), **pt(q / 100, k)})
    labels = []
    for k in range(n):
        c = math.cos(ang[k])
        labels.append({"t": wrap(F[k]["label"]), "a": "left" if c > 0.2 else "right" if c < -0.2 else "center",
                       "b": "bottom" if math.sin(ang[k]) > 0.5 else "top" if math.sin(ang[k]) < -0.5 else "middle", **pt(1.12, k)})
    r = "max(60, min(width / 2 - min(112, width / 4), height / 2 - 56))"
    sx = {"domain": {"expr": f"[-width / 2 / {r}, width / 2 / {r}]"}, "nice": False, "zero": False}
    sy = {"domain": {"expr": f"[-height / 2 / {r}, height / 2 / {r}]"}, "nice": False, "zero": False}
    X = {"field": "x", "type": "quantitative", "axis": None, "scale": sx}
    Y = {"field": "y", "type": "quantitative", "axis": None, "scale": sy}
    names = [s[0] for s in series]
    legend = {"orient": "bottom", "direction": "vertical", "symbolType": "stroke", "symbolStrokeWidth": 2, "symbolSize": 400}
    layers = [
        {"data": {"values": rings}, "mark": {"type": "line", "color": C["hair"], "strokeWidth": 1},
         "encoding": {"x": X, "y": Y, "detail": {"field": "ring"}, "order": {"field": "o"}}},
        {"data": {"values": spokes}, "mark": {"type": "rule", "color": C["hair"], "strokeWidth": 1},
         "encoding": {"x": X, "y": Y, "x2": {"field": "x2"}, "y2": {"field": "y2"}}},
        {"data": {"values": ticks}, "mark": {"type": "text", "font": MONO, "fontSize": 10, "color": C["ink2"], "dx": 4, "align": "left"},
         "encoding": {"x": X, "y": Y, "text": {"field": "t"}}},
        {"data": {"values": poly}, "mark": {"type": "line", "strokeWidth": 2},
         "encoding": {"x": X, "y": Y, "order": {"field": "o"},
                      "color": {"field": "s", "type": "nominal", "title": None, "sort": names,
                                "scale": {"domain": names, "range": [C["ink"], C["ox"], C["blue"]]},
                                "legend": legend},
                      "strokeDash": {"field": "s", "type": "nominal", "title": None, "sort": names, "legend": legend,
                                     "scale": {"domain": names, "range": [[1, 0], [7, 3], [2, 2]]}}}},
        {"data": {"values": [p for p in poly if p["s"] == names[0]][:n]}, "mark": {"type": "point", "filled": True, "size": 40, "color": C["ink"]},
         "encoding": {"x": X, "y": Y, "tooltip": [{"field": "f", "title": "Feature"}, {"field": "v", "title": "Value"},
                                                  {"field": "pct", "title": "Percentile"}]}},
    ]
    for a, b in sorted({(q["a"], q["b"]) for q in labels}):
        layers.append({"data": {"values": [q for q in labels if (q["a"], q["b"]) == (a, b)]},
                       "mark": {"type": "text", "font": FONT, "fontSize": 13, "color": C["ink"], "lineHeight": 15, "align": a, "baseline": b},
                       "encoding": {"x": X, "y": Y, "text": {"field": "t"}}})
    return {"width": "container", "height": 500, "layer": layers, "config": axis_config()}, poly[: n + 1]


def donut(share, color):
    rows = [{"k": "a", "v": share}, {"k": "b", "v": 1 - share}]
    return {"width": 150, "height": 150, "config": axis_config(), "layer": [
        {"data": {"values": rows}, "mark": {"type": "arc", "innerRadius": 48, "outerRadius": 72, "stroke": C["paper"], "strokeWidth": 1},
         "encoding": {"theta": {"field": "v", "type": "quantitative", "stack": True, "sort": None},
                      "order": {"field": "k"},
                      "color": {"field": "k", "type": "nominal", "legend": None,
                                "scale": {"domain": ["a", "b"], "range": [color, C["paper3"]]}}}},
        {"data": {"values": [{"t": core.pct(share)}]}, "mark": {"type": "text", "font": DISPLAY, "fontSize": 24, "fontWeight": 500, "color": C["ink"]},
         "encoding": {"text": {"field": "t"}}},
    ]}


def score(pairs):
    cell = ('<div style="flex: 1 1 11rem; min-width: 0"><dt style="font-size: 0.875rem; margin: 0 0 0.25rem">{}</dt>'
            '<dd style="font-size: 2.25rem; line-height: 1.15; margin: 0">{}</dd></div>')
    return ('<dl style="display: flex; flex-wrap: wrap; gap: 0.75rem 2rem; margin: 0 0 1rem">'
            + "".join(cell.format(k, v) for k, v in pairs) + "</dl>")


def boxes(j, v):
    ft, st_ = F[j], POP["features"][j]
    rows = [{"g": "Churned", **st_["churned"]}, {"g": "Retained", **st_["retained"]}]
    groups = ["Churned", "Retained"]
    days = ft["key"] == core.DAYS
    lo = min(v, *(r["lo"] for r in rows))
    hi = max(v, *(r["hi"] for r in rows))
    if days:
        scale, axis = {"type": "linear", "domain": [0, hi], "nice": True}, {"tickCount": 5, "format": "d"}
    else:
        ticks = [t for t in (-100000, -10000, -1000, 0, 1000, 2000, 5000, 10000, 20000, 50000, 100000) if lo <= t <= hi]
        scale = {"type": "symlog", "constant": 100, "domain": [lo, hi], "nice": False}
        axis = {"values": ticks, "labelExpr": "datum.value === 0 ? '0' : replace(format(datum.value / 1000, ',') + 'k', '-', '\u2212')"}
    X = {"field": "lo", "type": "quantitative", "title": "days" if days else "account currency units, symmetric log scale", "scale": scale,
         "axis": {**axis, "grid": False}}
    Y = {"field": "g", "type": "nominal", "title": None, "sort": groups, "scale": {"type": "band", "paddingInner": 0.25}, "axis": {"labelFont": FONT, "labelFontSize": 13, "labelColor": C["ink"], "ticks": False, "domain": False}}
    col = {"field": "g", "type": "nominal", "legend": None, "scale": {"domain": groups, "range": [C["ox"], C["blue"]]}}
    tip = [{"field": "g", "title": "Group"}, {"field": "n", "title": "Customers", "format": ","}, {"field": "lo", "title": "Lower whisker"},
           {"field": "q1", "title": "First quartile"}, {"field": "median", "title": "Median"}, {"field": "q3", "title": "Third quartile"},
           {"field": "hi", "title": "Upper whisker"}]
    return {"width": "container", "height": 64, "autosize": {"type": "fit-x", "contains": "padding"},
            "title": {"text": ft["label"], "subtitle": f"This customer: {core.value_text(ft['key'], v)} (dashed line)", "anchor": "start",
                      "font": FONT, "fontSize": 15, "fontWeight": 600, "color": C["ink"], "subtitleFont": FONT, "subtitleFontSize": 13,
                      "subtitleColor": C["ink2"], "offset": 10},
            "config": axis_config(), "layer": [
                {"data": {"values": rows}, "layer": [
                    {"mark": {"type": "rule", "color": C["ink2"]}, "encoding": {"x": X, "x2": {"field": "hi"}, "y": Y}},
                    {"mark": {"type": "bar", "size": 22, "stroke": C["ink"], "strokeWidth": 1}, "encoding": {"x": {**X, "field": "q1"}, "x2": {"field": "q3"}, "y": Y, "color": col, "tooltip": tip}},
                    {"mark": {"type": "tick", "thickness": 2, "size": 22, "color": C["paper"]}, "encoding": {"x": {**X, "field": "median"}, "y": Y}},
                ]},
                {"data": {"values": [{"lo": v}]}, "mark": {"type": "rule", "color": C["ink"], "strokeWidth": 2, "strokeDash": [4, 2]},
                 "encoding": {"x": X}},
            ]}


if "scored" not in st.session_state:
    st.session_state.pick = DEFAULT
    pick(DEFAULT)

st.title("ChurnGuard")
st.markdown("*Locking in loyalty*")

left, mid, right = st.container(key="cg-layout").columns((1.75, 4.25, 2), gap="medium")

with left:
    st.markdown("#### Customer information")
    st.selectbox("Start from", range(len(NAMES)), format_func=lambda k: NAMES[k], key="pick", on_change=on_pick,
                 help="Customers A to H were held out from training, so their real outcome is known.")
    with st.form("customer"):
        for j, ft in enumerate(F):
            days = ft["key"] == core.DAYS
            st.number_input(ft["label"], min_value=0.0 if days else float(ft["min"]), max_value=365.0 if days else float(ft["max"]),
                            step=1.0, format="%.0f" if days else "%.2f", key=f"x{j}")
        if st.form_submit_button("Submit"):
            st.session_state.scored = [float(st.session_state[f"x{j}"]) for j in range(len(F))]
    x = st.session_state.scored
    k = st.session_state.pick
    if x != (SAMPLES[k]["x"] if k < MEDIAN else M["baseline"]):
        st.button(f"Restore {NAMES[k]}", on_click=on_pick)
    st.caption("Money is in account currency units; the dataset does not name a currency.")

p = core.proba(M["model"], x)

with mid:
    one, base = st.tabs(["This customer", "Customer base"])
    with one:
        t = round(st.slider("Flag for retention when the probability is at least", 0.02, 0.98, 0.30, 0.02, format="%.2f", key="t"), 2)
        flagged = p >= t
        st.markdown(score([("Probability this customer churns", core.pct(p)),
                           (f"Decision at {t:.2f}", "Flag for retention" if flagged else "No action")]), unsafe_allow_html=True)
        if k == MEDIAN:
            st.markdown("The median customer is not a real account, so there is no outcome to compare with.")
        elif x != SAMPLES[k]["x"]:
            st.markdown(f"{NAMES[k]} has been edited, so the real outcome no longer applies.")
        else:
            churned = SAMPLES[k]["churn"] == 1
            call = ("Flagged: caught." if flagged else "Not flagged: missed.") if churned else ("Flagged: false alarm." if flagged else "Not flagged: correct.")
            st.markdown(f"{NAMES[k]} {'churned' if churned else 'stayed'}. {call}")

        st.subheader("Where this customer sits")
        spec, mine = radar(x)
        st.vega_lite_chart(spec)
        st.caption("Each spoke is the customer's percentile among the "
                   f"{POP['meta']['rows']:,} training customers: "
                   + "; ".join(f"{q['f']} {q['pct']}" for q in mine[:-1]) + ". The centre is the lowest value, the rim the highest.")

        st.subheader("Why this score")
        st.caption("The hackathon build asked an LLM to write these notes. This browser edition writes them with fixed rules, "
                   "because a static page cannot hold an API key. They are ordered by how far the score moves when that "
                   "input is swapped for the median customer's value, the same what-if as Figure 3.")
        st.markdown("\n".join(f"{i + 1}. **{n['label']}** {n['text']}" for i, n in enumerate(core.notes(M, POP, x))))
    with base:
        st.subheader("Churned and retained customers")
        st.caption(f"Box plots of the {POP['meta']['rows']:,} training customers ({POP['meta']['churned']:,} churned, "
                   f"{POP['meta']['retained']:,} retained): the box spans the middle half, the pale line is the median and the "
                   "whiskers reach the furthest value within 1.5 box lengths. The dashed line is this customer.")
        for j in range(len(F)):
            st.vega_lite_chart(boxes(j, x[j]))

with right:
    rows = M["thresholds"]
    n = rows[0][1] + rows[0][2] + rows[0][3] + rows[0][4]
    churners = rows[0][1] + rows[0][3]
    tr = core.threshold_row(M, t)
    st.markdown("#### Held-out customers")
    st.markdown("**Observed churn rate**")
    st.vega_lite_chart(donut(churners / n, C["ox"]))
    st.caption(f"{churners:,} of {n:,} held-out customers churned.")
    st.markdown(f"**Flagged at {t:.2f}**")
    st.vega_lite_chart(donut(tr["flagged"] / n, C["ox"]))
    st.caption(f"{tr['flagged']:,} of {n:,} flagged; {tr['tp']:,} of them churned, {tr['tp'] / churners:.1%} of all churners.")
    with st.expander("About", expanded=True):
        hp, mt = META["hyperparameters"], META["metrics"]
        st.markdown(
            f"- ChurnGuard scores how likely a bank customer is to leave and says why.\n"
            f"- This is a browser edition of my team's winning entry at UTD FinHack 2024. Streamlit itself runs here on "
            f"Python compiled to WebAssembly, so nothing you enter leaves this tab.\n"
            f"- Data: a public [bank customer churn dataset](https://www.kaggle.com/datasets/pentakrishnakishore/bank-customer-churn-data) "
            f"as cleaned by the team, {META['rows_used']:,} customers: {META['split']['train']:,} to train, {META['split']['test']:,} held out.\n"
            f"- Model: {META['trees']} gradient-boosted trees of depth {hp['max_depth']} on {len(F)} account features, the same file as "
            f"Figure 3, walked tree by tree in Python. ROC-AUC {mt['test_roc_auc']:.2f} on the held-out customers.\n"
            f"- Different from the hackathon build: that one used tree ensembles (random forest, XGBoost) on ten features and an LLM "
            f"for the notes; this one uses scikit-learn gradient boosting on {len(F)} features and fixed rules.")
