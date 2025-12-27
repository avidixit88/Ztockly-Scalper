import time
import pandas as pd
import streamlit as st
import plotly.graph_objects as go

from av_client import AlphaVantageClient
from engine import scan_watchlist, fetch_bundle
from indicators import vwap as calc_vwap
from signals import compute_scalp_signal, PRESETS

st.set_page_config(page_title="Ztockly Scalping Scanner", layout="wide")

# ---------------- Session state ----------------
if "watchlist" not in st.session_state:
    st.session_state.watchlist = ["AAPL", "NVDA", "TSLA", "SPY", "QQQ"]

if "last_alert_ts" not in st.session_state:
    # symbol -> unix timestamp of last in-app alert
    st.session_state.last_alert_ts = {}

if "alerts" not in st.session_state:
    # list of alert dicts (most recent first)
    st.session_state.alerts = []


# ---------------- Sidebar: watchlist + config ----------------
st.sidebar.title("Scalping Scanner")

watchlist_text = st.sidebar.text_area(
    "Watchlist (comma or newline separated)",
    value="\n".join(st.session_state.watchlist),
    height=150,
)

interval = st.sidebar.selectbox("Intraday interval", ["1min", "5min"], index=0)
mode = st.sidebar.selectbox("Signal mode", list(PRESETS.keys()), index=list(PRESETS.keys()).index("Cleaner signals"))

st.sidebar.markdown("### Time-of-day filter (ET)")
allow_opening = st.sidebar.checkbox("Opening 90 min (09:30–11:00)", value=True)
allow_midday = st.sidebar.checkbox("Midday chop (11:00–15:00)", value=False)
allow_power = st.sidebar.checkbox("Power hour (15:00–16:00)", value=True)

st.sidebar.markdown("### In‑App Alerts")
cooldown_minutes = st.sidebar.slider("Cooldown minutes (per ticker)", 1, 30, 7, 1)
alert_threshold = st.sidebar.slider(
    "Alert score threshold",
    60, 100,
    int(PRESETS[mode]["min_actionable_score"]),
    1
)
capture_alerts = st.sidebar.checkbox("Capture alerts in-app", value=True)
max_alerts_kept = st.sidebar.slider("Max alerts kept", 10, 300, 60, 10)

st.sidebar.markdown("### API pacing / refresh")
min_between_calls = st.sidebar.slider("Seconds between API calls", 0.5, 5.0, 1.0, 0.5)
auto_refresh = st.sidebar.checkbox("Auto-refresh scanner", value=False)
refresh_seconds = st.sidebar.slider("Refresh every (seconds)", 10, 180, 30, 5) if auto_refresh else None

st.sidebar.markdown("---")
st.sidebar.caption("Required env var: ALPHAVANTAGE_API_KEY")

symbols = [s.strip().upper() for s in watchlist_text.replace(",", "\n").splitlines() if s.strip()]
st.session_state.watchlist = symbols


# ---------------- Main header ----------------
st.title("Ztockly — Intraday Reversal Scalping Engine")
st.caption("Confluence scoring (VWAP reclaim/rejection + RSI‑5 event + MACD histogram turn + volume). Includes time‑of‑day filters + cooldown + in‑app alerts.")


# ---------------- Client init ----------------
@st.cache_resource
def get_client(min_seconds_between_calls: float):
    client = AlphaVantageClient()
    client.cfg.min_seconds_between_calls = float(min_seconds_between_calls)
    return client

client = get_client(min_between_calls)


# ---------------- Helpers ----------------
def _now_et_label() -> str:
    # Streamlit Cloud may run in UTC; show a human label without requiring tz libs
    return pd.Timestamp.utcnow().strftime("%Y-%m-%d %H:%M:%S UTC")


def can_alert(symbol: str, now_ts: float, cooldown_min: int) -> bool:
    last = st.session_state.last_alert_ts.get(symbol)
    if last is None:
        return True
    return (now_ts - float(last)) >= cooldown_min * 60.0


def add_in_app_alert(row: dict) -> None:
    """
    Store an alert "card" in session_state.alerts.
    """
    alert = {
        "ts_unix": time.time(),
        "time": _now_et_label(),
        "symbol": row["Symbol"],
        "bias": row["Bias"],
        "score": int(row["Score"]),
        "session": row.get("Session"),
        "last": row.get("Last"),
        "entry": row.get("Entry"),
        "stop": row.get("Stop"),
        "t1": row.get("Target 1R"),
        "t2": row.get("Target 2R"),
        "why": row.get("Why"),
        "as_of": row.get("AsOf"),
        "mode": mode,
        "interval": interval,
    }
    # newest first
    st.session_state.alerts.insert(0, alert)
    # truncate
    st.session_state.alerts = st.session_state.alerts[: int(max_alerts_kept)]


def render_alerts_panel():
    st.subheader("🚨 Live Alerts")
    left, right = st.columns([2, 1])

    with right:
        st.metric("Alerts stored", len(st.session_state.alerts))
        if st.button("Clear alerts", type="secondary"):
            st.session_state.alerts = []
            st.session_state.last_alert_ts = {}
            st.rerun()

        st.markdown("**Filters**")
        f_bias = st.multiselect("Bias", ["LONG", "SHORT"], default=["LONG", "SHORT"])
        min_score = st.slider("Min score", 0, 100, 80, 1)
        st.caption("These filters only affect what you see, not what gets captured.")

    with left:
        alerts = [
            a for a in st.session_state.alerts
            if a["bias"] in f_bias and a["score"] >= min_score
        ]

        if not alerts:
            st.info("No alerts matching your filters yet. Turn on auto-refresh + in‑app alerts and let it scan.")
            return

        # Compact "cards"
        for a in alerts[:30]:
            # Color-esque styling via emoji + markdown (Streamlit doesn't allow full CSS on cloud)
            badge = "🟢" if a["bias"] == "LONG" else "🔴"
            title = f"{badge} **{a['symbol']}** — **{a['bias']}** — Score **{a['score']}** ({a.get('session','')})"
            with st.container(border=True):
                st.markdown(title)
                cols = st.columns(5)
                cols[0].metric("Last", f"{a['last']:.4f}" if a["last"] is not None else "N/A")
                cols[1].metric("Entry", f"{a['entry']:.4f}" if a["entry"] is not None else "—")
                cols[2].metric("Stop", f"{a['stop']:.4f}" if a["stop"] is not None else "—")
                cols[3].metric("1R", f"{a['t1']:.4f}" if a["t1"] is not None else "—")
                cols[4].metric("2R", f"{a['t2']:.4f}" if a["t2"] is not None else "—")
                st.caption(f"{a['time']} • interval={a['interval']} • mode={a['mode']} • as_of={a.get('as_of')}")
                st.write(a.get("why") or "")
                with st.expander("Raw alert payload"):
                    st.json(a)


# ---------------- Layout: alerts panel + scanner ----------------
tab_scan, tab_alerts = st.tabs(["📡 Scanner", "🚨 Alerts"])

with tab_alerts:
    render_alerts_panel()

with tab_scan:
    # Controls row
    col_a, col_b, col_c, col_d = st.columns([1, 1, 2, 1])
    with col_a:
        scan_now = st.button("Scan Watchlist", type="primary")
    with col_b:
        if st.button("Capture test alert"):
            add_in_app_alert({
                "Symbol": "TEST",
                "Bias": "LONG",
                "Score": 95,
                "Session": "OPENING",
                "Last": 0.0,
                "Entry": 0.0,
                "Stop": 0.0,
                "Target 1R": 0.0,
                "Target 2R": 0.0,
                "Why": "Test alert card",
                "AsOf": str(pd.Timestamp.utcnow()),
            })
            st.success("Test alert added — open the Alerts tab.")
    with col_c:
        st.write("Tip: Keep watchlist small (5–15) to stay within API limits and keep it snappy.")
    with col_d:
        st.write(f"Now: { _now_et_label() }")

    # Run scan
    def run_scan():
        if not symbols:
            st.warning("Add at least one ticker to your watchlist.")
            return []
        with st.spinner("Scanning watchlist..."):
            return scan_watchlist(
                client, symbols,
                interval=interval,
                mode=mode,
                allow_opening=allow_opening,
                allow_midday=allow_midday,
                allow_power=allow_power,
            )

    results = []
    if auto_refresh:
        results = run_scan()
        st.info(f"Auto-refresh is ON — rerunning every ~{refresh_seconds}s.")
    else:
        if scan_now:
            results = run_scan()

    # If results, show table and (optional) capture in-app alerts
    if results:
        df = pd.DataFrame([{
            "Symbol": r.symbol,
            "Bias": r.bias,
            "Score": r.setup_score,
            "Session": r.session,
            "Last": r.last_price,
            "Entry": r.entry,
            "Stop": r.stop,
            "Target 1R": r.target_1r,
            "Target 2R": r.target_2r,
            "Why": r.reason,
            "AsOf": str(r.timestamp) if r.timestamp is not None else None
        } for r in results])

        st.subheader("Ranked Setups")
        st.dataframe(
            df,
            use_container_width=True,
            hide_index=True,
            column_config={
                "Score": st.column_config.ProgressColumn("Score", min_value=0, max_value=100),
            },
        )

        top = results[0]
        st.success(f"Top setup: **{top.symbol}** — **{top.bias}** (Score {top.setup_score}, {top.session})")
        st.caption(f"Cooldown: {cooldown_minutes} min per ticker. Alert threshold: {alert_threshold}. Mode: {mode}.")

        # Capture alerts for any ticker crossing threshold, respecting cooldown
        if capture_alerts:
            now = time.time()
            for r in results:
                if r.bias in ["LONG", "SHORT"] and r.setup_score >= alert_threshold:
                    if can_alert(r.symbol, now, cooldown_minutes):
                        add_in_app_alert({
                            "Symbol": r.symbol,
                            "Bias": r.bias,
                            "Score": r.setup_score,
                            "Session": r.session,
                            "Last": r.last_price,
                            "Entry": r.entry,
                            "Stop": r.stop,
                            "Target 1R": r.target_1r,
                            "Target 2R": r.target_2r,
                            "Why": r.reason,
                            "AsOf": str(r.timestamp) if r.timestamp is not None else None
                        })
                        st.session_state.last_alert_ts[r.symbol] = now

        # Drilldown chart
        st.subheader("Chart & Signal Detail")
        pick = st.selectbox("Select ticker", [r.symbol for r in results], index=0)

        with st.spinner(f"Loading chart data for {pick}..."):
            ohlcv, rsi5, rsi14, macd_hist, quote = fetch_bundle(client, pick, interval=interval)

        sig = compute_scalp_signal(
            pick, ohlcv, rsi5, rsi14, macd_hist,
            mode=mode,
            allow_opening=allow_opening,
            allow_midday=allow_midday,
            allow_power=allow_power,
        )

        plot_df = ohlcv.sort_index().copy()
        plot_df["vwap"] = calc_vwap(plot_df)
        plot_df = plot_df.tail(220)

        fig = go.Figure(data=[
            go.Candlestick(
                x=plot_df.index,
                open=plot_df["open"],
                high=plot_df["high"],
                low=plot_df["low"],
                close=plot_df["close"],
                name="Price",
            ),
            go.Scatter(
                x=plot_df.index,
                y=plot_df["vwap"],
                mode="lines",
                name="VWAP",
            )
        ])

        if sig.entry and sig.stop:
            fig.add_hline(y=sig.entry, line_dash="dot", annotation_text="Entry", annotation_position="top left")
            fig.add_hline(y=sig.stop, line_dash="dash", annotation_text="Stop", annotation_position="bottom left")
        if sig.target_1r:
            fig.add_hline(y=sig.target_1r, line_dash="dot", annotation_text="1R", annotation_position="top right")
        if sig.target_2r:
            fig.add_hline(y=sig.target_2r, line_dash="dot", annotation_text="2R", annotation_position="top right")

        fig.update_layout(height=520, xaxis_rangeslider_visible=False, margin=dict(l=10, r=10, t=30, b=10))
        st.plotly_chart(fig, use_container_width=True)

        c1, c2, c3, c4 = st.columns(4)
        with c1:
            st.metric("Bias", sig.bias)
        with c2:
            st.metric("Score", sig.setup_score)
        with c3:
            st.metric("Session", sig.session)
        with c4:
            lp = quote if quote is not None else sig.last_price
            st.metric("Last", f"{lp:.4f}" if lp is not None else "N/A")

        st.write("**Reasoning:**", sig.reason)

        st.markdown("### Trade Plan")
        if sig.bias in ["LONG", "SHORT"] and sig.entry and sig.stop:
            st.write(f"- **Entry:** {sig.entry:.4f}")
            st.write(f"- **Stop:** {sig.stop:.4f} (invalidation)")
            st.write(f"- **Scale out:** 1R = {sig.target_1r:.4f}, 2R = {sig.target_2r:.4f}")
            st.write("- **Fail-safe exit:** if price loses VWAP and MACD histogram turns against you, flatten remainder.")
            st.warning("This is an analytics tool, not financial advice. Always position-size and respect stops.")
        else:
            st.info("No clean confluence signal right now (or time-of-day filter blocking).")

    else:
        st.info("Add your watchlist in the sidebar, then click **Scan Watchlist** or enable auto-refresh.")

    # auto-refresh loop at end of tab
    if auto_refresh:
        time.sleep(refresh_seconds)
        st.rerun()
