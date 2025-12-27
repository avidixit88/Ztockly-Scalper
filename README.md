# Ztockly Scalping Scanner v3 (In‑App Alerts)

What you get:
- Watchlist scanner (ranked)
- Two modes:
  1) **Fast scalp**
  2) **Cleaner signals**
- **Time-of-day filter** (ET): Opening 90 min / Midday / Power hour
- **Cooldown rule** so the same ticker doesn’t “spam” you
- **In‑app alerts panel** (no email/webhook): signals are captured and shown inside Streamlit

> This is an analytics tool, not financial advice.

---

## 1) Install
```bash
pip install -r requirements.txt
```

## 2) Set your Alpha Vantage key
macOS/Linux:
```bash
export ALPHAVANTAGE_API_KEY="YOUR_KEY"
```

Windows (PowerShell):
```powershell
setx ALPHAVANTAGE_API_KEY "YOUR_KEY"
```

## 3) Run
```bash
streamlit run app.py
```

---

## In‑App Alerts
In the sidebar:
- Turn on **Capture alerts in-app**
- Set **Alert score threshold** (e.g., 80+)
- Set **Cooldown minutes** (per ticker)

Then:
- Use **Auto-refresh** to continuously scan
- Open the **Alerts** tab to see new alert “cards” in real time
- Use “Clear alerts” to reset

Notes:
- Alerts are stored only in the current Streamlit session (not persisted after restart).
- Keep watchlists small (5–15) to stay within API limits.
