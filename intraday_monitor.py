import datetime
import json
import os
import glob
import pandas as pd
import requests
import yfinance as yf
import warnings

warnings.simplefilter(action='ignore', category=FutureWarning)

CACHE_DIR = "stock_cache"
os.makedirs(CACHE_DIR, exist_ok=True)

TELEGRAM_BOT_TOKEN = os.environ.get("TELEGRAM_BOT_TOKEN", "")
TELEGRAM_CHAT_ID = os.environ.get("TELEGRAM_CHAT_ID", "")

# 手動強制監控清單（即使 JSON 尚未更新，也會隨時被納入監控）
CUSTOM_WATCH_LIST = [
    "4912", "5508", "5483", "1321", "5465", "1334", 
    "3631", "6277", "8171", "5426", "4577", "6525"
]


def to_float(x, default=0.0):
    try:
        return float(str(x).replace(",", "").replace("%", "").strip())
    except Exception:
        return default


def send_telegram_message(message: str):
    if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID:
        print("⚠ 未設定 Telegram Token 或 Chat ID，略過發送。")
        return
    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
    payload = {
        "chat_id": TELEGRAM_CHAT_ID,
        "text": message,
        "parse_mode": "Markdown",
        "disable_web_page_preview": True
    }
    try:
        resp = requests.post(url, json=payload, timeout=10)
        if resp.status_code == 200:
            print("📲 Telegram 訊息發送成功！")
        else:
            print(f"⚠ Telegram 發送失敗: {resp.text}")
    except Exception as e:
        print(f"⚠ Telegram 連線錯誤: {e}")


def load_sent_alerts(date_str: str) -> set:
    path = os.path.join(CACHE_DIR, f"alerts_{date_str}.json")
    if os.path.exists(path):
        try:
            with open(path, "r", encoding="utf-8") as f:
                return set(json.load(f))
        except Exception:
            return set()
    return set()


def save_sent_alerts(date_str: str, alerts_set: set):
    path = os.path.join(CACHE_DIR, f"alerts_{date_str}.json")
    try:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(list(alerts_set), f, ensure_ascii=False)
    except Exception:
        pass


def load_watchlist_fast():
    """
    優先讀取 intraday_watchlist.json；
    若無 JSON 或手動自選股有新增，自動從最近 6 份 market_*.csv 補齊資料。
    """
    watchlist_path = os.path.join(CACHE_DIR, "intraday_watchlist.json")
    stocks_map = {}
    base_date = ""

    # 1. 讀取主程式匯出的 JSON
    if os.path.exists(watchlist_path):
        try:
            with open(watchlist_path, "r", encoding="utf-8") as f:
                data = json.load(f)
                base_date = str(data.get("base_date", ""))
                for item in data.get("stocks", []):
                    code = str(item.get("股票代號", "")).strip()
                    if code:
                        stocks_map[code] = item
            print(f"⚡ 成功載入輕量快取清單 (基準日: {base_date}，共 {len(stocks_map)} 檔)")
        except Exception as e:
            print(f"⚠ 讀取 intraday_watchlist.json 失敗: {e}")

    # 2. 檢查是否有不在 JSON 內的 CUSTOM_WATCH_LIST，若有則從最近 6 個 CSV 快速補齊
    missing_codes = [c for c in CUSTOM_WATCH_LIST if c not in stocks_map]
    if missing_codes or not stocks_map:
        csv_files = sorted(glob.glob(os.path.join(CACHE_DIR, "market_*.csv")), reverse=True)[:6]
        if len(csv_files) >= 2:
            if not base_date:
                base_date = os.path.basename(csv_files[0]).replace("market_", "").replace(".csv", "")
            dfs = []
            for idx, fpath in enumerate(csv_files):
                try:
                    df = pd.read_csv(fpath, dtype={"股票代號": str})
                    df = df[df["股票代號"].isin(missing_codes)]
                    if idx == 0:
                        df = df[["市場", "股票代號", "股票名稱", "收盤價"]].rename(columns={"收盤價": "收盤價_0"})
                    else:
                        df = df[["股票代號", "收盤價"]].rename(columns={"收盤價": f"收盤價_{idx}"})
                    dfs.append(df)
                except Exception:
                    pass
            if dfs:
                merged = dfs[0]
                for d in dfs[1:]:
                    merged = pd.merge(merged, d, on="股票代號", how="left")
                prev_cols = [c for c in merged.columns if c.startswith("收盤價_") and c != "收盤價_0"]
                merged["5日最高收盤"] = merged[prev_cols].max(axis=1)
                if "收盤價_1" not in merged.columns:
                    merged["收盤價_1"] = merged["收盤價_0"]
                for row in merged.to_dict("records"):
                    stocks_map[str(row["股票代號"]).strip()] = row
                print(f"🔧 已從歷史 CSV 自動補齊 {len(merged)} 檔自選股基準價。")

    return pd.DataFrame(list(stocks_map.values())), base_date


def check_all_intraday_signals(df_5m, df_15m, prev_close, is_volume_safe, is_above_5d_high):
    """【盤中訊號核心引擎 2.1】"""
    if df_5m is None or len(df_5m) < 3 or not is_volume_safe or not is_above_5d_high:
        return [], {}

    c0_5m = float(df_5m['Close'].iloc[-1])
    c1_5m = float(df_5m['Close'].iloc[-2])
    c2_5m = float(df_5m['Close'].iloc[-3])

    o0_5m = float(df_5m['Open'].iloc[-1])
    o1_5m = float(df_5m['Open'].iloc[-2])

    pct_change = ((c0_5m - prev_close) / prev_close) * 100 if prev_close > 0 else 0.0

    # 漲停與平盤下過濾
    if pct_change >= 9.4 or pct_change <= 0.0 or (c0_5m - c1_5m) / c1_5m * 100 >= 9.2:
        return [], {}

    # 計算盤中 VWAP
    typical_price = (df_5m['High'] + df_5m['Low'] + df_5m['Close']) / 3.0
    total_vol = float(df_5m['Volume'].sum())
    vwap = float((typical_price * df_5m['Volume']).sum() / total_vol) if total_vol > 0 else c0_5m

    if c0_5m < (vwap * 0.995):
        return [], {}

    triggered_signals = []
    signal_data = {"c0": c0_5m, "pct_change": pct_change, "vwap": vwap}

    # 策略一：雙時框連續收盤向上墊高
    cond_5m_up = (c0_5m > c1_5m) and (c1_5m > c2_5m)
    cond_15m_up = False
    if df_15m is not None and len(df_15m) >= 3:
        c0_15m = float(df_15m.iloc[-1])
        c1_15m = float(df_15m.iloc[-2])
        c2_15m = float(df_15m.iloc[-3])
        cond_15m_up = (c0_15m > c1_15m) and (c1_15m > c2_15m)

    if cond_5m_up and cond_15m_up:
        triggered_signals.append(1)

    # 策略五：5分K連續跳空兩次
    gap1 = (o1_5m > c2_5m)
    gap2 = (o0_5m > c1_5m)
    strong_k = (c1_5m >= o1_5m) and (c0_5m >= o0_5m)
    if gap1 and gap2 and strong_k:
        triggered_signals.append(5)

    # 策略三：主力階梯推升型態
    if len(df_5m) >= 5:
        c3_5m = float(df_5m['Close'].iloc[-4])
        is_step_up = cond_5m_up and (c2_5m > c3_5m)
        is_price_stable = (c0_5m / c1_5m) <= 1.035
        if is_step_up and is_price_stable:
            triggered_signals.append(3)

    # 策略二：20分鐘高點突破
    if len(df_5m) >= 6:
        recent_4_closes = df_5m['Close'].iloc[-5:-1]
        recent_4_highs = df_5m['High'].iloc[-5:-1]
        max_high_20m = float(recent_4_highs.max())
        min_close_20m = float(recent_4_closes.min())

        is_holding_high = min_close_20m >= (max_high_20m * 0.988)
        is_breaking_out = c0_5m > max_high_20m
        if is_holding_high and is_breaking_out:
            triggered_signals.append(2)
            signal_data["max_h20"] = max_high_20m

    # 策略四：高檔鎖碼橫盤突破
    if len(df_5m) >= 8:
        recent_highs = df_5m['High'].iloc[:-1]
        day_max_high = float(recent_highs.max())
        recent_closes_4 = df_5m['Close'].iloc[-5:-1]
        min_recent_close_4 = float(recent_closes_4.min())

        is_high_plateau = min_recent_close_4 >= (day_max_high * 0.985)
        is_challenging_high = c0_5m >= (day_max_high * 0.99)
        if is_high_plateau and is_challenging_high:
            triggered_signals.append(4)
            signal_data["day_max"] = day_max_high

    return triggered_signals, signal_data


def intraday_check_strategy(res_df: pd.DataFrame, date_0: str):
    if res_df is None or res_df.empty:
        print("ℹ 目前無符合的監控標的。", flush=True)
        return

    today_str = datetime.date.today().strftime("%Y%m%d")
    sent_alerts = load_sent_alerts(today_str)

    # 判斷目前讀取的快取是「昨收(盤中執行)」還是「今收(盤後測試)」
    is_after_hours_data = (str(date_0) == today_str)

    total_len = len(res_df)
    print(f"🚀 啟動盤中極速五大策略引擎 (基準日:{date_0} / 共 {total_len} 檔)...", flush=True)

    records = res_df.to_dict("records")
    batch_size = 25
    triggered_total = 0

    for b_start in range(0, total_len, batch_size):
        batch_rows = records[b_start:b_start + batch_size]
        b_end = min(b_start + batch_size, total_len)
        print(f"⏳ 正在批次掃描第 {b_start + 1} ~ {b_end} / {total_len} 檔...", flush=True)

        ticker_map = {}
        for row in batch_rows:
            code = str(row['股票代號']).strip()
            market = str(row.get('市場', '上市')).strip()
            suffix = ".TWO" if "上櫃" in market else ".TW"
            ticker_map[f"{code}{suffix}"] = row

        tickers_list = list(ticker_map.keys())
        try:
            batch_data = yf.download(
                tickers=tickers_list,
                period="1d",
                interval="5m",
                group_by="ticker",
                threads=True,
                progress=False,
                timeout=10
            )
        except Exception as e:
            print(f"⚠ 批次下載發生例外: {e}", flush=True)
            continue

        if batch_data is None or batch_data.empty:
            continue

        for tk_sym, row in ticker_map.items():
            code = str(row['股票代號']).strip()
            name = str(row.get('股票名稱', '')).strip()

            try:
                if isinstance(batch_data.columns, pd.MultiIndex):
                    if tk_sym not in batch_data.columns.get_level_values(0):
                        continue
                    df_5m = batch_data[tk_sym].copy()
                else:
                    df_5m = batch_data.copy()

                df_5m = df_5m.dropna(subset=['Close'])
                if df_5m is None or len(df_5m) < 3:
                    continue

                c0_daily = to_float(row.get('收盤價_0', 0))
                c1_daily = to_float(row.get('收盤價_1', 0))
                max5_prev = to_float(row.get('5日最高收盤', 0))

                if is_after_hours_data:
                    prev_close = c1_daily
                    max_5d_close = max5_prev
                else:
                    prev_close = c0_daily
                    max_5d_close = max(c0_daily, max5_prev)

                if prev_close <= 0 or max_5d_close <= 0:
                    continue

                current_price = float(df_5m['Close'].iloc[-1])
                is_above_5d_high = current_price > max_5d_close
                if not is_above_5d_high:
                    continue

                df_15m_close = df_5m['Close'].resample('15min').last().dropna()

                v0_5m = float(df_5m['Volume'].iloc[-1]) / 1000.0
                v1_5m = float(df_5m['Volume'].iloc[-2]) / 1000.0
                effective_vol = max(v0_5m, v1_5m)

                c0_5m = float(df_5m['Close'].iloc[-1])
                o0_5m = float(df_5m['Open'].iloc[-1])
                h0_5m = float(df_5m['High'].iloc[-1])

                upper_shadow = h0_5m - max(c0_5m, o0_5m)
                body = abs(c0_5m - o0_5m)

                is_blowoff = (v1_5m > 0 and v0_5m > v1_5m * 4.0 and c0_5m < o0_5m) or (
                    v1_5m > 0 and (v0_5m > v1_5m * 1.8) and (upper_shadow > max(body * 1.5, c0_5m * 0.015))
                )

                is_liquid = effective_vol >= 10.0
                is_volume_safe = is_liquid and (v1_5m > 0) and not is_blowoff

                triggered_list, sig_data = check_all_intraday_signals(
                    df_5m, df_15m_close, prev_close, is_volume_safe, is_above_5d_high
                )
                if not triggered_list:
                    continue

                c0_s = sig_data.get("c0")
                pct_s = sig_data.get("pct_change")
                vwap_s = sig_data.get("vwap", 0)
                sign = "+" if pct_s >= 0 else ""
                yh_link = f"https://tw.stock.yahoo.com/quote/{code}/technical-analysis"

                strategy_titles = {
                    1: ("策略一：雙時框連拉", f"突破5日高＋站穩均價線 (`{vwap_s:.2f}`) 雙時框向上！"),
                    2: ("策略二：20分鐘高點突破", f"高檔抗跌帶量突破 20分高 (`{sig_data.get('max_h20', 0):.2f}`)！"),
                    3: ("策略三：主力階梯推升", f"站穩均價線 (`{vwap_s:.2f}`)，階梯式穩健墊高！"),
                    4: ("策略四：全日高檔鎖碼突破", f"全日高姿態整理，再度挑戰當日高 (`{sig_data.get('day_max', 0):.2f}`)！"),
                    5: ("策略五：5分K連續雙跳空", f"突破5日高且 5分K 連兩根跳空強攻！")
                }

                for s_id in triggered_list:
                    if s_id == 3 and 1 in triggered_list:
                        continue
                    if s_id == 4 and 2 in triggered_list:
                        continue
                    alert_key = f"{code}_S{s_id}"
                    if alert_key in sent_alerts:
                        continue

                    title_str, status_str = strategy_titles[s_id]
                    msg = (
                        f"🚨 *【{title_str}】*\n"
                        f"• 股票: {name} ({code})\n"
                        f"• 漲跌幅: `{sign}{pct_s:.2f}%` (現價: `{c0_s:.2f}` / 均價: `{vwap_s:.2f}`)\n"
                        f"• 狀態: {status_str}\n"
                        f"• 5分K量: `{effective_vol:.0f}張`\n"
                        f"🔗 [Yahoo 技術線圖]({yh_link})"
                    )
                    print(msg, flush=True)
                    send_telegram_message(msg)
                    sent_alerts.add(alert_key)
                    triggered_total += 1

            except Exception as e:
                print(f"⚠ 檢查股票 {code} ({name}) 時發生錯誤: {e}", flush=True)

        save_sent_alerts(today_str, sent_alerts)

    print(f"✅ 盤中即時監控掃描完畢！本次共觸發 {triggered_total} 則新訊號。", flush=True)


if __name__ == "__main__":
    target_df, base_date = load_watchlist_fast()
    intraday_check_strategy(target_df, base_date)
