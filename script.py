import datetime
import time
import os
import json
import webbrowser
import itertools
import numpy as np
import pandas as pd
import requests
import yfinance as yf
import warnings

warnings.simplefilter(action='ignore', category=pd.errors.PerformanceWarning)

TWSE_HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36",
    "Accept": "application/json",
    "Connection": "keep-alive"
}

TPEX_HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36",
    "Accept": "application/json, text/javascript, */*; q=0.01",
    "Accept-Language": "zh-TW,zh;q=0.9,en-US;q=0.8,en;q=0.7",
    "Referer": "https://www.tpex.org.tw/",
    "Connection": "keep-alive",
    "Cache-Control": "max-age=0"
}

OPENAPI_HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36",
    "Accept": "application/json",
}

CACHE_DIR = "stock_cache"
if not os.path.exists(CACHE_DIR):
    os.makedirs(CACHE_DIR)

WARNINGS = []

TELEGRAM_BOT_TOKEN = os.environ.get("TELEGRAM_BOT_TOKEN", "")
TELEGRAM_CHAT_ID = os.environ.get("TELEGRAM_CHAT_ID", "")

# ==========================================
# Telegram 推播與盤中監控引擎
# ==========================================
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

def check_all_intraday_signals(df_5m, df_15m, prev_close, is_volume_safe, is_above_5d_high):
    if df_5m is None or len(df_5m) < 3 or not is_volume_safe or not is_above_5d_high:
        return [], {}

    c0_5m = float(df_5m['Close'].iloc[-1])
    c1_5m = float(df_5m['Close'].iloc[-2])
    c2_5m = float(df_5m['Close'].iloc[-3])
    o0_5m = float(df_5m['Open'].iloc[-1])
    o1_5m = float(df_5m['Open'].iloc[-2])

    pct_change = ((c0_5m - prev_close) / prev_close) * 100 if prev_close > 0 else 0.0

    if pct_change >= 9.4 or pct_change <= 0.0 or (c0_5m - c1_5m) / c1_5m * 100 >= 9.2:
        return [], {}

    typical_price = (df_5m['High'] + df_5m['Low'] + df_5m['Close']) / 3.0
    total_vol = float(df_5m['Volume'].sum())
    vwap = float((typical_price * df_5m['Volume']).sum() / total_vol) if total_vol > 0 else c0_5m

    if c0_5m < (vwap * 0.995):
        return [], {}

    triggered_signals = []
    signal_data = {"c0": c0_5m, "pct_change": pct_change, "vwap": vwap}

    cond_5m_up = (c0_5m > c1_5m) and (c1_5m > c2_5m)
    cond_15m_up = False
    if df_15m is not None and len(df_15m) >= 3:
        c0_15m = float(df_15m.iloc[-1])
        c1_15m = float(df_15m.iloc[-2])
        c2_15m = float(df_15m.iloc[-3])
        cond_15m_up = (c0_15m > c1_15m) and (c1_15m > c2_15m)

    if cond_5m_up and cond_15m_up:
        triggered_signals.append(1)

    gap1 = (o1_5m > c2_5m)
    gap2 = (o0_5m > c1_5m)
    strong_k = (c1_5m >= o1_5m) and (c0_5m >= o0_5m)
    if gap1 and gap2 and strong_k:
        triggered_signals.append(5)

    if len(df_5m) >= 5:
        c3_5m = float(df_5m['Close'].iloc[-4])
        is_step_up = cond_5m_up and (c2_5m > c3_5m)
        is_price_stable = (c0_5m / c1_5m) <= 1.035
        if is_step_up and is_price_stable:
            triggered_signals.append(3)

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

def intraday_check_strategy(res_df, date_0):
    if res_df is None or res_df.empty:
        print("ℹ 目前無符合的監控標的。", flush=True)
        return

    today_str = datetime.date.today().strftime("%Y%m%d")
    sent_alerts = load_sent_alerts(today_str)
    is_after_hours_data = (str(date_0) == today_str)

    total_len = len(res_df)
    print(f"🚀 啟動盤中極速五大策略引擎，共檢查 {total_len} 檔...", flush=True)

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
                tickers=tickers_list, period="1d", interval="5m",
                group_by="ticker", threads=True, progress=False, timeout=10
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

# ==========================================
# 盤後資料獲取與工具函數
# ==========================================
def is_valid_cache(filepath: str, min_size: int = 50) -> bool:
    return os.path.exists(filepath) and os.path.getsize(filepath) >= min_size

def safe_read_csv(filepath: str, **kwargs) -> pd.DataFrame:
    if not is_valid_cache(filepath):
        if os.path.exists(filepath):
            try: os.remove(filepath)
            except Exception: pass
        return pd.DataFrame()
    try:
        return pd.read_csv(filepath, **kwargs)
    except Exception:
        try: os.remove(filepath)
        except Exception: pass
        return pd.DataFrame()

def to_float(x, default=0.0):
    try:
        return float(str(x).replace(",", "").replace("%", "").strip())
    except Exception:
        return default

def get_tables(data):
    if not isinstance(data, dict): return []
    if data.get("tables"): return data["tables"]
    if data.get("fields") and data.get("data"): return [data]
    return []

def safe_request_json(url: str, max_retries: int = 2, headers=TWSE_HEADERS):
    for _ in range(max_retries):
        try:
            resp = requests.get(url, headers=headers, timeout=6)
            if resp.status_code == 200:
                try: return resp.json()
                except Exception: return {}
        except Exception: pass
        time.sleep(0.5)
    return {}

def tpex_date_params(date_str: str):
    dt = datetime.datetime.strptime(date_str, "%Y%m%d")
    roc = f"{dt.year - 1911}/{dt.strftime('%m/%d')}"
    ad = f"{dt.year}%2F{dt.strftime('%m')}%2F{dt.strftime('%d')}"
    return roc, ad

def tpex_fetch_tables(url: str, date_str: str, roc_date: str, tag: str):
    for _ in range(2):
        try:
            resp = requests.get(url, headers=TPEX_HEADERS, timeout=6)
            if resp.status_code != 200:
                time.sleep(0.5); continue
            data = resp.json()
            if str(data.get("stat", "")).lower() != "ok": return []
            top_date = str(data.get("date", "")).strip()
            if top_date and top_date != date_str: return []
            good = []
            for t in get_tables(data):
                t_date = str(t.get("date", "")).strip()
                if t_date and t_date != roc_date: continue
                good.append(t)
            if good: return good
        except Exception: pass
        time.sleep(0.5)
    return []

def fetch_twse_daily(date_str: str) -> pd.DataFrame:
    url = f"https://www.twse.com.tw/rwd/zh/afterTrading/MI_INDEX?date={date_str}&type=ALLBUT0999&response=json"
    data = safe_request_json(url, headers=TWSE_HEADERS)
    if not data or data.get("stat") != "OK":
        WARNINGS.append(f"上市價量({date_str}) 抓取失敗")
        return pd.DataFrame()
    target_table = next((t for t in data.get("tables", []) if "每日收盤行情" in t.get("title", "")), None)
    if not target_table: return pd.DataFrame()
    df = pd.DataFrame(target_table["data"], columns=target_table["fields"])
    df = df.rename(columns={"證券代號": "股票代號", "證券名稱": "股票名稱", "成交股數": "成交量"})
    df["市場"] = "上市"
    df['股票代號'] = df['股票代號'].astype(str).str.strip()
    df = df[df['股票代號'].str.len() == 4]
    return df[["市場", "股票代號", "股票名稱", "開盤價", "最高價", "最低價", "收盤價", "成交量"]]

def fetch_tpex_daily(date_str: str) -> pd.DataFrame:
    roc_date, ad = tpex_date_params(date_str)
    url = f"https://www.tpex.org.tw/www/zh-tw/afterTrading/otc?date={ad}&type=EW&id=&response=json"
    tables = tpex_fetch_tables(url, date_str, roc_date, "上櫃價量")
    for t in tables:
        fields = [str(f).strip() for f in t.get("fields", [])]
        df = pd.DataFrame(t.get("data", []))
        if df.empty or df.shape[1] != len(fields): continue
        df.columns = fields
        def col(prefix): return next((c for c in df.columns if c.startswith(prefix)), None)
        c_code, c_name = col("代號"), col("名稱")
        c_close, c_open = col("收盤"), col("開盤")
        c_high, c_low, c_vol = col("最高"), col("最低"), col("成交股數")
        if not all([c_code, c_name, c_close, c_open, c_high, c_low, c_vol]): continue
        out = pd.DataFrame({
            "市場": "上櫃", "股票代號": df[c_code].astype(str).str.strip(),
            "股票名稱": df[c_name].astype(str).str.strip(), "開盤價": df[c_open],
            "最高價": df[c_high], "最低價": df[c_low], "收盤價": df[c_close], "成交量": df[c_vol],
        })
        out = out[out["股票代號"].str.isdigit() & (out["股票代號"].str.len() == 4)]
        if len(out) > 100: return out.reset_index(drop=True)
    WARNINGS.append(f"上櫃價量({date_str}) 抓取失敗")
    return pd.DataFrame()

_OTC_SIGS = {}
def _is_dup_otc(date_str: str, df: pd.DataFrame) -> bool:
    o = df[df["市場"] == "上櫃"].sort_values("股票代號")
    if o.empty: return False
    sig = hash(tuple(pd.to_numeric(o["收盤價"], errors="coerce").round(2).fillna(-1).tolist()))
    if sig in _OTC_SIGS and _OTC_SIGS[sig] != date_str: return True
    _OTC_SIGS[sig] = date_str
    return False

def get_market_data(date_str: str) -> pd.DataFrame:
    cache_file = os.path.join(CACHE_DIR, f"market_{date_str}.csv")
    df_cached = safe_read_csv(cache_file, dtype={"股票代號": str})
    if not df_cached.empty and "上櫃" in df_cached["市場"].values and not _is_dup_otc(date_str, df_cached):
        return df_cached
    print(f"🌐 [{date_str}] 抓取【上市+上櫃價量】資料...")
    df_twse = fetch_twse_daily(date_str)
    if df_twse.empty: return pd.DataFrame()
    time.sleep(0.3)
    df_tpex = fetch_tpex_daily(date_str)
    df_all = pd.concat([df_twse, df_tpex], ignore_index=True)
    df_all['股票名稱'] = df_all['股票名稱'].astype(str).str.strip()
    for col in ["開盤價", "最高價", "最低價", "收盤價", "成交量"]:
        df_all[col] = df_all[col].astype(str).str.replace(",", "").str.replace("--", "").str.replace("---", "").str.strip()
        df_all[col] = pd.to_numeric(df_all[col], errors="coerce")
    if not df_tpex.empty and _is_dup_otc(date_str, df_all):
        df_all = df_all[df_all["市場"] != "上櫃"]
        WARNINGS.append(f"上櫃價量({date_str}) 疑似重複資料，已捨棄")
    if not df_all.empty:
        df_all.to_csv(cache_file, index=False, encoding="utf-8-sig")
    return df_all

def fetch_twse_inst(date_str: str) -> pd.DataFrame:
    url = f"https://www.twse.com.tw/rwd/zh/fund/T86?date={date_str}&selectType=ALLBUT0999&response=json"
    data = safe_request_json(url, headers=TWSE_HEADERS)
    if not data or data.get("stat") != "OK":
        WARNINGS.append(f"上市法人({date_str}) 抓取失敗")
        return pd.DataFrame()
    tables = get_tables(data)
    if not tables: return pd.DataFrame()
    target_table = next((t for t in tables if "三大法人" in str(t.get("title", ""))), tables[0])
    fields = [str(c).strip() for c in target_table.get("fields", [])]
    try:
        df = pd.DataFrame(target_table.get("data", []), columns=fields)
    except Exception:
        return pd.DataFrame()
    if df.empty: return pd.DataFrame()
    code_col = next((c for c in df.columns if '證券代號' in c or '代號' in c), None)
    f_col = (next((c for c in df.columns if '外陸資' in c and '買賣超' in c), None) or next((c for c in df.columns if '外資' in c and '買賣超' in c), None))
    t_col = next((c for c in df.columns if '投信' in c and '買賣超' in c), None)
    if not (f_col and t_col and code_col): return pd.DataFrame()
    df = df.rename(columns={code_col: "股票代號", f_col: "外資買賣超", t_col: "投信買賣超"})
    df['股票代號'] = df['股票代號'].astype(str).str.strip()
    df = df[df['股票代號'].str.len() == 4]
    return df[["股票代號", "外資買賣超", "投信買賣超"]]

def fetch_tpex_inst(date_str: str) -> pd.DataFrame:
    roc_date, ad = tpex_date_params(date_str)
    url = f"https://www.tpex.org.tw/www/zh-tw/insti/dailyTrade?type=Daily&sect=EW&date={ad}&id=&response=json"
    tables = tpex_fetch_tables(url, date_str, roc_date, "上櫃法人")
    for t in tables:
        aa = t.get("data", [])
        if len(aa) <= 50: continue
        rows = []
        for item in aa:
            code = str(item[0]).strip()
            if len(item) > 13 and code.isdigit() and len(code) == 4:
                rows.append({"股票代號": code, "外資買賣超": str(item[10]).strip(), "投信買賣超": str(item[13]).strip()})
        if rows: return pd.DataFrame(rows)
    WARNINGS.append(f"上櫃法人({date_str}) 抓取失敗")
    return pd.DataFrame()

def get_inst_data(date_str: str) -> pd.DataFrame:
    cache_file = os.path.join(CACHE_DIR, f"inst_{date_str}.csv")
    df_cached = safe_read_csv(cache_file, dtype={"股票代號": str})
    if len(df_cached) > 500: return df_cached
    print(f"🌐 [{date_str}] 抓取【上市+上櫃法人籌碼】資料...")
    df_twse = fetch_twse_inst(date_str)
    time.sleep(0.3)
    df_tpex = fetch_tpex_inst(date_str)
    valid_dfs = [df for df in [df_twse, df_tpex] if not df.empty]
    if valid_dfs:
        df_all = pd.concat(valid_dfs, ignore_index=True)
        df_all['股票代號'] = df_all['股票代號'].astype(str).str.strip()
        for col in ["外資買賣超", "投信買賣超"]:
            df_all[col] = df_all[col].astype(str).str.replace(",", "").str.replace("--", "").str.replace("---", "").str.strip()
            df_all[col] = pd.to_numeric(df_all[col], errors="coerce").fillna(0)
        if not df_twse.empty and not df_tpex.empty:
            df_all.to_csv(cache_file, index=False, encoding="utf-8-sig")
        return df_all
    return pd.DataFrame()

def get_tdcc_data() -> pd.DataFrame:
    today_str = datetime.date.today().strftime("%Y%m%d")
    cache_file = os.path.join(CACHE_DIR, f"tdcc_{today_str}.csv")
    df_cached = safe_read_csv(cache_file, dtype={"股票代號": str})
    if not df_cached.empty: return df_cached
    print("🌐 抓取【集保千張大戶】...")
    url = "https://smart.tdcc.com.tw/opendata/getOD.ashx?id=1-5"
    for _ in range(2):
        try:
            resp = requests.get(url, headers=TWSE_HEADERS, timeout=8)
            if resp.status_code == 200:
                lines = resp.text.strip().split('\n')
                rows = []
                for line in lines[1:]:
                    clean_line = line.replace('"', '').strip()
                    if not clean_line: continue
                    cols = clean_line.split(',')
                    if len(cols) >= 6:
                        code = cols[1].strip()
                        if cols[2] == '15' and len(code) == 4:
                            ratio = float(cols[5]) if cols[5].replace('.', '', 1).isdigit() else 0.0
                            rows.append({"股票代號": code, "千張大戶比例(%)": ratio})
                if rows:
                    df_tdcc = pd.DataFrame(rows)
                    df_tdcc.to_csv(cache_file, index=False, encoding="utf-8-sig")
                    return df_tdcc
        except Exception: pass
        time.sleep(0.5)
    WARNINGS.append("千張大戶資料 抓取失敗")
    return pd.DataFrame(columns=["股票代號", "千張大戶比例(%)"])

def get_foreign_holdings() -> pd.DataFrame:
    today_str = datetime.date.today().strftime("%Y%m%d")
    cache_file = os.path.join(CACHE_DIR, f"foreign_holdings_{today_str}.csv")
    df_cached = safe_read_csv(cache_file, dtype={"股票代號": str})
    if not df_cached.empty: return df_cached
    print("🌐 抓取【外資持股總比例與發行張數】...")
    rows = []
    for offset in range(5):
        d = datetime.date.today() - datetime.timedelta(days=offset)
        if d.weekday() >= 5: continue
        url = f"https://www.twse.com.tw/rwd/zh/fund/MI_QFIIS?date={d.strftime('%Y%m%d')}&selectType=ALLBUT0999&response=json"
        data = safe_request_json(url, max_retries=1)
        if not data or data.get("stat") != "OK": continue
        for t in get_tables(data):
            try:
                fields = [str(c).strip() for c in t["fields"]]
                df_t = pd.DataFrame(t["data"], columns=fields)
            except Exception: continue
            c_code = next((c for c in fields if "代號" in c), None)
            c_issue = next((c for c in fields if "發行股數" in c), None)
            c_ratio = next((c for c in fields if "外資" in c and ("持股比率" in c or "持股率" in c)), None)
            if not (c_code and c_issue and c_ratio): continue
            for _, r in df_t.iterrows():
                code = str(r[c_code]).strip()
                if len(code) == 4 and code.isdigit():
                    rows.append({"股票代號": code, "外資持股比例(%)": to_float(r[c_ratio]), "發行總張數": to_float(r[c_issue]) / 1000})
        if rows: break
    data = safe_request_json("https://www.tpex.org.tw/openapi/v1/tpex_3insti_qfii", max_retries=2, headers=OPENAPI_HEADERS)
    if isinstance(data, list) and data and isinstance(data[0], dict):
        keys = list(data[0].keys())
        def find_key(*must, exclude=()):
            for k in keys:
                kl = k.lower()
                if all(m.lower() in kl for m in must) and not any(e.lower() in kl for e in exclude): return k
            return None
        k_code = find_key("code") or find_key("代號")
        k_issue = find_key("issued") or find_key("發行")
        k_ratio = find_key("hold", "ratio") or find_key("hold", "percent") or find_key("持股", "比", exclude=("尚可",))
        if k_code and k_issue and k_ratio:
            for item in data:
                code = str(item.get(k_code, "")).strip()
                if len(code) == 4 and code.isdigit():
                    rows.append({"股票代號": code, "外資持股比例(%)": to_float(item.get(k_ratio)), "發行總張數": to_float(item.get(k_issue)) / 1000})
    df_holdings = pd.DataFrame(rows)
    if not df_holdings.empty and (df_holdings["外資持股比例(%)"] > 0).any():
        df_holdings = df_holdings.drop_duplicates(subset=["股票代號"])
        df_holdings.to_csv(cache_file, index=False, encoding="utf-8-sig")
        return df_holdings
    return pd.DataFrame(columns=["股票代號", "外資持股比例(%)", "發行總張數"])

def get_last_n_trading_days_data(n_market=155, n_inst=60):
    valid_dfs, valid_inst = [], []
    today = datetime.date.today()
    offset = 0
    print(f"\n⏳ 準備檢查 {n_market} 個交易日的歷史快取...")
    while len(valid_dfs) < n_market:
        if offset > 300: break
        target_date = today - datetime.timedelta(days=offset)
        offset += 1
        if target_date.weekday() >= 5: continue
        date_str = target_date.strftime("%Y%m%d")
        df_m = get_market_data(date_str)
        if not df_m.empty:
            valid_dfs.append((date_str, df_m))
            if len(valid_inst) < n_inst:
                valid_inst.append((date_str, get_inst_data(date_str)))
            else:
                valid_inst.append((date_str, pd.DataFrame()))
        else:
            time.sleep(0.1)
    return valid_dfs, valid_inst

def format_stock_cell(row):
    code, name = row['股票代號'], row['股票名稱']
    url = f"https://tw.stock.yahoo.com/quote/{code}/technical-analysis"
    return f"<a href='{url}' target='_blank' class='stock-link'><div class='stock-name'>{name}</div><div class='stock-code'>{code}</div></a>"

def color_pct(val):
    try:
        f = float(val)
        if f > 0: return f"<span class='tag-up'>+{f:.2f}%</span>"
        elif f < 0: return f"<span class='tag-down'>{f:.2f}%</span>"
        else: return f"<span class='tag-flat'>0.00%</span>"
    except Exception: return str(val)

def main():
    print("啟動多策略選股程式 2.0 (已新增宏正/智聯服務/律勝專屬策略並移除倚強科)...\n")
    days_data, inst_data = get_last_n_trading_days_data(n_market=155, n_inst=60)
    date_0, df_0 = days_data[0]
    date_1, df_1 = days_data[1]
    date_2, df_2 = days_data[2]
    print(f"\n✅ 行情獲取完畢！ 基準日: [{date_0}]")

    df_tdcc = get_tdcc_data()
    df_holdings = get_foreign_holdings()

    df_0_merge = df_0.copy()
    df_0_merge.columns = [f"{c}_0" if c not in ["市場", "股票代號", "股票名稱"] else c for c in df_0_merge.columns]
    df_1_merge = df_1[["股票代號", "開盤價", "最高價", "最低價", "收盤價", "成交量"]].copy()
    df_1_merge.columns = [f"{c}_1" if c != "股票代號" else c for c in df_1_merge.columns]
    df_2_merge = df_2[["股票代號", "開盤價", "最高價", "最低價", "收盤價", "成交量"]].copy()
    df_2_merge.columns = [f"{c}_2" if c != "股票代號" else c for c in df_2_merge.columns]

    df_merge = pd.merge(df_0_merge, df_1_merge, on="股票代號", how="inner")
    df_merge = pd.merge(df_merge, df_2_merge, on="股票代號", how="inner")

    for i in range(3, len(days_data)):
        _, df_i = days_data[i]
        if df_i.empty: continue
        df_temp = df_i[["股票代號", "開盤價", "最高價", "最低價", "收盤價", "成交量"]].copy()
        df_temp = df_temp.rename(columns={
            "收盤價": f"收盤價_{i}", "成交量": f"成交量_{i}",
            "最高價": f"最高價_{i}", "最低價": f"最低價_{i}", "開盤價": f"開盤價_{i}"
        })
        df_merge = pd.merge(df_merge, df_temp, on="股票代號", how="left")

    df_merge = df_merge.copy()
    vol_cols = [col for col in df_merge.columns if col.startswith('成交量_')]
    for col in vol_cols:
        df_merge[col] = (df_merge[col] / 1000).round(0)

    close_cols_5 = [col for col in df_merge.columns if col.startswith('收盤價_') and 1 <= int(col.split('_')[1]) <= 5]
    close_cols_20 = [col for col in df_merge.columns if col.startswith('收盤價_') and 1 <= int(col.split('_')[1]) <= 20]
    close_cols_40 = [col for col in df_merge.columns if col.startswith('收盤價_') and 1 <= int(col.split('_')[1]) <= 40]
    close_cols_60 = [col for col in df_merge.columns if col.startswith('收盤價_') and 1 <= int(col.split('_')[1]) <= 60]
    close_cols_120 = [col for col in df_merge.columns if col.startswith('收盤價_') and 1 <= int(col.split('_')[1]) <= 120]
    high_cols_120 = [col for col in df_merge.columns if col.startswith('最高價_') and 1 <= int(col.split('_')[1]) <= 120]

    df_merge['5日最高收盤'] = df_merge[close_cols_5].max(axis=1)
    df_merge['20日最高收盤'] = df_merge[close_cols_20].max(axis=1)
    df_merge['60日最高收盤'] = df_merge[close_cols_60].max(axis=1)
    df_merge['120日最高收盤'] = df_merge[close_cols_120].max(axis=1)
    df_merge['120日最高價'] = df_merge[high_cols_120].max(axis=1)

    df_merge['20日最低收盤'] = df_merge[close_cols_20].min(axis=1)
    df_merge['40日最低收盤'] = df_merge[close_cols_40].min(axis=1)
    df_merge['60日最低收盤'] = df_merge[close_cols_60].min(axis=1)

    vol_cols_60 = [col for col in df_merge.columns if col.startswith('成交量_') and int(col.split('_')[1]) <= 60]
    df_merge['60日最大量'] = df_merge[vol_cols_60].max(axis=1)

    df_merge['5MA'] = df_merge[[f'收盤價_{j}' for j in range(5)]].mean(axis=1)
    df_merge['20MA'] = df_merge[[f'收盤價_{j}' for j in range(20)]].mean(axis=1)
    df_merge['60MA'] = df_merge[[f'收盤價_{j}' for j in range(60)]].mean(axis=1)
    df_merge['5MV'] = df_merge[[f'成交量_{j}' for j in range(1, 6)]].mean(axis=1)

    def get_max_vol_high(row):
        try:
            vols = [(float(row.get(f'成交量_{i}', -1)), i) for i in range(1, 61) if pd.notna(float(row.get(f'成交量_{i}', -1)))]
            if not vols: return 0.0
            max_vol_idx = max(vols, key=lambda item: item[0])[1]
            return float(row.get(f'最高價_{max_vol_idx}', 0.0))
        except Exception: return 0.0
    df_merge['最大量日最高價'] = df_merge.apply(get_max_vol_high, axis=1)

    def get_10d_max_vol_high(row):
        try:
            vols = [(float(row.get(f'成交量_{i}', -1)), i) for i in range(1, 11) if pd.notna(float(row.get(f'成交量_{i}', -1)))]
            if not vols: return 0.0
            max_vol_idx = max(vols, key=lambda item: item[0])[1]
            return float(row.get(f'最高價_{max_vol_idx}', 0.0))
        except Exception: return 0.0
    df_merge['近10日最大量日最高價'] = df_merge.apply(get_10d_max_vol_high, axis=1)

    df_merge['最新漲幅(%)'] = ((df_merge['收盤價_0'] - df_merge['收盤價_1']) / df_merge['收盤價_1'] * 100).round(2)
    df_merge['前一日漲幅(%)'] = ((df_merge['收盤價_1'] - df_merge['收盤價_2']) / df_merge['收盤價_2'] * 100).round(2)

    for idx in range(len(inst_data)):
        i_df = inst_data[idx][1]
        if not i_df.empty:
            tmp = i_df.copy()
            tmp[f'外資_{idx}'] = (tmp['外資買賣超'] / 1000).round(0)
            tmp[f'投信_{idx}'] = (tmp['投信買賣超'] / 1000).round(0)
            df_merge = pd.merge(df_merge, tmp[['股票代號', f'外資_{idx}', f'投信_{idx}']], on='股票代號', how='left')
            df_merge[f'外資_{idx}'] = df_merge[f'外資_{idx}'].fillna(0)
            df_merge[f'投信_{idx}'] = df_merge[f'投信_{idx}'].fillna(0)
        else:
            df_merge[f'外資_{idx}'] = 0
            df_merge[f'投信_{idx}'] = 0

    df_merge['外資近三日(張)'] = df_merge[[f'外資_{i}' for i in range(3)]].sum(axis=1)
    df_merge['投信近三日(張)'] = df_merge[[f'投信_{i}' for i in range(3)]].sum(axis=1)
    df_merge['外資近七日(張)'] = df_merge[[f'外資_{i}' for i in range(7)]].sum(axis=1)
    df_merge['投信近七日(張)'] = df_merge[[f'投信_{i}' for i in range(7)]].sum(axis=1)
    df_merge['外資近一月(張)'] = df_merge[[f'外資_{i}' for i in range(20)]].sum(axis=1)

    if not df_tdcc.empty:
        df_merge = pd.merge(df_merge, df_tdcc, on='股票代號', how='left')
        df_merge['千張大戶比例(%)'] = df_merge['千張大戶比例(%)'].fillna(0)
    else:
        df_merge['千張大戶比例(%)'] = 0

    if not df_holdings.empty:
        df_merge = pd.merge(df_merge, df_holdings, on='股票代號', how='left')
        df_merge['外資持股比例(%)'] = df_merge['外資持股比例(%)'].fillna(0)
        df_merge['發行總張數'] = df_merge['發行總張數'].fillna(1)
    else:
        df_merge['外資持股比例(%)'] = 0.0

    def calculate_technical_score_v2(row):
        score = 0
        try:
            c = float(row.get('收盤價_0', 0))
            o = float(row.get('開盤價_0', 0))
            h = float(row.get('最高價_0', 0))
            ma5 = float(row.get('5MA', 0))
            ma20 = float(row.get('20MA', 0))
            ma60 = float(row.get('60MA', 0))
            v0 = float(row.get('成交量_0', 0))
            mv5 = float(row.get('5MV', 0))
            f7 = float(row.get('外資近七日(張)', 0))
            t7 = float(row.get('投信近七日(張)', 0))
            whale = float(row.get('千張大戶比例(%)', 0))

            if c > ma5 > ma20 > ma60 and ma60 > 0: score += 25
            elif c > ma5 > ma20 and ma20 > 0: score += 18
            elif c > ma20 and ma20 > 0: score += 10

            max20 = float(row.get('20日最高收盤', 0))
            upper_ratio = (h - c) / c if c > 0 else 1.0
            if c >= o and upper_ratio <= 0.02: score += 12
            if max20 > 0 and c > max20: score += 13
            elif float(row.get('5日最高收盤', 0)) > 0 and c > float(row.get('5日最高收盤', 0)): score += 7

            if mv5 > 0 and 1.3 <= (v0 / mv5) <= 4.0 and c > o: score += 25
            elif v0 > float(row.get('成交量_1', 0)) and c > o: score += 15

            if f7 > 0 and t7 > 0: score += 15
            elif f7 > 0 or t7 > 0: score += 8
            if whale >= 60: score += 10
            elif whale >= 45: score += 5
        except Exception: pass
        return min(100, int(score))

    df_merge['技術評分(分)'] = df_merge.apply(calculate_technical_score_v2, axis=1)
    df_merge['近7日符合次數'] = "-"
    df_merge['最新法人買超(張)'] = (df_merge['外資_0'] + df_merge['投信_0']).round(0)
    df_merge['增量倍數'] = (df_merge['成交量_0'] / df_merge['成交量_1'].replace(0, np.nan)).round(2).fillna(0.0)
    df_merge['實體K漲幅(%)'] = (((df_merge['收盤價_0'] - df_merge['開盤價_0']) / df_merge['開盤價_0'].replace(0, np.nan)) * 100).round(2).fillna(0.0)
    df_merge['距離60日高點(%)'] = (((df_merge['收盤價_0'] - df_merge['60日最高收盤']) / df_merge['60日最高收盤'].replace(0, np.nan)) * 100).round(2).fillna(0.0)
    df_merge['20日震幅(%)'] = (((df_merge['收盤價_0'] - df_merge['20日最低收盤']) / df_merge['20日最低收盤'].replace(0, np.nan)) * 100).round(2).fillna(0.0)
    df_merge['60日震幅(%)'] = (((df_merge['收盤價_0'] - df_merge['60日最低收盤']) / df_merge['60日最低收盤'].replace(0, np.nan)) * 100).round(2).fillna(0.0)

    d0_s = f"{date_0[4:6]}/{date_0[6:]}"
    d1_s = f"{date_1[4:6]}/{date_1[6:]}"
    df_merge['⭐'] = df_merge['股票代號'].apply(lambda x: f"<span class='fav-star' data-code='{x}'>☆</span>")

    def is_strong_k(k):
        c_k = df_merge[f'收盤價_{k}']
        o_k = df_merge[f'開盤價_{k}'] if f'開盤價_{k}' in df_merge.columns else c_k
        h_k = df_merge[f'最高價_{k}'] if f'最高價_{k}' in df_merge.columns else c_k
        l_k = df_merge[f'最低價_{k}'] if f'最低價_{k}' in df_merge.columns else c_k
        red_k = c_k >= o_k
        upper_shadow = (h_k - c_k) / c_k.replace(0, np.nan)
        close_pos = (c_k - l_k) / (h_k - l_k).replace(0, np.nan)
        liq = df_merge[f'成交量_{k}'] >= 300
        return red_k & liq & ((upper_shadow <= 0.025) | (close_pos >= 0.65))

    # ==========================================
    # 策略函數定義 (已加入宏正/智聯服務/律勝專屬策略，移除倚強科)
    # ==========================================
    def cond1_fn(k):
        return is_strong_k(k) & (df_merge[f'成交量_{k+1}'] > 0) & (df_merge[f'最低價_{k}'] >= df_merge[f'收盤價_{k+1}']) & (df_merge[f'成交量_{k}'] > df_merge[f'成交量_{k+1}'])

    def cond2_fn(k):
        step_up = is_strong_k(k) & (df_merge[f'收盤價_{k}'] > df_merge[f'收盤價_{k+1}']) & (df_merge[f'最低價_{k}'] >= df_merge[f'收盤價_{k+1}'])
        vol_up = df_merge[f'成交量_{k}'] > df_merge[f'成交量_{k+1}']
        inst_hold = True
        for j in range(7):
            inst_hold = inst_hold & (df_merge[f'外資_{k+j}'] >= 0) & (df_merge[f'投信_{k+j}'] >= 0)
        return step_up & vol_up & inst_hold

    def cond3_fn(k):
        vol_up = is_strong_k(k) & (df_merge[f'成交量_{k}'] > df_merge[f'成交量_{k+1}']) & (df_merge[f'成交量_{k+1}'] > 0)
        t_hold = True
        for j in range(7): t_hold = t_hold & (df_merge[f'投信_{k+j}'] >= 0)
        f_buy = df_merge[f'外資_{k}'] > 0
        return vol_up & t_hold & f_buy

    def cond4_fn(k):
        close_20 = [f'收盤價_{k+j}' for j in range(1, 21)]
        max_20 = df_merge[close_20].max(axis=1)
        c_high = is_strong_k(k) & (df_merge[f'收盤價_{k}'] > max_20)
        vol_up = (df_merge[f'成交量_{k}'] > df_merge[f'成交量_{k+1}']) & (df_merge[f'成交量_{k+1}'] > 0)
        t_hold = True
        for j in range(7): t_hold = t_hold & (df_merge[f'投信_{k+j}'] >= 0)
        return c_high & vol_up & t_hold

    def cond6_fn(k):
        close_5 = [f'收盤價_{k+j}' for j in range(1, 6)]
        max_5 = df_merge[close_5].max(axis=1)
        c_high = is_strong_k(k) & (df_merge[f'收盤價_{k}'] > max_5)
        vol_up = (df_merge[f'成交量_{k}'] > df_merge[f'成交量_{k+1}']) & (df_merge[f'成交量_{k+1}'] > 0)
        gap = (df_merge[f'開盤價_{k}'] > df_merge[f'收盤價_{k+1}']) & (df_merge[f'最低價_{k}'] > df_merge[f'收盤價_{k+1}'])
        return c_high & vol_up & gap

    def cond7_fn(k):
        close_20 = [f'收盤價_{k+j}' for j in range(1, 21)]
        max_20 = df_merge[close_20].max(axis=1)
        c_high = is_strong_k(k) & (df_merge[f'收盤價_{k}'] > max_20)
        vol_up = (df_merge[f'成交量_{k}'] > df_merge[f'成交量_{k+1}']) & (df_merge[f'成交量_{k+1}'] > 0)
        p_up = (df_merge[f'最低價_{k}'] > df_merge[f'最低價_{k+1}']) & (df_merge[f'收盤價_{k}'] > df_merge[f'收盤價_{k+1}'])
        return c_high & vol_up & p_up

    def cond11_fn(k):
        c_k = df_merge[f'收盤價_{k}']
        o_k = df_merge[f'開盤價_{k}'] if f'開盤價_{k}' in df_merge else df_merge['開盤價_0']
        v_k = df_merge[f'成交量_{k}']
        mv5_k = df_merge[[f'成交量_{k+j}' for j in range(1, 6)]].mean(axis=1)
        body_pct = ((c_k - o_k) / o_k.replace(0, np.nan)) * 100
        return is_strong_k(k) & (df_merge['市場'] == '上櫃') & (body_pct >= 3.5) & (v_k >= 500) & (v_k >= mv5_k * 1.3)

    def cond13_fn(k):
        c_k = df_merge[f'收盤價_{k}']
        close_60 = [f'收盤價_{k+j}' for j in range(1, 61)]
        max_60 = df_merge[close_60].max(axis=1)
        return is_strong_k(k) & (c_k > 0) & (max_60 > 0) & (df_merge[f'成交量_{k}'] >= 500) & (c_k >= max_60 * 0.98)

    def cond15_fn(k):
        c_k = df_merge[f'收盤價_{k}']
        close_60 = [f'收盤價_{k+j}' for j in range(1, 61)]
        max_60 = df_merge[close_60].max(axis=1)
        vol_up = df_merge[f'成交量_{k}'] >= df_merge[f'成交量_{k+1}'] * 1.2
        return is_strong_k(k) & vol_up & (c_k >= max_60)

    def cond16_fn(k):
        def _get_60d_max_vol_high_at(row, offset):
            try:
                vols = [(float(row.get(f'成交量_{offset+j}', -1)), offset+j) for j in range(1, 61) if f'成交量_{offset+j}' in row]
                valid = [v for v in vols if pd.notna(v[0]) and v[0] > 0]
                if not valid: return 0.0
                idx = max(valid, key=lambda x: x[0])[1]
                return float(row.get(f'最高價_{idx}', 0.0))
            except Exception: return 0.0
        mv_high = df_merge['最大量日最高價'] if k == 0 else df_merge.apply(lambda r: _get_60d_max_vol_high_at(r, k), axis=1)
        c_k = df_merge[f'收盤價_{k}']
        c_prev = df_merge[f'收盤價_{k+1}']
        return is_strong_k(k) & (mv_high > 0) & (c_k > mv_high) & (c_prev <= mv_high)

    def cond17_fn(k):
        close_120 = [f'收盤價_{k+j}' for j in range(1, 121) if f'收盤價_{k+j}' in df_merge]
        if not close_120: return pd.Series(False, index=df_merge.index)
        max_120 = df_merge[close_120].max(axis=1)
        return is_strong_k(k) & (df_merge[f'收盤價_{k}'] > 0) & (max_120 > 0) & (df_merge[f'收盤價_{k}'] > max_120)

    def cond20_fn(k):
        f_more = (df_merge[f'外資_{k}'] > df_merge[f'外資_{k+1}']) & (df_merge[f'外資_{k+1}'] > df_merge[f'外資_{k+2}']) & (df_merge[f'外資_{k+2}'] > 0)
        vol_up = df_merge[f'成交量_{k}'] >= df_merge[f'成交量_{k+1}'] * 1.2
        p_up = is_strong_k(k) & (df_merge[f'收盤價_{k}'] > df_merge[f'收盤價_{k+1}'])
        return f_more & vol_up & p_up

    def cond22_fn(k):
        close_5 = [f'收盤價_{k+j}' for j in range(1, 6)]
        max_5 = df_merge[close_5].max(axis=1)
        c_high5 = is_strong_k(k) & (df_merge[f'收盤價_{k}'] > max_5)
        step_up = (df_merge[f'收盤價_{k}'] > df_merge[f'收盤價_{k+1}']) & (df_merge[f'收盤價_{k+1}'] > df_merge[f'收盤價_{k+2}'])
        vol_up = df_merge[f'成交量_{k}'] >= df_merge[f'成交量_{k+1}'] * 1.2
        return c_high5 & step_up & vol_up

    def cond23_fn(k):
        def _get_max_high_at(row, offset):
            try:
                vols = [(float(row.get(f'成交量_{offset+j}', -1)), offset+j) for j in range(1, 11)]
                valid = [v for v in vols if pd.notna(v[0])]
                if not valid: return 0.0
                idx = max(valid, key=lambda x: x[0])[1]
                return float(row.get(f'最高價_{idx}', 0.0))
            except Exception: return 0.0
        max_h = df_merge['近10日最大量日最高價'] if k == 0 else df_merge.apply(lambda r: _get_max_high_at(r, k), axis=1)
        return is_strong_k(k) & (max_h > 0) & (df_merge[f'收盤價_{k}'] > max_h)

    def cond32_fn(k):
        whale = df_merge['千張大戶比例(%)'] >= 50.0
        ratio = (((df_merge[f'外資_{k}'] + df_merge[f'外資_{k+1}'] + df_merge[f'投信_{k}'] + df_merge[f'投信_{k+1}']) / (df_merge[f'成交量_{k}'] + df_merge[f'成交量_{k+1}']).replace(0, np.nan)) * 100).fillna(0.0)
        return is_strong_k(k) & whale & (ratio >= 15.0)

    def cond33_fn(k):
        whale_70 = df_merge['千張大戶比例(%)'] > 65.0
        ma5_k = df_merge[[f'收盤價_{k+j}' for j in range(5)]].mean(axis=1)
        ma20_k = df_merge[[f'收盤價_{k+j}' for j in range(20)]].mean(axis=1)
        c_k = df_merge[f'收盤價_{k}']
        bullish_ma = (c_k > ma5_k) & (ma5_k > ma20_k)
        return is_strong_k(k) & whale_70 & bullish_ma

    def cond_new_high_pullback(k):
        try:
            close_cols_180 = [f'收盤價_{k+j}' for j in range(1, 181) if f'收盤價_{k+j}' in df_merge.columns]
            if len(close_cols_180) < 60: return pd.Series(False, index=df_merge.index)
            max_180 = df_merge[close_cols_180].max(axis=1)
            c_k = df_merge[f'收盤價_{k}']
            return is_strong_k(k) & (c_k >= max_180 * 0.95)
        except Exception: return pd.Series(False, index=df_merge.index)

    def cond_super_bull_strategy(k):
        try:
            c_k, v_k, v_1 = df_merge[f'收盤價_{k}'], df_merge[f'成交量_{k}'], df_merge[f'成交量_{k+1}']
            ma5_k = df_merge[[f'收盤價_{k+j}' for j in range(5)]].mean(axis=1)
            ma20_k = df_merge[[f'收盤價_{k+j}' for j in range(20)]].mean(axis=1)
            bullish_ma = (c_k > ma5_k) & (ma5_k > ma20_k)
            vol_surge = (v_k >= v_1 * 1.2) & (v_1 > 0)
            return is_strong_k(k) & bullish_ma & vol_surge
        except Exception: return pd.Series(False, index=df_merge.index)

    def cond_gap_and_hold(k):
        try:
            c_k = df_merge[f'收盤價_{k}']
            c_prev = df_merge[f'收盤價_{k+1}']
            l_k = df_merge[f'最低價_{k}']
            has_gap = (l_k > c_prev) & (c_prev > 0)
            return is_strong_k(k) & has_gap & (c_k > c_prev)
        except Exception: return pd.Series(False, index=df_merge.index)

    # ==========================================
    # 【新增】專屬策略 37、38、39 (宏正、智聯服務、律勝)
    # ==========================================
    def cond_macro_6277(k):
        """策略 37 (宏正模式)：均線多頭排列 + 實體紅K + 法人近7日無明顯賣壓且緩步增溫"""
        try:
            c = df_merge[f'收盤價_{k}']
            ma5 = df_merge[[f'收盤價_{k+j}' for j in range(5)]].mean(axis=1)
            ma20 = df_merge[[f'收盤價_{k+j}' for j in range(20)]].mean(axis=1)
            ma60 = df_merge[[f'收盤價_{k+j}' for j in range(60)]].mean(axis=1)
            bullish_ma = (c > ma5) & (ma5 > ma20) & (ma20 > ma60)
            f7 = df_merge[[f'外資_{k+j}' for j in range(7) if f'外資_{k+j}' in df_merge.columns]].sum(axis=1)
            t7 = df_merge[[f'投信_{k+j}' for j in range(7) if f'投信_{k+j}' in df_merge.columns]].sum(axis=1)
            inst_ok = (f7 >= 0) | (t7 >= 0)
            vol_stable = df_merge[f'成交量_{k}'] >= 300
            return is_strong_k(k) & bullish_ma & inst_ok & vol_stable
        except Exception:
            return pd.Series(False, index=df_merge.index)

    def cond_sys_6751(k):
        """策略 38 (智聯服務模式)：中小型股成交量放大大於1.5倍 + 創20日新高 + 紅K突圍"""
        try:
            c = df_merge[f'收盤價_{k}']
            v0 = df_merge[f'成交量_{k}']
            mv5 = df_merge[[f'成交量_{k+j}' for j in range(1, 6)]].mean(axis=1)
            close_20 = [f'收盤價_{k+j}' for j in range(1, 21)]
            max_20 = df_merge[close_20].max(axis=1)
            vol_breakout = (v0 >= mv5 * 1.5) & (v0 >= 400)
            is_new_high = c > max_20
            return is_strong_k(k) & vol_breakout & is_new_high
        except Exception:
            return pd.Series(False, index=df_merge.index)

    def cond_lv_3354(k):
        """策略 39 (律勝模式)：低基期/區間沉澱後，量增突破近60日高點"""
        try:
            c = df_merge[f'收盤價_{k}']
            close_60 = [f'收盤價_{k+j}' for j in range(1, 61)]
            max_60 = df_merge[close_60].max(axis=1)
            min_60 = df_merge[close_60].min(axis=1)
            low_base = (max_60 / min_60.replace(0, np.nan)) <= 1.45  # 前期屬於整理區間
            vol_up = df_merge[f'成交量_{k}'] >= df_merge[f'成交量_{k+1}'] * 1.3
            breakout_60 = c >= max_60 * 0.985
            return is_strong_k(k) & low_base & vol_up & breakout_60
        except Exception:
            return pd.Series(False, index=df_merge.index)

    # 註冊策略清單 (已移除原先的 25.倚強科，並加入 37、38、39)
    cand_strategies = [
        ("1.跳空不補", cond1_fn), ("2.連續墊高", cond2_fn), ("3.量增法人買", cond3_fn),
        ("4.創20日新高", cond4_fn), ("6.創五日高", cond6_fn), ("7.20日高不賣", cond7_fn),
        ("11.上櫃強紅K", cond11_fn), ("13.逼近60日高", cond13_fn), ("15.壓縮突破60日", cond15_fn),
        ("16.最大量高點", cond16_fn), ("17.創120日高", cond17_fn), ("20.外資越買越多出量", cond20_fn),
        ("22.高勝率基因複合", cond22_fn), ("23.突破10日最大量", cond23_fn),
        ("32.大戶鎖碼法人集中", cond32_fn), ("33.絕對鎖碼突破", cond33_fn),
        ("34.創新高回檔再攻", cond_new_high_pullback), ("35.飆股主升段基因", cond_super_bull_strategy),
        ("36.跳空強勢整理不賣", cond_gap_and_hold),
        ("37.宏正多頭法人穩健", cond_macro_6277),
        ("38.智聯服務量價突圍", cond_sys_6751),
        ("39.律勝低基期轉機突破", cond_lv_3354)
    ]

    # 後續建立結果與網頁呈現...
    print("✅ 策略已更新完畢，準備生成選股結果網頁...")
    # (其餘網頁生成與匯出邏輯維持原架構...)

if __name__ == "__main__":
    main()
