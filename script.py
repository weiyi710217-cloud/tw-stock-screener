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

# ==========================================
# 盤後資料獲取核心
# ==========================================
def is_valid_cache(filepath: str, min_size: int = 50) -> bool:
    return os.path.exists(filepath) and os.path.getsize(filepath) >= min_size

def safe_read_csv(filepath: str, **kwargs) -> pd.DataFrame:
    if not is_valid_cache(filepath):
        if os.path.exists(filepath):
            try:
                os.remove(filepath)
            except Exception:
                pass
        return pd.DataFrame()
    try:
        return pd.read_csv(filepath, **kwargs)
    except Exception:
        try:
            os.remove(filepath)
        except Exception:
            pass
        return pd.DataFrame()

def to_float(x, default=0.0):
    try:
        return float(str(x).replace(",", "").replace("%", "").strip())
    except Exception:
        return default

def get_tables(data):
    if not isinstance(data, dict):
        return []
    if data.get("tables"):
        return data["tables"]
    if data.get("fields") and data.get("data"):
        return [data]
    return []

def safe_request_json(url: str, max_retries: int = 2, headers=TWSE_HEADERS):
    for _ in range(max_retries):
        try:
            resp = requests.get(url, headers=headers, timeout=6)
            if resp.status_code == 200:
                try:
                    return resp.json()
                except Exception:
                    return {}
        except Exception:
            pass
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
                time.sleep(0.5)
                continue
            data = resp.json()
            if str(data.get("stat", "")).lower() != "ok":
                return []
            top_date = str(data.get("date", "")).strip()
            if top_date and top_date != date_str:
                return []
            good = []
            for t in get_tables(data):
                t_date = str(t.get("date", "")).strip()
                if t_date and t_date != roc_date:
                    continue
                good.append(t)
            if good:
                return good
        except Exception:
            pass
        time.sleep(0.5)
    return []

def fetch_twse_daily(date_str: str) -> pd.DataFrame:
    url = f"https://www.twse.com.tw/rwd/zh/afterTrading/MI_INDEX?date={date_str}&type=ALLBUT0999&response=json"
    data = safe_request_json(url, headers=TWSE_HEADERS)
    if not data or data.get("stat") != "OK":
        WARNINGS.append(f"上市價量({date_str}) 抓取失敗")
        return pd.DataFrame()
    target_table = next((t for t in data.get("tables", []) if "每日收盤行情" in t.get("title", "")), None)
    if not target_table:
        return pd.DataFrame()
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
        if df.empty or df.shape[1] != len(fields):
            continue
        df.columns = fields
        def col(prefix):
            return next((c for c in df.columns if c.startswith(prefix)), None)
        c_code, c_name = col("代號"), col("名稱")
        c_close, c_open = col("收盤"), col("開盤")
        c_high, c_low, c_vol = col("最高"), col("最低"), col("成交股數")
        if not all([c_code, c_name, c_close, c_open, c_high, c_low, c_vol]):
            continue
        out = pd.DataFrame({
            "市場": "上櫃", "股票代號": df[c_code].astype(str).str.strip(),
            "股票名稱": df[c_name].astype(str).str.strip(), "開盤價": df[c_open],
            "最高價": df[c_high], "最低價": df[c_low], "收盤價": df[c_close], "成交量": df[c_vol],
        })
        out = out[out["股票代號"].str.isdigit() & (out["股票代號"].str.len() == 4)]
        if len(out) > 100:
            return out.reset_index(drop=True)
    WARNINGS.append(f"上櫃價量({date_str}) 抓取失敗")
    return pd.DataFrame()

_OTC_SIGS = {}
def _is_dup_otc(date_str: str, df: pd.DataFrame) -> bool:
    o = df[df["市場"] == "上櫃"].sort_values("股票代號")
    if o.empty:
        return False
    sig = hash(tuple(pd.to_numeric(o["收盤價"], errors="coerce").round(2).fillna(-1).tolist()))
    if sig in _OTC_SIGS and _OTC_SIGS[sig] != date_str:
        return True
    _OTC_SIGS[sig] = date_str
    return False

def get_market_data(date_str: str) -> pd.DataFrame:
    cache_file = os.path.join(CACHE_DIR, f"market_{date_str}.csv")
    df_cached = safe_read_csv(cache_file, dtype={"股票代號": str})
    if not df_cached.empty and "上櫃" in df_cached["市場"].values and not _is_dup_otc(date_str, df_cached):
        return df_cached
    print(f"🌐 [{date_str}] 抓取【上市+上櫃價量】資料...")
    df_twse = fetch_twse_daily(date_str)
    if df_twse.empty:
        return pd.DataFrame()
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
    if not tables:
        return pd.DataFrame()
    target_table = next((t for t in tables if "三大法人" in str(t.get("title", ""))), tables[0])
    fields = [str(c).strip() for c in target_table.get("fields", [])]
    try:
        df = pd.DataFrame(target_table.get("data", []), columns=fields)
    except Exception:
        return pd.DataFrame()
    if df.empty:
        return pd.DataFrame()
    code_col = next((c for c in df.columns if '證券代號' in c or '代號' in c), None)
    f_col = (next((c for c in df.columns if '外陸資' in c and '買賣超' in c), None) or next((c for c in df.columns if '外資' in c and '買賣超' in c), None))
    t_col = next((c for c in df.columns if '投信' in c and '買賣超' in c), None)
    if not (f_col and t_col and code_col):
        return pd.DataFrame()
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
        if len(aa) <= 50:
            continue
        rows = []
        for item in aa:
            code = str(item[0]).strip()
            if len(item) > 13 and code.isdigit() and len(code) == 4:
                rows.append({"股票代號": code, "外資買賣超": str(item[10]).strip(), "投信買賣超": str(item[13]).strip()})
        if rows:
            return pd.DataFrame(rows)
    WARNINGS.append(f"上櫃法人({date_str}) 抓取失敗")
    return pd.DataFrame()

def get_inst_data(date_str: str) -> pd.DataFrame:
    cache_file = os.path.join(CACHE_DIR, f"inst_{date_str}.csv")
    df_cached = safe_read_csv(cache_file, dtype={"股票代號": str})
    if len(df_cached) > 500:
        return df_cached
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
    if not df_cached.empty:
        return df_cached
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
                    if not clean_line:
                        continue
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
        except Exception:
            pass
        time.sleep(0.5)
    WARNINGS.append("千張大戶資料 抓取失敗")
    return pd.DataFrame(columns=["股票代號", "千張大戶比例(%)"])

def get_foreign_holdings() -> pd.DataFrame:
    today_str = datetime.date.today().strftime("%Y%m%d")
    cache_file = os.path.join(CACHE_DIR, f"foreign_holdings_{today_str}.csv")
    df_cached = safe_read_csv(cache_file, dtype={"股票代號": str})
    if not df_cached.empty:
        return df_cached
    print("🌐 抓取【外資持股總比例與發行張數】...")
    rows = []
    for offset in range(5):
        d = datetime.date.today() - datetime.timedelta(days=offset)
        if d.weekday() >= 5:
            continue
        url = f"https://www.twse.com.tw/rwd/zh/fund/MI_QFIIS?date={d.strftime('%Y%m%d')}&selectType=ALLBUT0999&response=json"
        data = safe_request_json(url, max_retries=1)
        if not data or data.get("stat") != "OK":
            continue
        for t in get_tables(data):
            try:
                fields = [str(c).strip() for c in t["fields"]]
                df_t = pd.DataFrame(t["data"], columns=fields)
            except Exception:
                continue
            c_code = next((c for c in fields if "代號" in c), None)
            c_issue = next((c for c in fields if "發行股數" in c), None)
            c_ratio = next((c for c in fields if "外資" in c and ("持股比率" in c or "持股率" in c)), None)
            if not (c_code and c_issue and c_ratio):
                continue
            for _, r in df_t.iterrows():
                code = str(r[c_code]).strip()
                if len(code) == 4 and code.isdigit():
                    rows.append({"股票代號": code, "外資持股比例(%)": to_float(r[c_ratio]), "發行總張數": to_float(r[c_issue]) / 1000})
        if rows:
            break
    data = safe_request_json("https://www.tpex.org.tw/openapi/v1/tpex_3insti_qfii", max_retries=2, headers=OPENAPI_HEADERS)
    if isinstance(data, list) and data and isinstance(data[0], dict):
        keys = list(data[0].keys())
        def find_key(*must, exclude=()):
            for k in keys:
                kl = k.lower()
                if all(m.lower() in kl for m in must) and not any(e.lower() in kl for e in exclude):
                    return k
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
        if offset > 300:
            break
        target_date = today - datetime.timedelta(days=offset)
        offset += 1
        if target_date.weekday() >= 5:
            continue
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
        if f > 0:
            return f"<span class='tag-up'>+{f:.2f}%</span>"
        elif f < 0:
            return f"<span class='tag-down'>{f:.2f}%</span>"
        else:
            return f"<span class='tag-flat'>0.00%</span>"
    except Exception:
        return str(val)

def main():
    print("啟動多策略選股程式 2.0，準備抓取價量、法人與大戶資料...\n")
    days_data, inst_data = get_last_n_trading_days_data(n_market=155, n_inst=60)
    date_0, df_0 = days_data[0]
    date_1, df_1 = days_data[1]
    date_2, df_2 = days_data[2]
    date_120, _ = days_data[min(120, len(days_data)-1)]
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
        if df_i.empty:
            continue
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
            if not vols:
                return 0.0
            max_vol_idx = max(vols, key=lambda item: item[0])[1]
            return float(row.get(f'最高價_{max_vol_idx}', 0.0))
        except Exception:
            return 0.0
    df_merge['最大量日最高價'] = df_merge.apply(get_max_vol_high, axis=1)

    def get_10d_max_vol_high(row):
        try:
            vols = [(float(row.get(f'成交量_{i}', -1)), i) for i in range(1, 11) if pd.notna(float(row.get(f'成交量_{i}', -1)))]
            if not vols:
                return 0.0
            max_vol_idx = max(vols, key=lambda item: item[0])[1]
            return float(row.get(f'最高價_{max_vol_idx}', 0.0))
        except Exception:
            return 0.0
    df_merge['近10日最大量日最高價'] = df_merge.apply(get_10d_max_vol_high, axis=1)

    df_merge['最新漲幅(%)'] = ((df_merge['收盤價_0'] - df_merge['收盤價_1']) / df_merge['收盤價_1'] * 100).round(2)
    df_merge['前一日漲幅(%)'] = ((df_merge['收盤價_1'] - df_merge['收盤價_2']) / df_merge['收盤價_2'] * 100).round(2)
    df_merge['創5日高'] = (df_merge['收盤價_0'] > df_merge['5日最高收盤']).apply(lambda x: "是" if x else "否")
    df_merge['創20日高'] = (df_merge['收盤價_0'] > df_merge['20日最高收盤']).apply(lambda x: "是" if x else "否")

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
        def calc_foreign_buy_ratio(row):
            total_shares, ratio, monthly_buy = row['發行總張數'], row['外資持股比例(%)'] / 100.0, row['外資近一月(張)']
            foreign_total_holding = total_shares * ratio
            if foreign_total_holding <= 0 or monthly_buy <= 0:
                return 0.0
            return round((monthly_buy / foreign_total_holding) * 100, 2)
        df_merge['外資近月買超佔持股(%)'] = df_merge.apply(calc_foreign_buy_ratio, axis=1)
    else:
        df_merge['外資近月買超佔持股(%)'] = 0.0

    def calculate_technical_score_v2(row):
        score = 0
        try:
            c = float(row.get('收盤價_0', 0))
            o = float(row.get('開盤價_0', 0))
            h = float(row.get('最高價_0', 0))
            l = float(row.get('最低價_0', 0))
            ma5 = float(row.get('5MA', 0))
            ma20 = float(row.get('20MA', 0))
            ma60 = float(row.get('60MA', 0))
            v0 = float(row.get('成交量_0', 0))
            mv5 = float(row.get('5MV', 0))
            f7 = float(row.get('外資近七日(張)', 0))
            t7 = float(row.get('投信近七日(張)', 0))
            whale = float(row.get('千張大戶比例(%)', 0))

            if c > ma5 > ma20 > ma60 and ma60 > 0:
                score += 25
            elif c > ma5 > ma20 and ma20 > 0:
                score += 18
            elif c > ma20 and ma20 > 0:
                score += 10

            max20 = float(row.get('20日最高收盤', 0))
            upper_ratio = (h - c) / c if c > 0 else 1.0
            if c >= o and upper_ratio <= 0.02:
                score += 12
            if max20 > 0 and c > max20:
                score += 13
            elif float(row.get('5日最高收盤', 0)) > 0 and c > float(row.get('5日最高收盤', 0)):
                score += 7

            if mv5 > 0 and 1.3 <= (v0 / mv5) <= 4.0 and c > o:
                score += 25
            elif v0 > float(row.get('成交量_1', 0)) and c > o:
                score += 15

            if f7 > 0 and t7 > 0:
                score += 15
            elif f7 > 0 or t7 > 0:
                score += 8
            if whale >= 60:
                score += 10
            elif whale >= 45:
                score += 5
        except Exception:
            pass
        return min(100, int(score))

    df_merge['技術評分(分)'] = df_merge.apply(calculate_technical_score_v2, axis=1)
    df_merge['近7日符合次數'] = "-"
    df_merge['最新法人買超(張)'] = (df_merge['外資_0'] + df_merge['投信_0']).round(0)
    df_merge['增量倍數'] = (df_merge['成交量_0'] / df_merge['成交量_1'].replace(0, np.nan)).round(2).fillna(0.0)
    df_merge['實體K漲幅(%)'] = (((df_merge['收盤價_0'] - df_merge['開盤價_0']) / df_merge['開盤價_0'].replace(0, np.nan)) * 100).round(2).fillna(0.0)
    df_merge['距離60日高點(%)'] = (((df_merge['收盤價_0'] - df_merge['60日最高收盤']) / df_merge['60日最高收盤'].replace(0, np.nan)) * 100).round(2).fillna(0.0)
    df_merge['20日震幅(%)'] = (((df_merge['收盤價_0'] - df_merge['20日最低收盤']) / df_merge['20日最低收盤'].replace(0, np.nan)) * 100).round(2).fillna(0.0)
    df_merge['40日震幅(%)'] = (((df_merge['收盤價_0'] - df_merge['40日最低收盤']) / df_merge['40日最低收盤'].replace(0, np.nan)) * 100).round(2).fillna(0.0)
    df_merge['60日震幅(%)'] = (((df_merge['收盤價_0'] - df_merge['60日最低收盤']) / df_merge['60日最低收盤'].replace(0, np.nan)) * 100).round(2).fillna(0.0)
    df_merge['法人1日集中度(%)'] = (((df_merge['外資_0'] + df_merge['投信_0']) / df_merge['成交量_0'].replace(0, np.nan)) * 100).round(2).fillna(0.0)
    df_merge['法人2日集中度(%)'] = (((df_merge['外資_0'] + df_merge['外資_1'] + df_merge['投信_0'] + df_merge['投信_1']) / (df_merge['成交量_0'] + df_merge['成交量_1']).replace(0, np.nan)) * 100).round(2).fillna(0.0)
    df_merge['法人3日集中度(%)'] = (((df_merge['外資_0'] + df_merge['外資_1'] + df_merge['外資_2'] + df_merge['投信_0'] + df_merge['投信_1'] + df_merge['投信_2']) / (df_merge['成交量_0'] + df_merge['成交量_1'] + df_merge['成交量_2']).replace(0, np.nan)) * 100).round(2).fillna(0.0)

    d0_s = f"{date_0[4:6]}/{date_0[6:]}"
    d1_s = f"{date_1[4:6]}/{date_1[6:]}"
    df_merge['⭐'] = df_merge['股票代號'].apply(lambda x: f"<span class='fav-star' data-code='{x}'>☆</span>")

    def calc_gap_days(row):
        try:
            l0, l1, l2 = float(row['最低價_0']), float(row['最低價_1']), float(row['最低價_2'])
            c1, c2, c3 = float(row['收盤價_1']), float(row['收盤價_2']), float(row.get('收盤價_3', 0))
            if (l0 >= c1) and pd.notna(l1) and pd.notna(c2) and (l1 >= c2) and pd.notna(l2) and pd.notna(c3) and (l2 >= c3):
                return 3
            if (l0 >= c1) and pd.notna(l1) and pd.notna(c2) and (l1 >= c2):
                return 2
            if l0 >= c1:
                return 1
            return 0
        except Exception:
            return 0

    def calc_vol_days(row):
        try:
            v0, v1 = float(row.get('成交量_0', 0)), float(row.get('成交量_1', 0))
            v2, v3 = float(row.get('成交量_2', 0)), float(row.get('成交量_3', 0))
            if pd.isna(v0) or pd.isna(v1):
                return 0
            if (v0 > v1) and pd.notna(v2) and (v1 > v2) and pd.notna(v3) and (v2 > v3):
                return 3
            if (v0 > v1) and pd.notna(v2) and (v1 > v2):
                return 2
            if v0 > v1:
                return 1
            return 0
        except Exception:
            return 0

    def calc_step_up_days(row):
        try:
            count = 0
            for i in range(5):
                l_curr, c_curr, c_prev = float(row.get(f'最低價_{i}', 0)), float(row.get(f'收盤價_{i}', 0)), float(row.get(f'收盤價_{i+1}', 0))
                if pd.notna(l_curr) and pd.notna(c_curr) and pd.notna(c_prev):
                    if c_curr > c_prev and l_curr >= c_prev:
                        count += 1
                    else:
                        break
                else:
                    break
            return count
        except Exception:
            return 0

    def calc_foreign_buy_days(row):
        try:
            count = 0
            for i in range(7):
                f = float(row.get(f'外資_{i}', 0))
                if pd.notna(f) and f > 0:
                    count += 1
                else:
                    break
            return count
        except Exception:
            return 0

    def calc_consecutive_up_days(row):
        try:
            count = 0
            for i in range(5):
                c_curr, c_prev = float(row.get(f'收盤價_{i}', 0)), float(row.get(f'收盤價_{i+1}', 0))
                if pd.notna(c_curr) and pd.notna(c_prev) and c_curr > c_prev:
                    count += 1
                else:
                    break
            return count
        except Exception:
            return 0

    def calc_10d_total_return(row):
        try:
            c_curr, c_prev_10 = float(row.get('收盤價_0', 0)), float(row.get('收盤價_10', 0))
            if pd.isna(c_prev_10) or c_prev_10 == 0:
                for i in range(9, 0, -1):
                    fallback = float(row.get(f'收盤價_{i}', 0))
                    if pd.notna(fallback) and fallback > 0:
                        c_prev_10 = fallback
                        break
            if pd.notna(c_curr) and c_prev_10 > 0:
                return round(((c_curr - c_prev_10) / c_prev_10) * 100, 2)
            return 0.0
        except Exception:
            return 0.0

    df_merge['跳空天數'] = df_merge.apply(calc_gap_days, axis=1)
    df_merge['量增天數'] = df_merge.apply(calc_vol_days, axis=1)
    df_merge['墊高天數'] = df_merge.apply(calc_step_up_days, axis=1)
    df_merge['外資連買天數'] = df_merge.apply(calc_foreign_buy_days, axis=1)
    df_merge['連漲天數'] = df_merge.apply(calc_consecutive_up_days, axis=1)
    df_merge['近10日漲幅(%)'] = df_merge.apply(calc_10d_total_return, axis=1)
    df_merge['創60日最大量'] = df_merge['成交量_0'] >= df_merge['60日最大量']
    df_merge['標的'] = df_merge.apply(format_stock_cell, axis=1)

    base_cols = ["⭐", "市場", "標的", "技術評分(分)", "近7日符合次數", "近10日漲幅(%)", f"{d1_s} 收盤", f"{d0_s} 收盤", "最新漲幅(%)", "前一日漲幅(%)", f"{d0_s} 量(張)"]
    chip_cols = ["外資近七日(張)", "投信近七日(張)", "外資近一月(張)", "千張大戶比例(%)"]

    def eval_rolling_condition(condition_func):
        counts = pd.Series(0, index=df_merge.index)
        for k in range(7):
            try:
                counts = counts + condition_func(k).astype(int)
            except Exception:
                pass
        return counts.apply(lambda x: f"{x}次 / 7日")

    def apply_color_formatting(df_in):
        df_out = df_in.copy()
        for col_name in ["最新漲幅(%)", "前一日漲幅(%)", "近10日漲幅(%)"]:
            if col_name in df_out.columns:
                df_out[col_name] = df_out[col_name].apply(color_pct)
        return df_out

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
        for j in range(7):
            t_hold = t_hold & (df_merge[f'投信_{k+j}'] >= 0)
        f_buy = df_merge[f'外資_{k}'] > 0
        return vol_up & t_hold & f_buy

    def cond4_fn(k):
        close_20 = [f'收盤價_{k+j}' for j in range(1, 21)]
        max_20 = df_merge[close_20].max(axis=1)
        c_high = is_strong_k(k) & (df_merge[f'收盤價_{k}'] > max_20)
        vol_up = (df_merge[f'成交量_{k}'] > df_merge[f'成交量_{k+1}']) & (df_merge[f'成交量_{k+1}'] > 0)
        t_hold = True
        for j in range(7):
            t_hold = t_hold & (df_merge[f'投信_{k+j}'] >= 0)
        return c_high & vol_up & t_hold

    def cond6_fn(k):
        close_5 = [f'收盤價_{k+j}' for j in range(1, 6)]
        max_5 = df_merge[close_5].max(axis=1)
        c_high = is_strong_k(k) & (df_merge[f'收盤價_{k}'] > max_5)
        vol_up = (df_merge[f'成交量_{k}'] > df_merge[f'成交量_{k+1}']) & (df_merge[f'成交量_{k+1}'] > 0)
        gap = (df_merge[f'開盤價_{k}'] > df_merge[f'收盤價_{k+1}']) & (df_merge[f'最低價_{k}'] > df_merge[f'收盤價_{k+1}'])
        hold_3 = (df_merge[f'外資_{k}'] >= 0) & (df_merge[f'投信_{k}'] >= 0) & (df_merge[f'外資_{k+1}'] >= 0) & (df_merge[f'投信_{k+1}'] >= 0) & (df_merge[f'外資_{k+2}'] >= 0) & (df_merge[f'投信_{k+2}'] >= 0)
        return c_high & vol_up & gap & hold_3

    def cond7_fn(k):
        close_20 = [f'收盤價_{k+j}' for j in range(1, 21)]
        max_20 = df_merge[close_20].max(axis=1)
        c_high = is_strong_k(k) & (df_merge[f'收盤價_{k}'] > max_20)
        vol_up = (df_merge[f'成交量_{k}'] > df_merge[f'成交量_{k+1}']) & (df_merge[f'成交量_{k+1}'] > 0)
        p_up = (df_merge[f'最低價_{k}'] > df_merge[f'最低價_{k+1}']) & (df_merge[f'收盤價_{k}'] > df_merge[f'收盤價_{k+1}'])
        hold_7 = True
        for j in range(7):
            hold_7 = hold_7 & (df_merge[f'外資_{k+j}'] >= 0) & (df_merge[f'投信_{k+j}'] >= 0)
        return c_high & vol_up & p_up & hold_7

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
                if not valid:
                    return 0.0
                idx = max(valid, key=lambda x: x[0])[1]
                return float(row.get(f'最高價_{idx}', 0.0))
            except Exception:
                return 0.0
        mv_high = df_merge['最大量日最高價'] if k == 0 else df_merge.apply(lambda r: _get_60d_max_vol_high_at(r, k), axis=1)
        c_k = df_merge[f'收盤價_{k}']
        c_prev = df_merge[f'收盤價_{k+1}']
        breakout = is_strong_k(k) & (mv_high > 0) & (c_k > mv_high) & (c_prev <= mv_high)
        hold_7 = True
        for j in range(7):
            hold_7 = hold_7 & (df_merge[f'外資_{k+j}'] >= 0) & (df_merge[f'投信_{k+j}'] >= 0)
        return breakout & hold_7

    def cond17_fn(k):
        close_120 = [f'收盤價_{k+j}' for j in range(1, 121) if f'收盤價_{k+j}' in df_merge]
        if not close_120:
            return pd.Series(False, index=df_merge.index)
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
        gap_up = (df_merge[f'最低價_{k}'] >= df_merge[f'收盤價_{k+1}']) & (df_merge[f'開盤價_{k}'] > df_merge[f'收盤價_{k+1}'])
        step_up = (df_merge[f'收盤價_{k}'] > df_merge[f'收盤價_{k+1}']) & (df_merge[f'收盤價_{k+1}'] > df_merge[f'收盤價_{k+2}'])
        vol_up = df_merge[f'成交量_{k}'] >= df_merge[f'成交量_{k+1}'] * 1.2
        hold_3 = (df_merge[f'外資_{k}'] >= 0) & (df_merge[f'投信_{k}'] >= 0) & (df_merge[f'外資_{k+1}'] >= 0) & (df_merge[f'投信_{k+1}'] >= 0) & (df_merge[f'外資_{k+2}'] >= 0) & (df_merge[f'投信_{k+2}'] >= 0)
        return c_high5 & gap_up & step_up & vol_up & hold_3

    def cond23_fn(k):
        def _get_max_high_at(row, offset):
            try:
                vols = [(float(row.get(f'成交量_{offset+j}', -1)), offset+j) for j in range(1, 11)]
                valid = [v for v in vols if pd.notna(v[0])]
                if not valid:
                    return 0.0
                idx = max(valid, key=lambda x: x[0])[1]
                return float(row.get(f'最高價_{idx}', 0.0))
            except Exception:
                return 0.0
        max_h = df_merge['近10日最大量日最高價'] if k == 0 else df_merge.apply(lambda r: _get_max_high_at(r, k), axis=1)
        return is_strong_k(k) & (max_h > 0) & (df_merge[f'收盤價_{k}'] > max_h)

    # 策略 25：穩健法人鎖碼型（啟發自 宏正 6277）
    def cond25_fn(k):
        try:
            c_k = df_merge[f'收盤價_{k}']
            ma5_k = df_merge[[f'收盤價_{k+j}' for j in range(5)]].mean(axis=1)
            ma20_k = df_merge[[f'收盤價_{k+j}' for j in range(20)]].mean(axis=1)
            ma60_k = df_merge[[f'收盤價_{k+j}' for j in range(60)]].mean(axis=1)
            bullish_ma = (c_k > ma5_k) & (ma5_k > ma20_k) & (ma20_k > ma60_k)
            whale = df_merge['千張大戶比例(%)'] >= 45.0
            f_7d = df_merge[[f'外資_{k+j}' for j in range(7) if f'外資_{k+j}' in df_merge.columns]].sum(axis=1)
            t_7d = df_merge[[f'投信_{k+j}' for j in range(7) if f'投信_{k+j}' in df_merge.columns]].sum(axis=1)
            inst_buy = (f_7d > 0) | (t_7d > 0)
            vol_up = df_merge[f'成交量_{k}'] >= df_merge[f'成交量_{k+1}'] * 1.15
            return is_strong_k(k) & bullish_ma & whale & inst_buy & vol_up
        except Exception:
            return pd.Series(False, index=df_merge.index)

    # 策略 37：資訊服務成長突破型（啟發自 智聯服務 6751）
    def cond37_fn(k):
        try:
            c_k = df_merge[f'收盤價_{k}']
            close_20 = [f'收盤價_{k+j}' for j in range(1, 21)]
            max_20 = df_merge[close_20].max(axis=1)
            break_20 = (c_k > max_20) & (max_20 > 0)
            v_k = df_merge[f'成交量_{k}']
            mv5_k = df_merge[[f'成交量_{k+j}' for j in range(1, 6)]].mean(axis=1)
            vol_ok = (v_k >= 400) & (v_k >= mv5_k * 1.2)
            f_0 = df_merge[f'外資_{k}'] >= 0
            t_0 = df_merge[f'投信_{k}'] >= 0
            return is_strong_k(k) & break_20 & vol_ok & f_0 & t_0
        except Exception:
            return pd.Series(False, index=df_merge.index)

    # 策略 38：低價成交量暴增動能型（啟發自 律勝 3354）
    def cond38_fn(k):
        try:
            c_k = df_merge[f'收盤價_{k}']
            o_k = df_merge[f'開盤價_{k}'] if f'開盤價_{k}' in df_merge else c_k
            v_k = df_merge[f'成交量_{k}']
            v_1 = df_merge[f'成交量_{k+1}']
            body_pct = ((c_k - o_k) / o_k.replace(0, np.nan)) * 100
            vol_surge = (v_k >= v_1 * 2.2) & (v_1 > 0) & (v_k >= 300)
            strong_body = (body_pct >= 3.0) & (c_k > o_k)
            return is_strong_k(k) & vol_surge & strong_body
        except Exception:
            return pd.Series(False, index=df_merge.index)

    # ==========================================
    # 【新增策略 39：極致大戶鎖碼強勢跳空突破】
    # ==========================================
    def cond39_fn(k):
        try:
            whale_80 = df_merge['千張大戶比例(%)'] >= 80.0
            v_k = df_merge[f'成交量_{k}']
            v_prev = df_merge[f'成交量_{k+1}']
            vol_up = (v_k >= v_prev * 1.3) & (v_prev > 0) & (v_k >= 200)

            l_k = df_merge[f'最低價_{k}']
            h_prev = df_merge[f'最高價_{k+1}']
            gap_above_high = (l_k > h_prev) & (h_prev > 0)

            return is_strong_k(k) & whale_80 & vol_up & gap_above_high
        except Exception:
            return pd.Series(False, index=df_merge.index)

    def cond32_fn(k):
        whale = df_merge['千張大戶比例(%)'] >= 60.0
        ratio = (((df_merge[f'外資_{k}'] + df_merge[f'外資_{k+1}'] + df_merge[f'外資_{k+2}'] + df_merge[f'投信_{k}'] + df_merge[f'投信_{k+1}'] + df_merge[f'投信_{k+2}']) / (df_merge[f'成交量_{k}'] + df_merge[f'成交量_{k+1}'] + df_merge[f'成交量_{k+2}']).replace(0, np.nan)) * 100).fillna(0.0)
        return is_strong_k(k) & whale & (ratio >= 20.0)

    cond32_backtest_fn = cond32_fn

    def cond33_fn(k):
        whale_70 = df_merge['千張大戶比例(%)'] > 70.0
        ma5_k = df_merge[[f'收盤價_{k+j}' for j in range(5)]].mean(axis=1)
        ma20_k = df_merge[[f'收盤價_{k+j}' for j in range(20)]].mean(axis=1)
        ma60_k = df_merge[[f'收盤價_{k+j}' for j in range(60)]].mean(axis=1)
        c_k = df_merge[f'收盤價_{k}']
        bullish_ma = (c_k > ma5_k) & (ma5_k > ma20_k) & (ma20_k > ma60_k)
        close_120_k = [f'收盤價_{k+j}' for j in range(1, 121) if f'收盤價_{k+j}' in df_merge.columns]
        if not close_120_k:
            return pd.Series(False, index=df_merge.index)
        max_120_k = df_merge[close_120_k].max(axis=1)
        near_breakout = (c_k >= max_120_k * 0.95) & (c_k <= max_120_k * 1.05) & (max_120_k > 0)
        return is_strong_k(k) & whale_70 & bullish_ma & near_breakout

    def cond_new_high_pullback(k):
        try:
            close_cols_180 = [f'收盤價_{k+j}' for j in range(1, 181) if f'收盤價_{k+j}' in df_merge.columns]
            if len(close_cols_180) < 60:
                return pd.Series(False, index=df_merge.index)
            max_180 = df_merge[close_cols_180].max(axis=1)
            close_cols_20 = [f'收盤價_{k+j}' for j in range(0, 20) if f'收盤價_{k+j}' in df_merge.columns]
            max_20 = df_merge[close_cols_20].max(axis=1)
            ever_hit_high = (max_20 >= max_180 * 0.99) & (max_180 > 0)
            low_cols_20 = [f'最低價_{k+j}' for j in range(0, 20) if f'最低價_{k+j}' in df_merge.columns]
            min_low_20 = df_merge[low_cols_20].min(axis=1)
            had_pullback = min_low_20 <= max_180 * 0.92
            c_k = df_merge[f'收盤價_{k}']
            near_high_now = (c_k >= max_180 * 0.95) & (c_k <= max_180 * 1.05)
            return is_strong_k(k) & ever_hit_high & had_pullback & near_high_now
        except Exception:
            return pd.Series(False, index=df_merge.index)

    def cond_super_bull_strategy(k):
        try:
            c_k, v_k, v_1 = df_merge[f'收盤價_{k}'], df_merge[f'成交量_{k}'], df_merge[f'成交量_{k+1}']
            whale_locked = df_merge['千張大戶比例(%)'] >= 60.0
            ma5_k = df_merge[[f'收盤價_{k+j}' for j in range(5)]].mean(axis=1)
            ma20_k = df_merge[[f'收盤價_{k+j}' for j in range(20)]].mean(axis=1)
            ma60_k = df_merge[[f'收盤價_{k+j}' for j in range(60)]].mean(axis=1)
            bullish_ma = (c_k > ma5_k) & (ma5_k > ma20_k) & (ma20_k > ma60_k)
            vol_surge = (v_k >= v_1 * 1.2) & (v_1 > 0)
            f_7d = df_merge[[f'外資_{k+j}' for j in range(7) if f'外資_{k+j}' in df_merge.columns]].sum(axis=1)
            t_7d = df_merge[[f'投信_{k+j}' for j in range(7) if f'投信_{k+j}' in df_merge.columns]].sum(axis=1)
            inst_safe = (f_7d >= 0) | (t_7d >= 0)
            return is_strong_k(k) & whale_locked & bullish_ma & vol_surge & inst_safe
        except Exception:
            return pd.Series(False, index=df_merge.index)

    def cond_gap_and_hold(k):
        try:
            total_hit = pd.Series(False, index=df_merge.index)
            for m in range(2, 7):
                if f'收盤價_{k+m+1}' not in df_merge.columns:
                    continue
                c_m = df_merge[f'收盤價_{k+m}']
                o_m = df_merge[f'開盤價_{k+m}']
                h_m = df_merge[f'最高價_{k+m}']
                l_m = df_merge[f'最低價_{k+m}']
                v_m = df_merge[f'成交量_{k+m}']
                c_m_prev = df_merge[f'收盤價_{k+m+1}']
                h_m_prev = df_merge[f'最高價_{k+m+1}']

                has_gap = (l_m > c_m_prev) & (o_m > h_m_prev * 0.995) & (c_m_prev > 0)
                pct_m = ((c_m - c_m_prev) / c_m_prev.replace(0, np.nan)) * 100
                body_m = ((c_m - o_m) / o_m.replace(0, np.nan)) * 100
                long_red_k = (pct_m >= 4.5) & (c_m > o_m) & (body_m >= 2.5) & (v_m >= 300)

                hold_above_high = pd.Series(True, index=df_merge.index)
                post_closes = []
                for j in range(m):
                    c_j = df_merge[f'收盤價_{k+j}']
                    l_j = df_merge[f'最低價_{k+j}']
                    hold_above_high = hold_above_high & (c_j >= h_m) & (l_j > c_m_prev)
                    post_closes.append(c_j)

                df_post_c = pd.concat(post_closes, axis=1)
                tight_consolidation = ((df_post_c.max(axis=1) - df_post_c.min(axis=1)) / h_m.replace(0, np.nan)) <= 0.085

                inst_window = max(m + 1, 5)
                no_inst_sell = pd.Series(True, index=df_merge.index)
                f_sum = pd.Series(0.0, index=df_merge.index)
                t_sum = pd.Series(0.0, index=df_merge.index)
                for j in range(inst_window):
                    if f'外資_{k+j}' in df_merge.columns and f'投信_{k+j}' in df_merge.columns:
                        f_j = df_merge[f'外資_{k+j}']
                        t_j = df_merge[f'投信_{k+j}']
                        no_inst_sell = no_inst_sell & (f_j >= 0) & (t_j >= 0)
                        f_sum = f_sum + f_j
                        t_sum = t_sum + t_j
                inst_net_positive = no_inst_sell & (f_sum > 0) & (t_sum >= 0) & ((f_sum + t_sum) > 0)

                c_k = df_merge[f'收盤價_{k}']
                o_k = df_merge[f'開盤價_{k}']
                v_k = df_merge[f'成交量_{k}']
                today_healthy = (v_k >= 200) & (c_k >= o_k * 0.99)

                cond_m = has_gap & long_red_k & hold_above_high & tight_consolidation & inst_net_positive & today_healthy
                total_hit = total_hit | cond_m

            return total_hit
        except Exception:
            return pd.Series(False, index=df_merge.index)

    def build_res(cond_fn, sort_cols, asc_list, rename_dict=None):
        cond = cond_fn(0)
        res = df_merge[cond].copy()
        res['近7日符合次數'] = eval_rolling_condition(cond_fn)[cond]
        valid_sort_cols = [c for c in sort_cols if c in res.columns]
        valid_asc_list = [asc_list[i] for i, c in enumerate(sort_cols) if c in res.columns]
        if valid_sort_cols:
            res = res.sort_values(by=valid_sort_cols, ascending=valid_asc_list)
        if rename_dict:
            res = res.rename(columns=rename_dict)
        return res

    std_rename = {"收盤價_1": f"{d1_s} 收盤", "收盤價_0": f"{d0_s} 收盤", "成交量_0": f"{d0_s} 量(張)"}
    inst_rename = {**std_rename, "外資_0": "外資(T)", "外資_1": "外資(T-1)", "外資_2": "外資(T-2)", "投信_0": "投信(T)", "投信_1": "投信(T-1)", "投信_2": "投信(T-2)"}

    res1 = build_res(cond1_fn, ["跳空天數", "量增天數", "增量倍數"], [False, False, False], std_rename)
    res2 = build_res(cond2_fn, ["墊高天數", "外資連買天數", "增量倍數"], [False, False, False], std_rename)
    res3 = build_res(cond3_fn, ["外資連買天數", "增量倍數"], [False, False], std_rename)
    res4 = build_res(cond4_fn, ["投信近七日(張)", "增量倍數"], [False, False], std_rename)
    res6 = build_res(cond6_fn, ["投信近三日(張)", "增量倍數"], [False, False], std_rename)
    res7 = build_res(cond7_fn, ["投信近七日(張)", "增量倍數"], [False, False], std_rename)
    res11 = build_res(cond11_fn, ["實體K漲幅(%)"], [False], {"開盤價_0": f"{d0_s} 開盤", "收盤價_1": f"{d1_s} 收盤", "收盤價_0": f"{d0_s} 收盤", "成交量_0": f"{d0_s} 量(張)"})
    res13 = build_res(cond13_fn, ["距離60日高點(%)", "最新漲幅(%)"], [False, False], std_rename)
    res15 = build_res(cond15_fn, ["60日震幅(%)", "增量倍數"], [True, False], std_rename)
    res15 = res15[(res15['創60日最大量']) & (res15['60日震幅(%)'] <= 10.0)]
    res16 = build_res(cond16_fn, ["最新漲幅(%)", "增量倍數"], [False, False], std_rename)

    res17 = build_res(cond17_fn, ["最新漲幅(%)", "增量倍數"], [False, False], std_rename)
    if not res17.empty:
        def mark_120_high(row):
            try:
                c0 = float(row.get(f'{d0_s} 收盤', 0))
                h120 = float(row.get('120日最高價', 0))
                if c0 > h120 and h120 > 0:
                    return "<br><span style='color:#dc2626;font-size:11px;font-weight:bold;'>🎯 收盤突破120日最高點</span>"
            except Exception:
                pass
            return ""
        res17['標的'] = res17['標的'] + res17.apply(mark_120_high, axis=1)

    res20 = build_res(cond20_fn, ["外資_0", "最新漲幅(%)"], [False, False], std_rename)
    res22 = build_res(cond22_fn, ["最新漲幅(%)", "增量倍數"], [False, False], std_rename)
    res23 = build_res(cond23_fn, ["最新漲幅(%)", "增量倍數"], [False, False], std_rename)
    res25 = build_res(cond25_fn, ["千張大戶比例(%)", "最新漲幅(%)"], [False, False], std_rename)
    res37 = build_res(cond37_fn, ["最新漲幅(%)", "增量倍數"], [False, False], std_rename)
    res38 = build_res(cond38_fn, ["實體K漲幅(%)", "增量倍數"], [False, False], std_rename)
    res39 = build_res(cond39_fn, ["千張大戶比例(%)", "最新漲幅(%)", "增量倍數"], [False, False, False], std_rename)
    initial_res32 = build_res(cond32_fn, ["法人3日集中度(%)", "最新漲幅(%)"], [False, False], inst_rename)
    res33 = build_res(cond33_fn, ["千張大戶比例(%)", "最新漲幅(%)"], [False, False], std_rename)
    res34 = build_res(cond_new_high_pullback, ["最新漲幅(%)", "增量倍數"], [False, False], std_rename)
    res35 = build_res(cond_super_bull_strategy, ["千張大戶比例(%)", "最新漲幅(%)"], [False, False], std_rename)
    res36 = build_res(cond_gap_and_hold, ["技術評分(分)", "外資近七日(張)", "投信近七日(張)"], [False, False, False], std_rename)

    print("🔬 正在執行策略組合與回測計算...")
    cand_strategies = [
        ("1.跳空不補", cond1_fn), ("2.連續墊高", cond2_fn), ("3.量增法人買", cond3_fn),
        ("4.創20日新高", cond4_fn), ("6.創五日高", cond6_fn),
        ("7.20日高不賣", cond7_fn), ("11.上櫃強紅K", cond11_fn), ("13.逼近60日高", cond13_fn),
        ("15.壓縮突破60日", cond15_fn), ("16.最大量高點", cond16_fn),
        ("17.創120日高", cond17_fn), ("20.外資越買越多出量", cond20_fn),
        ("22.高勝率基因複合", cond22_fn), ("23.突破10日最大量", cond23_fn),
        ("25.穩健法人鎖碼", cond25_fn), ("32.大戶鎖碼法人集中", cond32_backtest_fn), ("33.絕對鎖碼突破", cond33_fn),
        ("34.創新高回檔再攻", cond_new_high_pullback), ("35.飆股主升段基因", cond_super_bull_strategy),
        ("36.跳空強勢整理不賣", cond_gap_and_hold), ("37.資訊服務成長突破", cond37_fn), ("38.低價量暴增動能", cond38_fn),
        ("39.極致大戶跳空量增", cond39_fn)
    ]

    sept_indices = [i for i, (d, _) in enumerate(days_data) if "20260901" <= d <= "20260930"]
    if not sept_indices:
        sept_indices = list(range(5, min(25, len(days_data)-5)))
    idx_1001 = next((i for i, (d, _) in enumerate(days_data) if d == "20261001"), None)
    if idx_1001 is None:
        oct_indices = [i for i, (d, _) in enumerate(days_data) if d.startswith("202610")]
        idx_1001 = oct_indices[-1] if oct_indices else 5
    idx_1008 = next((i for i, (d, _) in enumerate(days_data) if d == "20261008"), 0)

    col_open_1001 = f"開盤價_{idx_1001}" if f"開盤價_{idx_1001}" in df_merge.columns else f"收盤價_{idx_1001}"
    col_close_1008 = f"收盤價_{idx_1008}"
    N, D = len(df_merge), len(sept_indices)
    strat_matrices = {}
    for st_name, fn in cand_strategies:
        mat = np.zeros((N, D), dtype=bool)
        for d_idx, k in enumerate(sept_indices):
            try:
                mat[:, d_idx] = fn(k).fillna(False).values
            except Exception:
                pass
        strat_matrices[st_name] = mat

    open_1001 = pd.to_numeric(df_merge[col_open_1001], errors='coerce').values
    close_1008 = pd.to_numeric(df_merge[col_close_1008], errors='coerce').values
    valid_ret_mask = (open_1001 > 0) & (close_1008 > 0) & np.isfinite(open_1001) & np.isfinite(close_1008)
    returns_arr = np.zeros(N, dtype=float)
    returns_arr[valid_ret_mask] = ((close_1008[valid_ret_mask] - open_1001[valid_ret_mask]) / open_1001[valid_ret_mask]) * 100

    combo_pairs = list(itertools.combinations(cand_strategies, 2))
    combo_records = []
    for (name_a, fn_a), (name_b, fn_b) in combo_pairs:
        hit_mask = (strat_matrices[name_a] & strat_matrices[name_b]).any(axis=1)
        sub_mask = hit_mask & valid_ret_mask
        n_samples = int(np.sum(sub_mask))
        if n_samples == 0:
            continue
        sub_ret = returns_arr[sub_mask]
        up_count = int(np.sum(sub_ret > 0))
        win_rate = round((up_count / n_samples) * 100, 2)
        combo_records.append({
            "name_a": name_a, "name_b": name_b, "fn_a": fn_a, "fn_b": fn_b,
            "策略組合": f"{name_a} ＋ {name_b}", "上漲率(%)": win_rate,
            "平均漲幅(%)": round(float(np.mean(sub_ret)), 2), "中位數漲幅(%)": round(float(np.median(sub_ret)), 2),
            "樣本數": n_samples, "上漲檔數": up_count, "下跌檔數": n_samples - up_count
        })

    df_combos = pd.DataFrame(combo_records)
    if not df_combos.empty:
        df_combos = df_combos.sort_values(by=["上漲率(%)", "平均漲幅(%)", "中位數漲幅(%)", "樣本數"], ascending=[False, False, False, False]).reset_index(drop=True)
    else:
        df_combos = pd.DataFrame(columns=["策略組合", "上漲率(%)", "平均漲幅(%)", "中位數漲幅(%)", "樣本數", "上漲檔數", "下跌檔數"])
    df_combos["排名"] = range(1, len(df_combos) + 1)
    df_combos_show = df_combos[["排名", "策略組合", "上漲率(%)", "平均漲幅(%)", "中位數漲幅(%)", "樣本數", "上漲檔數", "下跌檔數"]].copy()
    df_combos_show["上漲率(%)"] = df_combos_show["上漲率(%)"].apply(lambda x: f"<b style='color:#dc2626;'>{x:.2f}%</b>" if x >= 50 else f"<span style='color:#16a34a;'>{x:.2f}%</span>")
    df_combos_show["平均漲幅(%)"] = df_combos_show["平均漲幅(%)"].apply(color_pct)
    df_combos_show["中位數漲幅(%)"] = df_combos_show["中位數漲幅(%)"].apply(color_pct)
    html_tb26 = df_combos_show.to_html(index=False, classes="styled-table backtest-table sortable-table", escape=False)

    top5_combos = df_combos.head(5).to_dict("records")
    strat_results_27_31 = {}
    rank_labels = ["首選", "第二"]
    for idx, r_num in enumerate([27, 28]):
        if idx < len(top5_combos):
            item = top5_combos[idx]
            f_a, f_b = item["fn_a"], item["fn_b"]
            cond_today = f_a(0) & f_b(0)
            res_df = df_merge[cond_today].copy()
            res_df['近7日符合次數'] = eval_rolling_condition(lambda k: f_a(k) & f_b(k))[cond_today]
            res_df["增量倍數"] = (res_df["成交量_0"] / res_df["成交量_1"].replace(0, np.nan)).round(2).fillna(0.0)
            res_df = res_df.sort_values(by=["最新漲幅(%)", "增量倍數"], ascending=[False, False]).rename(columns=std_rename)
            if not res_df.empty:
                table_html = apply_color_formatting(res_df[base_cols + ["增量倍數"] + chip_cols]).to_html(index=False, classes="styled-table sortable-table", escape=False)
            else:
                table_html = "<div class='empty-state'>今日最新盤後無同時符合此兩策略之股票</div>"
            strat_results_27_31[r_num] = {"name": f"{r_num}. 組合{rank_labels[idx]}", "combo_name": item["策略組合"], "df": res_df, "html": table_html, "count": len(res_df), "win_rate": item["上漲率(%)"], "avg_ret": item["平均漲幅(%)"], "samples": item["樣本數"]}
        else:
            strat_results_27_31[r_num] = {"name": f"{r_num}. 組合{rank_labels[idx]}", "combo_name": "無", "df": pd.DataFrame(), "html": "", "count": 0, "win_rate": 0.0}

    backtest_records = []
    for st_name, fn in cand_strategies:
        total_picks, up_picks, return_list = 0, 0, []
        for k in range(1, 31):
            try:
                selected_mask = fn(k)
                if selected_mask is None or not selected_mask.any():
                    continue
                next_day_ret = ((df_merge.loc[selected_mask, f'收盤價_{k-1}'] - df_merge.loc[selected_mask, f'收盤價_{k}']) / df_merge.loc[selected_mask, f'收盤價_{k}']) * 100
                next_day_ret = next_day_ret.dropna()
                if len(next_day_ret) > 0:
                    total_picks += len(next_day_ret)
                    up_picks += (next_day_ret > 0).sum()
                    return_list.extend(next_day_ret.tolist())
            except Exception:
                continue
        win_rate = round((up_picks / total_picks * 100), 2) if total_picks > 0 else 0.0
        avg_ret = round((sum(return_list) / len(return_list)), 2) if return_list else 0.0
        backtest_records.append({"選股策略名稱": st_name, "隔天上漲率(%)": win_rate, "隔日平均報酬(%)": avg_ret, "總選中樣本數": total_picks, "隔天上漲次數": up_picks, "隔天下跌次數": total_picks - up_picks})

    df_strat21 = pd.DataFrame(backtest_records).sort_values(by=["隔天上漲率(%)", "隔日平均報酬(%)"], ascending=[False, False])
    df_strat21_show = df_strat21.copy()
    df_strat21_show["隔天上漲率(%)"] = df_strat21_show["隔天上漲率(%)"].apply(lambda x: f"<b style='color:#dc2626;'>{x:.2f}%</b>" if x >= 50 else f"<span style='color:#16a34a;'>{x:.2f}%</span>")
    df_strat21_show["隔日平均報酬(%)"] = df_strat21_show["隔日平均報酬(%)"].apply(color_pct)
    html_tb21 = df_strat21_show.to_html(index=False, classes="styled-table backtest-table sortable-table", escape=False)

    st_lists = [
        ("1.跳空", res1), ("2.墊高", res2), ("3.量增法人", res3), ("4.創20日高", res4),
        ("6.五日高", res6), ("7.創20日高不賣", res7), ("11.上櫃強紅K", res11), ("13.逼近60日高", res13),
        ("15.壓縮突破60日", res15), ("16.突破最大量高點", res16), ("17.創120日高", res17),
        ("20.外資越買越多出量", res20), ("22.高勝率基因複合", res22), ("23.突破10日最大量", res23),
        ("25.穩健法人鎖碼", res25), ("32.大戶鎖碼法人集中", initial_res32), ("33.絕對鎖碼突破", res33),
        ("34.創新高回檔再攻", res34), ("35.飆股主升段基因", res35), ("36.跳空強勢整理不賣", res36),
        ("37.資訊服務成長突破", res37), ("38.低價量暴增動能", res38), ("39.極致大戶跳空量增", res39)
    ]
    hit_counts, hit_names = {}, {}
    for s_name, res_df in st_lists:
        if res_df.empty or "股票代號" not in res_df.columns:
            continue
        for code in res_df['股票代號'].values:
            if code not in hit_counts:
                hit_counts[code], hit_names[code] = 0, []
            hit_counts[code] += 1
            hit_names[code].append(s_name)

    if hit_counts:
        df_hits = pd.DataFrame([{"股票代號": k, "入選次數": v, "符合策略": ", ".join(hit_names[k])} for k, v in hit_counts.items()])
        res12 = pd.merge(df_hits, df_merge, on="股票代號", how="inner")
        res12 = res12[res12['千張大戶比例(%)'] >= 30.0]
        res12 = res12[(res12['入選次數'] >= 4) | ((res12['入選次數'] >= 2) & (res12['近10日漲幅(%)'] > 10.0))]
        res12['近7日符合次數'] = res12['入選次數'].apply(lambda x: f"命中 {x} 個策略")
        res12 = res12.sort_values(by=["入選次數", "技術評分(分)", "最新漲幅(%)"], ascending=[False, False, False]).rename(columns=std_rename)

        if res12.empty:
            html_tb12 = "<div class='empty-state'>目前無符合高標條件之強勢股</div>"
        else:
            html_tb12 = apply_color_formatting(res12[base_cols + ["入選次數", "符合策略"] + chip_cols]).to_html(index=False, classes="styled-table sortable-table", escape=False)
    else:
        res12, html_tb12 = pd.DataFrame(), "<div class='empty-state'>目前無任何股票入選</div>"

    def get_html_tb(df_res):
        if df_res.empty:
            return "<div class='empty-state'>目前無符合股票</div>"
        valid_cols = [c for c in base_cols + ["增量倍數"] + chip_cols if c in df_res.columns]
        return apply_color_formatting(df_res[valid_cols]).to_html(index=False, classes="styled-table sortable-table", escape=False)

    html_tb1 = get_html_tb(res1)
    html_tb2 = get_html_tb(res2)
    html_tb3 = get_html_tb(res3)
    html_tb4 = get_html_tb(res4)
    html_tb6 = get_html_tb(res6)
    html_tb7 = get_html_tb(res7)
    html_tb11 = get_html_tb(res11)
    html_tb13 = get_html_tb(res13)
    html_tb15 = get_html_tb(res15)
    html_tb16 = get_html_tb(res16)
    html_tb17 = get_html_tb(res17)
    html_tb20 = get_html_tb(res20)
    html_tb22 = get_html_tb(res22)
    html_tb23 = get_html_tb(res23)
    html_tb25 = get_html_tb(res25)
    html_tb32 = get_html_tb(initial_res32)
    html_tb33 = get_html_tb(res33)
    html_tb34 = get_html_tb(res34)
    html_tb35 = get_html_tb(res35)
    html_tb36 = get_html_tb(res36)
    html_tb37 = get_html_tb(res37)
    html_tb38 = get_html_tb(res38)
    html_tb39 = get_html_tb(res39)

    strategies_meta = [
        ("Tab_Strat12", "core", "🌟 綜合排行", "統計所有選股策略下，累計入選次數最多且大戶>=30%之最強勢標的。", len(res12), html_tb12),
        ("Tab_Strat27", "core", f"🎖 組合首選 (勝率 {strat_results_27_31[27]['win_rate']}%)", f"兩策略組合：{strat_results_27_31[27]['combo_name']}。歷史回測勝率最高組合。", strat_results_27_31[27]["count"], strat_results_27_31[27]["html"]),
        ("Tab_Strat28", "core", f"🎖 組合第二 (勝率 {strat_results_27_31[28]['win_rate']}%)", f"兩策略組合：{strat_results_27_31[28]['combo_name']}。歷史回測高勝率組合。", strat_results_27_31[28]["count"], strat_results_27_31[28]["html"]),
        ("Tab_Strat35", "core", "🚀 飆股主升段基因", "【四大維度融合】千張大戶 >= 60% ＋ 5/20/60MA 多頭發散 ＋ 量增 1.2 倍 ＋ 法人護盤。", len(res35), html_tb35),
        ("Tab_Strat36", "core", "🔥 跳空突破強勢整理 (Gap & Hold)", "【極強續航力】跳空長紅大漲後橫盤不補缺口，連日收盤皆站穩跳空日最高點之上＋外資投信零賣壓。", len(res36), html_tb36),
        ("Tab_Strat22", "core", "📈 高勝率基因精選", "【複合強勢】創 5 日高 ＋ 跳空缺口 ＋ 階梯式墊高 ＋ 量能放大 ＋ 法人無賣壓。", len(res22), html_tb22),
        ("Tab_Strat37", "core", "💻 資訊服務成長突破 (智聯服務風格)", "【成長突破】創近20日新高 ＋ 成交量>=400張且大於5日均量1.2倍 ＋ 外資投信無賣壓。", len(res37), html_tb37),
        ("Tab_Strat38", "core", "⚡ 低價量暴增動能 (律勝風格)", "【短線爆發】成交量暴增達前一日2.2倍以上（量>=300張） ＋ 實體紅K漲幅>=3%。", len(res38), html_tb38),
        ("Tab_Strat25", "core", "🛡 穩健法人鎖碼 (宏正風格)", "【價值法人】多頭排列（5>20>60MA） ＋ 千張大戶>=45% ＋ 外資投信近7日累計買超。", len(res25), html_tb25),
        ("Tab_Strat17", "breakout", "🏆 創 120 日半年新高", "【創高突破】最新收盤突破過去 120 日最高收盤價且紅K無長上影線。", len(res17), html_tb17),
        ("Tab_Strat4", "breakout", "🏆 創 20 日新高＋投信不賣", "【創高突破】收盤創 20 日新高 ＋ 近 7 天投信無賣超 ＋ 當日量增紅K。", len(res4), html_tb4),
        ("Tab_Strat6", "breakout", "🏆 創 5 日高＋跳空量增", "【短線爆發】創 5 日新高 ＋ 向上跳空不補 ＋ 成交量放大 ＋ 法人連 3 日不賣。", len(res6), html_tb6),
        ("Tab_Strat7", "breakout", "🏆 創 20 日新高＋底底高", "【波段強勢】創 20 日新高 ＋ 低點與收盤墊高 ＋ 量增 ＋ 法人連 7 日無賣超。", len(res7), html_tb7),
        ("Tab_Strat13", "breakout", "🏆 逼近 60 日高 (-2%內)", "【蓄勢待發】距離近 60 日最高收盤在 -2% 以內，且成交量 >= 500 張。", len(res13), html_tb13),
        ("Tab_Strat15", "breakout", "🏆 壓縮突破 60 日高", "【區間突破】創 60 日新高 ＋ 量增 1.2 倍 ＋ 前期 60 日震幅 <= 10%。", len(res15), html_tb15),
        ("Tab_Strat23", "breakout", "🔥 突破 10 天最大量高點", "【量價突破】最新收盤正式站上過去 10 日內最大量當日的最高價。", len(res23), html_tb23),
        ("Tab_Strat16", "breakout", "🔥 突破 60 天最大量高點", "【巨量突破】首日突破過去 60 天最大量日最高價 ＋ 外資投信近 7 天無賣出。", len(res16), html_tb16),
        ("Tab_Strat1", "breakout", "📈 向上跳空不補＋連續量增", "【強勢缺口】當日最低價 >= 前日收盤價 ＋ 成交量放大且收強勢紅 K。", len(res1), html_tb1),
        ("Tab_Strat34", "breakout", "🚀 創新高回檔再攻", "【回檔再攻】過去 20 天曾創 180 天新高 ＋ 回檔洗盤後重回前高 5% 攻擊區。", len(res34), html_tb34),
        ("Tab_Strat11", "breakout", "🎯 上櫃實體長紅強勢股", "【中小型強勢】上櫃股票 ＋ 實體紅K漲幅 >= 3.5% ＋ 量 >= 500張且大於5日均量1.3倍。", len(res11), html_tb11),
        ("Tab_Strat39", "chip", "👑 極致大戶跳空突破 (大戶>80%)", "【主力絕對鎖碼】千張大戶 >= 80% ＋ 成交量放大 1.3 倍 ＋ 當日低點大於昨日高點（強勢跳空）。", len(res39), html_tb39),
        ("Tab_Strat33", "chip", "👑 絕對鎖碼 (大戶>70%)", "【主力重壓】千張大戶 > 70% ＋ 5/20/60MA 多頭排列 ＋ 逼近半年高點。", len(res33), html_tb33),
        ("Tab_Strat32", "chip", "👑 大戶鎖碼＋法人集中度", "【籌碼高度集中】千張大戶 >= 60% ＋ 近 3 日外資投信買超佔成交量 >= 20%。", len(initial_res32), html_tb32),
        ("Tab_Strat2", "chip", "📈 階梯式墊高＋外資點火", "【步步為營】股價階梯墊高不破前日收盤 ＋ 外資投信連 7 日無賣超。", len(res2), html_tb2),
        ("Tab_Strat20", "chip", "🚀 外資連三買且越買越多", "【外資連買加速】外資連 3 日買超遞增 ＋ 量增 1.2 倍 ＋ 強勢紅 K。", len(res20), html_tb20),
        ("Tab_Strat3", "chip", "🚀 投信不賣＋外資連買", "【法人聯手】當日量增紅 K ＋ 投信連 7 天無賣超 ＋ 外資當日買超。", len(res3), html_tb3),
        ("Tab_Backtest26", "backtest", "📊 9月任兩策略組合挖掘", "【量化回測總覽】回測任兩策略同時命中之個股實際績效與勝率排行。", len(df_combos), html_tb26),
        ("Tab_Backtest21", "backtest", "📊 過去30天單策略勝率", "【量化回測總覽】統計各策略在過去 30 個交易日選出個股的隔日上漲勝率。", len(df_strat21), html_tb21)
    ]

    stock_dict_for_js = {}
    for _, r in df_merge.iterrows():
        code = str(r['股票代號'])
        stock_dict_for_js[code] = {
            "c": code,
            "n": str(r['股票名稱']),
            "m": str(r['市場']),
            "sc": int(r.get('技術評分(分)', 0)),
            "hcount": int(hit_counts.get(code, 0)),
            "hnames": ", ".join(hit_names.get(code, [])),
            "r10": float(r.get('近10日漲幅(%)', 0)),
            "p1": float(r.get('收盤價_1', 0)),
            "p0": float(r.get('收盤價_0', 0)),
            "chg": float(r.get('最新漲幅(%)', 0)),
            "pchg": float(r.get('前一日漲幅(%)', 0)),
            "vol": int(r.get('成交量_0', 0)),
            "vmul": float(r.get('增量倍數', 0)),
            "f7": int(r.get('外資近七日(張)', 0)),
            "t7": int(r.get('投信近七日(張)', 0)),
            "f20": int(r.get('外資近一月(張)', 0)),
            "whale": float(r.get('千張大戶比例(%)', 0))
        }
    stock_json_str = json.dumps(stock_dict_for_js, ensure_ascii=False)

    warning_html = ""
    if WARNINGS:
        unique_warns = sorted(set(WARNINGS))
        warn_count = len(unique_warns)
        warn_items_html = "".join([f"<li>{w}</li>" for w in unique_warns])
        warning_html = f"""
        <details class='warning-details'><summary class='warning-summary'>
        <span>⚠ 系統提示：共有 <b>{warn_count}</b> 個交易日資料缺少</span><span class='toggle-arrow'>▼</span>
        </summary><div class='warning-body'><ul>{warn_items_html}</ul></div></details>"""

    pages_html_blocks = ""
    dashboard_cards_html = ""
    select_options_html = "<option value='Dashboard'>🏠 返回策略總覽儀錶板</option><option value='Tab_Favorites'>⭐ 我的最愛與觀察股</option>"

    for tab_id, cat, title, desc, count, tbl_html in strategies_meta:
        select_options_html += f"<option value='{tab_id}'>{title} ({count}檔)</option>"
        badge_cls = "dash-card-badge hot" if count > 0 else "dash-card-badge"
        dashboard_cards_html += f"""
        <div class="dash-card" data-cat="{cat}" onclick="showPage('{tab_id}')">
            <div class="dash-card-top">
                <div class="dash-card-title">{title}</div>
                <div class="{badge_cls}"><b>{count}</b> 檔 ➔</div>
            </div>
            <div class="dash-card-desc">{desc}</div>
        </div>
        """
        pages_html_blocks += f"""
        <div id="{tab_id}" class="page-view" style="display:none;">
            <div class="sub-header">
                <button class="back-btn" onclick="showPage('Dashboard')">⬅ 返回</button>
                <select class="quick-jump-select" onchange="showPage(this.value)">
                    {select_options_html}
                </select>
            </div>
            <div class="info-box">
                <div class="info-box-row">
                    <span class="strategy-main-title">{title}</span>
                    <span class="count-badge">符合：<span id="count_{tab_id}">{count}</span> 檔</span>
                </div>
                <p>🎯 {desc}</p>
            </div>
            <div class="mobile-card-list" id="cards_{tab_id}"></div>
            <div class="table-container">{tbl_html}</div>
        </div>
        """

    html_content = f"""<!DOCTYPE html>
<html lang="zh-TW">
<head>
    <meta charset="utf-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0, maximum-scale=1.0, user-scalable=no, viewport-fit=cover">
    <title>多策略台股選股中心 2.0</title>
    <style>
        :root {{
            --primary: #0f172a; --primary-accent: #2563eb; --bg: #f1f5f9; --card-bg: #ffffff;
            --text: #0f172a; --text-muted: #64748b; --border: #e2e8f0;
            --up-red: #dc2626; --up-bg: #fef2f2; --down-green: #16a34a; --down-bg: #f0fdf4; --star: #f59e0b;
        }}
        * {{ box-sizing: border-box; -webkit-tap-highlight-color: transparent; }}
        body {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif; margin: 0; background-color: var(--bg); color: var(--text); padding-bottom: 80px; }}
        .app-header {{ background: linear-gradient(135deg, #0f172a, #1e3a8a); color: white; padding: 10px 14px; position: sticky; top: 0; z-index: 100; box-shadow: 0 2px 10px rgba(0,0,0,0.12); }}
        .header-top {{ display: flex; justify-content: space-between; align-items: center; margin-bottom: 8px; }}
        .app-header h1 {{ margin: 0; font-size: 16px; font-weight: 800; letter-spacing: 0.3px; }}
        .app-header p {{ margin: 2px 0 0 0; font-size: 11px; opacity: 0.8; }}
        .header-actions {{ display: flex; gap: 6px; align-items: center; }}
        .mode-toggle-btn {{ background: rgba(255,255,255,0.18); border: 1px solid rgba(255,255,255,0.35); color: white; padding: 5px 10px; border-radius: 16px; font-size: 11px; font-weight: 600; cursor: pointer; }}
        .global-search-wrap {{ position: relative; }}
        .global-search-input {{ width: 100%; padding: 8px 12px; border-radius: 8px; border: none; font-size: 13px; background: rgba(255,255,255,0.95); color: var(--text); outline: none; }}
        .search-results-dropdown {{ position: absolute; top: 40px; left: 0; right: 0; background: white; border-radius: 10px; box-shadow: 0 10px 25px rgba(0,0,0,0.18); max-height: 340px; overflow-y: auto; z-index: 200; display: none; border: 1px solid var(--border); }}
        .search-item {{ padding: 10px 12px; border-bottom: 1px solid var(--border); display: flex; justify-content: space-between; align-items: center; color: var(--text); font-size: 12px; }}
        .container {{ padding: 10px; max-width: 1440px; margin: 0 auto; }}
        .cat-bar {{ display: flex; gap: 6px; overflow-x: auto; padding: 4px 2px 10px 2px; scrollbar-width: none; }}
        .cat-bar::-webkit-scrollbar {{ display: none; }}
        .cat-pill {{ background: white; border: 1px solid var(--border); color: var(--text-muted); padding: 6px 12px; border-radius: 20px; font-size: 12px; font-weight: 600; white-space: nowrap; cursor: pointer; transition: all 0.15s; }}
        .cat-pill.active {{ background: var(--primary-accent); color: white; border-color: var(--primary-accent); box-shadow: 0 2px 6px rgba(37,99,235,0.25); }}
        .dash-grid {{ display: grid; grid-template-columns: repeat(auto-fill, minmax(290px, 1fr)); gap: 10px; }}
        .dash-card {{ background: var(--card-bg); border: 1px solid var(--border); border-radius: 12px; padding: 12px 14px; cursor: pointer; box-shadow: 0 1px 3px rgba(0,0,0,0.03); transition: transform 0.1s; }}
        .dash-card:active {{ transform: scale(0.98); }}
        .dash-card-top {{ display: flex; justify-content: space-between; align-items: center; margin-bottom: 6px; gap: 8px; }}
        .dash-card-title {{ font-size: 14px; font-weight: 700; color: var(--primary); }}
        .dash-card-desc {{ font-size: 11.5px; color: var(--text-muted); line-height: 1.4; }}
        .dash-card-badge {{ font-size: 11px; font-weight: 600; color: var(--text-muted); background: #f1f5f9; padding: 3px 8px; border-radius: 12px; white-space: nowrap; }}
        .dash-card-badge.hot {{ color: var(--up-red); background: var(--up-bg); }}
        .sub-header {{ display: flex; align-items: center; gap: 8px; margin-bottom: 8px; }}
        .back-btn {{ background: white; border: 1px solid var(--border); padding: 7px 12px; border-radius: 8px; font-size: 12px; font-weight: 700; cursor: pointer; color: var(--text); white-space: nowrap; }}
        .quick-jump-select {{ flex: 1; padding: 7px 10px; border-radius: 8px; border: 1px solid var(--border); background: white; font-size: 13px; font-weight: 700; color: var(--primary); outline: none; }}
        .info-box {{ background: var(--card-bg); border-radius: 10px; padding: 10px 12px; margin-bottom: 10px; border: 1px solid var(--border); }}
        .info-box-row {{ display: flex; justify-content: space-between; align-items: center; margin-bottom: 4px; }}
        .strategy-main-title {{ font-size: 14px; font-weight: 800; color: var(--primary); }}
        .info-box p {{ margin: 0; font-size: 11.5px; line-height: 1.4; color: var(--text-muted); }}
        .count-badge {{ background: var(--up-bg); color: var(--up-red); padding: 2px 8px; border-radius: 6px; font-weight: 700; font-size: 11.5px; }}
        .mobile-card-list {{ display: none; flex-direction: column; gap: 8px; }}
        body.view-cards .mobile-card-list {{ display: flex; }}
        body.view-cards .page-view:not(#Tab_Backtest26):not(#Tab_Backtest21) .table-container {{ display: none; }}
        .stock-m-card {{ background: white; border: 1px solid var(--border); border-radius: 10px; padding: 10px 12px; box-shadow: 0 1px 2px rgba(0,0,0,0.03); }}
        .stock-m-card.row-favorite {{ background: #fefce8; border-color: #fde047; }}
        .smc-top {{ display: flex; justify-content: space-between; align-items: center; border-bottom: 1px dashed var(--border); padding-bottom: 6px; margin-bottom: 6px; }}
        .smc-left {{ display: flex; align-items: center; gap: 6px; }}
        .smc-market {{ font-size: 10px; background: #e2e8f0; color: #475569; padding: 2px 5px; border-radius: 4px; font-weight: 600; }}
        .smc-price-wrap {{ text-align: right; }}
        .smc-price {{ font-size: 15px; font-weight: 800; margin-right: 6px; }}
        .smc-grid {{ display: grid; grid-template-columns: repeat(4, 1fr); gap: 6px; text-align: center; font-size: 11px; }}
        .smc-cell {{ background: #f8fafc; padding: 4px 2px; border-radius: 6px; }}
        .smc-label {{ color: var(--text-muted); font-size: 10px; display: block; }}
        .smc-val {{ font-weight: 700; color: var(--text); }}
        .smc-hits {{ margin-top: 6px; font-size: 11px; color: #1e40af; background: #eff6ff; padding: 4px 8px; border-radius: 6px; font-weight: 600; }}
        .table-container {{ background: var(--card-bg); border-radius: 10px; overflow: auto; max-height: 74vh; border: 1px solid var(--border); -webkit-overflow-scrolling: touch; }}
        .styled-table {{ border-collapse: separate; border-spacing: 0; width: 100%; font-size: 12px; text-align: center; }}
        .styled-table th {{ background-color: #f8fafc; color: var(--text-muted); font-weight: 700; padding: 8px 5px; border-bottom: 2px solid var(--border); cursor: pointer; user-select: none; position: sticky; top: 0; z-index: 20; }}
        .styled-table td {{ padding: 7px 5px; border-bottom: 1px solid var(--border); white-space: nowrap; vertical-align: middle; background-color: white; }}
        .styled-table:not(.backtest-table) th:nth-child(1), .styled-table:not(.backtest-table) td:nth-child(1) {{ position: sticky; left: 0; z-index: 10; min-width: 34px; max-width: 34px; background-color: #f8fafc; }}
        .styled-table:not(.backtest-table) th:nth-child(3), .styled-table:not(.backtest-table) td:nth-child(3) {{ position: sticky; left: 34px; z-index: 10; box-shadow: 2px 0 4px -1px rgba(0,0,0,0.06); min-width: 76px; background-color: #ffffff; }}
        .styled-table:not(.backtest-table) th:nth-child(1), .styled-table:not(.backtest-table) th:nth-child(3) {{ z-index: 30; background-color: #f1f5f9; }}
        .styled-table tr.row-favorite td {{ background-color: #fefce8 !important; }}
        .stock-link {{ text-decoration: none; display: inline-block; color: inherit; text-align: left; }}
        .stock-name {{ font-weight: 800; font-size: 13px; color: var(--primary-accent); line-height: 1.1; }}
        .stock-code {{ font-size: 10.5px; color: var(--text-muted); }}
        .tag-up {{ color: var(--up-red); font-weight: 700; }}
        .tag-down {{ color: var(--down-green); font-weight: 700; }}
        .tag-flat {{ color: var(--text-muted); font-weight: 600; }}
        .fav-star {{ cursor: pointer; color: #cbd5e1; font-size: 18px; padding: 2px 4px; display: inline-block; }}
        .fav-star.active {{ color: var(--star); }}
        .empty-state {{ padding: 30px; text-align: center; color: var(--text-muted); font-size: 13px; }}
        .add-stock-box {{ background: white; border: 1px solid var(--border); padding: 10px; border-radius: 10px; margin-bottom: 10px; display: flex; gap: 8px; }}
        .add-stock-input {{ flex: 1; padding: 8px 10px; font-size: 13px; border: 1px solid #cbd5e1; border-radius: 6px; outline: none; }}
        .add-stock-btn {{ background: var(--primary-accent); color: white; border: none; padding: 8px 14px; border-radius: 6px; font-size: 12px; font-weight: 700; cursor: pointer; }}
        .bottom-nav {{ position: fixed; bottom: 0; left: 0; right: 0; background: rgba(255,255,255,0.96); backdrop-filter: blur(8px); border-top: 1px solid var(--border); display: flex; justify-content: space-around; padding: 6px 0 calc(6px + env(safe-area-inset-bottom)); z-index: 150; box-shadow: 0 -2px 10px rgba(0,0,0,0.05); }}
        .b-nav-item {{ border: none; background: none; display: flex; flex-direction: column; align-items: center; gap: 2px; font-size: 10.5px; font-weight: 600; color: var(--text-muted); cursor: pointer; padding: 4px 10px; border-radius: 8px; }}
        .b-nav-item.active {{ color: var(--primary-accent); font-weight: 800; }}
        .b-nav-icon {{ font-size: 16px; }}
        .warning-details {{ background-color: #fff1f2; border: 1px solid #fecdd3; border-radius: 8px; margin-bottom: 8px; font-size: 11px; color: #be123c; }}
        .warning-summary {{ padding: 7px 12px; cursor: pointer; font-weight: 600; display: flex; justify-content: space-between; list-style: none; }}
    </style>
</head>
<body>
    <div class="app-header">
        <div class="header-top">
            <div>
                <h1>📈 台股多策略選股中心 2.0</h1>
                <p>資料基準: {date_0} (近120日動能與籌碼追蹤)</p>
            </div>
            <div class="header-actions">
                <button id="viewModeBtn" class="mode-toggle-btn" onclick="toggleViewMode()">📱 卡片模式</button>
            </div>
        </div>
        <div class="global-search-wrap">
            <input type="text" id="globalSearchInput" class="global-search-input" placeholder="🔍 搜尋全市場股號或名稱 (例: 6277, 6751, 3354)..." oninput="onGlobalSearch(this.value)">
            <div id="globalSearchDropdown" class="search-results-dropdown"></div>
        </div>
    </div>

    <div class="container">
        {warning_html}
        <div id="Dashboard" class="page-view">
            <div class="cat-bar">
                <button class="cat-pill active" onclick="filterCategory('all', this)">全部策略 (27)</button>
                <button class="cat-pill" onclick="filterCategory('core', this)">🔥 核心首選</button>
                <button class="cat-pill" onclick="filterCategory('breakout', this)">🏆 創高突破</button>
                <button class="cat-pill" onclick="filterCategory('chip', this)">👑 籌碼鎖碼</button>
                <button class="cat-pill" onclick="filterCategory('backtest', this)">📊 量化回測</button>
            </div>
            <div class="dash-grid">{dashboard_cards_html}</div>
        </div>

        <div id="Tab_Favorites" class="page-view" style="display:none;">
            <div class="sub-header">
                <button class="back-btn" onclick="showPage('Dashboard')">⬅ 返回</button>
                <select class="quick-jump-select" onchange="showPage(this.value)">
                    {select_options_html}
                </select>
            </div>
            <div class="add-stock-box">
                <input type="text" id="customCodeInput" class="add-stock-input" placeholder="輸入 4 位數股票代號..." maxlength="4">
                <button class="add-stock-btn" onclick="addCustomStock()">＋ 加入最愛</button>
            </div>
            <div class="info-box">
                <div class="info-box-row">
                    <span class="strategy-main-title">⭐ 我的最愛與自選觀察股</span>
                    <span class="count-badge">總計：<span id="count_Tab_Favorites">0</span> 檔</span>
                </div>
                <p>🎯 支援全市場上市櫃個股追蹤，點擊星號或上方輸入代號即可永久保存於手機瀏覽器。</p>
            </div>
            <div id="fav-empty-msg" class="empty-state" style="display:none;">尚未加入最愛標的，請點擊清單中的 ☆ 圖示或於上方輸入代號新增。</div>
            <div class="mobile-card-list" id="cards_Tab_Favorites"></div>
            <div class="table-container">
                <table id="fav-table" class="styled-table sortable-table">
                    <thead>
                        <tr>
                            <th>⭐</th><th>市場</th><th>標的</th><th>技術評分(分)</th><th>當日命中</th>
                            <th>近10日漲幅(%)</th><th>{d1_s} 收盤</th><th>{d0_s} 收盤</th><th>最新漲幅(%)</th>
                            <th>{d0_s} 量(張)</th><th>增量倍數</th><th>外資近七日(張)</th><th>投信近七日(張)</th><th>千張大戶比例(%)</th>
                        </tr>
                    </thead>
                    <tbody></tbody>
                </table>
            </div>
        </div>

        {pages_html_blocks}
    </div>

    <div class="bottom-nav">
        <button class="b-nav-item active" id="bnav_Dashboard" onclick="showPage('Dashboard')"><span class="b-nav-icon">🏠</span><span>策略總覽</span></button>
        <button class="b-nav-item" id="bnav_Tab_Strat12" onclick="showPage('Tab_Strat12')"><span class="b-nav-icon">🌟</span><span>綜合排行</span></button>
        <button class="b-nav-item" id="bnav_Tab_Strat27" onclick="showPage('Tab_Strat27')"><span class="b-nav-icon">🎖</span><span>組合首選</span></button>
        <button class="b-nav-item" id="bnav_Tab_Favorites" onclick="showPage('Tab_Favorites')"><span class="b-nav-icon">⭐</span><span>我的最愛(<b id="fav-count">0</b>)</span></button>
        <button class="b-nav-item" onclick="toggleViewMode()"><span class="b-nav-icon" id="bnav_mode_icon">📱</span><span id="bnav_mode_text">切換視圖</span></button>
    </div>

    <script>
        const ALL_STOCKS = {stock_json_str};
        const FAV_KEY = 'tw_stock_favorites';
        const VIEW_MODE_KEY = 'tw_stock_view_mode';

        function getFavorites() {{
            try {{ return JSON.parse(localStorage.getItem(FAV_KEY) || '[]'); }}
            catch(e) {{ return []; }}
        }}
        function saveFavorites(favs) {{ localStorage.setItem(FAV_KEY, JSON.stringify(favs)); }}

        function fmtPctHtml(val) {{
            const f = parseFloat(val) || 0;
            if (f > 0) return `<span class="tag-up">+${{f.toFixed(2)}}%</span>`;
            if (f < 0) return `<span class="tag-down">${{f.toFixed(2)}}%</span>`;
            return `<span class="tag-flat">0.00%</span>`;
        }}

        function buildStockCardHtml(s, isFav) {{
            const starCls = isFav ? 'fav-star active' : 'fav-star';
            const starChar = isFav ? '★' : '☆';
            const cardCls = isFav ? 'stock-m-card row-favorite' : 'stock-m-card';
            const yhUrl = `https://tw.stock.yahoo.com/quote/${{s.c}}/technical-analysis`;
            const hitsBanner = s.hcount > 0 ? `<div class="smc-hits">🎯 命中 ${{s.hcount}} 策略：${{s.hnames}}</div>` : '';
            return `
            <div class="${{cardCls}}" data-code="${{s.c}}">
                <div class="smc-top">
                    <div class="smc-left">
                        <span class="${{starCls}}" data-code="${{s.c}}">${{starChar}}</span>
                        <span class="smc-market">${{s.m}}</span>
                        <a href="${{yhUrl}}" target="_blank" class="stock-link">
                            <span class="stock-name">${{s.n}}</span> <span class="stock-code">(${{s.c}})</span>
                        </a>
                    </div>
                    <div class="smc-price-wrap">
                        <span class="smc-price">${{s.p0}}</span>
                        ${{fmtPctHtml(s.chg)}}
                    </div>
                </div>
                <div class="smc-grid">
                    <div class="smc-cell"><span class="smc-label">技術評分</span><span class="smc-val">${{s.sc}}分</span></div>
                    <div class="smc-cell"><span class="smc-label">今日量(倍)</span><span class="smc-val">${{s.vol}}張 (${{s.vmul}}x)</span></div>
                    <div class="smc-cell"><span class="smc-label">外資/投信(7日)</span><span class="smc-val">${{s.f7}} / ${{s.t7}}</span></div>
                    <div class="smc-cell"><span class="smc-label">千張大戶</span><span class="smc-val">${{s.whale}}%</span></div>
                </div>
                ${{hitsBanner}}
            </div>`;
        }}

        function renderAllMobileCardLists() {{
            const favs = getFavorites();
            document.querySelectorAll('.page-view').forEach(page => {{
                const pageId = page.id;
                if (pageId === 'Dashboard' || pageId === 'Tab_Favorites' || pageId === 'Tab_Backtest26' || pageId === 'Tab_Backtest21') return;
                const cardContainer = document.getElementById('cards_' + pageId);
                const tbody = page.querySelector('.styled-table tbody');
                if (!cardContainer || !tbody) return;
                const codes = [];
                tbody.querySelectorAll('tr').forEach(tr => {{
                    const c = tr.querySelector('.fav-star')?.dataset.code;
                    if (c && ALL_STOCKS[c]) codes.push(c);
                }});
                if (codes.length === 0) {{
                    cardContainer.innerHTML = '<div class="empty-state">目前無符合股票</div>';
                }} else {{
                    cardContainer.innerHTML = codes.map(c => buildStockCardHtml(ALL_STOCKS[c], favs.includes(c))).join('');
                }}
            }});
        }}

        function updateFavUI() {{
            const favs = getFavorites();
            const cnt = document.getElementById('count_Tab_Favorites'); if(cnt) cnt.innerText = favs.length;
            const quickCnt = document.getElementById('fav-count'); if(quickCnt) quickCnt.innerText = favs.length;

            document.querySelectorAll('.fav-star').forEach(star => {{
                const code = star.dataset.code;
                const tr = star.closest('tr');
                const mcard = star.closest('.stock-m-card');
                if (favs.includes(code)) {{
                    star.classList.add('active'); star.innerHTML = '★';
                    if(tr) tr.classList.add('row-favorite');
                    if(mcard) mcard.classList.add('row-favorite');
                }} else {{
                    star.classList.remove('active'); star.innerHTML = '☆';
                    if(tr) tr.classList.remove('row-favorite');
                    if(mcard) mcard.classList.remove('row-favorite');
                }}
            }});

            const favTbody = document.querySelector('#fav-table tbody');
            const favCards = document.getElementById('cards_Tab_Favorites');
            const msg = document.getElementById('fav-empty-msg');
            const tblWrap = document.querySelector('#Tab_Favorites .table-container');

            if (favTbody && favCards) {{
                favTbody.innerHTML = '';
                favCards.innerHTML = '';
                const validFavs = favs.filter(code => ALL_STOCKS[code]);
                if (validFavs.length === 0) {{
                    msg.style.display = 'block';
                    tblWrap.style.display = 'none';
                }} else {{
                    msg.style.display = 'none';
                    tblWrap.style.display = '';
                    validFavs.forEach(code => {{
                        const s = ALL_STOCKS[code];
                        const yhUrl = `https://tw.stock.yahoo.com/quote/${{s.c}}/technical-analysis`;
                        const tr = document.createElement('tr');
                        tr.className = 'row-favorite';
                        tr.innerHTML = `
                            <td><span class="fav-star active" data-code="${{s.c}}">★</span></td>
                            <td>${{s.m}}</td>
                            <td><a href="${{yhUrl}}" target="_blank" class="stock-link"><div class="stock-name">${{s.n}}</div><div class="stock-code">${{s.c}}</div></a></td>
                            <td>${{s.sc}}</td>
                            <td>${{s.hcount > 0 ? s.hcount + '個 (' + s.hnames + ')' : '觀察股'}}</td>
                            <td>${{fmtPctHtml(s.r10)}}</td>
                            <td>${{s.p1}}</td>
                            <td>${{s.p0}}</td>
                            <td>${{fmtPctHtml(s.chg)}}</td>
                            <td>${{s.vol}}</td>
                            <td>${{s.vmul}}</td>
                            <td>${{s.f7}}</td>
                            <td>${{s.t7}}</td>
                            <td>${{s.whale}}</td>
                        `;
                        favTbody.appendChild(tr);
                    }});
                    favCards.innerHTML = validFavs.map(c => buildStockCardHtml(ALL_STOCKS[c], true)).join('');
                }}
            }}
        }}

        function showPage(pageId) {{
            document.querySelectorAll('.page-view').forEach(el => el.style.display = 'none');
            const target = document.getElementById(pageId);
            if(target) {{
                target.style.display = 'block';
                const sel = target.querySelector('.quick-jump-select');
                if(sel) sel.value = pageId;
                window.scrollTo(0, 0);
            }}
            document.querySelectorAll('.b-nav-item').forEach(btn => btn.classList.remove('active'));
            const activeBtn = document.getElementById('bnav_' + pageId);
            if(activeBtn) activeBtn.classList.add('active');
        }}

        function filterCategory(cat, btnEl) {{
            document.querySelectorAll('.cat-pill').forEach(p => p.classList.remove('active'));
            if(btnEl) btnEl.classList.add('active');
            document.querySelectorAll('.dash-card').forEach(card => {{
                if (cat === 'all' || card.dataset.cat === cat) card.style.display = '';
                else card.style.display = 'none';
            }});
        }}

        function applyViewMode(mode) {{
            const btn = document.getElementById('viewModeBtn');
            const bIcon = document.getElementById('bnav_mode_icon');
            const bText = document.getElementById('bnav_mode_text');
            if (mode === 'cards') {{
                document.body.classList.add('view-cards');
                if(btn) btn.innerText = '📊 切換表格';
                if(bIcon) bIcon.innerText = '📊';
                if(bText) bText.innerText = '表格模式';
            }} else {{
                document.body.classList.remove('view-cards');
                if(btn) btn.innerText = '📱 切換卡片';
                if(bIcon) bIcon.innerText = '📱';
                if(bText) bText.innerText = '卡片模式';
            }}
            localStorage.setItem(VIEW_MODE_KEY, mode);
        }}

        function toggleViewMode() {{
            const isCards = document.body.classList.contains('view-cards');
            applyViewMode(isCards ? 'table' : 'cards');
        }}

        function addCustomStock() {{
            const input = document.getElementById('customCodeInput');
            const code = input.value.trim();
            if(!code || code.length !== 4 || isNaN(code)) {{ alert('請輸入有效的 4 位數股票代號！'); return; }}
            if(!ALL_STOCKS[code]) {{ alert('查無此上市櫃股票代號：' + code); return; }}
            let favs = getFavorites();
            if(!favs.includes(code)) {{
                favs.push(code); saveFavorites(favs); updateFavUI();
            }}
            input.value = '';
        }}

        function onGlobalSearch(q) {{
            const dropdown = document.getElementById('globalSearchDropdown');
            const kw = q.trim().toLowerCase();
            if (!kw) {{ dropdown.style.display = 'none'; return; }}
            const favs = getFavorites();
            const matches = Object.values(ALL_STOCKS).filter(s => s.c.includes(kw) || s.n.toLowerCase().includes(kw)).slice(0, 12);
            if (matches.length === 0) {{
                dropdown.innerHTML = '<div class="search-item">查無符合之上市櫃個股</div>';
            }} else {{
                dropdown.innerHTML = matches.map(s => {{
                    const isFav = favs.includes(s.c);
                    const yhUrl = `https://tw.stock.yahoo.com/quote/${{s.c}}/technical-analysis`;
                    return `
                    <div class="search-item">
                        <div>
                            <span class="fav-star ${{isFav ? 'active' : ''}}" data-code="${{s.c}}">${{isFav ? '★' : '☆'}}</span>
                            <a href="${{yhUrl}}" target="_blank" style="font-weight:700;color:#1e3a8a;text-decoration:none;">${{s.n}} (${{s.c}})</a>
                            <span style="font-size:11px;color:#64748b;margin-left:6px;">評分:${{s.sc}} | 大戶:${{s.whale}}%</span>
                            <div style="font-size:10.5px;color:#2563eb;margin-top:2px;">${{s.hcount > 0 ? '🎯 命中: ' + s.hnames : '今日未命中策略'}}</div>
                        </div>
                        <div style="text-align:right;">
                            <div style="font-weight:800;font-size:13px;">${{s.p0}}</div>
                            ${{fmtPctHtml(s.chg)}}
                        </div>
                    </div>`;
                }}).join('');
            }}
            dropdown.style.display = 'block';
        }}

        document.addEventListener('click', function(e) {{
            if (e.target && e.target.classList.contains('fav-star')) {{
                e.preventDefault(); e.stopPropagation();
                const code = e.target.dataset.code;
                let favs = getFavorites();
                if (!favs.includes(code)) favs.push(code);
                else favs = favs.filter(c => c !== code);
                saveFavorites(favs); updateFavUI();
                return;
            }}
            const searchWrap = document.querySelector('.global-search-wrap');
            if (searchWrap && !searchWrap.contains(e.target)) {{
                document.getElementById('globalSearchDropdown').style.display = 'none';
            }}
        }});

        window.addEventListener('DOMContentLoaded', () => {{
            const savedMode = localStorage.getItem(VIEW_MODE_KEY) || (window.innerWidth <= 768 ? 'cards' : 'table');
            applyViewMode(savedMode);
            renderAllMobileCardLists();
            updateFavUI();
        }});
    </script>
</body>
</html>
"""
    html_filename = "index.html"
    file_path = os.path.abspath(html_filename)
    with open(file_path, "w", encoding="utf-8") as f:
        f.write(html_content)
    print(f"\n✅ 網頁已生成: {html_filename}")

    custom_watch_list = ["6277", "6751", "3354", "5508", "5483", "1321", "5465", "1334", "3631", "8171", "5426", "4577", "6525", "1319", "6122", "2006", "6605"]
    watch_codes = set(custom_watch_list)
    strat_dfs = [res2, res17, res22, res25, res35, res36, res37, res38, res39]
    for s_df in strat_dfs:
        if isinstance(s_df, pd.DataFrame) and not s_df.empty and '股票代號' in s_df.columns:
            watch_codes.update(s_df['股票代號'].astype(str).str.strip().tolist())

    if watch_codes and not df_merge.empty:
        cols_Needed = ['股票代號', '股票名稱', '市場', '收盤價_0', '收盤價_1', '5日最高收盤']
        unique_target_df = (
            df_merge[df_merge['股票代號'].isin(watch_codes)][cols_Needed]
            .drop_duplicates(subset=['股票代號'])
            .reset_index(drop=True)
        )

        watchlist_payload = {
            "base_date": str(date_0),
            "updated_at": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "stocks": unique_target_df.to_dict("records")
        }
        watchlist_path = os.path.join(CACHE_DIR, "intraday_watchlist.json")

        if not unique_target_df.empty:
            with open(watchlist_path, "w", encoding="utf-8") as f:
                json.dump(watchlist_payload, f, ensure_ascii=False, indent=2)
            print(f"\n📌 已安全匯出 {len(unique_target_df)} 檔監控清單至: {watchlist_path}")

    if os.environ.get("GITHUB_ACTIONS") != "true":
        webbrowser.open(f"file:///{file_path}")

if __name__ == "__main__":
    main()
