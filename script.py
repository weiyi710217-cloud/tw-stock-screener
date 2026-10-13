import datetime
import time
import os
import webbrowser
import itertools
import numpy as np
import pandas as pd
import requests
import yfinance as yf
import warnings

warnings.simplefilter(action='ignore', category=pd.errors.PerformanceWarning)

TWSE_HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    "Accept": "application/json",
    "Connection": "keep-alive"
}

TPEX_HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    "Accept": "application/json, text/javascript, */*; q=0.01",
    "Accept-Language": "zh-TW,zh;q=0.9,en-US;q=0.8,en;q=0.7",
    "Referer": "https://www.tpex.org.tw/",
    "Connection": "keep-alive",
    "Cache-Control": "max-age=0"
}

OPENAPI_HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
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
        "parse_mode": "Markdown"
    }
    try:
        resp = requests.post(url, json=payload, timeout=10)
        if resp.status_code == 200:
            print("📲 Telegram 訊息發送成功！")
        else:
            print(f"⚠ Telegram 發送失敗: {resp.text}")
    except Exception as e:
        print(f"⚠ Telegram 連線錯誤: {e}")

def check_all_intraday_signals(df_5m, df_15m, prev_close, is_volume_safe, is_above_10d_high):
    """
    【盤中訊號核心引擎】一次性計算共通變數，並判斷是否觸發四大策略
    回傳: (觸發的策略編號清單, 共通數據字典)
    """
    if df_5m is None or len(df_5m) < 3 or not is_volume_safe or not is_above_10d_high:
        return [], {}

    c0_5m = float(df_5m['Close'].iloc[-1])
    c1_5m = float(df_5m['Close'].iloc[-2])
    
    pct_change = ((c0_5m - prev_close) / prev_close) * 100 if prev_close > 0 else 0.0
    
    # 共通漲停過濾（已漲停或當根暴漲鎖死，全數不通知）
    if pct_change >= 9.5 or (c0_5m - c1_5m) / c1_5m * 100 >= 9.5:
        return [], {}

    triggered_signals = []
    signal_data = {"c0": c0_5m, "pct_change": pct_change}

    # 策略一：雙時框連續收盤向上墊高 (需 >= 3根)
    c2_5m = float(df_5m['Close'].iloc[-3])
    cond_5m_up = (c0_5m > c1_5m) and (c1_5m > c2_5m)
    
    cond_15m_up = False
    if df_15m is not None and len(df_15m) >= 3:
        c0_15m = float(df_15m['Close'].iloc[-1])
        c1_15m = float(df_15m['Close'].iloc[-2])
        c2_15m = float(df_15m['Close'].iloc[-3])
        cond_15m_up = (c0_15m > c1_15m) and (c1_15m > c2_15m)

    if cond_5m_up and cond_15m_up:
        triggered_signals.append(1)

    # 策略三：主力階梯推升型態 (需 >= 5根)
    if len(df_5m) >= 5:
        c3_5m = float(df_5m['Close'].iloc[-4])
        is_step_up = cond_5m_up and (c2_5m > c3_5m)
        is_price_stable = (c0_5m / c1_5m) <= 1.035
        if is_step_up and is_price_stable:
            triggered_signals.append(3)

    # 策略二：20分鐘高點突破 (需 >= 6根)
    if len(df_5m) >= 6:
        recent_4_closes = df_5m['Close'].iloc[-5:-1]
        recent_4_highs = df_5m['High'].iloc[-5:-1]
        
        max_high_20m = float(recent_4_highs.max())
        min_close_20m = float(recent_4_closes.min())
        
        is_holding_high = min_close_20m >= (max_high_20m * 0.99)
        is_breaking_out = c0_5m > max_high_20m
        if is_holding_high and is_breaking_out:
            triggered_signals.append(2)
            signal_data["max_h20"] = max_high_20m

    # 策略四：高檔鎖碼橫盤突破 (需 >= 8根)
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

def intraday_check_strategy(res_df):
    if res_df is None or res_df.empty:
        print("ℹ 目前無符合的監控標的。")
        return

    print(f"🚀 啟動盤中即時四大策略整合引擎偵測，準備檢查 {len(res_df)} 檔強勢股...")
    
    for _, row in res_df.iterrows():
        code = str(row['股票代號']).strip()
        name = str(row['股票名稱']).strip() if '股票名稱' in row else ""
        market = str(row['市場']).strip() if '市場' in row else "上市"
        
        prev_close = float(row.get('收盤價_1', 0))
        suffix = ".TWO" if "上櫃" in market else ".TW"
        ticker_symbol = f"{code}{suffix}"
        
        try:
            stock = yf.Ticker(ticker_symbol)
            
            df_daily = stock.history(period="15d")
            if df_daily is None or len(df_daily) < 10:
                continue
                
            if prev_close <= 0:
                prev_close = float(df_daily['Close'].iloc[-2])
                
            recent_10_closes = df_daily['Close'].iloc[-11:-1] if len(df_daily) >= 11 else df_daily['Close'].iloc[:-1]
            max_10d_close = float(recent_10_closes.max())

            df_5m = stock.history(period="1d", interval="5m")
            if (df_5m is None or len(df_5m) < 6) and suffix == ".TWO":
                stock = yf.Ticker(f"{code}.TW")
                df_5m = stock.history(period="1d", interval="5m")

            if df_5m is None or len(df_5m) < 6:
                continue
                
            df_15m = stock.history(period="1d", interval="15m")
            current_price = float(df_5m['Close'].iloc[-1])
            is_above_10d_high = current_price > max_10d_close

            v0_5m = float(df_5m['Volume'].iloc[-1]) / 100.0
            v1_5m = float(df_5m['Volume'].iloc[-2]) / 100.0
            is_volume_safe = (v1_5m <= 0) or (v0_5m <= v1_5m * 2.0)

            triggered_list, sig_data = check_all_intraday_signals(df_5m, df_15m, prev_close, is_volume_safe, is_above_10d_high)
            
            if not triggered_list:
                continue
                
            c0_s = sig_data.get("c0")
            pct_s = sig_data.get("pct_change")
            sign = "+" if pct_s >= 0 else ""
            
            if 1 in triggered_list:
                msg1 = f"🚨 *【策略一：雙時框連拉通知】*\n" \
                       f"• 股票: `{name} ({code})`\n" \
                       f"• 漲跌幅: `{sign}{pct_s:.2f}%` (現價: `{c0_s:.2f}`)\n" \
                       f"• 狀態: 突破10日高且雙時框向上！\n" \
                       f"• 成交量: `{v0_5m:.1f}張`"
                print(msg1)
                send_telegram_message(msg1)

            if 2 in triggered_list:
                msg2 = f"🚨 *【策略二：20分鐘高點突破】*\n" \
                       f"• 股票: `{name} ({code})`\n" \
                       f"• 漲跌幅: `{sign}{pct_s:.2f}%` (現價: `{c0_s:.2f}`)\n" \
                       f"• 狀態: 高檔抗跌後帶量突破！\n" \
                       f"• 20分高點: `{sig_data.get('max_h20'):.2f}`"
                print(msg2)
                send_telegram_message(msg2)

            if 3 in triggered_list and 1 not in triggered_list: 
                msg3 = f"🚨 *【策略三：主力階梯推升】*\n" \
                       f"• 股票: `{name} ({code})`\n" \
                       f"• 漲跌幅: `{sign}{pct_s:.2f}%` (現價: `{c0_s:.2f}`)\n" \
                       f"• 狀態: 階梯式穩健墊高，主力點火！"
                print(msg3)
                send_telegram_message(msg3)

            if 4 in triggered_list and 2 not in triggered_list:
                msg4 = f"🚨 *【策略四：全日高檔鎖碼突破】*\n" \
                       f"• 股票: `{name} ({code})`\n" \
                       f"• 漲跌幅: `{sign}{pct_s:.2f}%` (現價: `{c0_s:.2f}`)\n" \
                       f"• 狀態: 全日高姿態整理，準備再攻！\n" \
                       f"• 當日高點: `{sig_data.get('day_max'):.2f}`"
                print(msg4)
                send_telegram_message(msg4)
                
            time.sleep(0.5)
        except Exception as e:
            print(f"⚠ 檢查股票 {code} ({name}) 時發生錯誤: {e}")

# ==========================================
# 盤後資料獲取核心
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
    for attempt in range(max_retries):
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
    for attempt in range(2):
        try:
            resp = requests.get(url, headers=TPEX_HEADERS, timeout=6)
            if resp.status_code != 200:
                time.sleep(0.5)
                continue
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
    try: df = pd.DataFrame(target_table.get("data", []), columns=fields)
    except Exception: return pd.DataFrame()
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
    else: return pd.DataFrame()

def get_tdcc_data() -> pd.DataFrame:
    today_str = datetime.date.today().strftime("%Y%m%d")
    cache_file = os.path.join(CACHE_DIR, f"tdcc_{today_str}.csv")
    df_cached = safe_read_csv(cache_file, dtype={"股票代號": str})
    if not df_cached.empty: return df_cached
    print("🌐 抓取【集保千張大戶】...")
    url = "https://smart.tdcc.com.tw/opendata/getOD.ashx?id=1-5"
    for attempt in range(2):
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
        k_code, k_issue, k_ratio = find_key("code") or find_key("代號"), find_key("issued") or find_key("發行"), (find_key("hold", "ratio") or find_key("hold", "percent") or find_key("持股", "比", exclude=("尚可",)))
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
        else: time.sleep(0.1)
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
    print("啟動多策略選股程式，準備抓取價量、法人與大戶資料...\n")
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
        if df_i.empty: continue
        df_temp = df_i[["股票代號", "開盤價", "最高價", "最低價", "收盤價", "成交量"]].copy()
        df_temp = df_temp.rename(columns={"收盤價": f"收盤價_{i}", "成交量": f"成交量_{i}", "最高價": f"最高價_{i}", "最低價": f"最低價_{i}", "開盤價": f"開盤價_{i}"})
        df_merge = pd.merge(df_merge, df_temp, on="股票代號", how="left")

    df_merge = df_merge.copy() 
    vol_cols = [col for col in df_merge.columns if col.startswith('成交量_')]
    for col in vol_cols: df_merge[col] = (df_merge[col] / 1000).round(0)

    close_cols_5 = [col for col in df_merge.columns if col.startswith('收盤價_') and 1 <= int(col.split('_')[1]) <= 5]
    close_cols_20 = [col for col in df_merge.columns if col.startswith('收盤價_') and 1 <= int(col.split('_')[1]) <= 20]
    close_cols_40 = [col for col in df_merge.columns if col.startswith('收盤價_') and 1 <= int(col.split('_')[1]) <= 40]
    close_cols_60 = [col for col in df_merge.columns if col.startswith('收盤價_') and 1 <= int(col.split('_')[1]) <= 60]
    close_cols_120 = [col for col in df_merge.columns if col.startswith('收盤價_') and 1 <= int(col.split('_')[1]) <= 120]

    df_merge['5日最高收盤'] = df_merge[close_cols_5].max(axis=1)
    df_merge['20日最高收盤'] = df_merge[close_cols_20].max(axis=1)
    df_merge['60日最高收盤'] = df_merge[close_cols_60].max(axis=1)
    df_merge['120日最高收盤'] = df_merge[close_cols_120].max(axis=1)
    
    df_merge['20日最低收盤'] = df_merge[close_cols_20].min(axis=1)
    df_merge['40日最低收盤'] = df_merge[close_cols_40].min(axis=1)
    df_merge['60日最低收盤'] = df_merge[close_cols_60].min(axis=1)

    vol_cols_60 = [col for col in df_merge.columns if col.startswith('成交量_') and int(col.split('_')[1]) <= 60]
    df_merge['60日最大量'] = df_merge[vol_cols_60].max(axis=1)

    df_merge['5MA'] = df_merge[[f'收盤價_{j}' for j in range(5)]].mean(axis=1)
    df_merge['20MA'] = df_merge[[f'收盤價_{j}' for j in range(20)]].mean(axis=1)
    df_merge['60MA'] = df_merge[[f'收盤價_{j}' for j in range(60)]].mean(axis=1)
    
    def get_max_vol_high(row):
        try:
            vols = [(float(row.get(f'成交量_{i}', -1)), i) for i in range(60) if pd.notna(float(row.get(f'成交量_{i}', -1)))]
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

    def calculate_technical_score(row):
        score = 0
        try:
            c, ma20, ma60 = float(row.get('收盤價_0', 0)), float(row.get('20MA', 0)), float(row.get('60MA', 0))
            if c > ma20 and ma20 > ma60: score += 25
            elif c > ma20: score += 12
            closes = [float(row.get(f'收盤價_{j}', 0)) for j in range(20)]
            if len(closes) == 20 and ma20 > 0:
                lower_band = ma20 - 2 * np.std(closes)
                if lower_band > 0:
                    if (c - lower_band) / lower_band < 0.03: score += 25
                    elif c < ma20: score += 10
            lows_9, highs_9 = [float(row.get(f'最低價_{j}', 0)) for j in range(9)], [float(row.get(f'最高價_{j}', 0)) for j in range(9)]
            if min(lows_9) > 0 and max(highs_9) > min(lows_9):
                rsv = (c - min(lows_9)) / (max(highs_9) - min(lows_9)) * 100
                k_val = rsv * 0.33 + 50 * 0.67
                d_val = k_val * 0.33 + 50 * 0.67
                if k_val > d_val and k_val < 80: score += 25
                elif k_val > d_val: score += 10
            ema12, ema26 = float(row.get('5MA', c)), float(row.get('20MA', c))
            dif, dea = ema12 - ema26, (ema12 - ema26) * 0.8
            macd_hist = (dif - dea) * 2
            if macd_hist > 0 and dif > dea: score += 25
            elif macd_hist > 0: score += 10
        except Exception: pass
        return int(score)

    df_merge['技術評分(分)'] = df_merge.apply(calculate_technical_score, axis=1)
    df_merge['近7日符合次數'] = "-"
    df_merge['最新法人買超(張)'] = (df_merge['外資_0'] + df_merge['投信_0']).round(0)
    df_merge['增量倍數'] = (df_merge['成交量_0'] / df_merge['成交量_1'].replace(0, np.nan)).round(2).fillna(0.0)
    df_merge['實體K漲幅(%)'] = (((df_merge['收盤價_0'] - df_merge['開盤價_0']) / df_merge['開盤價_0'].replace(0, np.nan)) * 100).round(2).fillna(0.0)
    df_merge['距離60日高點(%)'] = (((df_merge['收盤價_0'] - df_merge['60日最高收盤']) / df_merge['60日最高收盤'].replace(0, np.nan)) * 100).round(2).fillna(0.0)
    df_merge['20日震幅(%)'] = (((df_merge['收盤價_0'] - df_merge['20日最低收盤']) / df_merge['20日最低收盤'].replace(0, np.nan)) * 100).round(2).fillna(0.0)
    df_merge['40日震幅(%)'] = (((df_merge['收盤價_0'] - df_merge['40日最低收盤']) / df_merge['40日最低收盤'].replace(0, np.nan)) * 100).round(2).fillna(0.0)
    df_merge['60日震幅(%)'] = (((df_merge['收盤價_0'] - df_merge['60日最低收盤']) / df_merge['60日最低收盤'].replace(0, np.nan)) * 100).round(2).fillna(0.0)
    df_merge['最新日外資(張)'] = df_merge['外資_0']
    df_merge['前1日外資(張)'] = df_merge['外資_1']
    df_merge['前2日外資(張)'] = df_merge['外資_2']
    df_merge['法人1日集中度(%)'] = (((df_merge['外資_0'] + df_merge['投信_0']) / df_merge['成交量_0'].replace(0, np.nan)) * 100).round(2).fillna(0.0)
    df_merge['法人2日集中度(%)'] = (((df_merge['外資_0'] + df_merge['外資_1'] + df_merge['投信_0'] + df_merge['投信_1']) / (df_merge['成交量_0'] + df_merge['成交量_1']).replace(0, np.nan)) * 100).round(2).fillna(0.0)
    df_merge['法人3日集中度(%)'] = (((df_merge['外資_0'] + df_merge['外資_1'] + df_merge['外資_2'] + df_merge['投信_0'] + df_merge['投信_1'] + df_merge['投信_2']) / (df_merge['成交量_0'] + df_merge['成交量_1'] + df_merge['成交量_2']).replace(0, np.nan)) * 100).round(2).fillna(0.0)

    if not df_tdcc.empty:
        df_merge = pd.merge(df_merge, df_tdcc, on='股票代號', how='left')
        df_merge['千張大戶比例(%)'] = df_merge['千張大戶比例(%)'].fillna(0)
    else: df_merge['千張大戶比例(%)'] = 0

    if not df_holdings.empty:
        df_merge = pd.merge(df_merge, df_holdings, on='股票代號', how='left')
        df_merge['外資持股比例(%)'] = df_merge['外資持股比例(%)'].fillna(0)
        df_merge['發行總張數'] = df_merge['發行總張數'].fillna(1)
        def calc_foreign_buy_ratio(row):
            total_shares, ratio, monthly_buy = row['發行總張數'], row['外資持股比例(%)'] / 100.0, row['外資近一月(張)']
            foreign_total_holding = total_shares * ratio
            if foreign_total_holding <= 0 or monthly_buy <= 0: return 0.0
            return round((monthly_buy / foreign_total_holding) * 100, 2)
        df_merge['外資近月買超佔持股(%)'] = df_merge.apply(calc_foreign_buy_ratio, axis=1)
    else: df_merge['外資近月買超佔持股(%)'] = 0.0

    d0_s = f"{date_0[4:6]}/{date_0[6:]}"
    d1_s = f"{date_1[4:6]}/{date_1[6:]}"
    df_merge['⭐'] = df_merge['股票代號'].apply(lambda x: f"<span class='fav-star' data-code='{x}'>☆</span>")

    def calc_gap_days(row):
        try:
            l0, l1, l2 = float(row['最低價_0']), float(row['最低價_1']), float(row['最低價_2'])
            c1, c2, c3 = float(row['收盤價_1']), float(row['收盤價_2']), float(row.get('收盤價_3', 0))
            if (l0 >= c1) and pd.notna(l1) and pd.notna(c2) and (l1 >= c2) and pd.notna(l2) and pd.notna(c3) and (l2 >= c3): return 3
            if (l0 >= c1) and pd.notna(l1) and pd.notna(c2) and (l1 >= c2): return 2
            if (l0 >= c1): return 1
            return 0
        except Exception: return 0

    def calc_vol_days(row):
        try:
            v0, v1 = float(row.get('成交量_0', 0)), float(row.get('成交量_1', 0))
            v2, v3 = float(row.get('成交量_2', 0)), float(row.get('成交量_3', 0))
            if pd.isna(v0) or pd.isna(v1): return 0
            if (v0 > v1) and pd.notna(v2) and (v1 > v2) and pd.notna(v3) and (v2 > v3): return 3
            if (v0 > v1) and pd.notna(v2) and (v1 > v2): return 2
            if (v0 > v1): return 1
            return 0
        except Exception: return 0

    def calc_step_up_days(row):
        try:
            count = 0
            for i in range(5):
                l_curr, c_curr, c_prev = float(row.get(f'最低價_{i}', 0)), float(row.get(f'收盤價_{i}', 0)), float(row.get(f'收盤價_{i+1}', 0))
                if pd.notna(l_curr) and pd.notna(c_curr) and pd.notna(c_prev):
                    if c_curr > c_prev and l_curr >= c_prev: count += 1
                    else: break
                else: break
            return count
        except Exception: return 0

    def calc_foreign_buy_days(row):
        try:
            count = 0
            for i in range(7):
                f = float(row.get(f'外資_{i}', 0))
                if pd.notna(f) and f > 0: count += 1
                else: break
            return count
        except Exception: return 0

    def calc_consecutive_up_days(row):
        try:
            count = 0
            for i in range(5):
                c_curr, c_prev = float(row.get(f'收盤價_{i}', 0)), float(row.get(f'收盤價_{i+1}', 0))
                if pd.notna(c_curr) and pd.notna(c_prev) and c_curr > c_prev: count += 1
                else: break
            return count
        except Exception: return 0

    def calc_10d_total_return(row):
        try:
            c_curr, c_prev_10 = float(row.get('收盤價_0', 0)), float(row.get('收盤價_10', 0))
            if pd.isna(c_prev_10) or c_prev_10 == 0:
                for i in range(9, 0, -1):
                    fallback = float(row.get(f'收盤價_{i}', 0))
                    if pd.notna(fallback) and fallback > 0:
                        c_prev_10 = fallback
                        break
            if pd.notna(c_curr) and c_prev_10 > 0: return round(((c_curr - c_prev_10) / c_prev_10) * 100, 2)
            return 0.0
        except Exception: return 0.0

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
            try: counts = counts + condition_func(k).astype(int)
            except Exception: pass
        return counts.apply(lambda x: f"{x}次 / 7日")

    def apply_color_formatting(df_in):
        df_out = df_in.copy()
        for col_name in ["最新漲幅(%)", "前一日漲幅(%)", "近10日漲幅(%)"]:
            if col_name in df_out.columns:
                df_out[col_name] = df_out[col_name].apply(color_pct)
        return df_out

    # --- 策略條件函數 ---
    def cond1_fn(k): return (df_merge[f'成交量_{k+1}'] > 0) & (df_merge[f'最低價_{k}'] >= df_merge[f'收盤價_{k+1}']) & (df_merge[f'成交量_{k}'] > df_merge[f'成交量_{k+1}'])
    def cond2_fn(k): 
        step_up = (df_merge[f'收盤價_{k}'] > df_merge[f'收盤價_{k+1}']) & (df_merge[f'最低價_{k}'] >= df_merge[f'收盤價_{k+1}'])
        vol_up = df_merge[f'成交量_{k}'] > df_merge[f'成交量_{k+1}']
        inst_hold = True
        for j in range(7): inst_hold = inst_hold & (df_merge[f'外資_{k+j}'] >= 0) & (df_merge[f'投信_{k+j}'] >= 0)
        return step_up & vol_up & inst_hold
    def cond3_fn(k):
        vol_up = (df_merge[f'成交量_{k}'] > df_merge[f'成交量_{k+1}']) & (df_merge[f'成交量_{k+1}'] > 0)
        t_hold = True
        for j in range(7): t_hold = t_hold & (df_merge[f'投信_{k+j}'] >= 0)
        f_buy = df_merge[f'外資_{k}'] > 0
        return vol_up & t_hold & f_buy
    def cond4_fn(k):
        close_20 = [f'收盤價_{k+j}' for j in range(1, 21)]
        max_20 = df_merge[close_20].max(axis=1)
        c_high = df_merge[f'收盤價_{k}'] > max_20
        vol_up = (df_merge[f'成交量_{k}'] > df_merge[f'成交量_{k+1}']) & (df_merge[f'成交量_{k+1}'] > 0)
        t_hold = True
        for j in range(7): t_hold = t_hold & (df_merge[f'投信_{k+j}'] >= 0)
        return c_high & vol_up & t_hold
    def cond6_fn(k):
        close_5 = [f'收盤價_{k+j}' for j in range(1, 6)]
        max_5 = df_merge[close_5].max(axis=1)
        c_high = df_merge[f'收盤價_{k}'] > max_5
        vol_up = (df_merge[f'成交量_{k}'] > df_merge[f'成交量_{k+1}']) & (df_merge[f'成交量_{k+1}'] > 0)
        gap = (df_merge[f'開盤價_{k}'] > df_merge[f'收盤價_{k+1}']) & (df_merge[f'最低價_{k}'] > df_merge[f'收盤價_{k+1}'])
        hold_3 = (df_merge[f'外資_{k}'] >= 0) & (df_merge[f'投信_{k}'] >= 0) & (df_merge[f'外資_{k+1}'] >= 0) & (df_merge[f'投信_{k+1}'] >= 0) & (df_merge[f'外資_{k+2}'] >= 0) & (df_merge[f'投信_{k+2}'] >= 0)
        return c_high & vol_up & gap & hold_3
    def cond7_fn(k):
        close_20 = [f'收盤價_{k+j}' for j in range(1, 21)]
        max_20 = df_merge[close_20].max(axis=1)
        c_high = df_merge[f'收盤價_{k}'] > max_20
        vol_up = (df_merge[f'成交量_{k}'] > df_merge[f'成交量_{k+1}']) & (df_merge[f'成交量_{k+1}'] > 0)
        p_up = (df_merge[f'最低價_{k}'] > df_merge[f'最低價_{k+1}']) & (df_merge[f'收盤價_{k}'] > df_merge[f'收盤價_{k+1}'])
        hold_7 = True
        for j in range(7): hold_7 = hold_7 & (df_merge[f'外資_{k+j}'] >= 0) & (df_merge[f'投信_{k+j}'] >= 0)
        return c_high & vol_up & p_up & hold_7
    def cond11_fn(k):
        c_k = df_merge[f'收盤價_{k}']
        o_k = df_merge[f'開盤價_{k}'] if f'開盤價_{k}' in df_merge else df_merge['開盤價_0']
        return (df_merge['市場'] == '上櫃') & (c_k > o_k) & (o_k > 0)
    def cond13_fn(k):
        c_k = df_merge[f'收盤價_{k}']
        close_60 = [f'收盤價_{k+j}' for j in range(1, 61)]
        max_60 = df_merge[close_60].max(axis=1)
        return (c_k > 0) & (max_60 > 0) & (df_merge[f'成交量_{k}'] >= 500) & (c_k >= max_60 * 0.98)
    def cond15_fn(k):
        c_k = df_merge[f'收盤價_{k}']
        close_60 = [f'收盤價_{k+j}' for j in range(1, 61)]
        max_60 = df_merge[close_60].max(axis=1)
        vol_up = df_merge[f'成交量_{k}'] >= df_merge[f'成交量_{k+1}'] * 1.2
        return vol_up & (c_k >= max_60)
    def cond16_fn(k):
        mv_high = df_merge['最大量日最高價']
        c_k = df_merge[f'收盤價_{k}']
        c_prev = df_merge[f'收盤價_{k+1}']
        breakout = (mv_high > 0) & (c_k > mv_high) & (c_prev <= mv_high)
        hold_7 = True
        for j in range(7): hold_7 = hold_7 & (df_merge[f'外資_{k+j}'] >= 0) & (df_merge[f'投信_{k+j}'] >= 0)
        return breakout & hold_7
    def cond17_fn(k):
        close_120 = [f'收盤價_{k+j}' for j in range(1, 121) if f'收盤價_{k+j}' in df_merge]
        if not close_120: return pd.Series(False, index=df_merge.index)
        max_120 = df_merge[close_120].max(axis=1)
        return (df_merge[f'收盤價_{k}'] > 0) & (max_120 > 0) & (df_merge[f'收盤價_{k}'] > max_120)
    def cond20_fn(k):
        f_more = (df_merge[f'外資_{k}'] > df_merge[f'外資_{k+1}']) & (df_merge[f'外資_{k+1}'] > df_merge[f'外資_{k+2}']) & (df_merge[f'外資_{k+2}'] > 0)
        vol_up = df_merge[f'成交量_{k}'] >= df_merge[f'成交量_{k+1}'] * 1.2
        p_up = df_merge[f'收盤價_{k}'] > df_merge[f'收盤價_{k+1}']
        return f_more & vol_up & p_up
    def cond22_fn(k):
        close_5 = [f'收盤價_{k+j}' for j in range(1, 6)]
        max_5 = df_merge[close_5].max(axis=1)
        c_high5 = df_merge[f'收盤價_{k}'] > max_5
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
                if not valid: return 0.0
                idx = max(valid, key=lambda x: x[0])[1]
                return float(row.get(f'最高價_{idx}', 0.0))
            except Exception: return 0.0
        max_h = df_merge.apply(lambda r: _get_max_high_at(r, k), axis=1)
        return (max_h > 0) & (df_merge[f'收盤價_{k}'] > max_h)
    def cond25_fn(k):
        c_k, c_10ago = df_merge[f'收盤價_{k}'], df_merge[f'收盤價_{k+10}'] if f'收盤價_{k+10}' in df_merge else df_merge[f'收盤價_{k+9}']
        ret_10d = ((c_k - c_10ago) / c_10ago) * 100
        whale_locked = df_merge['千張大戶比例(%)'] >= 60.0
        ma5_k = df_merge[[f'收盤價_{k+j}' for j in range(5)]].mean(axis=1)
        ma20_k = df_merge[[f'收盤價_{k+j}' for j in range(20)]].mean(axis=1)
        bullish_ma = (c_k > ma5_k) & (ma5_k > ma20_k)
        vol_surge = df_merge[f'成交量_{k}'] >= df_merge[f'成交量_{k+1}'] * 1.5
        launch_zone = (ret_10d >= 3.0) & (ret_10d <= 25.0)
        inst_safe = (df_merge[f'外資_{k}'] >= 0) | (df_merge[f'投信_{k}'] >= 0)
        return whale_locked & bullish_ma & vol_surge & launch_zone & inst_safe
    def cond32_fn(k): return df_merge['千張大戶比例(%)'] >= 60.0
    def cond32_backtest_fn(k):
        whale = df_merge['千張大戶比例(%)'] >= 60.0
        ratio = (((df_merge[f'外資_{k}'] + df_merge[f'外資_{k+1}'] + df_merge[f'外資_{k+2}'] + df_merge[f'投信_{k}'] + df_merge[f'投信_{k+1}'] + df_merge[f'投信_{k+2}']) / (df_merge[f'成交量_{k}'] + df_merge[f'成交量_{k+1}'] + df_merge[f'成交量_{k+2}']).replace(0, np.nan)) * 100).fillna(0.0)
        return whale & (ratio >= 20.0)
    def cond33_fn(k):
        whale_70 = df_merge['千張大戶比例(%)'] > 70.0
        ma5_k = df_merge[[f'收盤價_{k+j}' for j in range(5)]].mean(axis=1)
        ma20_k = df_merge[[f'收盤價_{k+j}' for j in range(20)]].mean(axis=1)
        ma60_k = df_merge[[f'收盤價_{k+j}' for j in range(60)]].mean(axis=1)
        c_k = df_merge[f'收盤價_{k}']
        bullish_ma = (c_k > ma5_k) & (ma5_k > ma20_k) & (ma20_k > ma60_k)
        close_120_k = [f'收盤價_{k+j}' for j in range(1, 121) if f'收盤價_{k+j}' in df_merge.columns]
        if not close_120_k: return pd.Series(False, index=df_merge.index)
        max_120_k = df_merge[close_120_k].max(axis=1)
        near_breakout = (c_k >= max_120_k * 0.95) & (c_k <= max_120_k * 1.05) & (max_120_k > 0)
        return whale_70 & bullish_ma & near_breakout
    def cond_new_high_pullback(k):
        try:
            close_cols_180 = [f'收盤價_{k+j}' for j in range(1, 181) if f'收盤價_{k+j}' in df_merge.columns]
            if len(close_cols_180) < 60: return pd.Series(False, index=df_merge.index)
            max_180 = df_merge[close_cols_180].max(axis=1)
            close_cols_20 = [f'收盤價_{k+j}' for j in range(0, 20) if f'收盤價_{k+j}' in df_merge.columns]
            max_20 = df_merge[close_cols_20].max(axis=1)
            ever_hit_high = (max_20 >= max_180 * 0.99) & (max_180 > 0)
            low_cols_20 = [f'最低價_{k+j}' for j in range(0, 20) if f'最低價_{k+j}' in df_merge.columns]
            min_low_20 = df_merge[low_cols_20].min(axis=1)
            had_pullback = min_low_20 <= max_180 * 0.92
            c_k = df_merge[f'收盤價_{k}']
            near_high_now = (c_k >= max_180 * 0.95) & (c_k <= max_180 * 1.05)
            return ever_hit_high & had_pullback & near_high_now
        except Exception: return pd.Series(False, index=df_merge.index)
    def cond_super_bull_strategy(k):
        try:
            c_k, v_k, v_1 = df_merge[f'收盤價_{k}'], df_merge[f'成交量_{k}'], df_merge[f'成交量_{k+1}']
            whale_locked = df_merge['千張大戶比例(%)'] >= 60.0
            ma5_k = df_merge[[f'收盤價_{k+j}' for j in range(5)]].mean(axis=1)
            ma20_k = df_merge[[f'收盤價_{k+j}' for j in range(20)]].mean(axis=1)
            ma60_k = df_merge[[f'收盤價_{k+j}' for j in range(60)]].mean(axis=1)
            bullish_ma = (c_k > ma5_k) & (ma5_k > ma20_k) & (ma20_k > ma60_k)
            vol_surge = (v_k >= v_1 * 1.2) & (v_1 > 0)
            f_7d, t_7d = df_merge['外資近七日(張)'], df_merge['投信近七日(張)']
            inst_safe = (f_7d >= 0) | (t_7d >= 0)
            return whale_locked & bullish_ma & vol_surge & inst_safe
        except Exception: return pd.Series(False, index=df_merge.index)

    def build_res(cond_fn, sort_cols, asc_list, rename_dict=None, extra_cols=None):
        cond = cond_fn(0)
        res = df_merge[cond].copy()
        res['近7日符合次數'] = eval_rolling_condition(cond_fn)[cond]
        valid_sort_cols = [c for c in sort_cols if c in res.columns]
        valid_asc_list = [asc_list[i] for i, c in enumerate(sort_cols) if c in res.columns]
        if valid_sort_cols: res = res.sort_values(by=valid_sort_cols, ascending=valid_asc_list)
        if rename_dict: res = res.rename(columns=rename_dict)
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
    res20 = build_res(cond20_fn, ["外資_0", "最新漲幅(%)"], [False, False], std_rename)
    res22 = build_res(cond22_fn, ["最新漲幅(%)", "增量倍數"], [False, False], std_rename)
    res23 = build_res(cond23_fn, ["最新漲幅(%)", "增量倍數"], [False, False], std_rename)
    res25 = build_res(cond25_fn, ["千張大戶比例(%)", "最新漲幅(%)"], [False, False], std_rename)
    initial_res32 = build_res(cond32_fn, ["法人3日集中度(%)", "最新漲幅(%)"], [False, False], inst_rename)
    res33 = build_res(cond33_fn, ["千張大戶比例(%)", "最新漲幅(%)"], [False, False], std_rename)
    res34 = build_res(cond_new_high_pullback, ["最新漲幅(%)", "增量倍數"], [False, False], std_rename)
    res35 = build_res(cond_super_bull_strategy, ["千張大戶比例(%)", "最新漲幅(%)"], [False, False], std_rename)

    # === 打標籤與合併 ===
    if 'res1' in locals() and not res1.empty: res1['觸發型態'] = '跳空不補'
    if 'res2' in locals() and not res2.empty: res2['觸發型態'] = '階梯墊高'
    if 'res3' in locals() and not res3.empty: res3['觸發型態'] = '投信外資聯手'
    if 'res4' in locals() and not res4.empty: res4['觸發型態'] = '創20日高'
    if 'res6' in locals() and not res6.empty: res6['觸發型態'] = '創5日高'
    if 'res7' in locals() and not res7.empty: res7['觸發型態'] = '底底高'
    if 'res11' in locals() and not res11.empty: res11['觸發型態'] = '實體紅K'
    if 'res13' in locals() and not res13.empty: res13['觸發型態'] = '逼近60日高'
    if 'res15' in locals() and not res15.empty: res15['觸發型態'] = '壓縮突破60日'
    if 'res16' in locals() and not res16.empty: res16['觸發型態'] = '突破60日最大量'
    if 'res17' in locals() and not res17.empty: res17['觸發型態'] = '創120日高'
    if 'res20' in locals() and not res20.empty: res20['觸發型態'] = '外資連買'
    if 'res22' in locals() and not res22.empty: res22['觸發型態'] = '高勝率複合'
    if 'res23' in locals() and not res23.empty: res23['觸發型態'] = '突破10日最大量'
    if 'res25' in locals() and not res25.empty: res25['觸發型態'] = '倚強科模式'
    if 'initial_res32' in locals() and not initial_res32.empty: initial_res32['觸發型態'] = '大戶集中'
    if 'res33' in locals() and not res33.empty: res33['觸發型態'] = '絕對鎖碼'
    if 'res34' in locals() and not res34.empty: res34['觸發型態'] = '回檔再攻'
    if 'res35' in locals() and not res35.empty: res35['觸發型態'] = '主升段基因'

    def merge_strategies(df_list, sort_by="最新漲幅(%)"):
        valid_dfs = [df for df in df_list if isinstance(df, pd.DataFrame) and not df.empty]
        if not valid_dfs:
            return pd.DataFrame(), "<p style='text-align:center;'>目前無符合股票</p>"
        merged = pd.concat(valid_dfs, ignore_index=True)
        merged['觸發型態'] = merged.groupby('股票代號')['觸發型態'].transform(lambda x: ' ＋ '.join(x.unique()))
        merged = merged.drop_duplicates(subset=['股票代號']).reset_index(drop=True)
        if sort_by in merged.columns: merged = merged.sort_values(by=sort_by, ascending=False)
        display_cols = ["⭐", "市場", "標的", "觸發型態"] + [c for c in base_cols if c not in ["⭐", "市場", "標的"]] + ["增量倍數"] + chip_cols
        valid_cols = [c for c in display_cols if c in merged.columns]
        html = apply_color_formatting(merged[valid_cols]).to_html(index=False, classes="styled-table sortable-table", escape=False)
        return merged, html

    df_trend, html_trend = merge_strategies([res17, res13, res15, res4, res7])
    df_short, html_short = merge_strategies([res1, res2, res6, res11])
    df_vol, html_vol     = merge_strategies([res16, res23])
    df_chip, html_chip   = merge_strategies([res3, res20, initial_res32, res33])
    df_gene, html_gene   = merge_strategies([res22, res25, res34, res35])

    print("🔬 啟動策略 26：9 月任意兩策略組合高勝率回測挖掘...")
    cand_strategies = [
        ("1.跳空不補", cond1_fn), ("2.連續墊高", cond2_fn), ("3.量增法人買", cond3_fn),
        ("4.創20日新高", cond4_fn), ("6.創五日高", cond6_fn),
        ("7.20日高不賣", cond7_fn), ("11.上櫃紅K", cond11_fn), ("13.逼近60日高", cond13_fn),
        ("15.壓縮突破60日", cond15_fn), ("16.最大量高點", cond16_fn),
        ("17.創120日高", cond17_fn), ("20.外資越買越多出量", cond20_fn),
        ("22.高勝率基因複合", cond22_fn), ("23.突破10日最大量", cond23_fn),
        ("32.大戶鎖碼法人集中", cond32_backtest_fn), ("33.絕對鎖碼突破", cond33_fn),
        ("34.創新高回檔再攻", cond_new_high_pullback), ("35.飆股主升段基因", cond_super_bull_strategy)
    ]
    sept_indices = [i for i, (d, _) in enumerate(days_data) if "20260901" <= d <= "20260930"]
    if not sept_indices: sept_indices = list(range(5, min(25, len(days_data)-5)))
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
            try: mat[:, d_idx] = fn(k).fillna(False).values
            except Exception: pass
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
        if n_samples == 0: continue
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
    else: df_combos = pd.DataFrame(columns=["策略組合", "上漲率(%)", "平均漲幅(%)", "中位數漲幅(%)", "樣本數", "上漲檔數", "下跌檔數"])
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
            if not res_df.empty: table_html = apply_color_formatting(res_df[base_cols + ["增量倍數"] + chip_cols]).to_html(index=False, classes="styled-table sortable-table", escape=False)
            else: table_html = "<div style='padding: 30px; text-align: center; color: #64748b; font-size: 13px;'>今日最新盤後無同時符合此兩策略之股票</div>"
            strat_results_27_31[r_num] = {"name": f"{r_num}. 組合{rank_labels[idx]}", "combo_name": item["策略組合"], "df": res_df, "html": table_html, "count": len(res_df), "win_rate": item["上漲率(%)"], "avg_ret": item["平均漲幅(%)"], "samples": item["樣本數"]}
        else:
            strat_results_27_31[r_num] = {"name": f"{r_num}. 組合{rank_labels[idx]}", "combo_name": "無", "df": pd.DataFrame(), "html": "", "count": 0, "win_rate": 0.0}

    print("🔬 正在執行過去 30 天單策略隔日勝率回測計算...")
    backtest_records = []
    for st_name, fn in cand_strategies:
        total_picks, up_picks, return_list = 0, 0, []
        for k in range(1, 31):
            try:
                selected_mask = fn(k)
                if selected_mask is None or not selected_mask.any(): continue
                next_day_ret = ((df_merge.loc[selected_mask, f'收盤價_{k-1}'] - df_merge.loc[selected_mask, f'收盤價_{k}']) / df_merge.loc[selected_mask, f'收盤價_{k}']) * 100
                next_day_ret = next_day_ret.dropna()
                if len(next_day_ret) > 0:
                    total_picks += len(next_day_ret)
                    up_picks += (next_day_ret > 0).sum()
                    return_list.extend(next_day_ret.tolist())
            except Exception: continue
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
        ("6.五日高", res6), ("7.創20日高不賣", res7), ("11.上櫃實體紅K", res11), ("13.逼近60日高", res13),
        ("15.壓縮突破60日", res15), ("16.突破最大量高點", res16), ("17.創120日高", res17),
        ("20.外資越買越多出量", res20), ("22.高勝率基因複合", res22), ("23.突破10日最大量", res23),
        ("25.倚強科模式複製", res25), ("32.大戶鎖碼法人集中", initial_res32), ("33.絕對鎖碼突破", res33),
        ("34.創新高回檔再攻", res34), ("35.飆股主升段基因", res35)
    ]
    hit_counts, hit_names = {}, {}
    for s_name, res_df in st_lists:
        if res_df.empty or "股票代號" not in res_df.columns: continue
        for code in res_df['股票代號'].values:
            if code not in hit_counts:
                hit_counts[code], hit_names[code] = 0, []
            hit_counts[code] += 1
            hit_names[code].append(s_name)
            
    if hit_counts:
        df_hits = pd.DataFrame([{"股票代號": k, "入選次數": v, "符合策略": ", ".join(hit_names[k])} for k, v in hit_counts.items()])
        res12 = pd.merge(df_hits, df_merge, on="股票代號", how="inner")
        res12['近7日符合次數'] = res12['入選次數'].apply(lambda x: f"命中 {x} 個策略")
        res12 = res12.sort_values(by=["入選次數", "最新漲幅(%)"], ascending=[False, False]).head(50).rename(columns=std_rename)
        html_tb12 = apply_color_formatting(res12[base_cols + ["入選次數", "符合策略"] + chip_cols]).to_html(index=False, classes="styled-table sortable-table", escape=False)
    else:
        res12, html_tb12 = pd.DataFrame(), "<p style='text-align:center;'>目前無任何股票入選</p>"

    # === 全新精簡版 8 大選單 ===
    strategies_meta = [
        ("Tab_Strat12", "🌟 綜合強勢排行", "統計所有策略累計入選次數最多的前 50 檔最強勢標的排序。", len(res12), html_tb12),
        ("Tab_Trend", "🏆 長線波段與創高", "包含：創半年新高、逼近/壓縮突破60日高、創20日高、底底高等趨勢表態型態。", len(df_trend), html_trend),
        ("Tab_Short", "📈 短線強勢與跳空", "包含：跳空缺口不補、階梯式墊高、創5日短高、實體紅K等極短線強勢表態。", len(df_short), html_short),
        ("Tab_Vol", "🔥 關鍵量能突破", "包含：突破10日或60日最大量高點，成功化解套牢賣壓之發動股。", len(df_vol), html_vol),
        ("Tab_Chip", "👑 法人與大戶鎖碼", "包含：千張大戶絕對鎖碼(>70%)、法人買盤高度集中、外資/投信連續買超點火。", len(df_chip), html_chip),
        ("Tab_Gene", "🚀 飆股主升段基因", "包含：倚強科複製模式、創新高回檔再攻、均線多頭發散等高勝率飆股型態。", len(df_gene), html_gene),
        ("Tab_Backtest27", "🎖 最佳勝率組合", "歷史回測表現最優異的兩大策略組合（首選與次選）。", strat_results_27_31[27]["count"] + strat_results_27_31[28]["count"], strat_results_27_31[27]["html"] + "<br><h3>組合第二</h3>" + strat_results_27_31[28]["html"]),
        ("Tab_Backtest", "📊 策略回測實驗室", "9月份任兩策略組合回測績效與過去30天單一策略隔日勝率排行。", len(df_combos), html_tb26 + "<br><br><h3>過去30天單策略隔日勝率</h3>" + html_tb21)
    ]

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

    for tab_id, title, desc, count, tbl_html in strategies_meta:
        dashboard_cards_html += f"""
        <div class="dash-card" onclick="showPage('{tab_id}')">
            <div class="dash-card-title">{title}</div>
            <div class="dash-card-desc">{desc}</div>
            <div class="dash-card-badge">符合標的：<b>{count}</b> 檔 ➔</div>
        </div>
        """
        filter_toolbar = f"""
        <div class="filter-toolbar">
            <div class="filter-group"><label>市場別：</label>
                <select class="custom-select filter-market" onchange="applyFilters('{tab_id}')">
                    <option value="all" selected>全部</option><option value="上市">僅上市</option><option value="上櫃">僅上櫃</option>
                </select>
            </div>
            <div class="filter-group"><label>外資狀態：</label>
                <select class="custom-select filter-foreign" onchange="applyFilters('{tab_id}')">
                    <option value="all" selected>無限制</option><option value="ge0">近七日不賣 (>=0)</option><option value="ge100">大舉買超 (>100張)</option>
                </select>
            </div>
            <div class="filter-group"><label>投信狀態：</label>
                <select class="custom-select filter-trust" onchange="applyFilters('{tab_id}')">
                    <option value="all" selected>無限制</option><option value="ge0">近七日不賣 (>=0)</option><option value="ge50">積極認養 (>50張)</option>
                </select>
            </div>
            <div class="filter-group"><label>千張大戶：</label>
                <select class="custom-select filter-whale" onchange="applyFilters('{tab_id}')">
                    <option value="all" selected>無限制</option><option value="ge40">籌碼集中 (>40%)</option><option value="ge60">高度鎖碼 (>60%)</option><option value="ge70">絕對重壓 (>70%)</option>
                </select>
            </div>
        </div>
        """ if "Backtest" not in tab_id and "Strat12" not in tab_id else ""

        pages_html_blocks += f"""
        <div id="{tab_id}" class="page-view" style="display:none;">
            <div class="sub-header"><button class="back-btn" onclick="showPage('Dashboard')">⬅ 返回</button><h2>{title}</h2></div>
            <div class="info-box"><p>🎯 <b>說明：</b> {desc}</p><div class="count-badge">符合：<span id="count_{tab_id}">{count}</span> 檔</div></div>
            {filter_toolbar}
            <div class="table-container">{tbl_html}</div>
        </div>
        """

    html_content = f"""<!DOCTYPE html>
    <html lang="zh-TW">
    <head>
        <meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1.0, maximum-scale=1.0, user-scalable=no, viewport-fit=cover">
        <title>多策略台股選股中心</title>
        <style>
            :root {{--primary: #1e3a8a; --primary-light: #3b82f6; --bg: #f8fafc; --card-bg: #ffffff; --text: #0f172a; --text-muted: #64748b; --border: #e2e8f0; --up-red: #ef4444; --down-green: #22c55e; --star: #f59e0b;}}
            * {{ box-sizing: border-box; -webkit-tap-highlight-color: transparent; }}
            body {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Arial, sans-serif; margin: 0; background-color: var(--bg); color: var(--text); padding-bottom: 70px; }}
            .app-header {{ background: linear-gradient(135deg, var(--primary), var(--primary-light)); color: white; padding: 12px 16px; position: sticky; top: 0; z-index: 100; box-shadow: 0 2px 8px rgba(0,0,0,0.08); display: flex; justify-content: space-between; align-items: center; }}
            .app-header h1 {{ margin: 0; font-size: 16px; font-weight: 700; }}
            .app-header p {{ margin: 2px 0 0 0; font-size: 11px; opacity: 0.85; }}
            .fav-quick-btn {{ background: rgba(255,255,255,0.2); border: 1px solid rgba(255,255,255,0.4); color: white; padding: 6px 12px; border-radius: 20px; font-size: 12px; font-weight: 600; cursor: pointer; }}
            .container {{ padding: 10px; max-width: 1500px; margin: 0 auto; }}
            .dash-grid {{ display: grid; grid-template-columns: repeat(auto-fill, minmax(280px, 1fr)); gap: 10px; margin-top: 10px; }}
            .dash-card {{ background: var(--card-bg); border: 1px solid var(--border); border-radius: 10px; padding: 14px; cursor: pointer; box-shadow: 0 1px 3px rgba(0,0,0,0.04); transition: transform 0.1s ease, box-shadow 0.1s ease; }}
            .dash-card:active {{ transform: scale(0.98); }}
            .dash-card:hover {{ border-color: var(--primary-light); box-shadow: 0 4px 12px rgba(59, 130, 246, 0.15); }}
            .dash-card-title {{ font-size: 14px; font-weight: 700; color: var(--primary); margin-bottom: 4px; }}
            .dash-card-desc {{ font-size: 11px; color: var(--text-muted); line-height: 1.3; margin-bottom: 8px; height: 32px; overflow: hidden; }}
            .dash-card-badge {{ font-size: 12px; font-weight: 600; color: var(--up-red); display: flex; justify-content: space-between; align-items: center; }}
            .sub-header {{ display: flex; align-items: center; gap: 10px; margin-bottom: 10px; }}
            .back-btn {{ background: #e2e8f0; border: none; padding: 6px 12px; border-radius: 6px; font-size: 12px; font-weight: 600; cursor: pointer; color: var(--text); }}
            .sub-header h2 {{ margin: 0; font-size: 15px; color: var(--primary); }}
            .filter-toolbar {{ background: white; border: 1px solid var(--border); padding: 8px 12px; border-radius: 8px; margin-bottom: 8px; display: flex; flex-wrap: wrap; gap: 10px; align-items: center; font-size: 11px; }}
            .filter-group {{ display: flex; align-items: center; gap: 4px; }}
            .filter-group label {{ font-weight: 600; color: var(--text-muted); }}
            .custom-select {{ padding: 4px 8px; font-size: 11px; font-weight: 600; border-radius: 6px; border: 1px solid #cbd5e1; background-color: white; color: var(--text); outline: none; }}
            .add-stock-box {{ background: white; border: 1px solid var(--border); padding: 10px; border-radius: 8px; margin-bottom: 10px; display: flex; gap: 8px; }}
            .add-stock-input {{ flex: 1; padding: 6px 10px; font-size: 12px; border: 1px solid #cbd5e1; border-radius: 6px; outline: none; }}
            .add-stock-btn {{ background: var(--primary); color: white; border: none; padding: 6px 14px; border-radius: 6px; font-size: 12px; font-weight: 600; cursor: pointer; }}
            .warning-details {{ background-color: #fff1f2; border: 1px solid #fecdd3; border-radius: 8px; margin-bottom: 8px; font-size: 11px; color: #be123c; overflow: hidden; }}
            .warning-summary {{ padding: 7px 12px; cursor: pointer; font-weight: 600; display: flex; justify-content: space-between; align-items: center; user-select: none; list-style: none; }}
            .warning-summary::-webkit-details-marker {{ display: none; }}
            .toggle-arrow {{ font-size: 9px; transition: transform 0.2s; }}
            details[open] .toggle-arrow {{ transform: rotate(180deg); }}
            .warning-body {{ padding: 4px 14px 8px 14px; border-top: 1px dashed #fecdd3; color: #9f1239; max-height: 120px; overflow-y: auto; }}
            .warning-body ul {{ margin: 4px 0 0 16px; padding: 0; }}
            .warning-body li {{ margin-bottom: 2px; }}
            .info-box {{ background: var(--card-bg); border-radius: 8px; padding: 10px 12px; margin-bottom: 8px; border: 1px solid var(--border); box-shadow: 0 1px 3px rgba(0,0,0,0.04); }}
            .info-box p {{ margin: 0; font-size: 12px; line-height: 1.4; color: var(--text-muted); }}
            .count-badge {{ display: inline-flex; align-items: center; background: #fee2e2; color: var(--up-red); padding: 1px 6px; border-radius: 4px; font-weight: 700; font-size: 12px; margin-top: 6px; }}
            .table-container {{ background: var(--card-bg); border-radius: 8px; overflow: auto; max-height: 75vh; border: 1px solid var(--border); box-shadow: 0 2px 8px rgba(0,0,0,0.04); position: relative; -webkit-overflow-scrolling: touch; }}
            .styled-table {{ border-collapse: separate; border-spacing: 0; width: 100%; font-size: 12px; text-align: center; }}
            .styled-table th {{ background-color: #f8fafc; color: var(--text-muted); font-weight: 600; padding: 6px 4px; white-space: normal; word-break: keep-all; border-bottom: 2px solid var(--border); cursor: pointer; user-select: none; position: sticky; top: 0; z-index: 20; line-height: 1.25; }}
            .styled-table th::after {{ content: ' ↕'; opacity: 0.3; font-size: 9px; }}
            .styled-table th.th-sort-asc::after {{ content: ' ↑'; opacity: 1; color: var(--primary-light); }}
            .styled-table th.th-sort-desc::after {{ content: ' ↓'; opacity: 1; color: var(--primary-light); }}
            .styled-table td {{ padding: 6px 4px; border-bottom: 1px solid var(--border); white-space: nowrap; vertical-align: middle; background-color: white; line-height: 1.2; }}
            .styled-table tr:last-child td {{ border-bottom: none; }}
            .styled-table:not(.backtest-table) th:nth-child(1), .styled-table:not(.backtest-table) td:nth-child(1) {{ position: sticky; left: 0; z-index: 10; background-color: #f8fafc; min-width: 36px; max-width: 36px; }}
            .styled-table:not(.backtest-table) th:nth-child(3), .styled-table:not(.backtest-table) td:nth-child(3) {{ position: sticky; left: 36px; z-index: 10; background-color: #f8fafc; box-shadow: 2px 0 4px -1px rgba(0,0,0,0.08); min-width: 72px; max-width: 80px; }}
            .styled-table:not(.backtest-table) th:nth-child(1), .styled-table:not(.backtest-table) th:nth-child(3) {{ z-index: 30; background-color: #f1f5f9; }}
            .styled-table:not(.backtest-table) td:nth-child(1), .styled-table:not(.backtest-table) td:nth-child(3) {{ background-color: #ffffff; }}
            .backtest-table th, .backtest-table td {{ position: static !important; box-shadow: none !important; white-space: normal !important; padding: 8px 6px !important; }}
            .backtest-table th {{ position: sticky !important; top: 0 !important; z-index: 25 !important; background-color: #f1f5f9 !important; }}
            .styled-table tr.row-favorite td {{ background-color: #fefce8 !important; }}
            .stock-link {{ text-decoration: none; display: block; color: inherit; }}
            .stock-name {{ font-weight: 700; font-size: 13px; color: var(--text); line-height: 1.1; margin-bottom: 2px; }}
            .stock-code {{ font-size: 10px; color: var(--text-muted); line-height: 1; }}
            .stock-link:hover .stock-name {{ color: var(--primary-light); }}
            .tag-up {{ color: #dc2626; font-weight: 700; display: inline-block; }}
            .tag-down {{ color: #16a34a; font-weight: 700; display: inline-block; }}
            .tag-flat {{ color: #64748b; font-weight: 600; display: inline-block; }}
            .fav-star {{ cursor: pointer; color: #cbd5e1; font-size: 18px; padding: 4px; display: inline-block; transition: transform 0.15s ease; }}
            .fav-star:active {{ transform: scale(1.2); }}
            .fav-star.active {{ color: var(--star); font-weight: bold; }}
            #fav-table tbody tr {{ display: none; }}
            #fav-table tbody tr.row-favorite {{ display: table-row; }}
        </style>
    </head>
    <body>
        <div class="app-header">
            <div><h1>📈 多策略選股中心</h1><p>歷史區間: T-120日 ({date_120}) ➔ 最新日 ({date_0})</p></div>
            <button class="fav-quick-btn" onclick="showPage('Tab_Favorites')">⭐ 我的最愛 (<span id="fav-count">0</span>)</button>
        </div>
        <div class="container">
            {warning_html}
            <div id="Dashboard" class="page-view">
                <div class="info-box"><p>🎯 <b>行動端儀錶板：</b> 點擊下方任意卡片進入明細，並可使用外資、投信、大戶即時篩選器。</p></div>
                <div class="dash-grid">{dashboard_cards_html}</div>
            </div>
            <div id="Tab_Favorites" class="page-view" style="display:none;">
                <div class="sub-header"><button class="back-btn" onclick="showPage('Dashboard')">⬅ 返回儀錶板</button><h2>⭐ 我的最愛與觀察股</h2></div>
                <div class="add-stock-box">
                    <input type="text" id="customCodeInput" class="add-stock-input" placeholder="輸入 4 位數代號..." maxlength="4">
                    <button class="add-stock-btn" onclick="addCustomStock()">新增 +</button>
                </div>
                <div class="info-box"><div class="count-badge">💖 總計觀察：<span id="count_Tab_Favorites">0</span> 檔</div></div>
                <div id="fav-empty-msg" style="text-align:center; padding: 40px; color:#64748b; font-size:14px; display:none;">尚未加入最愛標的，請點擊清單中的 ⭐ 圖示新增。</div>
                <div class="table-container"><table id="fav-table" class="styled-table sortable-table"><tbody></tbody></table></div>
            </div>
            {pages_html_blocks}
        </div>
        <script>
            const FAV_KEY = 'tw_stock_favorites';
            function getFavorites() {{ return JSON.parse(localStorage.getItem(FAV_KEY) || '[]'); }}
            function saveFavorites(favs) {{ localStorage.setItem(FAV_KEY, JSON.stringify(favs)); }}
            function updateFavUI() {{
                const favs = getFavorites();
                const cnt = document.getElementById('count_Tab_Favorites'); if(cnt) cnt.innerText = favs.length;
                const quickCnt = document.getElementById('fav-count'); if(quickCnt) quickCnt.innerText = favs.length;
                document.querySelectorAll('.fav-star').forEach(star => {{
                    const code = star.dataset.code; const tr = star.closest('tr');
                    if (favs.includes(code)) {{ star.classList.add('active'); star.innerHTML = '★'; if(tr) tr.classList.add('row-favorite'); }}
                    else {{ star.classList.remove('active'); star.innerHTML = '☆'; if(tr) tr.classList.remove('row-favorite'); }}
                }});
                const favTbody = document.querySelector('#fav-table tbody');
                if(favTbody) {{
                    favTbody.innerHTML = '';
                    const allFavRows = document.querySelectorAll('.row-favorite');
                    const added = new Set();
                    allFavRows.forEach(r => {{
                        const code = r.querySelector('.fav-star')?.dataset.code;
                        if(code && !added.has(code) && r.closest('table').id !== 'fav-table') {{
                            favTbody.appendChild(r.cloneNode(true));
                            added.add(code);
                        }}
                    }});
                    const msg = document.getElementById('fav-empty-msg');
                    const tbl = document.getElementById('fav-table');
                    if(added.size === 0) {{ msg.style.display = 'block'; tbl.style.display = 'none'; }}
                    else {{ msg.style.display = 'none'; tbl.style.display = 'table'; }}
                }}
            }}
            function showPage(pageId) {{
                document.querySelectorAll('.page-view').forEach(el => el.style.display = 'none');
                const target = document.getElementById(pageId);
                if(target) {{ target.style.display = 'block'; window.scrollTo(0, 0); }}
            }}
            function addCustomStock() {{
                const input = document.getElementById('customCodeInput'); const code = input.value.trim();
                if(!code || code.length !== 4 || isNaN(code)) {{ alert('請輸入有效的代號！'); return; }}
                let favs = getFavorites();
                if(!favs.includes(code)) {{ favs.push(code); saveFavorites(favs); updateFavUI(); alert(`成功新增 ${{code}}`); }}
                else alert('該股票已在清單中！');
                input.value = '';
            }}
            function applyFilters(pageId) {{
                const page = document.getElementById(pageId); if(!page) return;
                const table = page.querySelector('table'); if(!table) return;
                const rows = table.querySelectorAll('tbody tr');
                const headers = Array.from(table.querySelectorAll('thead th'));
                const mktVal = page.querySelector('.filter-market')?.value || 'all';
                const forVal = page.querySelector('.filter-foreign')?.value || 'all';
                const truVal = page.querySelector('.filter-trust')?.value || 'all';
                const whaVal = page.querySelector('.filter-whale')?.value || 'all';
                let idxMkt = -1, idxFor = -1, idxTru = -1, idxWha = -1;
                headers.forEach((th, idx) => {{
                    let txt = th.innerText;
                    if(txt.includes('市場')) idxMkt = idx;
                    if(txt.includes('外資近七日') || txt.includes('外資')) idxFor = idx;
                    if(txt.includes('投信近七日') || txt.includes('投信')) idxTru = idx;
                    if(txt.includes('千張大戶')) idxWha = idx;
                }});
                let visibleCount = 0;
                rows.forEach(row => {{
                    let pass = true; let cells = row.querySelectorAll('td');
                    if(cells.length < 5) return;
                    if(mktVal !== 'all' && idxMkt >= 0) if(!cells[idxMkt].innerText.includes(mktVal)) pass = false;
                    if(forVal !== 'all' && idxFor >= 0) {{
                        let v = parseFloat(cells[idxFor].innerText.replace(/,/g, '').replace(/%/g, ''));
                        if(!isNaN(v)) {{ if(forVal === 'ge0' && v < 0) pass = false; if(forVal === 'ge100' && v < 100) pass = false; }}
                    }}
                    if(truVal !== 'all' && idxTru >= 0) {{
                        let v = parseFloat(cells[idxTru].innerText.replace(/,/g, '').replace(/%/g, ''));
                        if(!isNaN(v)) {{ if(truVal === 'ge0' && v < 0) pass = false; if(truVal === 'ge50' && v < 50) pass = false; }}
                    }}
                    if(whaVal !== 'all' && idxWha >= 0) {{
                        let v = parseFloat(cells[idxWha].innerText.replace(/,/g, '').replace(/%/g, ''));
                        if(!isNaN(v)) {{ if(whaVal === 'ge40' && v < 40) pass = false; if(whaVal === 'ge60' && v < 60) pass = false; if(whaVal === 'ge70' && v < 70) pass = false; }}
                    }}
                    row.style.display = pass ? '' : 'none';
                    if(pass) visibleCount++;
                }});
                const badge = page.querySelector('.count-badge span'); if(badge) badge.innerText = visibleCount;
            }}
            document.addEventListener('click', function(e) {{
                if (e.target && e.target.classList.contains('fav-star')) {{
                    e.preventDefault(); e.stopPropagation();
                    const star = e.target; const code = star.dataset.code;
                    let favs = getFavorites(); const isAdding = !star.classList.contains('active');
                    if (isAdding) {{ if(!favs.includes(code)) favs.push(code); }}
                    else favs = favs.filter(c => c !== code);
                    saveFavorites(favs); updateFavUI();
                }}
            }});
            function sortTbody(tbody, colIndex = -1, isAscending = false) {{
                const rows = Array.from(tbody.querySelectorAll('tr'));
                rows.sort((rowA, rowB) => {{
                    const favA = rowA.querySelector('.fav-star')?.classList.contains('active') ? 1 : 0;
                    const favB = rowB.querySelector('.fav-star')?.classList.contains('active') ? 1 : 0;
                    if (favA !== favB) return favB - favA;
                    if (colIndex >= 0) {{
                        let cellAStr = rowA.querySelectorAll('td')[colIndex].innerText.trim();
                        let cellBStr = rowB.querySelectorAll('td')[colIndex].innerText.trim();
                        let numA = parseFloat(cellAStr.replace(/,/g, '').replace(/%/g, '').replace(/\\+/g, '').split('/')[0]);
                        let numB = parseFloat(cellBStr.replace(/,/g, '').replace(/%/g, '').replace(/\\+/g, '').split('/')[0]);
                        const mul = isAscending ? 1 : -1;
                        if (!isNaN(numA) && !isNaN(numB)) return (numA - numB) * mul;
                        return cellAStr.localeCompare(cellBStr, 'zh-Hant') * mul;
                    }}
                    return 0;
                }});
                rows.forEach(row => tbody.appendChild(row));
            }}
            function enableTableSorting() {{
                document.querySelectorAll('.styled-table th').forEach(header => {{
                    if (header.cellIndex === 0 && header.innerText.includes("⭐")) return;
                    header.addEventListener('click', function() {{
                        const table = this.closest('table'); const tbody = table.querySelector('tbody');
                        const isAsc = this.classList.contains('th-sort-asc'); const colIdx = this.cellIndex;
                        table.querySelectorAll('th').forEach(th => th.classList.remove('th-sort-asc', 'th-sort-desc'));
                        if (isAsc) {{ this.classList.add('th-sort-desc'); sortTbody(tbody, colIdx, false); }}
                        else {{ this.classList.add('th-sort-asc'); sortTbody(tbody, colIdx, true); }}
                    }});
                }});
            }}
            window.addEventListener('DOMContentLoaded', () => {{
                updateFavUI(); enableTableSorting();
                document.querySelectorAll('.styled-table tbody').forEach(tbody => sortTbody(tbody));
            }});
        </script>
    </body>
    </html>
    """
    html_filename = "index.html"
    file_path = os.path.abspath(html_filename)
    with open(file_path, "w", encoding="utf-8") as f: f.write(html_content)
    print(f"\n✅ 網頁已生成: {html_filename}")

    # ==========================================
    # 執行盤中即時檢查 (收集所有策略中的股票去重)
    # ==========================================
    target_dfs = []
    strat_names = ['res17',  'res22',  'res35']
    for df_name in strat_names:
        if df_name in locals() and isinstance(locals()[df_name], pd.DataFrame) and not locals()[df_name].empty:
            target_dfs.append(locals()[df_name])
            
    if target_dfs:
        combined_df = pd.concat(target_dfs, ignore_index=True)
        unique_target_df = combined_df.drop_duplicates(subset=['股票代號']).reset_index(drop=True)
        print(f"\n🎯 總計擷取 {len(unique_target_df)} 檔不重複的強勢標的，送入即時引擎監控...")
        intraday_check_strategy(unique_target_df)
    else:
        print("\nℹ 目前無任何策略資料可供盤中監控。")

    if os.environ.get("GITHUB_ACTIONS") != "true": webbrowser.open(f"file:///{file_path}")

if __name__ == "__main__":
    main()
