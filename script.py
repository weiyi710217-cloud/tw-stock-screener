import datetime
import time
import random
import os
import webbrowser
import itertools
import numpy as np
import pandas as pd
import requests

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
    for attempt in range(max_retries):
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
    for attempt in range(2):
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
            "市場": "上櫃",
            "股票代號": df[c_code].astype(str).str.strip(),
            "股票名稱": df[c_name].astype(str).str.strip(),
            "開盤價": df[c_open], "最高價": df[c_high], "最低價": df[c_low],
            "收盤價": df[c_close], "成交量": df[c_vol],
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
    f_col = (next((c for c in df.columns if '外陸資' in c and '買賣超' in c), None)
             or next((c for c in df.columns if '外資' in c and '買賣超' in c), None))
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
                rows.append({
                    "股票代號": code,
                    "外資買賣超": str(item[10]).strip(),
                    "投信買賣超": str(item[13]).strip()
                })
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
    else:
        return pd.DataFrame()

def get_tdcc_data() -> pd.DataFrame:
    today_str = datetime.date.today().strftime("%Y%m%d")
    cache_file = os.path.join(CACHE_DIR, f"tdcc_{today_str}.csv")

    df_cached = safe_read_csv(cache_file, dtype={"股票代號": str})
    if not df_cached.empty:
        return df_cached

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
                    if not clean_line:
                        continue
                    cols = clean_line.split(',')
                    if len(cols) >= 6:
                        code = cols[1].strip()
                        if cols[2] == '15' and len(code) == 4:
                            ratio = float(cols[5]) if cols[5].replace('.', '', 1).isdigit() else 0.0
                            rows.append({
                                "股票代號": code,
                                "千張大戶比例(%)": ratio
                            })
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
        url = (f"https://www.twse.com.tw/rwd/zh/fund/MI_QFIIS"
               f"?date={d.strftime('%Y%m%d')}&selectType=ALLBUT0999&response=json")
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
                    rows.append({"股票代號": code,
                                 "外資持股比例(%)": to_float(r[c_ratio]),
                                 "發行總張數": to_float(r[c_issue]) / 1000})
        if rows:
            break

    data = safe_request_json("https://www.tpex.org.tw/openapi/v1/tpex_3insti_qfii",
                             max_retries=2, headers=OPENAPI_HEADERS)
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
        k_ratio = (find_key("hold", "ratio") or find_key("hold", "percent")
                   or find_key("持股", "比", exclude=("尚可",)))
        if k_code and k_issue and k_ratio:
            for item in data:
                code = str(item.get(k_code, "")).strip()
                if len(code) == 4 and code.isdigit():
                    rows.append({"股票代號": code,
                                 "外資持股比例(%)": to_float(item.get(k_ratio)),
                                 "發行總張數": to_float(item.get(k_issue)) / 1000})

    df_holdings = pd.DataFrame(rows)
    if not df_holdings.empty and (df_holdings["外資持股比例(%)"] > 0).any():
        df_holdings = df_holdings.drop_duplicates(subset=["股票代號"])
        df_holdings.to_csv(cache_file, index=False, encoding="utf-8-sig")
        return df_holdings

    return pd.DataFrame(columns=["股票代號", "外資持股比例(%)", "發行總張數"])

def fetch_month_revenue(year_roc: int, month: int) -> pd.DataFrame:
    cache_file = os.path.join(CACHE_DIR, f"rev_{year_roc}_{month:02d}.csv")
    df_cached = safe_read_csv(cache_file, dtype={"股票代號": str})
    if not df_cached.empty:
        return df_cached

    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
    }
    dfs = []
    for skey in [0, 1]:
        url = f"https://mops.twse.com.tw/nas/t21/skey{skey}/t21sc03_{year_roc}_{month}_0.html"
        try:
            resp = requests.get(url, headers=headers, timeout=5)
            if resp.status_code == 200:
                resp.encoding = 'big5'
                tables = pd.read_html(resp.text)
                for t in tables:
                    if t.shape[1] >= 11 and "公司代號" in str(t.values):
                        for r_idx in range(len(t)):
                            row_vals = [str(x).strip() for x in t.iloc[r_idx].values]
                            code = row_vals[0]
                            if code.isdigit() and len(code) == 4:
                                rev = to_float(row_vals[2])
                                dfs.append({"股票代號": code, f"營收_{year_roc}_{month:02d}": rev})
        except Exception:
            pass

    if dfs:
        df_res = pd.DataFrame(dfs).drop_duplicates(subset=["股票代號"])
        df_res.to_csv(cache_file, index=False, encoding="utf-8-sig")
        return df_res
    return pd.DataFrame(columns=["股票代號", f"營收_{year_roc}_{month:02d}"])

def get_revenue_analysis() -> pd.DataFrame:
    today = datetime.date.today()
    cur_year = today.year - 1911
    cur_month = today.month

    months = []
    start_offset = 1 if today.day >= 12 else 2
    for i in range(start_offset, start_offset + 4):
        m = cur_month - i
        y = cur_year
        while m <= 0:
            m += 12
            y -= 1
        months.append((y, m))

    print(f"📊 檢查近 4 個月營收區間: {[f'{y}/{m:02d}' for y, m in months]}...")
    df_merged = None
    rev_cols = []
    for y, m in months:
        df_m = fetch_month_revenue(y, m)
        col_name = f"營收_{y}_{m:02d}"
        rev_cols.append(col_name)
        if df_merged is None:
            df_merged = df_m
        else:
            if not df_m.empty:
                df_merged = pd.merge(df_merged, df_m, on="股票代號", how="outer")

    if df_merged is None or len(rev_cols) < 4:
        return pd.DataFrame(columns=["股票代號", "營收連3月月增", "最新營收月增率(%)", "營收爆發1.5倍"])

    rev_cols_sorted = list(reversed(rev_cols))
    m3, m2, m1, m0 = rev_cols_sorted[0], rev_cols_sorted[1], rev_cols_sorted[2], rev_cols_sorted[3]

    for c in [m3, m2, m1, m0]:
        if c not in df_merged.columns:
            df_merged[c] = 0.0
        df_merged[c] = pd.to_numeric(df_merged[c], errors="coerce").fillna(0)

    cond_growth_3m = (
        (df_merged[m0] > df_merged[m1]) &
        (df_merged[m1] > df_merged[m2]) &
        (df_merged[m2] > df_merged[m3]) &
        (df_merged[m3] > 0)
    )

    df_merged['最新營收月增率(%)'] = (((df_merged[m0] - df_merged[m1]) / df_merged[m1].replace(0, float('nan'))) * 100).round(2).fillna(0.0)
    cond_surge_1_5x = (df_merged[m1] > 0) & (df_merged[m0] >= df_merged[m1] * 1.5)

    df_merged['營收連3月月增'] = cond_growth_3m
    df_merged['營收爆發1.5倍'] = cond_surge_1_5x

    return df_merged[["股票代號", "營收連3月月增", "最新營收月增率(%)", "營收爆發1.5倍"]]

def get_last_n_trading_days_data(n_market=155, n_inst=60):
    valid_dfs = []
    valid_inst = []
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
                df_i = get_inst_data(date_str)
                valid_inst.append((date_str, df_i))
            else:
                valid_inst.append((date_str, pd.DataFrame()))
        else:
            time.sleep(0.1)
    return valid_dfs, valid_inst

def format_stock_cell(row):
    code = row['股票代號']
    name = row['股票名稱']
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
    print("啟動多策略選股程式，準備抓取價量、法人與大戶資料...\n")

    days_data, inst_data = get_last_n_trading_days_data(n_market=155, n_inst=60)

    date_0, df_0 = days_data[0]
    date_1, df_1 = days_data[1]
    date_2, df_2 = days_data[2]
    date_120, _ = days_data[min(120, len(days_data)-1)]

    print(f"\n✅ 行情獲取完畢！ 基準日: [{date_0}]")

    df_tdcc = get_tdcc_data()
    df_holdings = get_foreign_holdings()
    df_revenue = get_revenue_analysis()

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

        cols_to_get = ["股票代號", "開盤價", "最高價", "最低價", "收盤價", "成交量"]
        df_temp = df_i[cols_to_get].copy()
        rename_dict = {
            "收盤價": f"收盤價_{i}", "成交量": f"成交量_{i}",
            "最高價": f"最高價_{i}", "最低價": f"最低價_{i}",
            "開盤價": f"開盤價_{i}"
        }
        df_temp = df_temp.rename(columns=rename_dict)
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
            vols = []
            for i in range(60):
                v = float(row.get(f'成交量_{i}', -1))
                if pd.notna(v): vols.append((v, i))
            if not vols: return 0.0
            max_vol_idx = max(vols, key=lambda item: item[0])[1]
            return float(row.get(f'最高價_{max_vol_idx}', 0.0))
        except Exception:
            return 0.0
            
    df_merge['最大量日最高價'] = df_merge.apply(get_max_vol_high, axis=1)

    def get_10d_max_vol_high(row):
        try:
            vols = []
            for i in range(1, 11):
                v = float(row.get(f'成交量_{i}', -1))
                if pd.notna(v): vols.append((v, i))
            if not vols: return 0.0
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

    df_merge = df_merge.copy() 

    df_merge['外資近三日(張)'] = df_merge[[f'外資_{i}' for i in range(3)]].sum(axis=1)
    df_merge['投信近三日(張)'] = df_merge[[f'投信_{i}' for i in range(3)]].sum(axis=1)
    df_merge['外資近七日(張)'] = df_merge[[f'外資_{i}' for i in range(7)]].sum(axis=1)
    df_merge['投信近七日(張)'] = df_merge[[f'投信_{i}' for i in range(7)]].sum(axis=1)
    df_merge['外資近一月(張)'] = df_merge[[f'外資_{i}' for i in range(20)]].sum(axis=1)

    # 計算技術評分
    def calculate_technical_score(row):
        score = 0
        try:
            c = float(row.get('收盤價_0', 0))
            ma20 = float(row.get('20MA', 0))
            ma60 = float(row.get('60MA', 0))
            if c > ma20 and ma20 > ma60: score += 25
            elif c > ma20: score += 12

            closes = [float(row.get(f'收盤價_{j}', 0)) for j in range(20)]
            if len(closes) == 20 and ma20 > 0:
                std20 = np.std(closes)
                lower_band = ma20 - 2 * std20
                if lower_band > 0:
                    dist_lower = (c - lower_band) / lower_band
                    if dist_lower < 0.03: score += 25
                    elif c < ma20: score += 10

            lows_9 = [float(row.get(f'最低價_{j}', 0)) for j in range(9)]
            highs_9 = [float(row.get(f'最高價_{j}', 0)) for j in range(9)]
            if min(lows_9) > 0 and max(highs_9) > min(lows_9):
                rsv = (c - min(lows_9)) / (max(highs_9) - min(lows_9)) * 100
                k_val = rsv * 0.33 + 50 * 0.67
                d_val = k_val * 0.33 + 50 * 0.67
                if k_val > d_val and k_val < 80: score += 25
                elif k_val > d_val: score += 10

            ema12 = float(row.get('5MA', c))
            ema26 = float(row.get('20MA', c))
            dif = ema12 - ema26
            dea = dif * 0.8
            macd_hist = (dif - dea) * 2
            if macd_hist > 0 and dif > dea: score += 25
            elif macd_hist > 0: score += 10
        except Exception:
            pass
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
    else:
        df_merge['千張大戶比例(%)'] = 0

    if not df_holdings.empty:
        df_merge = pd.merge(df_merge, df_holdings, on='股票代號', how='left')
        df_merge['外資持股比例(%)'] = df_merge['外資持股比例(%)'].fillna(0)
        df_merge['發行總張數'] = df_merge['發行總張數'].fillna(1)

        def calc_foreign_buy_ratio(row):
            total_shares = row['發行總張數']
            ratio = row['外資持股比例(%)'] / 100.0
            foreign_total_holding = total_shares * ratio
            monthly_buy = row['外資近一月(張)']

            if foreign_total_holding <= 0 or monthly_buy <= 0:
                return 0.0
            return round((monthly_buy / foreign_total_holding) * 100, 2)

        df_merge['外資近月買超佔持股(%)'] = df_merge.apply(calc_foreign_buy_ratio, axis=1)
    else:
        df_merge['外資近月買超佔持股(%)'] = 0.0

    if not df_revenue.empty:
        df_merge = pd.merge(df_merge, df_revenue, on='股票代號', how='left')
        df_merge['營收連3月月增'] = df_merge['營收連3月月增'].fillna(False)
        df_merge['最新營收月增率(%)'] = df_merge['最新營收月增率(%)'].fillna(0.0)
        df_merge['營收爆發1.5倍'] = df_merge['營收爆發1.5倍'].fillna(False)
    else:
        df_merge['營收連3月月增'] = False
        df_merge['最新營收月增率(%)'] = 0.0
        df_merge['營收爆發1.5倍'] = False

    d0_s = f"{date_0[4:6]}/{date_0[6:]}"
    d1_s = f"{date_1[4:6]}/{date_1[6:]}"
    d2_s = f"{date_2[4:6]}/{date_2[6:]}"

    df_merge['⭐'] = df_merge['股票代號'].apply(lambda x: f"<span class='fav-star' data-code='{x}'>☆</span>")

    def calc_gap_days(row):
        try:
            l0, l1, l2 = float(row['最低價_0']), float(row['最低價_1']), float(row['最低價_2'])
            c1, c2, c3 = float(row['收盤價_1']), float(row['收盤價_2']), float(row.get('收盤價_3', 0))
            d1 = (l0 >= c1)
            d2 = d1 and pd.notna(l1) and pd.notna(c2) and (l1 >= c2)
            d3 = d2 and pd.notna(l2) and pd.notna(c3) and (l2 >= c3)
            if d3: return 3
            if d2: return 2
            if d1: return 1
            return 0
        except Exception: return 0

    def calc_vol_days(row):
        try:
            v0 = float(row.get('成交量_0', 0))
            v1 = float(row.get('成交量_1', 0))
            v2 = float(row.get('成交量_2', 0))
            v3 = float(row.get('成交量_3', 0))
            if pd.isna(v0) or pd.isna(v1): return 0
            d1 = (v0 > v1)
            d2 = d1 and pd.notna(v2) and (v1 > v2)
            d3 = d2 and pd.notna(v3) and (v2 > v3)
            if d3: return 3
            if d2: return 2
            if d1: return 1
            return 0
        except Exception: return 0

    def calc_step_up_days(row):
        try:
            count = 0
            for i in range(5):
                l_curr = float(row.get(f'最低價_{i}', 0))
                c_curr = float(row.get(f'收盤價_{i}', 0))
                c_prev = float(row.get(f'收盤價_{i+1}', 0))
                if pd.notna(l_curr) and pd.notna(c_curr) and pd.notna(c_prev):
                    if c_curr > c_prev and l_curr >= c_prev:
                        count += 1
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
                c_curr = float(row.get(f'收盤價_{i}', 0))
                c_prev = float(row.get(f'收盤價_{i+1}', 0))
                if pd.notna(c_curr) and pd.notna(c_prev) and c_curr > c_prev:
                    count += 1
                else: break
            return count
        except Exception: return 0

    def calc_5d_red_days(row):
        try:
            count = 0
            for i in range(5):
                c_curr = float(row.get(f'收盤價_{i}', 0))
                c_prev = float(row.get(f'收盤價_{i+1}', 0))
                if pd.notna(c_curr) and pd.notna(c_prev) and c_prev > 0:
                    if c_curr >= c_prev:
                        count += 1
            return f"{count}/5"
        except Exception: 
            return "0/5"
            
    def calc_10d_red_days(row):
        try:
            count = 0
            for i in range(10):
                c_curr = float(row.get(f'收盤價_{i}', 0))
                c_prev = float(row.get(f'收盤價_{i+1}', 0))
                if pd.notna(c_curr) and pd.notna(c_prev) and c_prev > 0:
                    if c_curr >= c_prev:
                        count += 1
            return f"{count}/10"
        except Exception: 
            return "0/10"

    def calc_10d_total_return(row):
        try:
            c_curr = float(row.get('收盤價_0', 0))
            c_prev_10 = float(row.get('收盤價_10', 0))
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
    df_merge['近5日紅盤'] = df_merge.apply(calc_5d_red_days, axis=1)
    df_merge['近10日紅盤'] = df_merge.apply(calc_10d_red_days, axis=1)
    df_merge['近10日漲幅(%)'] = df_merge.apply(calc_10d_total_return, axis=1)
    df_merge['創60日最大量'] = df_merge['成交量_0'] >= df_merge['60日最大量']

    df_merge['標的'] = df_merge.apply(format_stock_cell, axis=1)

    base_cols = ["⭐", "市場", "標的", "技術評分(分)", "近7日符合次數", "近5日紅盤", "近10日紅盤", "近10日漲幅(%)", f"{d1_s} 收盤", f"{d0_s} 收盤", "最新漲幅(%)", "前一日漲幅(%)", f"{d0_s} 量(張)"]
    chip_cols = ["外資近七日(張)", "投信近七日(張)", "外資近一月(張)", "外資近月買超佔持股(%)", "千張大戶比例(%)"]

    def eval_rolling_condition(condition_func):
        counts = pd.Series(0, index=df_merge.index)
        for k in range(7):
            try:
                c = condition_func(k)
                counts = counts + c.astype(int)
            except Exception:
                pass
        return counts.apply(lambda x: f"{x}次 / 7日")

    def apply_color_formatting(df_in):
        df_out = df_in.copy()
        for col_name in ["最新漲幅(%)", "前一日漲幅(%)", "近10日漲幅(%)", "最新營收月增率(%)"]:
            if col_name in df_out.columns:
                df_out[col_name] = df_out[col_name].apply(color_pct)
        return df_out

    def cond1_fn(k):
        l_k, c_prev = df_merge[f'最低價_{k}'], df_merge[f'收盤價_{k+1}']
        v_k, v_prev = df_merge[f'成交量_{k}'], df_merge[f'成交量_{k+1}']
        return (v_prev > 0) & (l_k >= c_prev) & (v_k > v_prev)

    def cond2_fn(k):
        step_up = (df_merge[f'收盤價_{k}'] > df_merge[f'收盤價_{k+1}']) & (df_merge[f'最低價_{k}'] >= df_merge[f'收盤價_{k+1}'])
        vol_up = df_merge[f'成交量_{k}'] > df_merge[f'成交量_{k+1}']
        inst_hold = True
        for j in range(7):
            inst_hold = inst_hold & (df_merge[f'外資_{k+j}'] >= 0) & (df_merge[f'投信_{k+j}'] >= 0)
        return step_up & vol_up & inst_hold

    def cond3_fn(k):
        vol_up = (df_merge[f'成交量_{k}'] > df_merge[f'成交量_{k+1}']) & (df_merge[f'成交量_{k+1}'] > 0)
        t_hold = True
        for j in range(7):
            t_hold = t_hold & (df_merge[f'投信_{k+j}'] >= 0)
        f_buy = df_merge[f'外資_{k}'] > 0
        return vol_up & t_hold & f_buy

    def cond4_fn(k):
        close_20 = [f'收盤價_{k+j}' for j in range(1, 21)]
        max_20 = df_merge[close_20].max(axis=1)
        c_high = df_merge[f'收盤價_{k}'] > max_20
        vol_up = (df_merge[f'成交量_{k}'] > df_merge[f'成交量_{k+1}']) & (df_merge[f'成交量_{k+1}'] > 0)
        t_hold = True
        for j in range(7):
            t_hold = t_hold & (df_merge[f'投信_{k+j}'] >= 0)
        return c_high & vol_up & t_hold

    def cond5_fn(k):
        vol_up = (df_merge[f'成交量_{k}'] > df_merge[f'成交量_{k+1}']) & (df_merge[f'成交量_{k+1}'] > 0)
        price_up = (df_merge[f'最低價_{k}'] > df_merge[f'最低價_{k+1}']) & (df_merge[f'收盤價_{k}'] > df_merge[f'收盤價_{k+1}'])
        zero_prev = True
        for j in range(1, 7):
            zero_prev = zero_prev & (df_merge[f'外資_{k+j}'] == 0) & (df_merge[f'投信_{k+j}'] == 0)
        inst_today = (df_merge[f'外資_{k}'] > 0) | (df_merge[f'投信_{k}'] > 0)
        return vol_up & price_up & zero_prev & inst_today

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
        for j in range(7):
            hold_7 = hold_7 & (df_merge[f'外資_{k+j}'] >= 0) & (df_merge[f'投信_{k+j}'] >= 0)
        return c_high & vol_up & p_up & hold_7

    def cond8_fn(k):
        c_up2 = (df_merge[f'收盤價_{k}'] > df_merge[f'收盤價_{k+1}']) & (df_merge[f'收盤價_{k+1}'] > df_merge[f'收盤價_{k+2}'])
        vol_up = (df_merge[f'成交量_{k}'] > df_merge[f'成交量_{k+1}']) & (df_merge[f'成交量_{k+1}'] > 0)
        hold_7 = True
        for j in range(7):
            hold_7 = hold_7 & (df_merge[f'外資_{k+j}'] >= 0) & (df_merge[f'投信_{k+j}'] >= 0)
        return c_up2 & vol_up & hold_7

    def cond9_fn(k):
        c_valid = df_merge[f'收盤價_{k+1}'] > 0
        hold_7 = True
        for j in range(7):
            hold_7 = hold_7 & (df_merge[f'外資_{k+j}'] >= 0) & (df_merge[f'投信_{k+j}'] >= 0)
        return c_valid & hold_7

    def cond10_fn(k):
        ret_k = ((df_merge[f'收盤價_{k}'] - df_merge[f'收盤價_{k+1}']) / df_merge[f'收盤價_{k+1}']) * 100
        return (df_merge['市場'] == '上櫃') & (ret_k > 0) & (df_merge[f'收盤價_{k+1}'] > 0)

    def cond11_fn(k):
        c_k = df_merge[f'收盤價_{k}']
        o_k = df_merge[f'開盤價_{k}'] if f'開盤價_{k}' in df_merge else df_merge['開盤價_0']
        return (df_merge['市場'] == '上櫃') & (c_k > o_k) & (o_k > 0)

    def cond13_fn(k):
        c_k = df_merge[f'收盤價_{k}']
        close_60 = [f'收盤價_{k+j}' for j in range(1, 61)]
        max_60 = df_merge[close_60].max(axis=1)
        return (c_k > 0) & (max_60 > 0) & (df_merge[f'成交量_{k}'] >= 500) & (c_k >= max_60 * 0.98)

    def cond14_fn(k):
        close_20 = [f'收盤價_{k+j}' for j in range(1, 21)]
        max_20 = df_merge[close_20].max(axis=1)
        c_high = df_merge[f'收盤價_{k}'] > max_20
        vol_up = (df_merge[f'成交量_{k}'] > df_merge[f'成交量_{k+1}']) & (df_merge[f'成交量_{k+1}'] > 0)
        hold_7 = True
        for j in range(7):
            hold_7 = hold_7 & (df_merge[f'外資_{k+j}'] >= 0) & (df_merge[f'投信_{k+j}'] >= 0)
        return c_high & vol_up & hold_7

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
        for j in range(7):
            hold_7 = hold_7 & (df_merge[f'外資_{k+j}'] >= 0) & (df_merge[f'投信_{k+j}'] >= 0)
        return breakout & hold_7

    def cond17_fn(k):
        close_120 = [f'收盤價_{k+j}' for j in range(1, 121) if f'收盤價_{k+j}' in df_merge]
        if not close_120: return pd.Series(False, index=df_merge.index)
        max_120 = df_merge[close_120].max(axis=1)
        return (df_merge[f'收盤價_{k}'] > 0) & (max_120 > 0) & (df_merge[f'收盤價_{k}'] > max_120)

    def cond18_fn(k):
        rev_ok = df_merge['營收連3月月增']
        vol_first = (df_merge[f'成交量_{k}'] >= df_merge[f'成交量_{k+1}'] * 1.2) & (df_merge[f'成交量_{k+1}'] <= df_merge[f'成交量_{k+2}'])
        p_up = df_merge[f'收盤價_{k}'] > df_merge[f'收盤價_{k+1}']
        return rev_ok & vol_first & p_up

    def cond19_fn(k):
        rev_1_5x = df_merge['營收爆發1.5倍']
        vol_first = (df_merge[f'成交量_{k}'] >= df_merge[f'成交量_{k+1}'] * 1.2) & (df_merge[f'成交量_{k+1}'] <= df_merge[f'成交量_{k+2}'])
        return rev_1_5x & vol_first

    def cond20_fn(k):
        f_more = (df_merge[f'外資_{k}'] > df_merge[f'外資_{k+1}']) & \
                 (df_merge[f'外資_{k+1}'] > df_merge[f'外資_{k+2}']) & \
                 (df_merge[f'外資_{k+2}'] > 0)
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
        hold_3 = (df_merge[f'外資_{k}'] >= 0) & (df_merge[f'投信_{k}'] >= 0) & \
                 (df_merge[f'外資_{k+1}'] >= 0) & (df_merge[f'投信_{k+1}'] >= 0) & \
                 (df_merge[f'外資_{k+2}'] >= 0) & (df_merge[f'投信_{k+2}'] >= 0)
        return c_high5 & gap_up & step_up & vol_up & hold_3

    def cond23_fn(k):
        def _get_max_high_at(row, offset):
            try:
                vols = [(float(row.get(f'成交量_{offset+j}', -1)), offset+j) for j in range(1, 11)]
                valid = [v for v in vols if pd.notna(v[0])]
                if not valid: return 0.0
                idx = max(valid, key=lambda x: x[0])[1]
                return float(row.get(f'最高價_{idx}', 0.0))
            except Exception:
                return 0.0
        max_h = df_merge.apply(lambda r: _get_max_high_at(r, k), axis=1)
        return (max_h > 0) & (df_merge[f'收盤價_{k}'] > max_h)

    def cond24_fn(k):
        c_k = df_merge[f'收盤價_{k}']
        c_10ago = df_merge[f'收盤價_{k+10}'] if f'收盤價_{k+10}' in df_merge else df_merge[f'收盤價_{k+9}']
        ret_10d = ((c_k - c_10ago) / c_10ago) * 100
        sweet_zone = (ret_10d >= 5.0) & (ret_10d <= 25.0)
        red_days = 0
        for j in range(5):
            c_curr = df_merge[f'收盤價_{k+j}']
            c_prev = df_merge[f'收盤價_{k+j+1}']
            red_days = red_days + (c_curr >= c_prev).astype(int)
        strong_red = red_days >= 4
        vol_up = df_merge[f'成交量_{k}'] >= df_merge[f'成交量_{k+1}'] * 1.2
        whale_hold = (df_merge['千張大戶比例(%)'] >= 35.0) | (df_merge[f'外資_{k}'] > 0)
        t_safe = (df_merge[f'投信_{k}'] >= 0) & (df_merge[f'投信_{k+1}'] >= 0) & (df_merge[f'投信_{k+2}'] >= 0)
        return sweet_zone & strong_red & vol_up & whale_hold & t_safe

    def cond25_fn(k):
        c_k = df_merge[f'收盤價_{k}']
        c_10ago = df_merge[f'收盤價_{k+10}'] if f'收盤價_{k+10}' in df_merge else df_merge[f'收盤價_{k+9}']
        ret_10d = ((c_k - c_10ago) / c_10ago) * 100
        whale_locked = df_merge['千張大戶比例(%)'] >= 60.0
        ma5_k = df_merge[[f'收盤價_{k+j}' for j in range(5)]].mean(axis=1)
        ma20_k = df_merge[[f'收盤價_{k+j}' for j in range(20)]].mean(axis=1)
        bullish_ma = (c_k > ma5_k) & (ma5_k > ma20_k)
        vol_surge = df_merge[f'成交量_{k}'] >= df_merge[f'成交量_{k+1}'] * 1.5
        launch_zone = (ret_10d >= 3.0) & (ret_10d <= 25.0)
        inst_safe = (df_merge[f'外資_{k}'] >= 0) | (df_merge[f'投信_{k}'] >= 0)
        return whale_locked & bullish_ma & vol_surge & launch_zone & inst_safe

    def cond32_fn(k):
        return df_merge['千張大戶比例(%)'] >= 60.0
        
    def cond32_backtest_fn(k):
        whale = df_merge['千張大戶比例(%)'] >= 60.0
        ratio = (((df_merge[f'外資_{k}'] + df_merge[f'外資_{k+1}'] + df_merge[f'外資_{k+2}'] + df_merge[f'投信_{k}'] + df_merge[f'投信_{k+1}'] + df_merge[f'投信_{k+2}']) / (df_merge[f'成交量_{k}'] + df_merge[f'成交量_{k+1}'] + df_merge[f'成交量_{k+2}']).replace(0, np.nan)) * 100).fillna(0.0)
        return whale & (ratio >= 20.0)

    # 策略 33：大戶>70% + 多頭排列 + 距離半年高點 -5% ~ +5% 以內 (即將或剛突破)
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

    def build_res(cond_fn, sort_cols, asc_list, rename_dict=None, extra_cols=None):
        cond = cond_fn(0)
        res = df_merge[cond].copy()
        res['近7日符合次數'] = eval_rolling_condition(cond_fn)[cond]
        valid_sort_cols = [c for c in sort_cols if c in res.columns]
        valid_asc_list = [asc_list[i] for i, c in enumerate(sort_cols) if c in res.columns]
        if valid_sort_cols:
            res = res.sort_values(by=valid_sort_cols, ascending=valid_asc_list)
        if rename_dict:
            res = res.rename(columns=rename_dict)
        cols = base_cols + (extra_cols if extra_cols is not None else ["增量倍數"]) + chip_cols
        valid_cols = [c for c in cols if c in res.columns]
        html = apply_color_formatting(res[valid_cols]).to_html(index=False, classes="styled-table sortable-table", escape=False)
        return res, html

    std_rename = {"收盤價_1": f"{d1_s} 收盤", "收盤價_0": f"{d0_s} 收盤", "成交量_0": f"{d0_s} 量(張)"}
    inst_rename = {**std_rename, "外資_0": "外資(T)", "外資_1": "外資(T-1)", "外資_2": "外資(T-2)", "投信_0": "投信(T)", "投信_1": "投信(T-1)", "投信_2": "投信(T-2)"}

    res_fav = df_merge.copy()
    res_fav = res_fav.sort_values(by="最新漲幅(%)", ascending=False)
    res_fav = res_fav.rename(columns=std_rename)
    cols_fav = base_cols + chip_cols
    html_tb_fav = apply_color_formatting(res_fav[cols_fav]).to_html(index=False, classes="styled-table sortable-table", escape=False, table_id="fav-table")

    res1, html_tb1 = build_res(cond1_fn, ["跳空天數", "量增天數", "增量倍數"], [False, False, False], std_rename, ["跳空天數", "量增天數", "增量倍數"])
    res2, html_tb2 = build_res(cond2_fn, ["墊高天數", "外資連買天數", "增量倍數"], [False, False, False], std_rename, ["墊高天數", "量增天數", "外資連買天數", "增量倍數"])
    res3, html_tb3 = build_res(cond3_fn, ["外資連買天數", "增量倍數"], [False, False], {**std_rename, "外資_0": "最新日外資(張)"}, ["最新日外資(張)", "外資連買天數", "連漲天數", "增量倍數"])
    res4, html_tb4 = build_res(cond4_fn, ["投信近七日(張)", "增量倍數"], [False, False], {**std_rename, "外資_0": "最新日外資(張)"}, ["最新日外資(張)", "增量倍數"])
    res5, html_tb5 = build_res(cond5_fn, ["最新法人買超(張)", "增量倍數"], [False, False], {"收盤價_0": f"{d0_s} 收盤", "成交量_0": f"{d0_s} 量(張)", "收盤價_1": f"{d1_s} 收盤"}, ["最新法人買超(張)", "增量倍數"])
    res6, html_tb6 = build_res(cond6_fn, ["投信近三日(張)", "增量倍數"], [False, False], std_rename)
    res7, html_tb7 = build_res(cond7_fn, ["投信近七日(張)", "增量倍數"], [False, False], std_rename)
    res8, html_tb8 = build_res(cond8_fn, ["投信近七日(張)", "增量倍數"], [False, False], {"收盤價_2": f"{d2_s} 收盤", "收盤價_1": f"{d1_s} 收盤", "收盤價_0": f"{d0_s} 收盤", "成交量_0": f"{d0_s} 量(張)"})
    res9, html_tb9 = build_res(cond9_fn, ["最新漲幅(%)"], [False], std_rename, ["創5日高", "創20日高"])
    res10, html_tb10 = build_res(cond10_fn, ["最新漲幅(%)"], [False], std_rename, [])
    res11, html_tb11 = build_res(cond11_fn, ["實體K漲幅(%)"], [False], {"開盤價_0": f"{d0_s} 開盤", "收盤價_1": f"{d1_s} 收盤", "收盤價_0": f"{d0_s} 收盤", "成交量_0": f"{d0_s} 量(張)"}, [f"{d0_s} 開盤", "實體K漲幅(%)"])
    res13, html_tb13 = build_res(cond13_fn, ["距離60日高點(%)", "最新漲幅(%)"], [False, False], std_rename, ["60日最高收盤", "距離60日高點(%)"])
    res14, html_tb14 = build_res(cond14_fn, ["最新漲幅(%)", "增量倍數"], [False, False], {**std_rename, "外資_0": "最新日外資(張)"}, ["最新日外資(張)", "外資連買天數", "連漲天數", "增量倍數"])
    res15, html_tb15 = build_res(cond15_fn, ["60日震幅(%)", "增量倍數"], [True, False], std_rename, ["20日震幅(%)", "40日震幅(%)", "60日震幅(%)", "增量倍數"])
    res15_strict = res15[(res15['創60日最大量']) & (res15['60日震幅(%)'] <= 10.0)]
    res16, html_tb16 = build_res(cond16_fn, ["最新漲幅(%)", "增量倍數"], [False, False], std_rename, ["最大量日最高價", "增量倍數"])
    res17, html_tb17 = build_res(cond17_fn, ["最新漲幅(%)", "增量倍數"], [False, False], std_rename, ["120日最高收盤", "增量倍數"])
    res18, html_tb18 = build_res(cond18_fn, ["最新營收月增率(%)", "最新漲幅(%)"], [False, False], std_rename, ["最新營收月增率(%)", "增量倍數"])
    res19, html_tb19 = build_res(cond19_fn, ["最新營收月增率(%)", "最新漲幅(%)"], [False, False], std_rename, ["最新營收月增率(%)", "增量倍數"])
    res20, html_tb20 = build_res(cond20_fn, ["外資_0", "最新漲幅(%)"], [False, False], {**std_rename, "外資_0": "最新日外資(張)", "外資_1": "前1日外資(張)", "外資_2": "前2日外資(張)"}, ["最新日外資(張)", "前1日外資(張)", "前2日外資(張)", "增量倍數"])
    res22, html_tb22 = build_res(cond22_fn, ["最新漲幅(%)", "增量倍數"], [False, False], std_rename)
    res23, html_tb23 = build_res(cond23_fn, ["最新漲幅(%)", "增量倍數"], [False, False], std_rename, ["近10日最大量日最高價", "增量倍數"])
    res24, html_tb24 = build_res(cond24_fn, ["最新漲幅(%)", "增量倍數"], [False, False], std_rename, ["千張大戶比例(%)", "增量倍數"])
    res25, html_tb25 = build_res(cond25_fn, ["千張大戶比例(%)", "最新漲幅(%)"], [False, False], std_rename, ["千張大戶比例(%)", "增量倍數"])
    res32, html_tb32 = build_res(cond32_fn, ["法人3日集中度(%)", "最新漲幅(%)"], [False, False], inst_rename, ["千張大戶比例(%)", "最新營收月增率(%)", "法人1日集中度(%)", "法人2日集中度(%)", "法人3日集中度(%)", "外資(T)", "外資(T-1)", "外資(T-2)", "投信(T)", "投信(T-1)", "投信(T-2)"])
    
    # 策略 33 輸出表 (顯示大戶比例、最新營收月增率)
    res33, html_tb33 = build_res(cond33_fn, ["千張大戶比例(%)", "最新漲幅(%)"], [False, False], std_rename, ["千張大戶比例(%)", "最新營收月增率(%)", "120日最高收盤"])

    # 策略 26: 9 月三策略組合挖掘回測
    print("🔬 啟動策略 26：9 月三策略組合高勝率回測...")
    cand_strategies = [
        ("1.跳空不補", cond1_fn), ("2.連續墊高", cond2_fn), ("3.量增法人買", cond3_fn),
        ("4.創20日新高", cond4_fn), ("5.旱地拔蔥", cond5_fn), ("6.創五日高", cond6_fn),
        ("7.20日高不賣", cond7_fn), ("8.連日墊高", cond8_fn), ("9.多重濾網", cond9_fn),
        ("10.上櫃強勢", cond10_fn), ("11.上櫃紅K", cond11_fn), ("13.逼近60日高", cond13_fn),
        ("14.創20日高不賣", cond14_fn), ("15.壓縮突破60日", cond15_fn), ("16.最大量高點", cond16_fn),
        ("17.創120日高", cond17_fn), ("20.外資越買越多出量", cond20_fn),
        ("22.高勝率基因複合", cond22_fn), ("23.突破10日最大量", cond23_fn),
        ("32.大戶鎖碼法人集中", cond32_backtest_fn), ("33.絕對鎖碼突破", cond33_fn)
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

    N = len(df_merge)
    D = len(sept_indices)
    strat_matrices = {}

    for st_name, fn in cand_strategies:
        mat = np.zeros((N, D), dtype=bool)
        for d_idx, k in enumerate(sept_indices):
            try:
                s = fn(k).fillna(False).values
                mat[:, d_idx] = s
            except Exception:
                pass
        strat_matrices[st_name] = mat

    open_1001 = pd.to_numeric(df_merge[col_open_1001], errors='coerce').values
    close_1008 = pd.to_numeric(df_merge[col_close_1008], errors='coerce').values
    valid_ret_mask = (open_1001 > 0) & (close_1008 > 0) & np.isfinite(open_1001) & np.isfinite(close_1008)
    returns_arr = np.zeros(N, dtype=float)
    returns_arr[valid_ret_mask] = ((close_1008[valid_ret_mask] - open_1001[valid_ret_mask]) / open_1001[valid_ret_mask]) * 100

    combo_triplets = list(itertools.combinations(cand_strategies, 3))
    combo_records = []

    for (name_a, fn_a), (name_b, fn_b), (name_c, fn_c) in combo_triplets:
        hit_mask = (strat_matrices[name_a] & strat_matrices[name_b] & strat_matrices[name_c]).any(axis=1)
        sub_mask = hit_mask & valid_ret_mask
        n_samples = int(np.sum(sub_mask))
        if n_samples == 0:
            continue

        sub_ret = returns_arr[sub_mask]
        up_count = int(np.sum(sub_ret > 0))
        win_rate = round((up_count / n_samples) * 100, 2)
        mean_ret = round(float(np.mean(sub_ret)), 2)
        median_ret = round(float(np.median(sub_ret)), 2)

        combo_records.append({
            "name_a": name_a, "name_b": name_b, "name_c": name_c,
            "fn_a": fn_a, "fn_b": fn_b, "fn_c": fn_c,
            "策略組合": f"{name_a} ＋ {name_b} ＋ {name_c}",
            "上漲率(%)": win_rate,
            "平均漲幅(%)": mean_ret,
            "中位數漲幅(%)": median_ret,
            "樣本數": n_samples,
            "上漲檔數": up_count,
            "下跌檔數": n_samples - up_count
        })

    df_combos = pd.DataFrame(combo_records)
    if not df_combos.empty:
        df_combos = df_combos.sort_values(
            by=["上漲率(%)", "平均漲幅(%)", "中位數漲幅(%)", "樣本數"],
            ascending=[False, False, False, False]
        ).reset_index(drop=True)
    else:
        df_combos = pd.DataFrame(columns=["策略組合", "上漲率(%)", "平均漲幅(%)", "中位數漲幅(%)", "樣本數", "上漲檔數", "下跌檔數"])

    df_combos["排名"] = range(1, len(df_combos) + 1)
    df_combos_show = df_combos[["排名", "策略組合", "上漲率(%)", "平均漲幅(%)", "中位數漲幅(%)", "樣本數", "上漲檔數", "下跌檔數"]].copy()
    df_combos_show["上漲率(%)"] = df_combos_show["上漲率(%)"].apply(lambda x: f"<b style='color:#dc2626;'>{x:.2f}%</b>" if x >= 50 else f"<span style='color:#16a34a;'>{x:.2f}%</span>")
    df_combos_show["平均漲幅(%)"] = df_combos_show["平均漲幅(%)"].apply(color_pct)
    df_combos_show["中位數漲幅(%)"] = df_combos_show["中位數漲幅(%)"].apply(color_pct)
    html_tb26 = df_combos_show.to_html(index=False, classes="styled-table backtest-table sortable-table", escape=False)

    # 建立策略 27~31
    top5_combos = df_combos.head(5).to_dict("records")
    strat_results_27_31 = {}
    rank_labels = ["首選", "第二", "第三", "第四", "第五"]

    for idx, r_num in enumerate([27, 28, 29, 30, 31]):
        if idx < len(top5_combos):
            item = top5_combos[idx]
            c_name = item["策略組合"]
            f_a, f_b, f_c = item["fn_a"], item["fn_b"], item["fn_c"]
            cond_today = f_a(0) & f_b(0) & f_c(0)
            res_df = df_merge[cond_today].copy()
            res_df['近7日符合次數'] = eval_rolling_condition(lambda k: f_a(k) & f_b(k) & f_c(k))[cond_today]
            res_df["增量倍數"] = (res_df["成交量_0"] / res_df["成交量_1"].replace(0, np.nan)).round(2).fillna(0.0)
            res_df = res_df.sort_values(by=["最新漲幅(%)", "增量倍數"], ascending=[False, False])
            res_df = res_df.rename(columns=std_rename)

            if not res_df.empty:
                table_html = apply_color_formatting(res_df[base_cols + ["增量倍數"] + chip_cols]).to_html(index=False, classes="styled-table sortable-table", escape=False)
            else:
                table_html = "<div style='padding: 30px; text-align: center; color: #64748b; font-size: 13px;'>今日最新盤後無同時符合此三策略之股票</div>"

            strat_results_27_31[r_num] = {
                "name": f"{r_num}. 組合{rank_labels[idx]}",
                "combo_name": c_name,
                "df": res_df,
                "html": table_html,
                "count": len(res_df),
                "win_rate": item["上漲率(%)"],
                "avg_ret": item["平均漲幅(%)"],
                "samples": item["樣本數"]
            }
        else:
            strat_results_27_31[r_num] = {
                "name": f"{r_num}. 組合{rank_labels[idx]}",
                "combo_name": "無足夠樣本",
                "df": pd.DataFrame(),
                "html": "<p style='text-align:center;'>目前無符合股票</p>",
                "count": 0,
                "win_rate": 0.0,
                "avg_ret": 0.0,
                "samples": 0
            }

    # 策略 21：單策略回測
    print("🔬 正在執行過去 30 天單策略隔日勝率回測計算...")
    backtest_funcs = [
        ("1.跳空不補", cond1_fn), ("2.連續墊高", cond2_fn), ("3.量增法人買", cond3_fn),
        ("4.創20日新高", cond4_fn), ("5.旱地拔蔥", cond5_fn), ("6.創五日高", cond6_fn),
        ("7.20日高不賣", cond7_fn), ("8.連日墊高", cond8_fn), ("9.多重濾網", cond9_fn),
        ("10.上櫃強勢", cond10_fn), ("11.上櫃紅K", cond11_fn), ("13.逼近60日高", cond13_fn),
        ("14.創20日高不賣", cond14_fn), ("15.壓縮突破60日", cond15_fn), ("16.最大量高點", cond16_fn),
        ("17.創120日高", cond17_fn), ("18.營收連三增啟動", cond18_fn), ("19.營收暴增1.5倍", cond19_fn),
        ("20.外資越買越多出量", cond20_fn), ("22.高勝率基因複合", cond22_fn), ("23.突破10日最大量", cond23_fn),
        ("24.飆股基因起漲", cond24_fn), ("25.倚強科模式複製", cond25_fn), ("32.大戶鎖碼法人集中", cond32_backtest_fn),
        ("33.絕對鎖碼突破", cond33_fn)
    ]

    backtest_records = []
    for st_name, fn in backtest_funcs:
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
            except Exception:
                continue

        win_rate = round((up_picks / total_picks * 100), 2) if total_picks > 0 else 0.0
        avg_ret = round((sum(return_list) / len(return_list)), 2) if return_list else 0.0
        backtest_records.append({
            "選股策略名稱": st_name,
            "隔天上漲率(%)": win_rate,
            "隔日平均報酬(%)": avg_ret,
            "總選中樣本數": total_picks,
            "隔天上漲次數": up_picks,
            "隔天下跌次數": total_picks - up_picks
        })

    df_strat21 = pd.DataFrame(backtest_records).sort_values(by=["隔天上漲率(%)", "隔日平均報酬(%)"], ascending=[False, False])
    df_strat21_show = df_strat21.copy()
    df_strat21_show["隔天上漲率(%)"] = df_strat21_show["隔天上漲率(%)"].apply(lambda x: f"<b style='color:#dc2626;'>{x:.2f}%</b>" if x >= 50 else f"<span style='color:#16a34a;'>{x:.2f}%</span>")
    df_strat21_show["隔日平均報酬(%)"] = df_strat21_show["隔日平均報酬(%)"].apply(color_pct)
    html_tb21 = df_strat21_show.to_html(index=False, classes="styled-table backtest-table sortable-table", escape=False)

    # 策略 12: 綜合排行
    st_lists = [
        ("1.跳空", res1), ("2.墊高", res2), ("3.量增法人", res3),
        ("4.創20日高", res4), ("5.拔蔥", res5), ("6.五日高", res6),
        ("7.創20日高不賣", res7), ("8.連兩日創高", res8), ("9.多重", res9),
        ("10.上櫃強勢", res10), ("11.上櫃實體紅K", res11), ("13.逼近60日高", res13),
        ("14.創20日高+法人七日不賣", res14), ("15.壓縮突破60日", res15_strict),
        ("16.突破最大量高點", res16), ("17.創120日高", res17),
        ("18.營收連三增啟動", res18), ("19.營收暴增1.5倍", res19),
        ("20.外資越買越多出量", res20), ("22.高勝率基因複合", res22),
        ("23.突破10日最大量", res23), ("24.飆股基因起漲", res24),
        ("25.倚強科模式複製", res25), ("32.大戶鎖碼法人集中", res32),
        ("33.絕對鎖碼突破", res33),
        ("27.組合首選", strat_results_27_31[27]["df"]),
        ("28.組合第二", strat_results_27_31[28]["df"]),
        ("29.組合第三", strat_results_27_31[29]["df"]),
        ("30.組合第四", strat_results_27_31[30]["df"]),
        ("31.組合第五", strat_results_27_31[31]["df"]),
    ]
    
    hit_counts, hit_names = {}, {}
    for s_name, res_df in st_lists:
        if res_df.empty or "股票代號" not in res_df.columns: continue
        for code in res_df['股票代號'].values:
            if code not in hit_counts:
                hit_counts[code] = 0
                hit_names[code] = []
            hit_counts[code] += 1
            hit_names[code].append(s_name)
            
    if hit_counts:
        df_hits = pd.DataFrame([{"股票代號": k, "入選次數": v, "符合策略": ", ".join(hit_names[k])} for k, v in hit_counts.items()])
        res12 = pd.merge(df_hits, df_merge, on="股票代號", how="inner")
        res12['近7日符合次數'] = res12['入選次數'].apply(lambda x: f"命中 {x} 個策略")
        res12 = res12.sort_values(by=["入選次數", "最新漲幅(%)"], ascending=[False, False]).head(50)
        res12 = res12.rename(columns=std_rename)
        cols12 = base_cols + ["入選次數", "符合策略"] + chip_cols
        html_tb12 = apply_color_formatting(res12[cols12]).to_html(index=False, classes="styled-table sortable-table", escape=False)
    else:
        html_tb12 = "<p style='text-align:center;'>目前無任何股票入選預設策略</p>"

    warning_html = ""
    if WARNINGS:
        unique_warns = sorted(set(WARNINGS))
        warn_count = len(unique_warns)
        warn_items_html = "".join([f"<li>{w}</li>" for w in unique_warns])
        warning_html = f"""
        <details class='warning-details'>
            <summary class='warning-summary'>
                <span>⚠ 系統提示：共有 <b>{warn_count}</b> 個交易日資料缺少（點擊展開/收合）</span>
                <span class='toggle-arrow'>▼</span>
            </summary>
            <div class='warning-body'>
                <p>下列日期可能為假日休市或連線超時：</p>
                <ul>{warn_items_html}</ul>
            </div>
        </details>
        """

    html_content = f"""
    <!DOCTYPE html>
    <html lang="zh-TW">
    <head>
        <meta charset="utf-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0, maximum-scale=1.0, user-scalable=no, viewport-fit=cover">
        <title>多策略台股選股中心</title>
        <style>
            :root {{
                --primary: #1e3a8a;
                --primary-light: #3b82f6;
                --bg: #f8fafc;
                --card-bg: #ffffff;
                --text: #0f172a;
                --text-muted: #64748b;
                --border: #e2e8f0;
                --up-red: #ef4444;
                --down-green: #22c55e;
                --star: #f59e0b;
            }}

            * {{ box-sizing: border-box; -webkit-tap-highlight-color: transparent; }}
            
            body {{
                font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif;
                margin: 0;
                background-color: var(--bg);
                color: var(--text);
                padding-bottom: 50px;
            }}

            .app-header {{
                background: linear-gradient(135deg, var(--primary), var(--primary-light));
                color: white;
                padding: 10px 16px;
                position: sticky;
                top: 0;
                z-index: 100;
                box-shadow: 0 2px 8px rgba(0,0,0,0.08);
            }}

            .app-header h1 {{
                margin: 0;
                font-size: 16px;
                font-weight: 700;
                letter-spacing: 0.5px;
                display: flex;
                align-items: center;
                gap: 6px;
            }}

            .app-header p {{
                margin: 2px 0 0 0;
                font-size: 11px;
                opacity: 0.85;
            }}

            .tabs-wrapper {{
                background: white;
                border-bottom: 1px solid var(--border);
                position: sticky;
                top: 48px;
                z-index: 90;
                padding: 6px 10px;
                display: flex;
                flex-wrap: nowrap;
                overflow-x: auto;
                -webkit-overflow-scrolling: touch;
                scrollbar-width: none;
                gap: 6px;
                box-shadow: 0 2px 4px rgba(0,0,0,0.02);
            }}
            .tabs-wrapper::-webkit-scrollbar {{ display: none; }}

            .tab-btn {{
                background: #f1f5f9;
                border: 1px solid var(--border);
                outline: none;
                cursor: pointer;
                padding: 5px 12px;
                border-radius: 16px;
                font-size: 12px;
                font-weight: 600;
                color: var(--text-muted);
                white-space: nowrap;
                flex-shrink: 0;
                transition: all 0.15s ease;
            }}

            .tab-btn.active {{
                background: var(--primary);
                color: white;
                border-color: var(--primary);
                box-shadow: 0 2px 4px rgba(30, 58, 138, 0.2);
            }}

            .container {{
                padding: 8px;
                max-width: 1500px;
                margin: 0 auto;
            }}

            .warning-details {{
                background-color: #fff1f2;
                border: 1px solid #fecdd3;
                border-radius: 8px;
                margin-bottom: 8px;
                font-size: 11px;
                color: #be123c;
                overflow: hidden;
            }}
            .warning-summary {{
                padding: 7px 12px;
                cursor: pointer;
                font-weight: 600;
                display: flex;
                justify-content: space-between;
                align-items: center;
                user-select: none;
                list-style: none;
            }}
            .warning-summary::-webkit-details-marker {{ display: none; }}
            .toggle-arrow {{ font-size: 9px; transition: transform 0.2s; }}
            details[open] .toggle-arrow {{ transform: rotate(180deg); }}
            .warning-body {{
                padding: 4px 14px 8px 14px;
                border-top: 1px dashed #fecdd3;
                color: #9f1239;
                max-height: 120px;
                overflow-y: auto;
            }}
            .warning-body ul {{ margin: 4px 0 0 16px; padding: 0; }}
            .warning-body li {{ margin-bottom: 2px; }}

            .module-nav-box {{
                background: var(--card-bg);
                border-radius: 8px;
                padding: 8px 12px;
                margin-bottom: 8px;
                border: 1px solid var(--border);
                box-shadow: 0 1px 3px rgba(0,0,0,0.04);
                display: flex;
                align-items: center;
                justify-content: space-between;
                flex-wrap: wrap;
                gap: 8px;
            }}

            .info-box {{
                background: var(--card-bg);
                border-radius: 8px;
                padding: 10px 12px;
                margin-bottom: 8px;
                border: 1px solid var(--border);
                box-shadow: 0 1px 3px rgba(0,0,0,0.04);
            }}

            .info-box p {{
                margin: 0;
                font-size: 12px;
                line-height: 1.4;
                color: var(--text-muted);
            }}

            .info-box b {{
                color: var(--text);
            }}

            .count-badge {{
                display: inline-flex;
                align-items: center;
                background: #fee2e2;
                color: var(--up-red);
                padding: 1px 6px;
                border-radius: 4px;
                font-weight: 700;
                font-size: 12px;
                margin-top: 6px;
            }}

            .custom-select {{
                padding: 5px 10px;
                font-size: 12px;
                font-weight: 600;
                border-radius: 6px;
                border: 1px solid #cbd5e1;
                background-color: white;
                color: var(--text);
                outline: none;
            }}

            .table-container {{
                background: var(--card-bg);
                border-radius: 8px;
                overflow: auto;
                max-height: 75vh;
                border: 1px solid var(--border);
                box-shadow: 0 2px 8px rgba(0,0,0,0.04);
                position: relative;
                -webkit-overflow-scrolling: touch;
            }}

            .styled-table {{
                border-collapse: separate;
                border-spacing: 0;
                width: 100%;
                font-size: 12px;
                text-align: center;
            }}

            .styled-table th {{
                background-color: #f8fafc;
                color: var(--text-muted);
                font-weight: 600;
                padding: 6px 4px;
                white-space: normal;
                word-break: keep-all;
                border-bottom: 2px solid var(--border);
                cursor: pointer;
                user-select: none;
                position: sticky;
                top: 0;
                z-index: 20;
                line-height: 1.25;
            }}

            .styled-table th::after {{ content: ' ↕'; opacity: 0.3; font-size: 9px; }}
            .styled-table th.th-sort-asc::after {{ content: ' ↑'; opacity: 1; color: var(--primary-light); }}
            .styled-table th.th-sort-desc::after {{ content: ' ↓'; opacity: 1; color: var(--primary-light); }}

            .styled-table td {{
                padding: 6px 4px;
                border-bottom: 1px solid var(--border);
                white-space: nowrap;
                vertical-align: middle;
                background-color: white;
                line-height: 1.2;
            }}

            .styled-table tr:last-child td {{
                border-bottom: none;
            }}

            .styled-table:not(.backtest-table) th:nth-child(1),
            .styled-table:not(.backtest-table) td:nth-child(1) {{
                position: sticky;
                left: 0;
                z-index: 10;
                background-color: #f8fafc;
                min-width: 36px;
                max-width: 36px;
            }}

            .styled-table:not(.backtest-table) th:nth-child(3),
            .styled-table:not(.backtest-table) td:nth-child(3) {{
                position: sticky;
                left: 36px;
                z-index: 10;
                background-color: #f8fafc;
                box-shadow: 2px 0 4px -1px rgba(0,0,0,0.08);
                min-width: 72px;
                max-width: 80px;
            }}

            .styled-table:not(.backtest-table) th:nth-child(1),
            .styled-table:not(.backtest-table) th:nth-child(3) {{
                z-index: 30;
                background-color: #f1f5f9;
            }}

            .styled-table:not(.backtest-table) td:nth-child(1),
            .styled-table:not(.backtest-table) td:nth-child(3) {{
                background-color: #ffffff;
            }}

            .backtest-table th, .backtest-table td {{
                position: static !important;
                box-shadow: none !important;
                white-space: normal !important;
                padding: 8px 6px !important;
            }}

            .backtest-table th {{
                position: sticky !important;
                top: 0 !important;
                z-index: 25 !important;
                background-color: #f1f5f9 !important;
            }}

            .styled-table tr.row-favorite td {{
                background-color: #fefce8 !important;
            }}

            .stock-link {{
                text-decoration: none;
                display: block;
                color: inherit;
            }}
            .stock-name {{
                font-weight: 700;
                font-size: 13px;
                color: var(--text);
                line-height: 1.1;
                margin-bottom: 2px;
            }}
            .stock-code {{
                font-size: 10px;
                color: var(--text-muted);
                line-height: 1;
            }}
            .stock-link:hover .stock-name {{
                color: var(--primary-light);
            }}

            .tag-up {{
                color: #dc2626;
                font-weight: 700;
                display: inline-block;
            }}
            .tag-down {{
                color: #16a34a;
                font-weight: 700;
                display: inline-block;
            }}
            .tag-flat {{
                color: #64748b;
                font-weight: 600;
                display: inline-block;
            }}

            .fav-star {{
                cursor: pointer;
                color: #cbd5e1;
                font-size: 18px;
                padding: 4px;
                display: inline-block;
                transition: transform 0.15s ease;
            }}
            .fav-star:active {{ transform: scale(1.2); }}
            .fav-star.active {{ color: var(--star); font-weight: bold; }}

            #fav-table tbody tr {{ display: none; }}
            #fav-table tbody tr.row-favorite {{ display: table-row; }}

            .tabcontent {{ display: none; }}
            .sub-tab-content {{ display: none; }}
        </style>
    </head>
    <body>
        <div class="app-header">
            <h1>📈 多策略選股中心</h1>
            <p>歷史區間: T-120日 ({date_120}) ➔ 最新日 ({date_0})</p>
        </div>

        <div class="tabs-wrapper" id="tabsHeader">
            <button class="tab-btn" onclick="openStrategy(event, 'Tab_Favorites')">💖 我的最愛</button>
            <button class="tab-btn active" onclick="openStrategy(event, 'Tab_Strat12')">🌟 綜合排行</button>
            <button class="tab-btn" onclick="openStrategy(event, 'Tab_TopCombos')">🎖 最佳回測組合</button>
            <button class="tab-btn" onclick="openStrategy(event, 'Tab_Breakout')">🏆 創高突破</button>
            <button class="tab-btn" onclick="openStrategy(event, 'Tab_MaxVol')">🔥 天量突破</button>
            <button class="tab-btn" onclick="openStrategy(event, 'Tab_StepGap')">📈 墊高跳空</button>
            <button class="tab-btn" onclick="openStrategy(event, 'Tab_Whale')">👑 主力鎖碼</button>
            <button class="tab-btn" onclick="openStrategy(event, 'Tab_Revenue')">💎 營收動能</button>
            <button class="tab-btn" onclick="openStrategy(event, 'Tab_Inst')">🚀 法人動能</button>
            <button class="tab-btn" onclick="openStrategy(event, 'Tab_OTC')">🎯 上櫃強勢</button>
            <button class="tab-btn" onclick="openStrategy(event, 'Tab_Backtest')">📊 量化回測總覽</button>
        </div>

        <div class="container">
            {warning_html}

            <!-- 0. 我的最愛 -->
            <div id="Tab_Favorites" class="tabcontent">
                <div class="info-box">
                    <p>🎯 <b>我的最愛：</b> 跨裝置（瀏覽器本地端）儲存的專屬觀察清單，點選表格中的 ⭐ 即可加入。</p>
                    <div class="count-badge">💖 自選清單：<span id="count_Tab_Favorites">0</span> 檔</div>
                </div>
                <div id="fav-empty-msg" style="text-align:center; padding: 40px; color:#64748b; font-size:14px; display:none;">
                    尚未加入任何最愛標的，請點擊各策略清單中的 ⭐ 圖示即可加入。
                </div>
                <div class="table-container">{html_tb_fav}</div>
            </div>

            <!-- 1. 綜合排行 -->
            <div id="Tab_Strat12" class="tabcontent" style="display: block;">
                <div class="info-box">
                    <p>🎯 <b>選股邏輯：</b> 統計所有策略預設條件下，選中最多次的股票排序。</p>
                    <div class="count-badge">✅ 前 50 名強勢標的</div>
                </div>
                <div class="table-container">{html_tb12}</div>
            </div>

            <!-- 2. 最佳回測組合 (策略 27~31 合併) -->
            <div id="Tab_TopCombos" class="tabcontent">
                <div class="module-nav-box">
                    <span style="font-size:12px; font-weight:700; color:var(--primary);">🎯 選擇回測最佳組合：</span>
                    <select class="custom-select" onchange="switchSubTab('Tab_TopCombos', this.value)">
                        <option value="sub_strat27" selected>27. 組合首選 (勝率 {strat_results_27_31[27]["win_rate"]}%)</option>
                        <option value="sub_strat28">28. 組合第二 (勝率 {strat_results_27_31[28]["win_rate"]}%)</option>
                        <option value="sub_strat29">29. 組合第三 (勝率 {strat_results_27_31[29]["win_rate"]}%)</option>
                        <option value="sub_strat30">30. 組合第四 (勝率 {strat_results_27_31[30]["win_rate"]}%)</option>
                        <option value="sub_strat31">31. 組合第五 (勝率 {strat_results_27_31[31]["win_rate"]}%)</option>
                    </select>
                </div>
                <div id="sub_strat27" class="sub-tab-content" style="display:block;">
                    <div class="info-box"><p><b>三策略搭配：</b> {strat_results_27_31[27]["combo_name"]}</p><p><b>歷史表現：</b> 上漲率 <b>{strat_results_27_31[27]["win_rate"]}%</b> | 平均漲幅 {strat_results_27_31[27]["avg_ret"]}% | 樣本數 {strat_results_27_31[27]["samples"]} 檔</p><div class="count-badge">今日符合：{strat_results_27_31[27]["count"]} 檔</div></div>
                    <div class="table-container">{strat_results_27_31[27]["html"]}</div>
                </div>
                <div id="sub_strat28" class="sub-tab-content">
                    <div class="info-box"><p><b>三策略搭配：</b> {strat_results_27_31[28]["combo_name"]}</p><p><b>歷史表現：</b> 上漲率 <b>{strat_results_27_31[28]["win_rate"]}%</b> | 平均漲幅 {strat_results_27_31[28]["avg_ret"]}% | 樣本數 {strat_results_27_31[28]["samples"]} 檔</p><div class="count-badge">今日符合：{strat_results_27_31[28]["count"]} 檔</div></div>
                    <div class="table-container">{strat_results_27_31[28]["html"]}</div>
                </div>
                <div id="sub_strat29" class="sub-tab-content">
                    <div class="info-box"><p><b>三策略搭配：</b> {strat_results_27_31[29]["combo_name"]}</p><p><b>歷史表現：</b> 上漲率 <b>{strat_results_27_31[29]["win_rate"]}%</b> | 平均漲幅 {strat_results_27_31[29]["avg_ret"]}% | 樣本數 {strat_results_27_31[29]["samples"]} 檔</p><div class="count-badge">今日符合：{strat_results_27_31[29]["count"]} 檔</div></div>
                    <div class="table-container">{strat_results_27_31[29]["html"]}</div>
                </div>
                <div id="sub_strat30" class="sub-tab-content">
                    <div class="info-box"><p><b>三策略搭配：</b> {strat_results_27_31[30]["combo_name"]}</p><p><b>歷史表現：</b> 上漲率 <b>{strat_results_27_31[30]["win_rate"]}%</b> | 平均漲幅 {strat_results_27_31[30]["avg_ret"]}% | 樣本數 {strat_results_27_31[30]["samples"]} 檔</p><div class="count-badge">今日符合：{strat_results_27_31[30]["count"]} 檔</div></div>
                    <div class="table-container">{strat_results_27_31[30]["html"]}</div>
                </div>
                <div id="sub_strat31" class="sub-tab-content">
                    <div class="info-box"><p><b>三策略搭配：</b> {strat_results_27_31[31]["combo_name"]}</p><p><b>歷史表現：</b> 上漲率 <b>{strat_results_27_31[31]["win_rate"]}%</b> | 平均漲幅 {strat_results_27_31[31]["avg_ret"]}% | 樣本數 {strat_results_27_31[31]["samples"]} 檔</p><div class="count-badge">今日符合：{strat_results_27_31[31]["count"]} 檔</div></div>
                    <div class="table-container">{strat_results_27_31[31]["html"]}</div>
                </div>
            </div>

            <!-- 3. 創高突破 (策略 4, 6, 7, 9, 13, 14, 15, 17 合併) -->
            <div id="Tab_Breakout" class="tabcontent">
                <div class="module-nav-box">
                    <span style="font-size:12px; font-weight:700; color:var(--primary);">🎯 選擇創高週期與條件：</span>
                    <select class="custom-select" onchange="switchSubTab('Tab_Breakout', this.value)">
                        <option value="sub_strat17" selected>17. 創 120 日半年新高</option>
                        <option value="sub_strat4">4. 創 20 日新高 ＋ 投信連七不賣</option>
                        <option value="sub_strat6">6. 創 5 日高 ＋ 跳空量增 ＋ 法人連三</option>
                        <option value="sub_strat7">7. 創 20 日高 ＋ 底底高 ＋ 法人連七</option>
                        <option value="sub_strat13">13. 逼近 60 日高 (-2%以內)</option>
                        <option value="sub_strat14">14. 創 20 日高 ＋ 法人七日不賣</option>
                        <option value="sub_strat15">15. 壓縮突破 60 日高</option>
                        <option value="sub_strat9">9. 多重濾網 (漲幅+創高+大戶)</option>
                    </select>
                </div>
                <div id="sub_strat17" class="sub-tab-content" style="display:block;">
                    <div class="info-box"><p>🎯 最新收盤價大於過去 120 個交易日內每一天的收盤價 (半年新高)。</p><div class="count-badge">符合：{len(res17)} 檔</div></div>
                    <div class="table-container">{html_tb17}</div>
                </div>
                <div id="sub_strat4" class="sub-tab-content">
                    <div class="info-box"><p>🎯 創 20 日新高、過去七天投信不賣、量大於前一日。</p><div class="count-badge">符合：{len(res4)} 檔</div></div>
                    <div class="table-container">{html_tb4}</div>
                </div>
                <div id="sub_strat6" class="sub-tab-content">
                    <div class="info-box"><p>🎯 創五日新高 & 跳空量增 & 法人連三日不賣。</p><div class="count-badge">符合：{len(res6)} 檔</div></div>
                    <div class="table-container">{html_tb6}</div>
                </div>
                <div id="sub_strat7" class="sub-tab-content">
                    <div class="info-box"><p>🎯 創 20 日新高 & 底底高 & 收高量增 & 法人七日不賣。</p><div class="count-badge">符合：{len(res7)} 檔</div></div>
                    <div class="table-container">{html_tb7}</div>
                </div>
                <div id="sub_strat13" class="sub-tab-content">
                    <div class="info-box"><p>🎯 最新收盤價距離「近60日最高收盤價」在 -2% 以內，且成交量大於 500 張。</p><div class="count-badge">符合：{len(res13)} 檔</div></div>
                    <div class="table-container">{html_tb13}</div>
                </div>
                <div id="sub_strat14" class="sub-tab-content">
                    <div class="info-box"><p>🎯 創 20 日新高、量增，且法人（外資、投信）過去 7 天皆無賣出。</p><div class="count-badge">符合：{len(res14)} 檔</div></div>
                    <div class="table-container">{html_tb14}</div>
                </div>
                <div id="sub_strat15" class="sub-tab-content">
                    <div class="info-box"><p>🎯 收盤價創 60 日新高，且量增 1.2 倍，底部區間壓縮突破。</p><div class="count-badge">符合：{len(res15)} 檔</div></div>
                    <div class="table-container">{html_tb15}</div>
                </div>
                <div id="sub_strat9" class="sub-tab-content">
                    <div class="info-box"><p>🎯 漲幅過濾 + 創高 + 大戶比例 + 法人連七不賣。</p><div class="count-badge">符合：{len(res9)} 檔</div></div>
                    <div class="table-container">{html_tb9}</div>
                </div>
            </div>

            <!-- 4. 天量突破 (策略 16, 23 合併) -->
            <div id="Tab_MaxVol" class="tabcontent">
                <div class="module-nav-box">
                    <span style="font-size:12px; font-weight:700; color:var(--primary);">🎯 選擇天量週期：</span>
                    <select class="custom-select" onchange="switchSubTab('Tab_MaxVol', this.value)">
                        <option value="sub_strat23" selected>23. 突破過去 10 天最大量高點</option>
                        <option value="sub_strat16">16. 突破過去 60 天最大量高點</option>
                    </select>
                </div>
                <div id="sub_strat23" class="sub-tab-content" style="display:block;">
                    <div class="info-box"><p>🎯 最新收盤價正式站上過去 10 個交易日內成交量最大當日的最高價。</p><div class="count-badge">符合：{len(res23)} 檔</div></div>
                    <div class="table-container">{html_tb23}</div>
                </div>
                <div id="sub_strat16" class="sub-tab-content">
                    <div class="info-box"><p>🎯 突破 60 日成交量最大當日高點、第一天剛突破且外資投信近 7 天無賣出。</p><div class="count-badge">符合：{len(res16)} 檔</div></div>
                    <div class="table-container">{html_tb16}</div>
                </div>
            </div>

            <!-- 5. 墊高跳空 (策略 1, 2, 8, 22 合併) -->
            <div id="Tab_StepGap" class="tabcontent">
                <div class="module-nav-box">
                    <span style="font-size:12px; font-weight:700; color:var(--primary);">🎯 選擇短線型態：</span>
                    <select class="custom-select" onchange="switchSubTab('Tab_StepGap', this.value)">
                        <option value="sub_strat22" selected>22. 高勝率基因精選 (創五日高+墊高+跳空)</option>
                        <option value="sub_strat1">1. 向上跳空不回補 ＋ 連續量增</option>
                        <option value="sub_strat2">2. 階梯式墊高 ＋ 外資點火</option>
                        <option value="sub_strat8">8. 連兩日收盤墊高 ＋ 法人連七不賣</option>
                    </select>
                </div>
                <div id="sub_strat22" class="sub-tab-content" style="display:block;">
                    <div class="info-box"><p>🎯 融合回測第 1 名「創五日高+跳空」、第 2 名「創120日高」、第 3 名「連續墊高」精華。</p><div class="count-badge">符合：{len(res22)} 檔</div></div>
                    <div class="table-container">{html_tb22}</div>
                </div>
                <div id="sub_strat1" class="sub-tab-content">
                    <div class="info-box"><p>🎯 向上跳空不回補、連續量增。</p><div class="count-badge">符合：{len(res1)} 檔</div></div>
                    <div class="table-container">{html_tb1}</div>
                </div>
                <div id="sub_strat2" class="sub-tab-content">
                    <div class="info-box"><p>🎯 階梯式墊高、外資點火、過去七天法人不賣出。</p><div class="count-badge">符合：{len(res2)} 檔</div></div>
                    <div class="table-container">{html_tb2}</div>
                </div>
                <div id="sub_strat8" class="sub-tab-content">
                    <div class="info-box"><p>🎯 連兩日墊高 & 量能放大 & 法人七日不賣。</p><div class="count-badge">符合：{len(res8)} 檔</div></div>
                    <div class="table-container">{html_tb8}</div>
                </div>
            </div>

            <!-- 6. 主力鎖碼 (策略 24, 25, 32, 33 合併) -->
            <div id="Tab_Whale" class="tabcontent">
                <div class="module-nav-box">
                    <span style="font-size:12px; font-weight:700; color:var(--primary);">🎯 選擇主力模式：</span>
                    <select class="custom-select" onchange="switchSubTab('Tab_Whale', this.value)">
                        <option value="sub_strat33" selected>33. 絕對鎖碼 (大戶>70% + 多頭排列 + 距半年高點-5%~+5%)</option>
                        <option value="sub_strat32">32. 大戶鎖碼 ＋ 法人集中度</option>
                        <option value="sub_strat25">25. 倚強科模式複製 (大戶&ge;60% ＋ 均線發散)</option>
                        <option value="sub_strat24">24. 飆股基因起漲 (近10日漲5%~25%起漲甜蜜區)</option>
                    </select>
                </div>
                <div id="sub_strat33" class="sub-tab-content" style="display:block;">
                    <div class="info-box">
                        <p>🎯 <b>策略 33：絕對鎖碼突破</b> 千張大戶比例 &gt; 70% ＋ 5/20/60MA 均線多頭排列 ＋ 最新收盤價距離半年(120日)高點在 -5% ~ +5% 以內（蓄勢待發或剛突破）。</p>
                        <div class="count-badge">符合：<span id="count_Strat33">{len(res33)}</span> 檔</div>
                    </div>
                    <div class="table-container">{html_tb33}</div>
                </div>
                <div id="sub_strat32" class="sub-tab-content">
                    <div class="info-box">
                        <p>🎯 <b>策略 32：大戶鎖碼 ＋ 法人集中度</b> 千張大戶佔比 > 60%，且過去 1~3 天「外資+投信」合計買超佔總成交量達一定比例。</p>
                        <div class="filter-container">
                            <select id="sel_Strat32_days" class="custom-select" onchange="filterStrat32()">
                                <option value="1">過去 1 天集中度</option>
                                <option value="2">過去 2 天集中度</option>
                                <option value="3" selected>過去 3 天集中度</option>
                            </select>
                            <select id="sel_Strat32_ratio" class="custom-select" onchange="filterStrat32()">
                                <option value="10">大於 10%</option>
                                <option value="20" selected>大於 20%</option>
                                <option value="30">大於 30%</option>
                            </select>
                        </div>
                        <div class="count-badge">符合：<span id="count_Strat32">0</span> 檔</div>
                    </div>
                    <div class="table-container">{html_tb32}</div>
                </div>
                <div id="sub_strat25" class="sub-tab-content">
                    <div class="info-box"><p>🎯 千張大戶 &ge; 60% ＋ 5MA>20MA ＋ 出量1.5倍 ＋ 漲幅3%~25%起漲區 ＋ 法人無賣壓。</p><div class="count-badge">符合：{len(res25)} 檔</div></div>
                    <div class="table-container">{html_tb25}</div>
                </div>
                <div id="sub_strat24" class="sub-tab-content">
                    <div class="info-box"><p>🎯 複製過去10天暴漲40%主力特徵：鎖定近10日漲幅 5%~25%、近5日紅盤&ge;4天、今日出量&ge;1.2倍、大戶高鎖碼。</p><div class="count-badge">符合：{len(res24)} 檔</div></div>
                    <div class="table-container">{html_tb24}</div>
                </div>
            </div>

            <!-- 7. 營收動能 (策略 18, 19 合併) -->
            <div id="Tab_Revenue" class="tabcontent">
                <div class="module-nav-box">
                    <span style="font-size:12px; font-weight:700; color:var(--primary);">🎯 選擇營收成長條件：</span>
                    <select class="custom-select" id="sel_Rev_Mode" onchange="filterRevenueMode()">
                        <option value="18" selected>18. 連續 3 個月營收月增 (MoM)</option>
                        <option value="19">19. 最新月營收暴增 1.5 倍以上</option>
                    </select>
                    <select class="custom-select" id="sel_Rev_Vol" onchange="filterRevenueMode()">
                        <option value="yes" selected>今日有出量 (&ge; 1.2倍)</option>
                        <option value="no">不限成交量</option>
                    </select>
                </div>
                <div id="sub_strat_rev" class="sub-tab-content" style="display:block;">
                    <div class="info-box"><p>🎯 篩選基本面營收強勢突破標的，可透過上方選單即時切換「營收成長條件」與「是否要求出量」。</p><div class="count-badge">符合：<span id="count_Revenue">0</span> 檔</div></div>
                    <div class="table-container" id="container_rev_table">{html_tb18}</div>
                </div>
            </div>

            <!-- 8. 法人動能 (策略 3, 5, 20 合併) -->
            <div id="Tab_Inst" class="tabcontent">
                <div class="module-nav-box">
                    <span style="font-size:12px; font-weight:700; color:var(--primary);">🎯 選擇法人籌碼行為：</span>
                    <select class="custom-select" onchange="switchSubTab('Tab_Inst', this.value)">
                        <option value="sub_strat20" selected>20. 外資連三日買超且越買越多</option>
                        <option value="sub_strat3">3. 最新日量增 ＋ 投信不賣 ＋ 外資連買</option>
                        <option value="sub_strat5">5. 旱地拔蔥 (前6天法人0，今日突大買)</option>
                    </select>
                </div>
                <div id="sub_strat20" class="sub-tab-content" style="display:block;">
                    <div class="info-box"><p>🎯 外資連三日買超且買超張數遞增 (越買越多) | 最新一日出量 (&ge; 1.2倍) | 股價收紅。</p><div class="count-badge">符合：{len(res20)} 檔</div></div>
                    <div class="table-container">{html_tb20}</div>
                </div>
                <div id="sub_strat3" class="sub-tab-content">
                    <div class="info-box"><p>🎯 最新日量增、投信七日不賣出、外資連續買超。</p><div class="count-badge">符合：{len(res3)} 檔</div></div>
                    <div class="table-container">{html_tb3}</div>
                </div>
                <div id="sub_strat5" class="sub-tab-content">
                    <div class="info-box"><p>🎯 旱地拔蔥 (前6天法人0，今日突介入)。</p><div class="count-badge">符合：{len(res5)} 檔</div></div>
                    <div class="table-container">{html_tb5}</div>
                </div>
            </div>

            <!-- 9. 上櫃強勢 (策略 10, 11 合併) -->
            <div id="Tab_OTC" class="tabcontent">
                <div class="module-nav-box">
                    <span style="font-size:12px; font-weight:700; color:var(--primary);">🎯 選擇上櫃強勢型態：</span>
                    <select class="custom-select" onchange="switchSubTab('Tab_OTC', this.value)">
                        <option value="sub_strat11" selected>11. 上櫃實體紅 K 強勢股 (收盤大於開盤)</option>
                        <option value="sub_strat10">10. 上櫃收盤強勢股 (收盤漲幅排行)</option>
                    </select>
                </div>
                <div id="sub_strat11" class="sub-tab-content" style="display:block;">
                    <div class="info-box"><p>🎯 限定上櫃股票 | 實體紅K強勢股 (收盤大於開盤)。</p><div class="count-badge">符合：{len(res11)} 檔</div></div>
                    <div class="table-container">{html_tb11}</div>
                </div>
                <div id="sub_strat10" class="sub-tab-content">
                    <div class="info-box"><p>🎯 限定上櫃股票 | 最新收盤漲幅排行。</p><div class="count-badge">符合：{len(res10)} 檔</div></div>
                    <div class="table-container">{html_tb10}</div>
                </div>
            </div>

            <!-- 10. 量化回測總覽 (策略 21, 26 合併) -->
            <div id="Tab_Backtest" class="tabcontent">
                <div class="module-nav-box">
                    <span style="font-size:12px; font-weight:700; color:var(--primary);">🎯 選擇回測研究模組：</span>
                    <select class="custom-select" onchange="switchSubTab('Tab_Backtest', this.value)">
                        <option value="sub_strat26" selected>26. 9 月三策略組合挖掘 (10/1~10/8 績效)</option>
                        <option value="sub_strat21">21. 過去 30 天各策略隔日勝率回測</option>
                    </select>
                </div>
                <div id="sub_strat26" class="sub-tab-content" style="display:block;">
                    <div class="info-box"><p>🎯 回測 2026/09/01～2026/09/30 期間任三策略同時命中之個股，以 <b>10/01 開盤至 10/08 收盤</b> 計算實際績效，依上漲機率（勝率）最高前五名搭配自動建構策略 27~31。</p><div class="count-badge">📊 969 組三策略搭配排序總覽</div></div>
                    <div class="table-container">{html_tb26}</div>
                </div>
                <div id="sub_strat21" class="sub-tab-content">
                    <div class="info-box"><p>🎯 統計各策略在過去 30 個交易日內選出的個股，在「隔天收盤為紅盤」的歷史機率與平均漲跌幅度。</p><div class="count-badge">📊 依上漲勝率排行</div></div>
                    <div class="table-container">{html_tb21}</div>
                </div>
            </div>
        </div>

        <script>
            const FAV_KEY = 'tw_stock_favorites';

            function getFavorites() {{
                return JSON.parse(localStorage.getItem(FAV_KEY) || '[]');
            }}

            function saveFavorites(favs) {{
                localStorage.setItem(FAV_KEY, JSON.stringify(favs));
            }}

            function updateFavEmptyState() {{
                const favs = getFavorites();
                const msg = document.getElementById('fav-empty-msg');
                const tbl = document.getElementById('fav-table');
                if(msg && tbl) {{
                    if(favs.length === 0) {{
                        msg.style.display = 'block';
                        tbl.style.display = 'none';
                    }} else {{
                        msg.style.display = 'none';
                        tbl.style.display = 'table';
                    }}
                }}
                
                const cnt = document.getElementById('count_Tab_Favorites');
                if(cnt) cnt.innerText = favs.length;
            }}

            function restoreFavorites() {{
                const favs = getFavorites();
                document.querySelectorAll('.fav-star').forEach(star => {{
                    const code = star.dataset.code;
                    if (favs.includes(code)) {{
                        star.classList.add('active');
                        star.innerHTML = '★';
                        star.closest('tr').classList.add('row-favorite');
                    }}
                }});
                updateFavEmptyState();
                
                document.querySelectorAll('.styled-table tbody').forEach(tbody => {{
                    sortTbody(tbody);
                }});
            }}

            document.addEventListener('click', function(e) {{
                if (e.target && e.target.classList.contains('fav-star')) {{
                    e.preventDefault();
                    e.stopPropagation();
                    
                    const star = e.target;
                    const code = star.dataset.code;
                    let favs = getFavorites();
                    
                    const isAdding = !star.classList.contains('active');
                    
                    if (isAdding) {{
                        if(!favs.includes(code)) favs.push(code);
                    }} else {{
                        favs = favs.filter(c => c !== code);
                    }}
                    saveFavorites(favs);
                    
                    document.querySelectorAll(`.fav-star[data-code="${{code}}"]`).forEach(s => {{
                        const tr = s.closest('tr');
                        if (isAdding) {{
                            s.classList.add('active');
                            s.innerHTML = '★';
                            tr.classList.add('row-favorite');
                        }} else {{
                            s.classList.remove('active');
                            s.innerHTML = '☆';
                            tr.classList.remove('row-favorite');
                        }}
                        sortTbody(tr.parentElement);
                    }});
                    
                    updateFavEmptyState();
                }}
            }});

            function openStrategy(evt, strategyName) {{
                document.querySelectorAll(".tabcontent").forEach(el => el.style.display = "none");
                document.querySelectorAll(".tab-btn").forEach(el => el.classList.remove("active"));
                document.getElementById(strategyName).style.display = "block";
                evt.currentTarget.classList.add("active");
                evt.currentTarget.scrollIntoView({{ behavior: 'smooth', inline: 'center', block: 'nearest' }});
                if(strategyName === 'Tab_Revenue') filterRevenueMode();
            }}

            function switchSubTab(parentTabId, subTargetId) {{
                const parent = document.getElementById(parentTabId);
                if (!parent) return;
                parent.querySelectorAll('.sub-tab-content').forEach(el => el.style.display = 'none');
                const target = parent.querySelector('#' + subTargetId);
                if (target) target.style.display = 'block';
                
                if(subTargetId === 'sub_strat32') {{
                    filterStrat32();
                }}
            }}

            function filterStrat32() {{
                let days = parseInt(document.getElementById('sel_Strat32_days').value);
                let minRatio = parseFloat(document.getElementById('sel_Strat32_ratio').value);
                let table = document.getElementById('sub_strat32').querySelector("table");
                if (!table) return;

                let headers = table.querySelectorAll("thead th");
                let idx1 = -1, idx2 = -1, idx3 = -1;
                headers.forEach((th, i) => {{
                    if (th.innerText.includes("法人1日")) idx1 = i;
                    if (th.innerText.includes("法人2日")) idx2 = i;
                    if (th.innerText.includes("法人3日")) idx3 = i;
                }});

                let count = 0;
                table.querySelectorAll("tbody tr").forEach(row => {{
                    let cells = row.querySelectorAll("td");
                    let val = 0;
                    if (days === 1 && idx1 >= 0) val = parseFloat(cells[idx1].innerText);
                    if (days === 2 && idx2 >= 0) val = parseFloat(cells[idx2].innerText);
                    if (days === 3 && idx3 >= 0) val = parseFloat(cells[idx3].innerText);

                    let show = (!isNaN(val) && val >= minRatio);
                    row.style.display = show ? "" : "none";
                    if(show) count++;
                }});
                updateVisibleCount('Strat32', count);
            }}

            function filterRevenueMode() {{
                let mode = document.getElementById('sel_Rev_Mode').value;
                let volOpt = document.getElementById('sel_Rev_Vol').value;
                
                let container = document.getElementById('container_rev_table');
                let countBadge = document.getElementById('count_Revenue');
                
                let tbl18 = `{html_tb18}`;
                let tbl19 = `{html_tb19}`;
                
                let targetHtml = (mode === '18') ? tbl18 : tbl19;
                container.innerHTML = targetHtml;
                
                let table = container.querySelector("table");
                if(!table) {{
                    countBadge.innerText = "0";
                    return;
                }}
                
                let headers = table.querySelectorAll("thead th");
                let idxVol = -1;
                headers.forEach((th, i) => {{ if (th.innerText.includes("增量倍數")) idxVol = i; }});

                let count = 0;
                table.querySelectorAll("tbody tr").forEach(row => {{
                    let show = true;
                    if (volOpt === 'yes' && idxVol >= 0) {{
                        let cells = row.querySelectorAll("td");
                        let vMul = parseFloat(cells[idxVol].innerText);
                        if (isNaN(vMul) || vMul < 1.2) show = false;
                    }}
                    row.style.display = show ? "" : "none";
                    if(show) count++;
                }});
                countBadge.innerText = count;
            }}

            function sortTbody(tbody, colIndex = -1, isAscending = false) {{
                const rows = Array.from(tbody.querySelectorAll('tr'));

                rows.sort((rowA, rowB) => {{
                    const starA = rowA.querySelector('.fav-star');
                    const starB = rowB.querySelector('.fav-star');
                    const favA = (starA && starA.classList.contains('active')) ? 1 : 0;
                    const favB = (starB && starB.classList.contains('active')) ? 1 : 0;

                    if (favA !== favB) return favB - favA;

                    if (colIndex >= 0) {{
                        let cellAStr = rowA.querySelectorAll('td')[colIndex].innerText.trim();
                        let cellBStr = rowB.querySelectorAll('td')[colIndex].innerText.trim();
                        
                        let numA, numB;
                        if (cellAStr.includes('/5') || cellAStr.includes('/10') || cellAStr.includes('/ 7日')) {{
                            numA = parseFloat(cellAStr.split('/')[0]);
                            numB = parseFloat(cellBStr.split('/')[0]);
                        }} else {{
                            numA = parseFloat(cellAStr.replace(/,/g, '').replace(/%/g, '').replace(/\\+/g, ''));
                            numB = parseFloat(cellBStr.replace(/,/g, '').replace(/%/g, '').replace(/\\+/g, ''));
                        }}

                        const multiplier = isAscending ? 1 : -1;

                        if (!isNaN(numA) && !isNaN(numB)) {{
                            return (numA - numB) * multiplier;
                        }} else {{
                            return cellAStr.localeCompare(cellBStr, 'zh-Hant') * multiplier;
                        }}
                    }}
                    return 0;
                }});

                rows.forEach(row => tbody.appendChild(row));
            }}

            function enableTableSorting() {{
                document.querySelectorAll('.styled-table th').forEach(header => {{
                    if (header.cellIndex === 0 && header.innerText.includes("⭐")) return;

                    header.addEventListener('click', function() {{
                        const table = this.closest('table');
                        const tbody = table.querySelector('tbody');
                        const isAscending = this.classList.contains('th-sort-asc');
                        const colIndex = this.cellIndex;

                        table.querySelectorAll('th').forEach(th => th.classList.remove('th-sort-asc', 'th-sort-desc'));

                        if (isAscending) {{
                            this.classList.add('th-sort-desc');
                            sortTbody(tbody, colIndex, false);
                        }} else {{
                            this.classList.add('th-sort-asc');
                            sortTbody(tbody, colIndex, true);
                        }}
                    }});
                }});
            }}

            function updateVisibleCount(tabId, visibleCount) {{
                let elem = document.getElementById("count_" + tabId);
                if(elem) elem.innerText = visibleCount;
            }}

            window.addEventListener('DOMContentLoaded', () => {{
                restoreFavorites();
                enableTableSorting();
                setTimeout(() => {{
                    filterStrat32();
                    filterRevenueMode();
                }}, 300);
            }});
        </script>
    </body>
    </html>
    """

    for code, row in res15.iterrows():
        is_max_vol = "true" if row['創60日最大量'] else "false"
        search_str = f"<tr>\n      <td><span class='fav-star' data-code='{code}'>☆</span></td>\n      <td>{row['市場']}</td>"
        replace_str = f"<tr data-is-max-vol='{is_max_vol}'>\n      <td><span class='fav-star' data-code='{code}'>☆</span></td>\n      <td>{row['市場']}</td>"
        html_content = html_content.replace(search_str, replace_str)

    html_filename = "index.html"
    file_path = os.path.abspath(html_filename)
    with open(file_path, "w", encoding="utf-8") as f:
        f.write(html_content)

    print(f"\n✅ 策略 18 營收量增選項與技術計分模組已整合完畢！檔案已生成: {html_filename}")
    if os.environ.get("GITHUB_ACTIONS") != "true":
        webbrowser.open(f"file:///{file_path}")

if __name__ == "__main__":
    main()
