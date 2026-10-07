import datetime
import time
import random
import os
import webbrowser
import pandas as pd
import requests

# ===== 關閉 Pandas 大量合併欄位時的效能警告 =====
import warnings
warnings.simplefilter(action='ignore', category=pd.errors.PerformanceWarning)
# ===============================================

# ==========================================
# 網路請求標頭設定 (強化防阻擋)
# ==========================================
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

def safe_request_json(url: str, max_retries: int = 3, headers=TWSE_HEADERS):
    for attempt in range(max_retries):
        try:
            resp = requests.get(url, headers=headers, timeout=15)
            if resp.status_code == 200:
                try:
                    return resp.json()
                except Exception:
                    return {}
        except (requests.exceptions.ConnectionError, requests.exceptions.Timeout):
            pass
        time.sleep(random.uniform(2, 4))
    return {}

def tpex_date_params(date_str: str):
    dt = datetime.datetime.strptime(date_str, "%Y%m%d")
    roc = f"{dt.year - 1911}/{dt.strftime('%m/%d')}"
    ad = f"{dt.year}%2F{dt.strftime('%m')}%2F{dt.strftime('%d')}"
    return roc, ad

def tpex_fetch_tables(url: str, date_str: str, roc_date: str, tag: str):
    for attempt in range(3):
        try:
            resp = requests.get(url, headers=TPEX_HEADERS, timeout=15)
            if resp.status_code != 200:
                time.sleep(1.5)
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
        time.sleep(1.5)
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
    if os.path.exists(cache_file):
        df_cached = pd.read_csv(cache_file, dtype={"股票代號": str})
        if "上櫃" in df_cached["市場"].values and not _is_dup_otc(date_str, df_cached):
            return df_cached

    print(f"🌐 [{date_str}] 抓取【上市+上櫃價量】資料...")
    df_twse = fetch_twse_daily(date_str)
    if df_twse.empty:
        return pd.DataFrame()
    time.sleep(random.uniform(1, 2))
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
    if os.path.exists(cache_file):
        df_cached = pd.read_csv(cache_file, dtype={"股票代號": str})
        if len(df_cached) > 500:
            return df_cached

    print(f"🌐 [{date_str}] 抓取【上市+上櫃法人籌碼】資料...")
    df_twse = fetch_twse_inst(date_str)
    time.sleep(random.uniform(1, 2))
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

    if os.path.exists(cache_file):
        return pd.read_csv(cache_file, dtype={"股票代號": str})

    print("🌐 抓取【集保千張大戶】...")
    url = "https://smart.tdcc.com.tw/opendata/getOD.ashx?id=1-5"
    for attempt in range(3):
        try:
            resp = requests.get(url, headers=TWSE_HEADERS, timeout=30)
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
                df_tdcc = pd.DataFrame(rows)
                df_tdcc.to_csv(cache_file, index=False, encoding="utf-8-sig")
                return df_tdcc
        except Exception:
            pass
        time.sleep(3)
    WARNINGS.append("千張大戶資料 抓取失敗")
    return pd.DataFrame(columns=["股票代號", "千張大戶比例(%)"])

def get_foreign_holdings() -> pd.DataFrame:
    today_str = datetime.date.today().strftime("%Y%m%d")
    cache_file = os.path.join(CACHE_DIR, f"foreign_holdings_{today_str}.csv")

    if os.path.exists(cache_file):
        return pd.read_csv(cache_file, dtype={"股票代號": str})

    print("🌐 抓取【外資持股總比例與發行張數】...")
    rows = []

    for offset in range(7):
        d = datetime.date.today() - datetime.timedelta(days=offset)
        if d.weekday() >= 5:
            continue
        url = (f"https://www.twse.com.tw/rwd/zh/fund/MI_QFIIS"
               f"?date={d.strftime('%Y%m%d')}&selectType=ALLBUT0999&response=json")
        data = safe_request_json(url, max_retries=2)
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
                             max_retries=3, headers=OPENAPI_HEADERS)
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

def get_last_n_trading_days_data(n_market=121, n_inst=20):
    valid_dfs = []
    valid_inst = []
    today = datetime.date.today()
    offset = 0
    
    print(f"\n⏳ 準備檢查 {n_market} 個交易日的歷史快取 (約半年)...")
    
    while len(valid_dfs) < n_market:
        if offset > 260:
            raise RuntimeError("連續 260 天都抓不到足夠的交易日資料，請檢查網路或 API。")
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
            time.sleep(0.5)
    return valid_dfs, valid_inst

# ==========================================
# 格式化小工具：合併代號名稱＋超連結、漲跌上色
# ==========================================
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

    days_data, inst_data = get_last_n_trading_days_data(n_market=121, n_inst=20)

    date_0, df_0 = days_data[0]
    date_1, df_1 = days_data[1]
    date_2, df_2 = days_data[2]
    date_120, _ = days_data[120]

    print(f"\n✅ 行情獲取完畢！ 日期區間: [{date_120}] ~ [{date_0}]")

    df_tdcc = get_tdcc_data()
    df_holdings = get_foreign_holdings()

    df_0_merge = df_0.copy()
    df_0_merge.columns = [f"{c}_0" if c not in ["市場", "股票代號", "股票名稱"] else c for c in df_0_merge.columns]

    df_1_merge = df_1[["股票代號", "開盤價", "最高價", "最低價", "收盤價", "成交量"]].copy()
    df_1_merge.columns = [f"{c}_1" if c != "股票代號" else c for c in df_1_merge.columns]

    df_2_merge = df_2[["股票代號", "最高價", "最低價", "收盤價", "成交量"]].copy()
    df_2_merge.columns = [f"{c}_2" if c != "股票代號" else c for c in df_2_merge.columns]

    df_merge = pd.merge(df_0_merge, df_1_merge, on="股票代號", how="inner")
    df_merge = pd.merge(df_merge, df_2_merge, on="股票代號", how="inner")

    for i in range(3, 121):
        _, df_i = days_data[i]
        if df_i.empty:
            continue

        cols_to_get = ["股票代號", "最高價", "最低價", "收盤價", "成交量"]
        df_temp = df_i[cols_to_get].copy()
        rename_dict = {"收盤價": f"收盤價_{i}", "成交量": f"成交量_{i}", "最高價": f"最高價_{i}", "最低價": f"最低價_{i}"}
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

    df_merge['最新漲幅(%)'] = ((df_merge['收盤價_0'] - df_merge['收盤價_1']) / df_merge['收盤價_1'] * 100).round(2)
    df_merge['前一日漲幅(%)'] = ((df_merge['收盤價_1'] - df_merge['收盤價_2']) / df_merge['收盤價_2'] * 100).round(2)
    
    df_merge['創5日高'] = (df_merge['收盤價_0'] > df_merge['5日最高收盤']).apply(lambda x: "是" if x else "否")
    df_merge['創20日高'] = (df_merge['收盤價_0'] > df_merge['20日最高收盤']).apply(lambda x: "是" if x else "否")

    for idx in range(20):
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

    d0_s = f"{date_0[4:6]}/{date_0[6:]}"
    d1_s = f"{date_1[4:6]}/{date_1[6:]}"
    d2_s = f"{date_2[4:6]}/{date_2[6:]}"

    df_merge['⭐'] = "<span class='fav-star'>☆</span>"

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

    # 合併名稱與代號成單一超連結欄位
    df_merge['標的'] = df_merge.apply(format_stock_cell, axis=1)

    # 輸出欄位模板
    base_cols = ["⭐", "市場", "標的", "近7日符合次數", "近5日紅盤", "近10日紅盤", "近10日漲幅(%)", f"{d1_s} 收盤", f"{d0_s} 收盤", "最新漲幅(%)", "前一日漲幅(%)", f"{d0_s} 量(張)"]
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
        for col_name in ["最新漲幅(%)", "前一日漲幅(%)", "近10日漲幅(%)"]:
            if col_name in df_out.columns:
                df_out[col_name] = df_out[col_name].apply(color_pct)
        return df_out

    # 策略 1: 跳空不回補
    def cond1_fn(k):
        l_k, c_prev = df_merge[f'最低價_{k}'], df_merge[f'收盤價_{k+1}']
        v_k, v_prev = df_merge[f'成交量_{k}'], df_merge[f'成交量_{k+1}']
        return (v_prev > 0) & (l_k >= c_prev) & (v_k > v_prev)
    res1_hits = eval_rolling_condition(cond1_fn)
    cond1 = (df_merge['成交量_1'] > 0) & (df_merge['跳空天數'] >= 1) & (df_merge['量增天數'] >= 1)
    res1 = df_merge[cond1].copy()
    res1['近7日符合次數'] = res1_hits[cond1]
    res1["增量倍數"] = (res1["成交量_0"] / res1["成交量_1"]).round(2)
    res1 = res1.sort_values(by=["跳空天數", "量增天數", "增量倍數"], ascending=[False, False, False])
    res1 = res1.rename(columns={"收盤價_1": f"{d1_s} 收盤", "收盤價_0": f"{d0_s} 收盤", "成交量_0": f"{d0_s} 量(張)"})
    cols1 = base_cols + ["跳空天數", "量增天數", "增量倍數"] + chip_cols
    html_tb1 = apply_color_formatting(res1[cols1]).to_html(index=False, classes="styled-table sortable-table", escape=False)

    # 策略 2: 連續收盤墊高
    def cond2_fn(k):
        step_up = (df_merge[f'收盤價_{k}'] > df_merge[f'收盤價_{k+1}']) & (df_merge[f'最低價_{k}'] >= df_merge[f'收盤價_{k+1}'])
        vol_up = df_merge[f'成交量_{k}'] > df_merge[f'成交量_{k+1}']
        inst_hold = True
        for j in range(7):
            inst_hold = inst_hold & (df_merge[f'外資_{k+j}'] >= 0) & (df_merge[f'投信_{k+j}'] >= 0)
        return step_up & vol_up & inst_hold
    res2_hits = eval_rolling_condition(cond2_fn)
    cond2 = (
        (df_merge['墊高天數'] >= 1) & (df_merge['量增天數'] >= 1) & (df_merge['外資連買天數'] >= 0) &
        (df_merge['外資_0'] >= 0) & (df_merge['投信_0'] >= 0) & (df_merge['外資_1'] >= 0) & (df_merge['投信_1'] >= 0) &
        (df_merge['外資_2'] >= 0) & (df_merge['投信_2'] >= 0) & (df_merge['外資_3'] >= 0) & (df_merge['投信_3'] >= 0) &
        (df_merge['外資_4'] >= 0) & (df_merge['投信_4'] >= 0) & (df_merge['外資_5'] >= 0) & (df_merge['投信_5'] >= 0) &
        (df_merge['外資_6'] >= 0) & (df_merge['投信_6'] >= 0)
    )
    res2 = df_merge[cond2].copy()
    res2['近7日符合次數'] = res2_hits[cond2]
    res2["增量倍數"] = (res2["成交量_0"] / res2["成交量_1"]).round(2)
    res2 = res2.sort_values(by=["墊高天數", "外資連買天數", "增量倍數"], ascending=[False, False, False])
    res2 = res2.rename(columns={"收盤價_1": f"{d1_s} 收盤", "收盤價_0": f"{d0_s} 收盤", "成交量_0": f"{d0_s} 量(張)"})
    cols2 = base_cols + ["墊高天數", "量增天數", "外資連買天數", "增量倍數"] + chip_cols
    html_tb2 = apply_color_formatting(res2[cols2]).to_html(index=False, classes="styled-table sortable-table", escape=False)

    # 策略 3: 量增+投信不賣+外資連買
    def cond3_fn(k):
        vol_up = (df_merge[f'成交量_{k}'] > df_merge[f'成交量_{k+1}']) & (df_merge[f'成交量_{k+1}'] > 0)
        t_hold = True
        for j in range(7):
            t_hold = t_hold & (df_merge[f'投信_{k+j}'] >= 0)
        f_buy = df_merge[f'外資_{k}'] > 0
        return vol_up & t_hold & f_buy
    res3_hits = eval_rolling_condition(cond3_fn)
    cond3 = (
        (df_merge['成交量_0'] > df_merge['成交量_1']) & (df_merge['成交量_1'] > 0) &
        (df_merge['投信_0'] >= 0) & (df_merge['投信_1'] >= 0) & (df_merge['投信_2'] >= 0) &
        (df_merge['投信_3'] >= 0) & (df_merge['投信_4'] >= 0) & (df_merge['投信_5'] >= 0) &
        (df_merge['投信_6'] >= 0) & (df_merge['外資連買天數'] >= 1)
    )
    res3 = df_merge[cond3].copy()
    res3['近7日符合次數'] = res3_hits[cond3]
    res3["增量倍數"] = (res3["成交量_0"] / res3["成交量_1"]).round(2)
    res3 = res3.sort_values(by=["外資連買天數", "增量倍數"], ascending=[False, False])
    res3 = res3.rename(columns={"收盤價_1": f"{d1_s} 收盤", "收盤價_0": f"{d0_s} 收盤", "成交量_0": f"{d0_s} 量(張)", "外資_0": "最新日外資(張)"})
    cols3 = base_cols + ["最新日外資(張)", "外資連買天數", "連漲天數", "增量倍數"] + chip_cols
    html_tb3 = apply_color_formatting(res3[cols3]).to_html(index=False, classes="styled-table sortable-table", escape=False)

    # 策略 4: 創 20 日高 + 投信七日不賣
    def cond4_fn(k):
        close_20 = [f'收盤價_{k+j}' for j in range(1, 21)]
        max_20 = df_merge[close_20].max(axis=1)
        c_high = df_merge[f'收盤價_{k}'] > max_20
        vol_up = (df_merge[f'成交量_{k}'] > df_merge[f'成交量_{k+1}']) & (df_merge[f'成交量_{k+1}'] > 0)
        t_hold = True
        for j in range(7):
            t_hold = t_hold & (df_merge[f'投信_{k+j}'] >= 0)
        return c_high & vol_up & t_hold
    res4_hits = eval_rolling_condition(cond4_fn)
    cond4 = (
        (df_merge['收盤價_0'] > df_merge['20日最高收盤']) & (df_merge['成交量_0'] > df_merge['成交量_1']) & (df_merge['成交量_1'] > 0) &
        (df_merge['投信_0'] >= 0) & (df_merge['投信_1'] >= 0) & (df_merge['投信_2'] >= 0) & (df_merge['投信_3'] >= 0) &
        (df_merge['投信_4'] >= 0) & (df_merge['投信_5'] >= 0) & (df_merge['投信_6'] >= 0)
    )
    res4 = df_merge[cond4].copy()
    res4['近7日符合次數'] = res4_hits[cond4]
    res4["增量倍數"] = (res4["成交量_0"] / res4["成交量_1"]).round(2)
    res4 = res4.sort_values(by=["投信近七日(張)", "增量倍數"], ascending=[False, False])
    res4 = res4.rename(columns={"收盤價_1": f"{d1_s} 收盤", "收盤價_0": f"{d0_s} 收盤", "成交量_0": f"{d0_s} 量(張)", "外資_0": "最新日外資(張)"})
    cols4 = base_cols + ["最新日外資(張)", "增量倍數"] + chip_cols
    html_tb4 = apply_color_formatting(res4[cols4]).to_html(index=False, classes="styled-table sortable-table", escape=False)

    # 策略 5: 旱地拔蔥
    def cond5_fn(k):
        vol_up = (df_merge[f'成交量_{k}'] > df_merge[f'成交量_{k+1}']) & (df_merge[f'成交量_{k+1}'] > 0)
        price_up = (df_merge[f'最低價_{k}'] > df_merge[f'最低價_{k+1}']) & (df_merge[f'收盤價_{k}'] > df_merge[f'收盤價_{k+1}'])
        zero_prev = True
        for j in range(1, 7):
            zero_prev = zero_prev & (df_merge[f'外資_{k+j}'] == 0) & (df_merge[f'投信_{k+j}'] == 0)
        inst_today = (df_merge[f'外資_{k}'] > 0) | (df_merge[f'投信_{k}'] > 0)
        return vol_up & price_up & zero_prev & inst_today
    res5_hits = eval_rolling_condition(cond5_fn)
    cond5 = (
        (df_merge['成交量_0'] > df_merge['成交量_1']) & (df_merge['成交量_1'] > 0) &
        (df_merge['最低價_0'] > df_merge['最低價_1']) & (df_merge['收盤價_0'] > df_merge['收盤價_1']) &
        (df_merge['外資_1'] == 0) & (df_merge['投信_1'] == 0) & (df_merge['外資_2'] == 0) & (df_merge['投信_2'] == 0) &
        (df_merge['外資_3'] == 0) & (df_merge['投信_3'] == 0) & (df_merge['外資_4'] == 0) & (df_merge['投信_4'] == 0) &
        (df_merge['外資_5'] == 0) & (df_merge['投信_5'] == 0) & (df_merge['外資_6'] == 0) & (df_merge['投信_6'] == 0) &
        ((df_merge['外資_0'] > 0) | (df_merge['投信_0'] > 0))
    )
    res5 = df_merge[cond5].copy()
    res5['近7日符合次數'] = res5_hits[cond5]
    res5["增量倍數"] = (res5["成交量_0"] / res5["成交量_1"]).round(2)
    res5['最新法人買超(張)'] = res5['外資_0'] + res5['投信_0']
    res5 = res5.sort_values(by=["最新法人買超(張)", "增量倍數"], ascending=[False, False])
    res5 = res5.rename(columns={"收盤價_0": f"{d0_s} 收盤", "成交量_0": f"{d0_s} 量(張)", "收盤價_1": f"{d1_s} 收盤"})
    cols5 = base_cols + ["最新法人買超(張)", "增量倍數"] + chip_cols
    html_tb5 = apply_color_formatting(res5[cols5]).to_html(index=False, classes="styled-table sortable-table", escape=False)

    # 策略 6: 創五日高+跳空不賣
    def cond6_fn(k):
        close_5 = [f'收盤價_{k+j}' for j in range(1, 6)]
        max_5 = df_merge[close_5].max(axis=1)
        c_high = df_merge[f'收盤價_{k}'] > max_5
        vol_up = (df_merge[f'成交量_{k}'] > df_merge[f'成交量_{k+1}']) & (df_merge[f'成交量_{k+1}'] > 0)
        gap = (df_merge[f'開盤價_{k}'] > df_merge[f'收盤價_{k+1}']) & (df_merge[f'最低價_{k}'] > df_merge[f'收盤價_{k+1}'])
        hold_3 = (df_merge[f'外資_{k}'] >= 0) & (df_merge[f'投信_{k}'] >= 0) & (df_merge[f'外資_{k+1}'] >= 0) & (df_merge[f'投信_{k+1}'] >= 0) & (df_merge[f'外資_{k+2}'] >= 0) & (df_merge[f'投信_{k+2}'] >= 0)
        return c_high & vol_up & gap & hold_3
    res6_hits = eval_rolling_condition(cond6_fn)
    cond6 = (
        (df_merge['成交量_0'] > df_merge['成交量_1']) & (df_merge['成交量_1'] > 0) &
        (df_merge['開盤價_0'] > df_merge['收盤價_1']) & (df_merge['最低價_0'] > df_merge['收盤價_1']) &
        (df_merge['外資_0'] >= 0) & (df_merge['投信_0'] >= 0) & (df_merge['外資_1'] >= 0) & (df_merge['投信_1'] >= 0) &
        (df_merge['外資_2'] >= 0) & (df_merge['投信_2'] >= 0) & (df_merge['收盤價_0'] > df_merge['5日最高收盤'])
    )
    res6 = df_merge[cond6].copy()
    res6['近7日符合次數'] = res6_hits[cond6]
    res6["增量倍數"] = (res6["成交量_0"] / res6["成交量_1"]).round(2)
    res6 = res6.sort_values(by=["投信近三日(張)", "增量倍數"], ascending=[False, False])
    res6 = res6.rename(columns={"收盤價_1": f"{d1_s} 收盤", "收盤價_0": f"{d0_s} 收盤", "成交量_0": f"{d0_s} 量(張)"})
    cols6 = base_cols + ["增量倍數"] + chip_cols
    html_tb6 = apply_color_formatting(res6[cols6]).to_html(index=False, classes="styled-table sortable-table", escape=False)

    # 策略 7: 創 20 日高 + 連七不賣
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
    res7_hits = eval_rolling_condition(cond7_fn)
    cond7 = (
        (df_merge['成交量_0'] > df_merge['成交量_1']) & (df_merge['成交量_1'] > 0) &
        (df_merge['最低價_0'] > df_merge['最低價_1']) & (df_merge['收盤價_0'] > df_merge['收盤價_1']) &
        (df_merge['收盤價_0'] > df_merge['20日最高收盤']) &
        (df_merge['外資_0'] >= 0) & (df_merge['投信_0'] >= 0) & (df_merge['外資_1'] >= 0) & (df_merge['投信_1'] >= 0) &
        (df_merge['外資_2'] >= 0) & (df_merge['投信_2'] >= 0) & (df_merge['外資_3'] >= 0) & (df_merge['投信_3'] >= 0) &
        (df_merge['外資_4'] >= 0) & (df_merge['投信_4'] >= 0) & (df_merge['外資_5'] >= 0) & (df_merge['投信_5'] >= 0) &
        (df_merge['外資_6'] >= 0) & (df_merge['投信_6'] >= 0)
    )
    res7 = df_merge[cond7].copy()
    res7['近7日符合次數'] = res7_hits[cond7]
    res7["增量倍數"] = (res7["成交量_0"] / res7["成交量_1"]).round(2)
    res7 = res7.sort_values(by=["投信近七日(張)", "增量倍數"], ascending=[False, False])
    res7 = res7.rename(columns={"收盤價_1": f"{d1_s} 收盤", "收盤價_0": f"{d0_s} 收盤", "成交量_0": f"{d0_s} 量(張)"})
    cols7 = base_cols + ["增量倍數"] + chip_cols
    html_tb7 = apply_color_formatting(res7[cols7]).to_html(index=False, classes="styled-table sortable-table", escape=False)

    # 策略 8: 連兩日墊高 + 連七不賣
    def cond8_fn(k):
        c_up2 = (df_merge[f'收盤價_{k}'] > df_merge[f'收盤價_{k+1}']) & (df_merge[f'收盤價_{k+1}'] > df_merge[f'收盤價_{k+2}'])
        vol_up = (df_merge[f'成交量_{k}'] > df_merge[f'成交量_{k+1}']) & (df_merge[f'成交量_{k+1}'] > 0)
        hold_7 = True
        for j in range(7):
            hold_7 = hold_7 & (df_merge[f'外資_{k+j}'] >= 0) & (df_merge[f'投信_{k+j}'] >= 0)
        return c_up2 & vol_up & hold_7
    res8_hits = eval_rolling_condition(cond8_fn)
    cond8 = (
        (df_merge['收盤價_0'] > df_merge['收盤價_1']) & (df_merge['收盤價_1'] > df_merge['收盤價_2']) &
        (df_merge['成交量_0'] > df_merge['成交量_1']) & (df_merge['成交量_1'] > 0) &
        (df_merge['外資_0'] >= 0) & (df_merge['投信_0'] >= 0) & (df_merge['外資_1'] >= 0) & (df_merge['投信_1'] >= 0) &
        (df_merge['外資_2'] >= 0) & (df_merge['投信_2'] >= 0) & (df_merge['外資_3'] >= 0) & (df_merge['投信_3'] >= 0) &
        (df_merge['外資_4'] >= 0) & (df_merge['投信_4'] >= 0) & (df_merge['外資_5'] >= 0) & (df_merge['投信_5'] >= 0) &
        (df_merge['外資_6'] >= 0) & (df_merge['投信_6'] >= 0)
    )
    res8 = df_merge[cond8].copy()
    res8['近7日符合次數'] = res8_hits[cond8]
    res8["增量倍數"] = (res8["成交量_0"] / res8["成交量_1"]).round(2)
    res8 = res8.sort_values(by=["投信近七日(張)", "增量倍數"], ascending=[False, False])
    res8 = res8.rename(columns={"收盤價_2": f"{d2_s} 收盤", "收盤價_1": f"{d1_s} 收盤", "收盤價_0": f"{d0_s} 收盤", "成交量_0": f"{d0_s} 量(張)"})
    cols8 = base_cols + ["增量倍數"] + chip_cols
    html_tb8 = apply_color_formatting(res8[cols8]).to_html(index=False, classes="styled-table sortable-table", escape=False)

    # 策略 9: 多重濾網 (連七不賣)
    def cond9_fn(k):
        c_valid = df_merge[f'收盤價_{k+1}'] > 0
        hold_7 = True
        for j in range(7):
            hold_7 = hold_7 & (df_merge[f'外資_{k+j}'] >= 0) & (df_merge[f'投信_{k+j}'] >= 0)
        return c_valid & hold_7
    res9_hits = eval_rolling_condition(cond9_fn)
    cond9 = (
        (df_merge['收盤價_1'] > 0) &
        (df_merge['外資_0'] >= 0) & (df_merge['投信_0'] >= 0) & (df_merge['外資_1'] >= 0) & (df_merge['投信_1'] >= 0) &
        (df_merge['外資_2'] >= 0) & (df_merge['投信_2'] >= 0) & (df_merge['外資_3'] >= 0) & (df_merge['投信_3'] >= 0) &
        (df_merge['外資_4'] >= 0) & (df_merge['投信_4'] >= 0) & (df_merge['外資_5'] >= 0) & (df_merge['投信_5'] >= 0) &
        (df_merge['外資_6'] >= 0) & (df_merge['投信_6'] >= 0)
    )
    res9 = df_merge[cond9].copy()
    res9['近7日符合次數'] = res9_hits[cond9]
    res9 = res9.sort_values(by=["最新漲幅(%)"], ascending=False)
    res9 = res9.rename(columns={"收盤價_1": f"{d1_s} 收盤", "收盤價_0": f"{d0_s} 收盤", "成交量_0": f"{d0_s} 量(張)"})
    cols9 = base_cols + ["創5日高", "創20日高"] + chip_cols
    html_tb9 = apply_color_formatting(res9[cols9]).to_html(index=False, classes="styled-table sortable-table", escape=False)

    # 策略 10: 上櫃強勢
    def cond10_fn(k):
        ret_k = ((df_merge[f'收盤價_{k}'] - df_merge[f'收盤價_{k+1}']) / df_merge[f'收盤價_{k+1}']) * 100
        return (df_merge['市場'] == '上櫃') & (ret_k > 0) & (df_merge[f'收盤價_{k+1}'] > 0)
    res10_hits = eval_rolling_condition(cond10_fn)
    cond10 = ((df_merge['市場'] == '上櫃') & (df_merge['最新漲幅(%)'] > 0) & (df_merge['收盤價_1'] > 0))
    res10 = df_merge[cond10].copy()
    res10['近7日符合次數'] = res10_hits[cond10]
    res10 = res10.sort_values(by=["最新漲幅(%)"], ascending=False)
    res10 = res10.rename(columns={"收盤價_1": f"{d1_s} 收盤", "收盤價_0": f"{d0_s} 收盤", "成交量_0": f"{d0_s} 量(張)"})
    cols10 = base_cols + chip_cols
    html_tb10 = apply_color_formatting(res10[cols10]).to_html(index=False, classes="styled-table sortable-table", escape=False)

    # 策略 11: 上櫃實體紅K
    def cond11_fn(k):
        c_k = df_merge[f'收盤價_{k}']
        o_k = df_merge[f'開盤價_{k}'] if f'開盤價_{k}' in df_merge else df_merge['開盤價_0']
        return (df_merge['市場'] == '上櫃') & (c_k > o_k) & (o_k > 0)
    res11_hits = eval_rolling_condition(cond11_fn)
    cond11 = ((df_merge['市場'] == '上櫃') & (df_merge['收盤價_0'] > 0) & (df_merge['開盤價_0'] > 0))
    res11 = df_merge[cond11].copy()
    res11["實體K漲幅(%)"] = ((res11["收盤價_0"] - res11["開盤價_0"]) / res11["開盤價_0"] * 100).round(2)
    res11 = res11[res11["實體K漲幅(%)"] > 0]
    res11 = res11[res11["實體K漲幅(%)"].notna()]
    res11['近7日符合次數'] = res11_hits.loc[res11.index]
    res11 = res11.sort_values(by=["實體K漲幅(%)"], ascending=False)
    res11 = res11.rename(columns={"開盤價_0": f"{d0_s} 開盤", "收盤價_1": f"{d1_s} 收盤", "收盤價_0": f"{d0_s} 收盤", "成交量_0": f"{d0_s} 量(張)"})
    cols11 = base_cols + [f"{d0_s} 開盤", "實體K漲幅(%)"] + chip_cols
    html_tb11 = apply_color_formatting(res11[cols11]).to_html(index=False, classes="styled-table sortable-table", escape=False)

    # 策略 13: 逼近60日新高 (-2%以上)
    def cond13_fn(k):
        c_k = df_merge[f'收盤價_{k}']
        close_60 = [f'收盤價_{k+j}' for j in range(1, 61)]
        max_60 = df_merge[close_60].max(axis=1)
        return (c_k > 0) & (max_60 > 0) & (df_merge[f'成交量_{k}'] >= 500) & (c_k >= max_60 * 0.98)
    res13_hits = eval_rolling_condition(cond13_fn)
    cond13 = (
        (df_merge['收盤價_0'] > 0) & 
        (df_merge['60日最高收盤'] > 0) & 
        (df_merge['成交量_0'] >= 500) &
        (df_merge['收盤價_0'] >= df_merge['60日最高收盤'] * 0.98)
    )
    res13 = df_merge[cond13].copy()
    res13['近7日符合次數'] = res13_hits[cond13]
    res13["距離60日高點(%)"] = ((res13["收盤價_0"] - res13["60日最高收盤"]) / res13["60日最高收盤"] * 100).round(2)
    res13 = res13.sort_values(by=["距離60日高點(%)", "最新漲幅(%)"], ascending=[False, False])
    res13 = res13.rename(columns={"收盤價_1": f"{d1_s} 收盤", "收盤價_0": f"{d0_s} 收盤", "成交量_0": f"{d0_s} 量(張)"})
    cols13 = base_cols + ["60日最高收盤", "距離60日高點(%)"] + chip_cols
    html_tb13 = apply_color_formatting(res13[cols13]).to_html(index=False, classes="styled-table sortable-table", escape=False)

    # 策略 14: 創 20 日高 + 法人七日不賣
    def cond14_fn(k):
        close_20 = [f'收盤價_{k+j}' for j in range(1, 21)]
        max_20 = df_merge[close_20].max(axis=1)
        c_high = df_merge[f'收盤價_{k}'] > max_20
        vol_up = (df_merge[f'成交量_{k}'] > df_merge[f'成交量_{k+1}']) & (df_merge[f'成交量_{k+1}'] > 0)
        hold_7 = True
        for j in range(7):
            hold_7 = hold_7 & (df_merge[f'外資_{k+j}'] >= 0) & (df_merge[f'投信_{k+j}'] >= 0)
        return c_high & vol_up & hold_7
    res14_hits = eval_rolling_condition(cond14_fn)
    cond14 = (
        (df_merge['成交量_0'] > df_merge['成交量_1']) & (df_merge['成交量_1'] > 0) &
        (df_merge['收盤價_0'] > df_merge['20日最高收盤']) &
        (df_merge['外資_0'] >= 0) & (df_merge['外資_1'] >= 0) & (df_merge['外資_2'] >= 0) &
        (df_merge['外資_3'] >= 0) & (df_merge['外資_4'] >= 0) & (df_merge['外資_5'] >= 0) &
        (df_merge['外資_6'] >= 0) & (df_merge['投信_0'] >= 0) & (df_merge['投信_1'] >= 0) &
        (df_merge['投信_2'] >= 0) & (df_merge['投信_3'] >= 0) & (df_merge['投信_4'] >= 0) &
        (df_merge['投信_5'] >= 0) & (df_merge['投信_6'] >= 0)
    )
    res14 = df_merge[cond14].copy()
    res14['近7日符合次數'] = res14_hits[cond14]
    res14["增量倍數"] = (res14["成交量_0"] / res14["成交量_1"]).round(2)
    res14 = res14.sort_values(by=["最新漲幅(%)", "增量倍數"], ascending=[False, False])
    res14 = res14.rename(columns={"收盤價_1": f"{d1_s} 收盤", "收盤價_0": f"{d0_s} 收盤", "成交量_0": f"{d0_s} 量(張)", "外資_0": "最新日外資(張)"})
    cols14 = base_cols + ["最新日外資(張)", "外資連買天數", "連漲天數", "增量倍數"] + chip_cols
    html_tb14 = apply_color_formatting(res14[cols14]).to_html(index=False, classes="styled-table sortable-table", escape=False)

    # 策略 15: 壓縮突破 60 日高
    def cond15_fn(k):
        c_k = df_merge[f'收盤價_{k}']
        close_60 = [f'收盤價_{k+j}' for j in range(1, 61)]
        max_60 = df_merge[close_60].max(axis=1)
        vol_up = df_merge[f'成交量_{k}'] >= df_merge[f'成交量_{k+1}'] * 1.2
        return vol_up & (c_k >= max_60)
    res15_hits = eval_rolling_condition(cond15_fn)
    cond15 = (
        (df_merge['成交量_0'] >= df_merge['成交量_1'] * 1.2) & 
        (df_merge['收盤價_0'] >= df_merge['60日最高收盤']) & 
        (df_merge['20日最低收盤'] > 0) 
    )
    res15 = df_merge[cond15].copy()
    res15['近7日符合次數'] = res15_hits[cond15]
    res15["增量倍數"] = (res15["成交量_0"] / res15["成交量_1"]).round(2)
    res15["20日震幅(%)"] = ((res15["收盤價_0"] - res15["20日最低收盤"]) / res15["20日最低收盤"] * 100).round(2)
    res15["40日震幅(%)"] = ((res15["收盤價_0"] - res15["40日最低收盤"]) / res15["40日最低收盤"] * 100).round(2)
    res15["60日震幅(%)"] = ((res15["收盤價_0"] - res15["60日最低收盤"]) / res15["60日最低收盤"] * 100).round(2)
    res15_strict = res15[(res15['創60日最大量']) & (res15['60日震幅(%)'] <= 10.0)]
    res15 = res15.sort_values(by=["60日震幅(%)", "增量倍數"], ascending=[True, False]) 
    res15 = res15.rename(columns={"收盤價_1": f"{d1_s} 收盤", "收盤價_0": f"{d0_s} 收盤", "成交量_0": f"{d0_s} 量(張)"})
    cols15 = base_cols + ["20日震幅(%)", "40日震幅(%)", "60日震幅(%)", "增量倍數"] + chip_cols
    html_tb15 = apply_color_formatting(res15[cols15]).to_html(index=False, classes="styled-table sortable-table", escape=False)

    # 策略 16: 突破最大量高點 (第一天突破) + 雙法人七日不賣
    def cond16_fn(k):
        mv_high = df_merge['最大量日最高價']
        c_k = df_merge[f'收盤價_{k}']
        c_prev = df_merge[f'收盤價_{k+1}']
        breakout = (mv_high > 0) & (c_k > mv_high) & (c_prev <= mv_high)
        hold_7 = True
        for j in range(7):
            hold_7 = hold_7 & (df_merge[f'外資_{k+j}'] >= 0) & (df_merge[f'投信_{k+j}'] >= 0)
        return breakout & hold_7
    res16_hits = eval_rolling_condition(cond16_fn)
    cond16 = (
        (df_merge['最大量日最高價'] > 0) &
        (df_merge['收盤價_0'] > df_merge['最大量日最高價']) &
        (df_merge['收盤價_1'] <= df_merge['最大量日最高價']) &
        (df_merge['外資_0'] >= 0) & (df_merge['外資_1'] >= 0) & (df_merge['外資_2'] >= 0) &
        (df_merge['外資_3'] >= 0) & (df_merge['外資_4'] >= 0) & (df_merge['外資_5'] >= 0) &
        (df_merge['外資_6'] >= 0) &
        (df_merge['投信_0'] >= 0) & (df_merge['投信_1'] >= 0) & (df_merge['投信_2'] >= 0) &
        (df_merge['投信_3'] >= 0) & (df_merge['投信_4'] >= 0) & (df_merge['投信_5'] >= 0) &
        (df_merge['投信_6'] >= 0)
    )
    res16 = df_merge[cond16].copy()
    res16['近7日符合次數'] = res16_hits[cond16]
    res16["增量倍數"] = (res16["成交量_0"] / res16["成交量_1"]).round(2)
    res16 = res16.sort_values(by=["最新漲幅(%)", "增量倍數"], ascending=[False, False])
    res16 = res16.rename(columns={"收盤價_1": f"{d1_s} 收盤", "收盤價_0": f"{d0_s} 收盤", "成交量_0": f"{d0_s} 量(張)"})
    cols16 = base_cols + ["最大量日最高價", "增量倍數"] + chip_cols
    html_tb16 = apply_color_formatting(res16[cols16]).to_html(index=False, classes="styled-table sortable-table", escape=False)

    # 策略 17: 收盤價創 120 日新高
    def cond17_fn(k):
        close_120 = [f'收盤價_{k+j}' for j in range(1, 121) if f'收盤價_{k+j}' in df_merge]
        if not close_120: return pd.Series(False, index=df_merge.index)
        max_120 = df_merge[close_120].max(axis=1)
        return (df_merge[f'收盤價_{k}'] > 0) & (max_120 > 0) & (df_merge[f'收盤價_{k}'] > max_120)
    res17_hits = eval_rolling_condition(cond17_fn)
    cond17 = (
        (df_merge['收盤價_0'] > 0) &
        (df_merge['120日最高收盤'] > 0) &
        (df_merge['收盤價_0'] > df_merge['120日最高收盤'])
    )
    res17 = df_merge[cond17].copy()
    res17['近7日符合次數'] = res17_hits[cond17]
    res17["增量倍數"] = (res17["成交量_0"] / res17["成交量_1"]).round(2)
    res17 = res17.sort_values(by=["最新漲幅(%)", "增量倍數"], ascending=[False, False])
    res17 = res17.rename(columns={"收盤價_1": f"{d1_s} 收盤", "收盤價_0": f"{d0_s} 收盤", "成交量_0": f"{d0_s} 量(張)"})
    cols17 = base_cols + ["120日最高收盤", "增量倍數"] + chip_cols
    html_tb17 = apply_color_formatting(res17[cols17]).to_html(index=False, classes="styled-table sortable-table", escape=False)

    # 策略 12: 綜合排行 (前 50 名)
    st_lists = [
        ("1.跳空", res1), ("2.墊高", res2), ("3.量增法人", res3),
        ("4.創20日高", res4), ("5.拔蔥", res5), ("6.五日高", res6),
        ("7.創20日高不賣", res7), ("8.連兩日創高", res8), ("9.多重", res9),
        ("10.上櫃強勢", res10), ("11.上櫃實體紅K", res11), ("13.逼近60日高", res13),
        ("14.創20日高+法人七日不賣", res14), ("15.壓縮突破60日高", res15_strict),
        ("16.突破最大量高點", res16), ("17.創120日新高", res17)
    ]
    
    hit_counts = {}
    hit_names = {}
    for s_name, res_df in st_lists:
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
        res12 = res12.rename(columns={"收盤價_1": f"{d1_s} 收盤", "收盤價_0": f"{d0_s} 收盤", "成交量_0": f"{d0_s} 量(張)"})
        cols12 = base_cols + ["入選次數", "符合策略"] + chip_cols
        html_tb12 = apply_color_formatting(res12[cols12]).to_html(index=False, classes="styled-table sortable-table", escape=False)
    else:
        html_tb12 = "<p style='text-align:center;'>目前無任何股票入選預設策略</p>"

    warning_html = ""
    if WARNINGS:
        warn_text = "、".join(sorted(set(WARNINGS)))
        warning_html = f"<div class='warning-box'><b>⚠ 系統警告：</b> 缺少部分資料 ({warn_text})，請檢查網路連線或該日是否為休市日。</div>"

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
                padding: 12px 16px;
                position: sticky;
                top: 0;
                z-index: 100;
                box-shadow: 0 4px 12px rgba(0,0,0,0.08);
            }}

            .app-header h1 {{
                margin: 0;
                font-size: 18px;
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

            /* 多列包覆排版按鈕 (Wrap) */
            .tabs-wrapper {{
                background: white;
                border-bottom: 1px solid var(--border);
                position: sticky;
                top: 55px;
                z-index: 90;
                padding: 8px 10px;
                display: flex;
                flex-wrap: wrap;
                gap: 4px;
                box-shadow: 0 2px 4px rgba(0,0,0,0.02);
            }}

            .tab-btn {{
                background: #f1f5f9;
                border: 1px solid var(--border);
                outline: none;
                cursor: pointer;
                padding: 4px 8px;
                border-radius: 6px;
                font-size: 11px;
                font-weight: 600;
                color: var(--text-muted);
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

            .filter-container {{
                display: flex;
                flex-wrap: wrap;
                gap: 6px;
                margin-top: 8px;
                padding-top: 8px;
                border-top: 1px dashed var(--border);
            }}

            .custom-select {{
                padding: 4px 8px;
                font-size: 11px;
                font-weight: 600;
                border-radius: 6px;
                border: 1px solid #cbd5e1;
                background-color: white;
                color: var(--text);
                outline: none;
                flex: 1 1 calc(50% - 6px);
                min-width: 110px;
            }}

            /* 表格容器：開啟高度限制與滾動，實現吸頂 */
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

            /* 表頭吸頂固定與換行設定 */
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

            /* 凍結前兩欄：星星 + 標的 (代號+名稱超連結) */
            .styled-table th:nth-child(1),
            .styled-table td:nth-child(1) {{
                position: sticky;
                left: 0;
                z-index: 10;
                background-color: #f8fafc;
                min-width: 36px;
                max-width: 36px;
            }}

            .styled-table th:nth-child(3),
            .styled-table td:nth-child(3) {{
                position: sticky;
                left: 36px;
                z-index: 10;
                background-color: #f8fafc;
                box-shadow: 2px 0 4px -1px rgba(0,0,0,0.08);
                min-width: 72px;
                max-width: 80px;
            }}

            /* 左上角交叉處擁有最高層級 (吸頂+吸左) */
            .styled-table th:nth-child(1),
            .styled-table th:nth-child(3) {{
                z-index: 30;
                background-color: #f1f5f9;
            }}

            .styled-table td:nth-child(1),
            .styled-table td:nth-child(3) {{
                background-color: #ffffff;
            }}

            .styled-table tr.row-favorite td {{
                background-color: #fefce8 !important;
            }}

            /* 合併代號與名稱的 cell 樣式 */
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

            /* 紅漲綠跌標籤樣式 */
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
                padding: 2px;
                display: inline-block;
                transition: transform 0.15s ease;
            }}
            .fav-star:active {{ transform: scale(1.2); }}
            .fav-star.active {{ color: var(--star); font-weight: bold; }}

            .warning-box {{
                background-color: #fff1f2;
                color: #be123c;
                padding: 8px 12px;
                border-radius: 6px;
                margin-bottom: 8px;
                font-size: 11px;
                font-weight: 600;
                border: 1px solid #fecdd3;
            }}

            .tabcontent {{ display: none; }}
        </style>
    </head>
    <body>
        <div class="app-header">
            <h1>📈 多策略選股中心</h1>
            <p>歷史區間: T-120日 ({date_120}) ➔ 最新日 ({date_0})</p>
        </div>

        <div class="tabs-wrapper">
            <button class="tab-btn active" onclick="openStrategy(event, 'Strat12')">🌟 12. 綜合排行</button>
            <button class="tab-btn" onclick="openStrategy(event, 'Strat17')">🏆 17. 創120日高</button>
            <button class="tab-btn" onclick="openStrategy(event, 'Strat16')">🔥 16. 最大量高點</button>
            <button class="tab-btn" onclick="openStrategy(event, 'Strat15')">🚀 15. 壓縮突破60日</button>
            <button class="tab-btn" onclick="openStrategy(event, 'Strat14')">🚀 14. 創20日高不賣</button>
            <button class="tab-btn" onclick="openStrategy(event, 'Strat13')">📈 13. 逼近60日高</button>
            <button class="tab-btn" onclick="openStrategy(event, 'Strat1')">1. 跳空不補</button>
            <button class="tab-btn" onclick="openStrategy(event, 'Strat2')">2. 連續墊高</button>
            <button class="tab-btn" onclick="openStrategy(event, 'Strat3')">3. 量增法人買</button>
            <button class="tab-btn" onclick="openStrategy(event, 'Strat4')">4. 創20日新高</button>
            <button class="tab-btn" onclick="openStrategy(event, 'Strat5')">5. 旱地拔蔥</button>
            <button class="tab-btn" onclick="openStrategy(event, 'Strat6')">6. 創五日高</button>
            <button class="tab-btn" onclick="openStrategy(event, 'Strat7')">7. 20日高不賣</button>
            <button class="tab-btn" onclick="openStrategy(event, 'Strat8')">8. 連日墊高</button>
            <button class="tab-btn" onclick="openStrategy(event, 'Strat9')">9. 多重濾網</button>
            <button class="tab-btn" onclick="openStrategy(event, 'Strat10')">10. 上櫃強勢</button>
            <button class="tab-btn" onclick="openStrategy(event, 'Strat11')">11. 上櫃紅K</button>
        </div>

        <div class="container">
            {warning_html}

            <div id="Strat12" class="tabcontent" style="display: block;">
                <div class="info-box">
                    <p>🎯 <b>選股邏輯：</b> 統計所有策略(1~11, 13~17)預設條件下，選中最多次的股票排序。</p>
                    <div class="count-badge">✅ 前 50 名強勢標的</div>
                </div>
                <div class="table-container">{html_tb12}</div>
            </div>

            <div id="Strat17" class="tabcontent">
                <div class="info-box">
                    <p>🎯 <b>選股邏輯：</b> 最新收盤價大於過去 120 個交易日內每一天的收盤價 (半年新高)。</p>
                    <div class="count-badge">符合：<span id="count_Strat17">{len(res17)}</span> 檔</div>
                </div>
                <div class="table-container">{html_tb17}</div>
            </div>

            <div id="Strat16" class="tabcontent">
                <div class="info-box">
                    <p>🎯 <b>選股邏輯：</b> 1. 突破60日成交量最大當日高點 | 2. 第一天剛突破 | 3. 外資與投信近七天無賣出。</p>
                    <div class="filter-container">
                        <select id="sel_Strat16_vol" class="custom-select" onchange="filterStrat16()">
                            <option value="0" selected>不考慮最新量能</option>
                            <option value="1.2">量增 &ge; 1.2 倍</option>
                            <option value="1.5">量增 &ge; 1.5 倍</option>
                        </select>
                    </div>
                    <div class="count-badge">符合：<span id="count_Strat16">0</span> 檔</div>
                </div>
                <div class="table-container">{html_tb16}</div>
            </div>

            <div id="Strat15" class="tabcontent">
                <div class="info-box">
                    <p>🎯 <b>選股邏輯：</b> 收盤價創 60 日新高，且量增 1.2 倍，底部區間壓縮突破。</p>
                    <div class="filter-container">
                        <select id="sel_Strat15_period" class="custom-select" onchange="filterStrat15()">
                            <option value="20">比較 20 日低點</option>
                            <option value="40">比較 40 日低點</option>
                            <option value="60" selected>比較 60 日低點</option>
                        </select>
                        <select id="sel_Strat15_range" class="custom-select" onchange="filterStrat15()">
                            <option value="10" selected>震幅 &le; 10%</option>
                            <option value="15">震幅 &le; 15%</option>
                            <option value="20">震幅 &le; 20%</option>
                            <option value="999">不限震幅</option>
                        </select>
                        <select id="sel_Strat15_vol" class="custom-select" onchange="filterStrat15()">
                            <option value="1.2" selected>量增 &ge; 1.2 倍</option>
                            <option value="1.5">量增 &ge; 1.5 倍</option>
                        </select>
                        <select id="sel_Strat15_maxvol" class="custom-select" onchange="filterStrat15()">
                            <option value="yes" selected>必須創 60 日最大量</option>
                            <option value="no">不限最大量</option>
                        </select>
                    </div>
                    <div class="count-badge">符合：<span id="count_Strat15">0</span> 檔</div>
                </div>
                <div class="table-container">{html_tb15}</div>
            </div>

            <div id="Strat14" class="tabcontent">
                <div class="info-box">
                    <p>🎯 <b>選股邏輯：</b> 創 20 日新高、量增，且法人（外資、投信）過去 7 天皆無賣出。</p>
                    <div class="filter-container">
                        <select id="sel_Strat14_vol" class="custom-select" onchange="filterStrat14()"><option value="1.5" selected>量增 &le; 1.5 倍</option><option value="2.0">量增 &le; 2.0 倍</option><option value="999">無上限</option></select>
                        <select id="sel_Strat14_fbuy" class="custom-select" onchange="filterStrat14()"><option value="0" selected>外資連買不限</option><option value="1">外資至少買 1 天</option><option value="2">外資連買 2 天</option></select>
                        <select id="sel_Strat14_updays" class="custom-select" onchange="filterStrat14()"><option value="0" selected>不考慮收盤關係</option><option value="1">最新日收高</option></select>
                    </div>
                    <div class="count-badge">符合：<span id="count_Strat14">0</span> 檔</div>
                </div>
                <div class="table-container">{html_tb14}</div>
            </div>

            <div id="Strat13" class="tabcontent">
                <div class="info-box">
                    <p>🎯 <b>選股邏輯：</b> 最新收盤價距離「近60日最高收盤價」在 -2% 以內，且成交量大於 500 張。</p>
                    <div class="filter-container">
                        <select id="sel_Strat13_dist" class="custom-select" onchange="filterStrat13()"><option value="-2" selected>距離高點 &ge; -2%</option><option value="0">創高 (&ge; 0%)</option></select>
                    </div>
                    <div class="count-badge">符合：<span id="count_Strat13">0</span> 檔</div>
                </div>
                <div class="table-container">{html_tb13}</div>
            </div>

            <div id="Strat1" class="tabcontent">
                <div class="info-box">
                    <p>🎯 <b>選股邏輯：</b> 向上跳空不回補、連續量增。</p>
                    <div class="filter-container">
                        <select id="sel_Strat1_vol" class="custom-select" onchange="filterStrat1()"><option value="1.2">量增 &le; 1.2 倍</option><option value="1.5" selected>量增 &le; 1.5 倍</option><option value="999">無上限</option></select>
                        <select id="sel_Strat1_gap" class="custom-select" onchange="filterStrat1()"><option value="1">跳空 &ge; 1 天</option><option value="2">跳空 &ge; 2 天</option><option value="3" selected>跳空 &ge; 3 天</option></select>
                        <select id="sel_Strat1_vol_days" class="custom-select" onchange="filterStrat1()"><option value="1" selected>量增 &ge; 1 天</option><option value="2">量增 &ge; 2 天</option></select>
                    </div>
                    <div class="count-badge">符合：<span id="count_Strat1">0</span> 檔</div>
                </div>
                <div class="table-container">{html_tb1}</div>
            </div>

            <div id="Strat2" class="tabcontent">
                <div class="info-box">
                    <p>🎯 <b>選股邏輯：</b> 階梯式墊高、外資點火、過去七天法人不賣出。</p>
                    <div class="filter-container">
                        <select id="sel_Strat2_step" class="custom-select" onchange="filterStrat2()"><option value="1">墊高 &ge; 1 天</option><option value="2" selected>墊高 &ge; 2 天</option></select>
                        <select id="sel_Strat2_voldays" class="custom-select" onchange="filterStrat2()"><option value="1" selected>量增 &ge; 1 天</option><option value="2">量增 &ge; 2 天</option></select>
                        <select id="sel_Strat2_fbuy" class="custom-select" onchange="filterStrat2()"><option value="0">外資不限</option><option value="1" selected>外資 &ge; 1 天</option></select>
                        <select id="sel_Strat2_vol" class="custom-select" onchange="filterStrat2()"><option value="1.5" selected>最新量增 &le; 1.5 倍</option><option value="999">無上限</option></select>
                    </div>
                    <div class="count-badge">符合：<span id="count_Strat2">0</span> 檔</div>
                </div>
                <div class="table-container">{html_tb2}</div>
            </div>

            <div id="Strat3" class="tabcontent">
                <div class="info-box">
                    <p>🎯 <b>選股邏輯：</b> 最新日量增、投信七日不賣出、外資連續買超。</p>
                    <div class="filter-container">
                        <select id="sel_Strat3_vol" class="custom-select" onchange="filterStrat3()"><option value="1.5" selected>量增 &le; 1.5 倍</option><option value="999">無上限</option></select>
                        <select id="sel_Strat3_fbuy" class="custom-select" onchange="filterStrat3()"><option value="1">外資 &ge; 1 天</option><option value="2">外資連買 2 天</option><option value="3" selected>外資連買 3 天</option></select>
                        <select id="sel_Strat3_updays" class="custom-select" onchange="filterStrat3()"><option value="0" selected>不限收盤</option><option value="1">最新日收高</option></select>
                    </div>
                    <div class="count-badge">符合：<span id="count_Strat3">0</span> 檔</div>
                </div>
                <div class="table-container">{html_tb3}</div>
            </div>

            <div id="Strat4" class="tabcontent">
                <div class="info-box">
                    <p>🎯 <b>選股邏輯：</b> 創 20 日新高、過去七天投信不賣、量大於前一日。</p>
                    <div class="filter-container">
                        <select id="sel_Strat4" class="custom-select" onchange="filterTable('Strat4', this.value)"><option value="1.5" selected>量增 &le; 1.5 倍</option><option value="999">無上限</option></select>
                    </div>
                    <div class="count-badge">符合：<span id="count_Strat4">0</span> 檔</div>
                </div>
                <div class="table-container">{html_tb4}</div>
            </div>

            <div id="Strat5" class="tabcontent"><div class="info-box"><p>🎯 <b>選股邏輯：</b> 旱地拔蔥 (前6天法人0，今日突介入)</p><select id="sel_Strat5" class="custom-select" onchange="filterTable('Strat5', this.value)"><option value="999" selected>量增無上限</option><option value="2.0">量增 &le; 2.0 倍</option></select><div class="count-badge">符合：<span id="count_Strat5">0</span> 檔</div></div><div class="table-container">{html_tb5}</div></div>
            <div id="Strat6" class="tabcontent"><div class="info-box"><p>🎯 <b>選股邏輯：</b> 創五日新高 & 跳空量增 & 法人連三日不賣</p><select id="sel_Strat6" class="custom-select" onchange="filterTable('Strat6', this.value)"><option value="1.5" selected>量增 &le; 1.5 倍</option><option value="999">無上限</option></select><div class="count-badge">符合：<span id="count_Strat6">0</span> 檔</div></div><div class="table-container">{html_tb6}</div></div>
            <div id="Strat7" class="tabcontent"><div class="info-box"><p>🎯 <b>選股邏輯：</b> 創 20 日新高 & 底底高 & 收高量增 & 法人七日不賣</p><select id="sel_Strat7" class="custom-select" onchange="filterTable('Strat7', this.value)"><option value="1.5" selected>量增 &le; 1.5 倍</option><option value="999">無上限</option></select><div class="count-badge">符合：<span id="count_Strat7">0</span> 檔</div></div><div class="table-container">{html_tb7}</div></div>
            <div id="Strat8" class="tabcontent"><div class="info-box"><p>🎯 <b>選股邏輯：</b> 連兩日墊高 & 量能放大 & 法人七日不賣</p><select id="sel_Strat8" class="custom-select" onchange="filterTable('Strat8', this.value)"><option value="1.5" selected>量增 &le; 1.5 倍</option><option value="999">無上限</option></select><div class="count-badge">符合：<span id="count_Strat8">0</span> 檔</div></div><div class="table-container">{html_tb8}</div></div>

            <div id="Strat9" class="tabcontent">
                <div class="info-box">
                    <p>🎯 <b>選股邏輯：</b> 漲幅過濾 + 創高 + 大戶比例 + 法人連七不賣</p>
                    <div class="filter-container">
                        <select id="sel9_pct" class="custom-select" onchange="filterStrat9()"><option value="0">漲幅 > 0%</option><option value="2" selected>漲幅 > 2%</option></select>
                        <select id="sel9_tdcc" class="custom-select" onchange="filterStrat9()"><option value="0">大戶不限</option><option value="40">大戶 > 40%</option></select>
                        <select id="sel9_high" class="custom-select" onchange="filterStrat9()"><option value="none" selected>創高不限</option><option value="20">創 20 日高</option></select>
                    </div>
                    <div class="count-badge">符合：<span id="count_Strat9">0</span> 檔</div>
                </div>
                <div class="table-container">{html_tb9}</div>
            </div>

            <div id="Strat10" class="tabcontent">
                <div class="info-box">
                    <p>🎯 <b>選股邏輯：</b> 限定上櫃股票 | 最新收盤漲幅大於自選百分比。</p>
                    <div class="filter-container">
                        <select id="sel_Strat10_pct" class="custom-select" onchange="filterStrat10()"><option value="1">大於 1%</option><option value="2" selected>大於 2%</option><option value="3">大於 3%</option></select>
                    </div>
                    <div class="count-badge">符合：<span id="count_Strat10">0</span> 檔</div>
                </div>
                <div class="table-container">{html_tb10}</div>
            </div>

            <div id="Strat11" class="tabcontent">
                <div class="info-box">
                    <p>🎯 <b>選股邏輯：</b> 限定上櫃股票 | 實體紅K強勢股 (收盤大於開盤)。</p>
                    <div class="filter-container">
                        <select id="sel_Strat11_k" class="custom-select" onchange="filterStrat11()"><option value="1">紅K > 1%</option><option value="3" selected>紅K > 3%</option></select>
                    </div>
                    <div class="count-badge">符合：<span id="count_Strat11">0</span> 檔</div>
                </div>
                <div class="table-container">{html_tb11}</div>
            </div>
        </div>

        <script>
            function openStrategy(evt, strategyName) {{
                document.querySelectorAll(".tabcontent").forEach(el => el.style.display = "none");
                document.querySelectorAll(".tab-btn").forEach(el => el.classList.remove("active"));
                document.getElementById(strategyName).style.display = "block";
                evt.currentTarget.classList.add("active");
            }}

            function setupFavorites() {{
                document.querySelectorAll('.fav-star').forEach(star => {{
                    if(star.dataset.bound) return;
                    star.dataset.bound = true;

                    star.addEventListener('click', function(e) {{
                        e.stopPropagation();
                        const tr = this.closest('tr');
                        const tbody = tr.parentElement;

                        if (this.classList.contains('active')) {{
                            this.classList.remove('active');
                            this.innerHTML = '☆';
                            tr.classList.remove('row-favorite');
                        }} else {{
                            this.classList.add('active');
                            this.innerHTML = '★';
                            tr.classList.add('row-favorite');
                        }}
                        sortTbody(tbody);
                    }});
                }});
            }}

            function sortTbody(tbody, colIndex = -1, isAscending = false) {{
                const rows = Array.from(tbody.querySelectorAll('tr'));

                rows.sort((rowA, rowB) => {{
                    const favA = rowA.querySelector('.fav-star').classList.contains('active') ? 1 : 0;
                    const favB = rowB.querySelector('.fav-star').classList.contains('active') ? 1 : 0;

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
                    if (header.cellIndex === 0) return;

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

            function filterTable(tabId, maxValStr) {{
                let maxVal = parseFloat(maxValStr);
                let table = document.getElementById(tabId).querySelector("table");
                if (!table) return;

                let headers = table.querySelectorAll("thead th");
                let colIdxs = [];
                headers.forEach((th, idx) => {{ if (th.innerText.includes("增量")) colIdxs.push(idx); }});

                let count = 0;
                table.querySelectorAll("tbody tr").forEach(row => {{
                    let show = true;
                    let cells = row.querySelectorAll("td");
                    for(let i = 0; i < colIdxs.length; i++) {{
                        if (parseFloat(cells[colIdxs[i]].innerText) > maxVal) {{ show = false; break; }}
                    }}
                    row.style.display = show ? "" : "none";
                    if(show) count++;
                }});
                updateVisibleCount(tabId, count);
            }}

            function filterStrat1() {{
                let maxVol = parseFloat(document.getElementById('sel_Strat1_vol').value);
                let minGap = parseInt(document.getElementById('sel_Strat1_gap').value);
                let minVolDays = parseInt(document.getElementById('sel_Strat1_vol_days').value);

                let table = document.getElementById('Strat1').querySelector("table");
                if (!table) return;

                let headers = table.querySelectorAll("thead th");
                let idxVol = -1, idxGap = -1, idxVolDays = -1;
                headers.forEach((th, i) => {{
                    if (th.innerText.includes("增量倍數")) idxVol = i;
                    if (th.innerText.includes("跳空天數")) idxGap = i;
                    if (th.innerText.includes("量增天數")) idxVolDays = i;
                }});

                let count = 0;
                table.querySelectorAll("tbody tr").forEach(row => {{
                    let cells = row.querySelectorAll("td");
                    let show = (parseFloat(cells[idxVol].innerText) <= maxVol) &&
                               (parseInt(cells[idxGap].innerText) >= minGap) &&
                               (parseInt(cells[idxVolDays].innerText) >= minVolDays);
                    row.style.display = show ? "" : "none";
                    if(show) count++;
                }});
                updateVisibleCount('Strat1', count);
            }}

            function filterStrat2() {{
                let maxVol = parseFloat(document.getElementById('sel_Strat2_vol').value);
                let minStep = parseInt(document.getElementById('sel_Strat2_step').value);
                let minVolDays = parseInt(document.getElementById('sel_Strat2_voldays').value);
                let minFBuy = parseInt(document.getElementById('sel_Strat2_fbuy').value);

                let table = document.getElementById('Strat2').querySelector("table");
                if (!table) return;

                let headers = table.querySelectorAll("thead th");
                let idxVol = -1, idxStep = -1, idxVolDays = -1, idxFBuy = -1;
                headers.forEach((th, i) => {{
                    if (th.innerText.includes("增量倍數")) idxVol = i;
                    if (th.innerText.includes("墊高天數")) idxStep = i;
                    if (th.innerText.includes("量增天數")) idxVolDays = i;
                    if (th.innerText.includes("外資連買天數")) idxFBuy = i;
                }});

                let count = 0;
                table.querySelectorAll("tbody tr").forEach(row => {{
                    let cells = row.querySelectorAll("td");
                    let show = (parseFloat(cells[idxVol].innerText) <= maxVol) &&
                               (parseInt(cells[idxStep].innerText) >= minStep) &&
                               (parseInt(cells[idxVolDays].innerText) >= minVolDays) &&
                               (parseInt(cells[idxFBuy].innerText) >= minFBuy);
                    row.style.display = show ? "" : "none";
                    if(show) count++;
                }});
                updateVisibleCount('Strat2', count);
            }}

            function filterStrat3() {{
                let maxVol = parseFloat(document.getElementById('sel_Strat3_vol').value);
                let minFbuy = parseInt(document.getElementById('sel_Strat3_fbuy').value);
                let minUpDays = parseInt(document.getElementById('sel_Strat3_updays').value);

                let table = document.getElementById('Strat3').querySelector("table");
                if (!table) return;

                let headers = table.querySelectorAll("thead th");
                let idxVol = -1, idxFbuy = -1, idxUpDays = -1;
                headers.forEach((th, i) => {{
                    if (th.innerText.includes("增量倍數")) idxVol = i;
                    if (th.innerText.includes("外資連買天數")) idxFbuy = i;
                    if (th.innerText.includes("連漲天數")) idxUpDays = i;
                }});

                let count = 0;
                table.querySelectorAll("tbody tr").forEach(row => {{
                    let cells = row.querySelectorAll("td");
                    let show = (parseFloat(cells[idxVol].innerText) <= maxVol) &&
                               (parseInt(cells[idxFbuy].innerText) >= minFbuy) &&
                               (parseInt(cells[idxUpDays].innerText) >= minUpDays);
                    row.style.display = show ? "" : "none";
                    if(show) count++;
                }});
                updateVisibleCount('Strat3', count);
            }}

            function filterStrat14() {{
                let maxVol = parseFloat(document.getElementById('sel_Strat14_vol').value);
                let minFbuy = parseInt(document.getElementById('sel_Strat14_fbuy').value);
                let minUpDays = parseInt(document.getElementById('sel_Strat14_updays').value);

                let table = document.getElementById('Strat14').querySelector("table");
                if (!table) return;

                let headers = table.querySelectorAll("thead th");
                let idxVol = -1, idxFbuy = -1, idxUpDays = -1;
                headers.forEach((th, i) => {{
                    if (th.innerText.includes("增量倍數")) idxVol = i;
                    if (th.innerText.includes("外資連買天數")) idxFbuy = i;
                    if (th.innerText.includes("連漲天數")) idxUpDays = i;
                }});

                let count = 0;
                table.querySelectorAll("tbody tr").forEach(row => {{
                    let cells = row.querySelectorAll("td");
                    let show = (parseFloat(cells[idxVol].innerText) <= maxVol) &&
                               (parseInt(cells[idxFbuy].innerText) >= minFbuy) &&
                               (parseInt(cells[idxUpDays].innerText) >= minUpDays);
                    row.style.display = show ? "" : "none";
                    if(show) count++;
                }});
                updateVisibleCount('Strat14', count);
            }}

            function filterStrat15() {{
                let minVol = parseFloat(document.getElementById('sel_Strat15_vol').value);
                let maxRange = parseFloat(document.getElementById('sel_Strat15_range').value);
                let checkMaxVol = document.getElementById('sel_Strat15_maxvol').value;
                let period = document.getElementById('sel_Strat15_period').value;

                let table = document.getElementById('Strat15').querySelector("table");
                if (!table) return;

                let headers = table.querySelectorAll("thead th");
                let idxVol = -1, idxAmp20 = -1, idxAmp40 = -1, idxAmp60 = -1;
                headers.forEach((th, i) => {{
                    if (th.innerText.includes("增量倍數")) idxVol = i;
                    if (th.innerText.includes("20日震幅")) idxAmp20 = i;
                    if (th.innerText.includes("40日震幅")) idxAmp40 = i;
                    if (th.innerText.includes("60日震幅")) idxAmp60 = i;
                }});

                let count = 0;
                table.querySelectorAll("tbody tr").forEach(row => {{
                    let cells = row.querySelectorAll("td");
                    let isMaxVol = row.getAttribute('data-is-max-vol') === "true"; 
                    let targetAmpIdx = period === "20" ? idxAmp20 : (period === "40" ? idxAmp40 : idxAmp60);
                    let ampVal = parseFloat(cells[targetAmpIdx].innerText.replace(/,/g, '').replace(/%/g, ''));

                    let show = (parseFloat(cells[idxVol].innerText) >= minVol) && (ampVal <= maxRange);
                    if (checkMaxVol === "yes" && !isMaxVol) show = false;
                               
                    row.style.display = show ? "" : "none";
                    if(show) count++;
                }});
                updateVisibleCount('Strat15', count);
            }}
            
            function filterStrat16() {{
                let minVol = parseFloat(document.getElementById('sel_Strat16_vol').value);
                let table = document.getElementById('Strat16').querySelector("table");
                if (!table) return;

                let headers = table.querySelectorAll("thead th");
                let idxVol = -1;
                headers.forEach((th, i) => {{ if (th.innerText.includes("增量倍數")) idxVol = i; }});

                let count = 0;
                table.querySelectorAll("tbody tr").forEach(row => {{
                    let cells = row.querySelectorAll("td");
                    let show = true;
                    if (minVol > 0 && parseFloat(cells[idxVol].innerText) < minVol) show = false;
                    row.style.display = show ? "" : "none";
                    if(show) count++;
                }});
                updateVisibleCount('Strat16', count);
            }}

            function filterStrat9() {{
                let pctMin = parseFloat(document.getElementById('sel9_pct').value);
                let tdccMin = parseFloat(document.getElementById('sel9_tdcc').value);
                let highCond = document.getElementById('sel9_high').value;

                let table = document.getElementById('Strat9').querySelector("table");
                if (!table) return;

                let headers = table.querySelectorAll("thead th");
                let idxPct = -1, idxTdcc = -1, idx5 = -1, idx20 = -1;
                headers.forEach((th, i) => {{
                    if (th.innerText.includes("最新漲幅")) idxPct = i;
                    if (th.innerText.includes("千張大戶")) idxTdcc = i;
                    if (th.innerText.includes("創5日高")) idx5 = i;
                    if (th.innerText.includes("創20日高")) idx20 = i;
                }});

                let count = 0;
                table.querySelectorAll("tbody tr").forEach(row => {{
                    let cells = row.querySelectorAll("td");
                    let is5High = cells[idx5].innerText.includes("是");
                    let is20High = cells[idx20].innerText.includes("是");

                    let valPct = parseFloat(cells[idxPct].innerText.replace(/,/g, '').replace(/%/g, '').replace(/\\+/g, ''));
                    let valTdcc = parseFloat(cells[idxTdcc].innerText.replace(/,/g, ''));

                    let show = (valPct >= pctMin) && (valTdcc >= tdccMin);

                    if (highCond === "5" && !is5High) show = false;
                    if (highCond === "20" && !is20High) show = false;

                    row.style.display = show ? "" : "none";
                    if(show) count++;
                }});
                updateVisibleCount('Strat9', count);
            }}

            function filterStrat10() {{
                let minPct = parseFloat(document.getElementById('sel_Strat10_pct').value);
                let table = document.getElementById('Strat10').querySelector("table");
                if (!table) return;

                let headers = table.querySelectorAll("thead th");
                let idxPct = -1;
                headers.forEach((th, i) => {{ if (th.innerText.includes("最新漲幅")) idxPct = i; }});

                let count = 0;
                table.querySelectorAll("tbody tr").forEach(row => {{
                    let valPct = parseFloat(row.querySelectorAll("td")[idxPct].innerText.replace(/,/g, '').replace(/%/g, '').replace(/\\+/g, ''));
                    let show = valPct >= minPct;
                    row.style.display = show ? "" : "none";
                    if(show) count++;
                }});
                updateVisibleCount('Strat10', count);
            }}

            function filterStrat11() {{
                let minK = parseFloat(document.getElementById('sel_Strat11_k').value);
                let table = document.getElementById('Strat11').querySelector("table");
                if (!table) return;

                let headers = table.querySelectorAll("thead th");
                let idxK = -1;
                headers.forEach((th, i) => {{ if (th.innerText.includes("實體K漲幅")) idxK = i; }});

                let count = 0;
                table.querySelectorAll("tbody tr").forEach(row => {{
                    let valK = parseFloat(row.querySelectorAll("td")[idxK].innerText.replace(/,/g, '').replace(/%/g, '').replace(/\\+/g, ''));
                    let show = valK >= minK;
                    row.style.display = show ? "" : "none";
                    if(show) count++;
                }});
                updateVisibleCount('Strat11', count);
            }}
            
            function filterStrat13() {{
                let minDist = parseFloat(document.getElementById('sel_Strat13_dist').value);
                let table = document.getElementById('Strat13').querySelector("table");
                if (!table) return;

                let headers = table.querySelectorAll("thead th");
                let idxDist = -1;
                headers.forEach((th, i) => {{ if (th.innerText.includes("距離60日高點(%)")) idxDist = i; }});

                let count = 0;
                table.querySelectorAll("tbody tr").forEach(row => {{
                    let valDist = parseFloat(row.querySelectorAll("td")[idxDist].innerText.replace(/,/g, '').replace(/%/g, ''));
                    let show = valDist >= minDist;
                    row.style.display = show ? "" : "none";
                    if(show) count++;
                }});
                updateVisibleCount('Strat13', count);
            }}

            window.addEventListener('DOMContentLoaded', () => {{
                setupFavorites();
                enableTableSorting();
                
                filterStrat1();
                filterStrat2();
                filterStrat3();
                filterStrat14();
                filterStrat16();
                
                filterTable('Strat4', document.getElementById('sel_Strat4').value);
                filterTable('Strat5', document.getElementById('sel_Strat5').value);
                filterTable('Strat6', document.getElementById('sel_Strat6').value);
                filterTable('Strat7', document.getElementById('sel_Strat7').value);
                filterTable('Strat8', document.getElementById('sel_Strat8').value);
                filterStrat9();
                filterStrat10();
                filterStrat11();
                filterStrat13();
            }});
            
            setTimeout(filterStrat15, 100);
        </script>
    </body>
    </html>
    """

    for code, row in res15.iterrows():
        is_max_vol = "true" if row['創60日最大量'] else "false"
        search_str = f"<tr>\n      <td><span class='fav-star'>☆</span></td>\n      <td>{row['市場']}</td>"
        replace_str = f"<tr data-is-max-vol='{is_max_vol}'>\n      <td><span class='fav-star'>☆</span></td>\n      <td>{row['市場']}</td>"
        html_content = html_content.replace(search_str, replace_str)

    html_filename = "index.html"
    file_path = os.path.abspath(html_filename)
    with open(file_path, "w", encoding="utf-8") as f:
        f.write(html_content)

    print(f"\n✅ 緊湊版網頁已生成: {html_filename}")
    if os.environ.get("GITHUB_ACTIONS") != "true":
        webbrowser.open(f"file:///{file_path}")

if __name__ == "__main__":
    main()
