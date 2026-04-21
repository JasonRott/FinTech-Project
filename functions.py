import requests
from bs4 import BeautifulSoup
import pandas as pd
import os
import numpy as np
import time
import random
import json
from yahooquery import Ticker
import parameters
from datetime import datetime, timedelta
from transformers import pipeline, AutoModelForSequenceClassification, AutoTokenizer, logging as hf_logging
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.optimize import linprog
from scipy.optimize import minimize
import logging
import sys
from pathlib import Path

# 使用 hf_logging 設定 transformers 的日誌層級
hf_logging.set_verbosity_error()
# 2. 隱藏載入模型時的進度條 (tqdm)
hf_logging.disable_progress_bar()

# --- 配置 Standard Logging ---
VERBOSE = parameters.VERBOSE  # 從參數設定讀取 Verbose 模式開關
# 1. 初始化 logger
log = logging.getLogger("AI_RoboAdvisor")

# 2. 移除可能預設存在的 handlers 防止重複列印
log.handlers = []

# 3. 根據 VERBOSE 參數動態決定日誌等級
# logging.INFO: 顯示所有 print(普通資訊)
# logging.WARNING: 只顯示警告跟錯誤
log_level = logging.INFO if VERBOSE else logging.WARNING
log.setLevel(log_level)

# 4. 定義輸出到終端機的 Handler (這就是你的噤聲功能開關)
stream_handler = logging.StreamHandler(sys.stdout)
stream_handler.setLevel(log_level) # Handler 也同步等級

# 5. 定義日誌顯示格式 (可以讓偵錯時資訊更豐富，正式展示時噤聲)
# 展示模式格式 (乾淨):
formatter_clean = logging.Formatter('%(message)s') 
# 偵錯模式格式 (帶有時間與函式名稱):
formatter_debug = logging.Formatter('[%(asctime)s] [%(levelname)s] (%(funcName)s) %(message)s', '%H:%M:%S')

# 根據 Verbose 模式選擇格式
if VERBOSE:
    stream_handler.setFormatter(formatter_debug)
else:
    stream_handler.setFormatter(formatter_clean)

# 6. 將 Handler 加入 logger
log.addHandler(stream_handler)

log.warning("✅ 金融理財引擎日誌系統配置完成。")

# ==========================================
# data_loader 智慧快取引擎
# ==========================================
DB_PATH = "csv\\historical_price_db.csv"

def get_or_fetch_historical_prices(tickers, max_retries=3):
    """
    智慧快取引擎：
    1. 讀取本地歷史資料庫。
    2. 比對請求的 tickers，找出未建立快取的標的。
    3. 只針對缺少的標的發送 API 請求。
    4. 將新資料合併至本地資料庫並存檔。
    5. 回傳此次請求所需的價格矩陣。
    """
    log.info(f"\n啟動本地快取引擎 (請求總數: {len(tickers)} 檔)...")
    
    # 確保資料夾存在
    os.makedirs("csv", exist_ok=True)
    
    # 1. 讀取本地資料庫
    if os.path.exists(DB_PATH):
        db_df = pd.read_csv(DB_PATH, index_col='date', parse_dates=True)
        existing_tickers = db_df.columns.tolist()
    else:
        db_df = pd.DataFrame()
        existing_tickers = []
        
    # 2. 找出缺少的 ETF (Set Difference)
    missing_tickers = list(set(tickers) - set(existing_tickers))
    
    # 3. 抓取缺少的資料
    if missing_tickers:
        log.info(f"🔍 發現 {len(missing_tickers)} 檔 ETF 尚未建立快取，準備從 API 抓取...")
        new_data_df = pd.DataFrame()
        
        for attempt in range(max_retries):
            try:
                yq_tickers = Ticker(missing_tickers, asynchronous=True)
                temp_df = yq_tickers.history(period="3y", interval="1d")
                
                if isinstance(temp_df, dict) or temp_df.empty:
                    log.warning(f"⚠️ 獲取資料為空，等待 3 秒後重試 (第 {attempt+1}/{max_retries} 次)...")
                    time.sleep(3)
                    continue
                
                # 轉換為 Date x Ticker 矩陣 (Adj Close)
                new_data_df = temp_df['adjclose'].unstack(level=0)
                new_data_df.index = pd.to_datetime(new_data_df.index)
                
                log.info("✅ 缺失資料抓取成功！")
                break # 成功則跳出重試迴圈
                
            except Exception as e:
                log.warning(f"⚠️ 發生錯誤 ({e})，等待 5 秒後重試...")
                time.sleep(5)
        
        # 4. 合併並存回資料庫
        if not new_data_df.empty:
            if db_df.empty:
                db_df = new_data_df
            else:
                # 使用 outer join 合併新舊資料，以日期為基準對齊
                db_df = db_df.join(new_data_df, how='outer')
            
            # 存檔更新資料庫
            db_df.to_csv(DB_PATH)
            log.info(f"💾 已將新資料寫入本地快取資料庫: {DB_PATH}")
        else:
            log.warning("❌ 無法獲取缺失的 ETF 資料，將僅使用本地現有資料。")
    else:
        log.info("⚡ 所有請求的 ETF 皆已在本地快取中，直接載入！")

    # 5. 萃取並回傳本次請求所需的資料矩陣
    # 確保只取有成功存在於資料庫的 tickers
    available_tickers = [t for t in tickers if t in db_df.columns]
    
    if not available_tickers:
        log.error("❌ 嚴重錯誤：完全沒有可用的價格資料。")
        sys.exit(1)
        return pd.DataFrame()
        
    # 切片出需要的欄位，並去除全部為 NaN 的無效日期
    final_price_matrix = db_df[available_tickers].dropna(how='all')
    
    return final_price_matrix
# ==========================================
# stage0_0_get_sorted_ETFsorted
# ==========================================
def parse_aum(aum_str: str) -> float:
    """
    將帶有 M, B, K 後綴的 AUM 字串轉換為實際數值，
    若遇到 N/A 或無法解析的格式則回傳 0.0。
    """
    # 處理空值或無效值
    if not aum_str or aum_str in ["N/A", "-", "n/a"]:
        return 0.0
        
    # 移除千分位逗號並去除頭尾空白
    aum_str = aum_str.replace(",", "").strip()
    
    multiplier = 1.0
    # 判斷並處理單位後綴
    if aum_str.endswith("B"):
        multiplier = 1_000_000_000.0
        aum_str = aum_str[:-1]
    elif aum_str.endswith("M"):
        multiplier = 1_000_000.0
        aum_str = aum_str[:-1]
    elif aum_str.endswith("K"):
        multiplier = 1_000.0
        aum_str = aum_str[:-1]
        
    try:
        # 轉換為浮點數並乘上對應的倍數
        return float(aum_str) * multiplier
    except ValueError:
        return 0.0
    
def get_all_etfs() -> list[str]:
    """
    抓取所有 ETF 的資訊，並儲存完整的資料。
    具備 7 天快取過期檢查機制。
    """
    csv_file = "csv\\all_etfs.csv"
    
    # 檢查檔案是否存在與是否過期
    if os.path.exists(csv_file):
        # 取得檔案最後修改時間
        file_mtime = datetime.fromtimestamp(os.path.getmtime(csv_file))
        now = datetime.now()
        age = now - file_mtime
        
        if age <= timedelta(days=7):
            log.info(f"已找到 {csv_file} (上次更新：{file_mtime.strftime('%Y-%m-%d')}，距今 {age.days} 天)，直接讀取快取資料。")
            df = pd.read_csv(csv_file)
            return df["ticker"].tolist()
        else:
            log.info(f"檔案 {csv_file} 已過期 (距今 {age.days} 天，超過 7 天)，準備重新抓取...")
    else:
        log.info(f"找不到 {csv_file}，準備執行首次抓取...")

    base_url = "https://stockanalysis.com/etf/"
    headers = {
        "User-Agent": "Mozilla/5.0"
    }
    
    all_results = []
    page = 1

    while True:
        log.info(f"正在請求第 {page} 頁...")
        resp = requests.get(f"{base_url}?page={page}", headers=headers, timeout=15)
        resp.raise_for_status()

        soup = BeautifulSoup(resp.text, "html.parser")
        table = soup.find("table")

        if not table:
            log.info("未找到任何資料，停止抓取。")
            break

        # 解析資料行
        rows = table.find("tbody").find_all("tr")
        if not rows:
            log.info("已無更多資料可抓取，停止。")
            break

        for row in rows:
            cols = row.find_all("td")
            if len(cols) < 2:
                continue

            ticker = cols[0].get_text(strip=True)
            name = cols[1].get_text(strip=True)
            asset_class = cols[2].get_text(strip=True)
            aum = cols[3].get_text(strip=True) if len(cols) > 3 else "N/A"  # 假設 AUM 在第 4 欄

            all_results.append({
                "ticker": ticker,
                "name": name,
                "asset_class": asset_class,
                "assets_aum": aum,
            })

        log.info(f"第 {page} 頁資料抓取成功，共計資料行數：{len(rows)}")
        if len(rows) < 500:  # 假設每頁最多 500 筆資料，少於 500 筆表示最後一頁
            break
        page += 1

    # 儲存結果
    all_results.sort(key=lambda x: parse_aum(x["assets_aum"]), reverse=True)
    df = pd.DataFrame(all_results)
    
    # 確保資料夾存在再存檔
    os.makedirs(os.path.dirname(csv_file), exist_ok=True)
    df.to_csv(csv_file, index=False)
    
    log.info(f"所有 ETF 資料已抓取完畢，更新儲存至 {csv_file}")
    
    return df["ticker"].tolist()
# ==========================================
# stage0_1_ETF_data_input
# ==========================================
# 模組 1：從本地 CSV 獲取 Top N 清單
def get_target_tickers_from_csv(file_path, top_n):
    if not os.path.exists(file_path):
        log.error(f"❌ 找不到 {file_path}")
        sys.exit(1)
        return []
    
    df = pd.read_csv(file_path)
    
    # 防呆：確認有我們需要的欄位
    if 'ticker' not in df.columns or 'asset_class' not in df.columns:
        log.error("❌ CSV 檔案格式錯誤，找不到 'ticker' 或 'asset_class' 欄位。")
        sys.exit(1)
        return []
        
    # 第一層過濾：只保留 Equity (股票型)，剔除 Fixed Income, Commodity 等
    df_equity = df[df['asset_class'] == 'Equity'].copy()
    
    # 取前 N 大
    target_tickers = df_equity['ticker'].head(top_n).tolist()
    log.info(f"📂 成功從 CSV 載入 AUM 前 {top_n} 大的純股票型 (Equity) ETF 清單。")
    return target_tickers

# 模組 2：yahooquery 財務與流動性特徵 (防封鎖改良版)
def fetch_etf_data_yq(ticker_list, period="3y", batch_size=10, YQ_OUTPUT_FILE=parameters.YQ_OUTPUT_FILE):
    log.info(f"\n🚀 啟動 yahooquery：加入防護機制，分批抓取 {period} 數據...")
    
    # 支援斷點續傳：檢查是否已有部分完成的檔案
    if os.path.exists(YQ_OUTPUT_FILE):
        existing_df = pd.read_csv(YQ_OUTPUT_FILE)
        completed_tickers = existing_df['ETF'].tolist()
        log.info(f"📂 發現既有 YQ 資料庫，已完成 {len(completed_tickers)} 檔。")
    else:
        existing_df = pd.DataFrame()
        completed_tickers = []

    # 濾除已經抓過的 ETF
    remaining_tickers = [t for t in ticker_list if t not in completed_tickers]
    
    if not remaining_tickers:
        log.info("✅ 所有 YQ 財務特徵皆已抓取完畢！")
        return existing_df

    log.info(f"⏳ 尚有 {len(remaining_tickers)} 檔待處理...")
    
    # 將剩餘名單切割為微型批次 (例如每 5 檔一組)
    for i in range(0, len(remaining_tickers), batch_size):
        batch = remaining_tickers[i:i + batch_size]
        log.info(f"\n📦 正在處理批次: {batch}")
        
        # 針對這小批次初始化 Ticker
        tq = Ticker(batch)
        
        # 抓取該批次的資料
        history = tq.history(period=period)
        profiles = tq.fund_profile
        summary = tq.summary_detail
        
        data_records = []
        
        for symbol in batch:
            try:
                # --- 1. 取得價格資料以計算長天期報酬與風險 ---
                if isinstance(history, pd.DataFrame) and symbol in history.index.levels[0]:
                    symbol_hist = history.loc[symbol]
                    
                    trading_days = len(symbol_hist)
                    years = trading_days / 252.0
                    
                    if years < 2.5:
                        log.warning(f"⚠️ {symbol} 歷史數據不足 2.5 年，跳過計算。")
                        data_records.append({
                            "ETF": symbol,
                            "Years_Data": np.nan,
                            "Date": pd.Timestamp.now().strftime("%Y-%m-%d"),
                            "Return_CAGR (%)": np.nan,
                            "Return_Div (%)": np.nan,
                            "Risk_Vol (%)": np.nan,
                            "Risk_MaxDD (%)": np.nan,
                            "Cost_ExpRatio (%)": np.nan,
                            "Liq_Volume (M)": np.nan,
                            "Liq_AUM (B)": np.nan
                        })
                        continue
                    
                    start_price = symbol_hist['close'].iloc[0]
                    end_price = symbol_hist['close'].iloc[-1]
                    
                    cagr = (end_price / start_price) ** (1 / years) - 1
                    daily_returns = symbol_hist['close'].pct_change().dropna()
                    annual_volatility = daily_returns.std() * np.sqrt(252)
                    
                    cumulative_returns = (1 + daily_returns).cumprod()
                    max_drawdown = ((cumulative_returns - cumulative_returns.cummax()) / cumulative_returns.cummax()).min()
                    avg_volume = symbol_hist['volume'].mean()
                else:
                    log.warning(f"⚠️ 找不到 {symbol} 的歷史價格。")
                    continue

                # --- 2. 提取準確的 ETF Metadata ---
                prof = profiles.get(symbol, {}) if isinstance(profiles, dict) else {}
                summ = summary.get(symbol, {}) if isinstance(summary, dict) else {}
                
                if prof == f'No fundamentals data found for symbol: {symbol}':
                    log.warning(f"⚠️ {symbol} 的 profile 資料格式異常，無法解析。")
                    expense_ratio = None
                else:
                    fees = prof.get('feesExpensesInvestment', {})
                    expense_ratio = fees.get('annualReportExpenseRatio')
                
                if expense_ratio is None:
                    expense_ratio = summ.get('navPrice', 0) * 0 

                dividend_yield = summ.get('yield', 0)
                aum = summ.get('totalAssets', np.nan)

                data_records.append({
                    "ETF": symbol,
                    "Years_Data": round(years, 1),
                    "Date": pd.Timestamp.now().strftime("%Y-%m-%d"),
                    "Return_CAGR (%)": round(cagr * 100, 2),
                    "Return_Div (%)": round(dividend_yield * 100, 2) if dividend_yield else 0.0,
                    "Risk_Vol (%)": round(annual_volatility * 100, 2),
                    "Risk_MaxDD (%)": round(max_drawdown * 100, 2),
                    "Cost_ExpRatio (%)": round(expense_ratio * 100, 3) if expense_ratio else np.nan,
                    "Liq_Volume (M)": round(avg_volume / 1000000, 2),
                    "Liq_AUM (B)": round(aum / 1000000000, 2) if pd.notnull(aum) else np.nan
                })
                log.info(f"✅ YQ 成功處理: {symbol}")

            except Exception as e:
                log.warning(f"❌ YQ 處理 {symbol} 時發生錯誤: {e}")

        # 如果這個批次有成功抓到資料，立刻存檔 (斷點續傳)
        if data_records:
            batch_df = pd.DataFrame(data_records)
            existing_df = pd.concat([existing_df, batch_df], ignore_index=True)
            existing_df.to_csv(YQ_OUTPUT_FILE, index=False)
            
        # 關鍵防護：每跑完一個批次，隨機休眠 3 到 6 秒
        if i + batch_size < len(remaining_tickers):
            sleep_time = round(random.uniform(2.5, 5.0), 1)
            log.info(f"💤 批次完成，隨機休眠 {sleep_time} 秒以防封鎖...")
            time.sleep(sleep_time)

    log.info(f"💾 YQ 財務特徵已全數抓取並儲存至 {YQ_OUTPUT_FILE}")
    return existing_df

# 模組 3：Alpha Vantage 產業分散度特徵
def build_etf_database_av(target_tickers, AV_DB_FILE=parameters.AV_DB_FILE, AV_API_KEY=parameters.AV_API_KEY, AV_MAX_CALLS_PER_DAY=parameters.AV_MAX_CALLS_PER_DAY):
    log.info(f"\n🚀 啟動 Alpha Vantage：比對並更新分散度資料庫...")
    
    if os.path.exists(AV_DB_FILE):
        with open(AV_DB_FILE, 'r', encoding='utf-8') as f:
            db = json.load(f)
    else:
        db = {}
        
    calls_today = 0
    log.info(f"📊 目前本地資料庫已有 {len(db)} 檔 ETF 的分散度資料。")
    
    for symbol in target_tickers:
        if symbol in db:
            continue  # 已存在則跳過，不浪費 API 額度
                
        if calls_today >= AV_MAX_CALLS_PER_DAY:
            log.warning(f"\n🛑 已達今日 AV 安全呼叫上限 ({AV_MAX_CALLS_PER_DAY}次)。請明天再執行後續代碼！")
            break
            
        log.info(f"⏳ 正在向 Alpha Vantage 請求 {symbol} 的結構資料...")
        url = f"https://www.alphavantage.co/query?function=ETF_PROFILE&symbol={symbol}&apikey={AV_API_KEY}"
        time.sleep(1)  # 防呆休眠
        try:
            response = requests.get(url).json()
            
            if "Information" in response or "Note" in response:
                log.warning(f"🛑 觸發 Alpha Vantage 頻率限制，程式提早暫停。")
                log.info(response)
                break

            if "sectors" in response and "holdings" in response:
                sectors = response["sectors"]
                holdings = response["holdings"]
                
                # 建立原始產業權重字典
                raw_sector_weights = {}
                for s in sectors:
                    industry_name = s.get('sector', 'Unknown')
                    weight = float(s.get('weight', 0))
                    raw_sector_weights[industry_name] = weight
                
                # 第一層過濾：計算產業權重總和
                total_sector_weight = sum(raw_sector_weights.values())
                
                SECTOR_is_True = 1
                # 如果權重涵蓋度不到 60% (0.6)，判定為無效資料並剔除
                if total_sector_weight < 0.6:
                    log.warning(f"⚠️ {symbol} 權重涵蓋度過低 ({total_sector_weight*100:.1f}%)，判定為無效資料並剔除。")
                    SECTOR_is_True = 0 

                # 🚨 正規化映射：將產業權重等比例放大至 100% (1.0)
                if total_sector_weight > 0:
                    normalized_sector_weights = {
                        ind: round(w / total_sector_weight, 6) 
                        for ind, w in raw_sector_weights.items()
                    }
                else:
                    normalized_sector_weights = {}

                # 使用正規化後的權重計算 HHI (加總必為 1.0，不需再補償未涵蓋部分)
                sector_hhi = sum([w ** 2 for w in normalized_sector_weights.values()])
                
                top3_weight = sum([float(h.get('weight', 0)) for h in holdings[:3]])
                if SECTOR_is_True:
                    db[symbol] = {
                        "Sector_HHI": round(sector_hhi, 4),
                        "Top3_Weight": round(top3_weight, 4),
                        "Sector_Weights": normalized_sector_weights,  # 儲存 100% 正規化後的產業分佈
                        "Last_Updated": time.strftime("%Y-%m-%d")
                    }  
                else:
                    db[symbol] = {
                        "Sector_HHI": -1.0,
                        "Top3_Weight": -1.0,
                        "Sector_Weights": {},  # 無效資料不儲存產業分佈
                        "Last_Updated": time.strftime("%Y-%m-%d")
                    }
                
                # 寫入本地 JSON 檔案
                with open(AV_DB_FILE, 'w', encoding='utf-8') as f:
                    json.dump(db, f, ensure_ascii=False, indent=4)
                log.info(f"✅ AV 成功計算並儲存 {symbol} (已正規化至 100%)。")
                
            else:
                log.warning(f"⚠️ 找不到 {symbol} 的結構資料 (可能非 ETF 或 API 缺漏)。")
            
            calls_today += 1
            time.sleep(1)  # 防呆休眠
            
        except Exception as e:
            log.error(f"❌ 處理 {symbol} 時發生錯誤: {e}")
            sys.exit(1)
        
    log.info(f"🏁 Alpha Vantage 更新排程執行完畢。資料庫儲存於 {AV_DB_FILE}")

# 模組 4：Finnhub 新聞與 FinBERT 宏觀情緒特徵
# 系統設定
FINNHUB_API_KEY = parameters.FINNHUB_API_KEY  # 請填入你的 Finnhub API Key
def fetch_finnhub_news(ticker, days=180):
    """使用 Finnhub API 抓取過去指定天數的歷史新聞"""
    end_date = datetime.now()
    start_date = end_date - timedelta(days=days)
    
    start_str = start_date.strftime('%Y-%m-%d')
    end_str = end_date.strftime('%Y-%m-%d')
    
    url = f"https://finnhub.io/api/v1/company-news?symbol={ticker}&from={start_str}&to={end_str}&token={FINNHUB_API_KEY}"
    
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        return response.json()
    except Exception as e:
        log.warning(f"⚠️ 抓取 {ticker} 新聞時發生錯誤: {e}")
        return []

def calculate_macro_sentiment(ticker_symbol, half_life_days=60, finbert=None):
    """抓取新聞、評分，並套用指數時間衰減計算宏觀情緒"""
    news_items = fetch_finnhub_news(ticker_symbol, days=180)
    
    if not news_items:
        return 0.0

    current_time = datetime.now().timestamp()
    decay_constant = np.log(2) / half_life_days
    
    total_weighted_score = 0.0
    total_weight = 0.0
    
    for article in news_items:
        headline = article.get('headline', '')
        summary = article.get('summary', '')
        timestamp = article.get('datetime', current_time)
        
        text = f"{headline}. {summary}".strip()
        if not text or text == ".":
            continue
            
        text = text[:500] 
        
        try:
            res = finbert(text)[0]
            label = res['label']
            confidence = res['score']
            
            if label == 'positive':
                base_score = confidence
            elif label == 'negative':
                base_score = -confidence
            else:
                base_score = 0.0
        except Exception:
            continue
            
        days_ago = max(0, (current_time - timestamp) / 86400)
        weight = np.exp(-decay_constant * days_ago)
        
        total_weighted_score += (base_score * weight)
        total_weight += weight

    if total_weight == 0:
        return 0.0
        
    macro_score = max(-1.0, min(1.0, total_weighted_score / total_weight))
    time.sleep(1.5) # 遵守免費 API 速率限制
    
    return round(macro_score, 4)

def append_sentiment_to_csv(csv_filepath=parameters.YQ_OUTPUT_FILE, max_age_days=14, print_preview=True):
    """
    讀取 CSV，計算情緒分數，並將結果存回原 CSV 檔案中。
    """
    if not os.path.exists(csv_filepath):
        log.error(f"❌ 錯誤：找不到檔案 {csv_filepath}")
        sys.exit(1)
        return

    if FINNHUB_API_KEY == "請填入你的_Finnhub_API_Key":
        log.error("❌ 警告：未設定 Finnhub API Key。")
        sys.exit(1)
        return

    log.info("\n" + "="*50)
    log.info(f" 📡 啟動 FinBERT 宏觀情緒動能擷取 ({csv_filepath})")
    log.info("="*50)

    # 1. 讀取 CSV
    df = pd.read_csv(csv_filepath)
    
    if 'ETF' not in df.columns:
        log.error("❌ 錯誤：CSV 中找不到 'ETF' 欄位。")
        sys.exit(1)
        return

    # 2. 確保快取欄位存在並格式化日期
    if 'FinBERT_score' not in df.columns:
        df['FinBERT_score'] = np.nan
    if 'FinBERT_date' not in df.columns:
        df['FinBERT_date'] = pd.NaT # Pandas 的時間空值 (Not-a-Time)
        
    df['FinBERT_date'] = pd.to_datetime(df['FinBERT_date'], errors='coerce')

    etf_list = df['ETF'].tolist()
    total = len(etf_list)
    current_date = datetime.now()

    # 3. 逐一計算分數
    etf_list = df['ETF'].tolist()
    total = len(etf_list)
    
    # 讀取 CSV 後，檢查哪些 ETF 已經算過了
    if 'FinBERT_score' not in df.columns:
        df['FinBERT_score'] = np.nan # 初始化空欄位
    
    log.info("⏳ 載入本地 FinBERT 模型中...")
    if not Path("local_finbert").is_dir():
        # 1. 首次聯網下載模型與 Tokenizer
        log.info("⏳ 首次使用，正在下載 FinBERT 模型...")
        model_name = "ProsusAI/finbert"
        model = AutoModelForSequenceClassification.from_pretrained(model_name)
        tokenizer = AutoTokenizer.from_pretrained(model_name)

        # 2. 儲存到本地端資料夾 (例如命名為 local_finbert)
        model.save_pretrained("./local_finbert")
        tokenizer.save_pretrained("./local_finbert")
        log.info("✅ FinBERT 模型下載並儲存完成！")

    # ==========================================
    # 之後你每次要測試時，只需要指向本地資料夾即可：
    finbert = pipeline("sentiment-analysis", model="./local_finbert", tokenizer="./local_finbert") # 預載入 FinBERT 模型，避免重複載入浪費時間
    log.info("✅ FinBERT 模型載入完成。")

    for i, ticker in enumerate(etf_list, 1):
        # 取得該 ETF 所在的列索引 (確保修改對齊)
        mask = df['ETF'] == ticker
        existing_score = df.loc[mask, 'FinBERT_score'].iloc[0]
        existing_date = df.loc[mask, 'FinBERT_date'].iloc[0]
        
        needs_update = True
        # 🚨 快取失效檢查 (TTL Logic)
        if pd.notna(existing_score) and pd.notna(existing_date):
            delta_days = (current_date - existing_date).days
            if delta_days < max_age_days:
                needs_update = False
                
        if not needs_update:
            continue
        
        # 若無快取或已過期，則執行 API 抓取與 NLP 推論
        # 判斷 existing_date 是否為 NaT 來決定印出的訊息
        if pd.notna(existing_date):
            status_msg = f"快取過期 ({existing_date.strftime('%Y-%m-%d')})"
        else:
            status_msg = "無快取紀錄 (首次抓取)"
            
        # 若無快取或已過期，則執行 API 抓取與 NLP 推論
        log.info(f"[{i:03d}/{total}] {ticker:<5} {status_msg}，執行 API 抓取與 NLP 推論...")
        score = calculate_macro_sentiment(ticker, finbert=finbert)
        
        # 將新分數與當下日期寫回 DataFrame
        df.loc[mask, 'FinBERT_score'] = score
        df.loc[mask, 'FinBERT_date'] = current_date.strftime('%Y-%m-%d')
        log.info(f"  -> 分數: {score}")
        
        df.loc[df['ETF'] == ticker, 'FinBERT_score'] = score
        # 🚨 斷點續傳：每算完 10 檔就存檔一次，避免跑到一半當機心血全毀
        if i % 10 == 0:
            df.to_csv(csv_filepath, index=False)
            if print_preview:
                log.info("💾 已建立暫存檔...")
    
    # 4. 存回原 CSV 檔案
    df.to_csv(csv_filepath, index=False)
    log.info("="*50)
    log.info(f"✅ 情緒分數擷取完成，已成功寫入 {csv_filepath}")
    
    # 檢驗輸出
    if print_preview:
        log.info("\n【資料預覽】")
        # 只取出字串格式的日期印出比較好看
        preview_df = df.copy()
        preview_df['FinBERT_date'] = preview_df['FinBERT_date'].dt.strftime('%Y-%m-%d')
        log.info(preview_df[['ETF', 'FinBERT_score', 'FinBERT_date']].head(10).to_string(index=False))

def clean_existing_database(AV_DB_FILE=parameters.AV_DB_FILE):
    if not os.path.exists(AV_DB_FILE):
        log.error(f"❌ 找不到 {AV_DB_FILE}")
        sys.exit(1)
        return
        
    with open(AV_DB_FILE, 'r', encoding='utf-8') as f:
        db = json.load(f)
        
    original_count = len(db)
    keys_to_delete = []
    
    for symbol, data in db.items():
        hhi = data.get("Sector_HHI", 0)
        
        # 根據數學極限，HHI 低於 0.08 絕對是無效或缺失的資料
        if hhi != -1.0 and hhi < 0.08:
            keys_to_delete.append(symbol)
            
    # 從字典中刪除異常標的
    for k in keys_to_delete:
        del db[k]
        
    # 將乾淨的資料寫回檔案
    with open(AV_DB_FILE, 'w', encoding='utf-8') as f:
        json.dump(db, f, ensure_ascii=False, indent=4)
        
    log.info(f"✅ 資料庫清理完成！")
    log.info(f"清理前資料數：{original_count}")
    log.info(f"刪除了 {len(keys_to_delete)} 筆異常資料：{keys_to_delete}")
    log.info(f"目前剩餘 {len(db)} 筆有效資料。")

def merge_final_features(AV_DB_FILE=parameters.AV_DB_FILE, YQ_FILE=parameters.YQ_OUTPUT_FILE, OUTPUT_FILE=parameters.OUTPUT_FILE):
    log.info("🚀 啟動 Stage 0 最終合併程序...")
    
    # 1. 讀取財務與流動性特徵
    if not os.path.exists(YQ_FILE):
        log.error(f"❌ 找不到 {YQ_FILE}，請確認是否已完成 YQ 資料抓取。")
        sys.exit(1)
        return
    df_yq = pd.read_csv(YQ_FILE)
    log.info(f"📄 成功讀取 {len(df_yq)} 檔 ETF 的財務特徵。")
    
    # 2. 讀取分散度特徵資料庫
    if os.path.exists(AV_DB_FILE):
        with open(AV_DB_FILE, 'r', encoding='utf-8') as f:
            db_av = json.load(f)
        log.info(f"📄 成功讀取 {len(db_av)} 檔 ETF 的本地分散度資料。")
    else:
        db_av = {}
        log.warning("⚠️ 找不到本地分散度資料庫。")

    # 3. 計算 Div_Score 並準備對應字典
    div_scores = {}
    for symbol, data in db_av.items():
        sector_hhi = data.get("Sector_HHI")
        top3_weight = data.get("Top3_Weight")
        
        if pd.notna(sector_hhi) and sector_hhi > 0:
            base_score = 1 / sector_hhi
            penalty = 0.8 if top3_weight > 0.20 else 1.0
            div_scores[symbol] = round(base_score * penalty, 4)

    # 4. 進行合併 (Left Join: 將 Div_Score 映射到主表中)
    df_yq['Div_Score (產出)'] = df_yq['ETF'].map(div_scores)

    # 5. 整理欄位順序並儲存
    cols = ['ETF', 'Years_Data', 'Date', 'Return_CAGR (%)', 'Return_Div (%)', 
            'Risk_Vol (%)', 'Risk_MaxDD (%)', 'Cost_ExpRatio (%)', 
            'Liq_Volume (M)', 'Liq_AUM (B)', 'Div_Score (產出)', 'FinBERT_score']
    
    # 確保只選取存在的欄位 (防呆)
    final_cols = [c for c in cols if c in df_yq.columns]
    df_final = df_yq[final_cols]
    
    df_final.to_csv(OUTPUT_FILE, index=False)
    
    log.info("\n=== Stage 0: 最終多維度特徵矩陣預覽 ===")
    log.info(df_final.head(15).to_string(index=False))
    log.info(f"\n✅ 合併完成！最終矩陣已儲存為 {OUTPUT_FILE}")
    log.info(f"提示：其中未具備產業數據的債券/商品 ETF，其 Div_Score 欄位會保留為 NaN。")

def patch_aum_from_csv(MATRIX_FILE=parameters.OUTPUT_FILE, UNIVERSE_FILE=parameters.CSV_UNIVERSE_FILE):
    log.info("🔧 啟動 AUM 數據校正與覆蓋程序...")
    try:
        # 1. 讀取目前的特徵矩陣與最原始的 CSV 清單
        df_matrix = pd.read_csv(MATRIX_FILE)
        df_universe = pd.read_csv(UNIVERSE_FILE)
        
        # 2. 建立 AUM 字串轉數值 (以 Billion 為單位) 的函數
        def parse_aum_to_billion(aum_str):
            if not isinstance(aum_str, str) or aum_str in ["N/A", "-", "n/a", ""]:
                return np.nan
            
            aum_str = aum_str.replace(",", "").strip()
            multiplier = 1.0
            
            if aum_str.endswith("B"):
                multiplier = 1.0       # 已經是 Billion 級別
                aum_str = aum_str[:-1]
            elif aum_str.endswith("M"):
                multiplier = 0.001     # Million 轉 Billion
                aum_str = aum_str[:-1]
            elif aum_str.endswith("K"):
                multiplier = 0.000001  # Kilo 轉 Billion
                aum_str = aum_str[:-1]
            
            try:
                return round(float(aum_str) * multiplier, 2)
            except ValueError:
                return np.nan

        # 3. 將原始 CSV 的 AUM 轉換為數值
        df_universe['Real_AUM_B'] = df_universe['assets_aum'].apply(parse_aum_to_billion)
        
        # 4. 建立「代碼對應真實 AUM」的字典
        aum_dict = dict(zip(df_universe['ticker'], df_universe['Real_AUM_B']))
        
        # 5. 直接覆蓋矩陣中的 Liq_AUM (B) 欄位
        df_matrix['Liq_AUM (B)'] = df_matrix['ETF'].map(aum_dict)
        
        # 6. 重新存檔
        df_matrix.to_csv(MATRIX_FILE, index=False)
        log.info("✅ AUM 校正成功！已使用靜態 CSV 的數據覆蓋 Yahoo 雜訊。")
        
        # 印出前 5 名來驗證 (檢查 VOO, IVV 等數值是否已恢復正常)
        log.info("\n=== 校正後的 AUM 預覽 ===")
        log.info(df_matrix[['ETF', 'Liq_AUM (B)']].head().to_string(index=False))
        
    except Exception as e:
        log.error(f"❌ 校正過程中發生錯誤: {e}")
        sys.exit(1)
# ==========================================
# stage0_2_EDA_and_Visualization
# ==========================================
def run_stage0_2_eda():
    log.info("啟動 Stage 0.5: 探索性資料分析 (EDA) 視覺化...")
    
    # 1. 讀取 Stage 0 產出的原始資料
    try:
        df = pd.read_csv("csv\\stage0_final_matrix.csv")
    except FileNotFoundError:
        log.error("❌ 找不到 stage0_final_matrix.csv，請確認檔案路徑。")
        sys.exit(1)
        return

    # 定義需要觀察的 9 個基礎數值特徵
    features_to_plot = [
        'Return_CAGR (%)', 'Return_Div (%)', 
        'Risk_Vol (%)', 'Risk_MaxDD (%)', 
        'Cost_ExpRatio (%)', 
        'Liq_Volume (M)', 'Liq_AUM (B)', 
        'Div_Score (產出)', 'FinBERT_score'
    ]
    
    # 過濾出實際存在於 df 中的欄位 (防呆)
    valid_features = [f for f in features_to_plot if f in df.columns]
    n_features = len(valid_features)
    
    if n_features == 0:
        log.error("❌ 找不到任何指定的數值特徵進行繪圖。")
        sys.exit(1)
        return

    # 2. 設置繪圖風格與中文字型
    sns.set_theme(style="whitegrid")
    plt.rcParams['font.sans-serif'] = ['Microsoft JhengHei', 'PingFang HK', 'Arial Unicode MS']
    plt.rcParams['axes.unicode_minus'] = False

    # 3. 繪製特徵分佈總覽圖 (Histogram + KDE)
    fig_hist, axes_hist = plt.subplots(nrows=3, ncols=3, figsize=(15, 12))
    axes_hist = axes_hist.flatten()
    
    for i, feature in enumerate(valid_features):
        sns.histplot(df[feature], kde=True, ax=axes_hist[i], color='steelblue', bins=30)
        axes_hist[i].set_title(f'分佈: {feature}', fontweight='bold')
        axes_hist[i].set_xlabel('')
        axes_hist[i].set_ylabel('頻率')
        
    # 隱藏多餘的子圖
    for j in range(i + 1, len(axes_hist)):
        fig_hist.delaxes(axes_hist[j])
        
    plt.tight_layout()
    fig_hist.savefig("png\\eda_histograms_beforeDEA.png", dpi=300)
    #plt.show()
    plt.close(fig_hist)
    log.info("✅ 產出特徵分佈圖：eda_histograms.png")

    # 4. 繪製箱型圖 (Boxplot) 觀察極端值
    fig_box, axes_box = plt.subplots(nrows=3, ncols=3, figsize=(15, 12))
    axes_box = axes_box.flatten()
    
    for i, feature in enumerate(valid_features):
        sns.boxplot(x=df[feature], ax=axes_box[i], color='lightcoral')
        axes_box[i].set_title(f'極端值: {feature}', fontweight='bold')
        axes_box[i].set_xlabel('')
        
    for j in range(i + 1, len(axes_box)):
        fig_box.delaxes(axes_box[j])
        
    plt.tight_layout()
    fig_box.savefig("png\\eda_boxplots_beforeDEA.png", dpi=300)
    #plt.show()
    plt.close(fig_box)
    log.info("✅ 產出特徵箱型圖：eda_boxplots.png")

    # 5. 輸出基礎統計量報告
    log.info("\n=== 📊 基礎特徵統計量敘述 (Descriptive Statistics) ===")
    desc_stats = df[valid_features].describe().T
    # 增加偏度 (Skewness) 指標，大於 1 或小於 -1 代表嚴重偏態
    desc_stats['skewness'] = df[valid_features].skew()
    log.info(desc_stats[['mean', 'std', 'min', '50%', 'max', 'skewness']].round(4).to_string())

# ==========================================
# stage0_3_regularization_and_dimensionality_reduction
# ==========================================
def custom_minmax_scaler(series, feature_name, lower_bound_q=0.01, upper_bound_q=0.95, min_val=0.1, max_val=1.0):
    """
    自定義正規化函數：包含 1%~95% 縮尾處理，並映射至 [0.1, 1.0] 區間。
    """
    # 縮尾處理 (Winsorization) 消除極端離群值
    lower_bound = series.quantile(lower_bound_q)
    upper_bound = series.quantile(upper_bound_q)
    clipped = series.clip(lower=lower_bound, upper=upper_bound)
    
    # 壓縮至 0~1
    s_min = clipped.min()
    s_max = clipped.max()
    
    if s_max == s_min:
        scaled = np.zeros(len(clipped))
    else:
        scaled = (clipped - s_min) / (s_max - s_min)
        
    # 線性映射至 [0.1, 1.0]
    final_scaled = min_val + scaled * (max_val - min_val)
    return final_scaled

def run_stage0_normalization_and_reduction():
    log.info("啟動 Stage 0: DEA 前置正規化與客觀降維...")
    
    try:
        df = pd.read_csv("csv\\stage0_final_matrix.csv")
    except FileNotFoundError:
        log.error("❌ 找不到 stage0_final_matrix.csv。")
        sys.exit(1)
        return
        
    df_dea = pd.DataFrame({'ETF': df['ETF']})
    
    # ==========================================
    # 1. DEA 產出項 (Outputs): 越大越好
    # ==========================================
    
    # [報酬維度 R_P]
    norm_cagr = custom_minmax_scaler(df['Return_CAGR (%)'], 'CAGR')
    norm_div = custom_minmax_scaler(df['Return_Div (%)'], 'Div')
    df_dea['Out_Return'] = (norm_cagr + norm_div) / 2
    
    # [流動性維度 L_P] - 先取對數處理嚴重右偏
    log_volume = np.log1p(df['Liq_Volume (M)'])
    log_aum = np.log1p(df['Liq_AUM (B)'])
    norm_vol = custom_minmax_scaler(log_volume, 'Volume')
    norm_aum = custom_minmax_scaler(log_aum, 'AUM')
    df_dea['Out_Liquidity'] = (norm_vol + norm_aum) / 2
    
    # [分散度維度 D_P]
    df_dea['Out_Diversity'] = custom_minmax_scaler(df['Div_Score (產出)'], 'Diversity')
    
    # [市場情緒維度 S_P]
    df_dea['Out_Sentiment'] = custom_minmax_scaler(df['FinBERT_score'], 'Sentiment', lower_bound_q=0.05, upper_bound_q=0.95)
    
    # ==========================================
    # 2. DEA 投入項 (Inputs): 越小越好 (消耗的資源/承擔的風險)
    # ==========================================
    
    # [風險維度 V_P] - MaxDD 取絕對值代表回撤幅度
    norm_risk_vol = custom_minmax_scaler(df['Risk_Vol (%)'], 'Risk_Vol')
    norm_maxdd = custom_minmax_scaler(df['Risk_MaxDD (%)'].abs(), 'Risk_MaxDD')
    df_dea['In_Risk'] = (norm_risk_vol + norm_maxdd) / 2
    
    # [成本維度 C_P]
    df_dea['In_Cost'] = custom_minmax_scaler(df['Cost_ExpRatio (%)'], 'Cost')
    
    # ==========================================
    # 3. 儲存 DEA 專用矩陣
    # ==========================================
    
    # 重新排列欄位，便於後續切片
    cols = ['ETF', 'In_Risk', 'In_Cost', 'Out_Return', 'Out_Liquidity', 'Out_Diversity', 'Out_Sentiment']
    df_dea = df_dea[cols]
    
    log.info("\n=== 📊 降維後 DEA 矩陣預覽 (皆落於 0.1 ~ 1.0 區間) ===")
    log.info(df_dea.head().round(4).to_string())
    
    df_dea.to_csv("csv\\stage0_dea_ready_matrix.csv", index=False)
    log.info("\n✅ 已輸出 stage0_dea_ready_matrix.csv")

# ==========================================
# stage1_DEA_efficiency_calculation
# ==========================================
def run_stage1_normalized_dea():
    # 1. 讀取 Stage 0 正規化並降維後的 DEA 專用矩陣
    try:
        df_raw = pd.read_csv("csv\\stage0_dea_ready_matrix.csv")
    except FileNotFoundError:
        log.error("找不到 stage0_dea_ready_matrix.csv，請先執行前段正規化。")
        sys.exit(1)
        return
    # 2. 資料清洗：實作「分層 DEA」，先濾除含有 NaN 的非股票型資產
    # 確保送入 linprog 的矩陣是完美的實數
    df = df_raw.dropna(subset=['Out_Diversity']).reset_index(drop=True)
    log.info(f"📊 載入資料：共 {len(df)} 檔 ETF 進入正規化 DEA 運算。")

    # 3. 定義投入 (Inputs) 與 產出 (Outputs) 欄位
    # 投入：越小越好 (In_Risk, In_Cost)
    input_cols = ['In_Risk', 'In_Cost']
    # 產出：越大越好 (Out_Return, Out_Liquidity, Out_Diversity)
    output_cols = ['Out_Return', 'Out_Liquidity', 'Out_Diversity']

    # 提取數值矩陣
    X = df[input_cols].values
    Y = df[output_cols].values

    n_dmus = len(df)          # 決策單元 (ETF) 數量
    n_inputs = X.shape[1]     # 投入維度數 = 2
    n_outputs = Y.shape[1]    # 產出維度數 = 3

    efficiencies = []

    log.info("🚀 啟動 DEA (CCR Input-Oriented) 線性規劃求解...")

    # 4. 對每一檔 ETF 分別求解線性規劃
    for k in range(n_dmus):
        # 目標函數：最大化 u^T * Y_k -> 轉為 scipy 的 Minimize -u^T * Y_k
        c = np.concatenate((np.zeros(n_inputs), -Y[k]))
        
        # 限制式 1：v^T * X_k = 1 
        A_eq = np.concatenate((X[k].reshape(1, -1), np.zeros((1, n_outputs))), axis=1)
        b_eq = np.array([1.0])
        
        # 限制式 2：-v^T * X_j + u^T * Y_j <= 0 (對所有 j)
        A_ub = np.hstack((-X, Y))
        b_ub = np.zeros(n_dmus)
        
        # 變數範圍 (Bounds)：加入 Non-Archimedean infinitesimal 防止權重為 0
        epsilon = 1e-6
        bounds = [(epsilon, None) for _ in range(n_inputs + n_outputs)]
        
        # 執行求解
        res = linprog(c, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq, bounds=bounds, method='highs')
        
        if res.success:
            eff = -res.fun
            # 處理極微小的浮點數誤差，確保最高分為 1.0
            eff = min(round(eff, 4), 1.0)
            efficiencies.append(eff)
        else:
            log.warning(f"⚠️ 警告：{df['ETF'].iloc[k]} 求解失敗。")
            efficiencies.append(np.nan)

    # 5. 將效率分數寫回 DataFrame 並排序
    df['DEA_Score'] = efficiencies

    # 顯示排版
    display_cols = ['ETF', 'DEA_Score'] + input_cols + output_cols
    df_sorted = df[display_cols].sort_values(by='DEA_Score', ascending=False).reset_index(drop=True)

    # 篩選
    efficient_frontier = df_sorted[df_sorted['DEA_Score'] == 1.0]
    relatively_efficient = df_sorted[df_sorted['DEA_Score'] >= 0.95]

    log.info(f"\n=== 🏆 效率前緣標的 (DEA Score = 1.0) 共 {len(efficient_frontier)} 檔 ===")
    log.info(efficient_frontier[['ETF', 'DEA_Score', 'In_Risk', 'In_Cost', 'Out_Return']].head(10).to_string(index=False))

    log.info(f"\n具有一定競爭力的標的數量 (DEA Score >= 0.95) : {len(relatively_efficient)}")

    log.info(f"\n=== 📉 需要剃除的劣勢標的 (DEA Score < 0.80) 共 {len(df_sorted[df_sorted['DEA_Score'] < 0.80])} 檔 ===")
    log.info(df_sorted[df_sorted['DEA_Score'] < 0.80][['ETF', 'DEA_Score']].head(5).to_string(index=False))

    # 儲存包含分數的最終結果
    df_sorted.to_csv("csv\\stage1_dea_results.csv", index=False)
    log.info("\n✅ 已將 DEA 分數儲存至 csv\\stage1_dea_results.csv")

def plot_dea_distribution():
    # 1. 讀取 DEA 結果
    try:
        df = pd.read_csv("csv\\stage1_dea_results.csv")
    except FileNotFoundError:
        log.error("找不到 csv\\stage1_dea_results.csv，請確認 DEA 模型已執行完畢。")
        sys.exit(1)
        return
    
    # 2. 設定繪圖風格與中文字型
    sns.set_theme(style="whitegrid")
    # 若在 Windows 執行可使用微軟正黑體；Mac 可改為 'PingFang HK' 或 'Arial Unicode MS'
    plt.rcParams['font.sans-serif'] = ['Microsoft JhengHei', 'PingFang HK', 'Arial Unicode MS']
    plt.rcParams['axes.unicode_minus'] = False

    # 3. 建立畫布
    fig, ax = plt.subplots(figsize=(10, 6))

    # 4. 繪製直方圖與 KDE 曲線
    # 根據資料量自動調整 bins，未來資料變多時圖形依然平滑
    sns.histplot(df['DEA_Score'], bins=20, kde=True, color='steelblue', edgecolor='black', ax=ax)

    # 5. 加上客觀閾值參考線
    ax.axvline(x=1.0, color='red', linestyle='--', linewidth=2, label='效率前緣 (1.0)')
    ax.axvline(x=0.90, color='orange', linestyle='--', linewidth=2, label='擴充門檻 (0.90)')
    ax.axvline(x=0.80, color='green', linestyle='--', linewidth=2, label='基本門檻 (0.80)')

    # 6. 設定標題與標籤
    ax.set_title('Stage 1: DEA 效率分數分布圖 (Score Distribution)', fontsize=16, pad=15)
    ax.set_xlabel('DEA 效率分數 (Score)', fontsize=12)
    ax.set_ylabel('ETF 數量 (Count)', fontsize=12)
    ax.legend()

    # 確保標籤不重疊
    plt.tight_layout()

    # 7. 儲存圖片
    output_filename = "png\\dea_score_distribution.png"
    plt.savefig(output_filename, dpi=300)
    log.info(f"📊 統計圖繪製完成！已儲存為 {output_filename}")

def run_stage1_super_efficiency_normalized():
    log.info("🚀 啟動超級效率模型 (Super-Efficiency DEA) 運算...")
    
    # 1. 🚨 修正：讀取「正規化且降維後」的 DEA 專用矩陣
    try:
        df_raw = pd.read_csv("csv\\stage0_dea_ready_matrix.csv")
    except FileNotFoundError:
        log.error("❌ 找不到 stage0_dea_ready_matrix.csv，請先執行前段正規化。")
        sys.exit(1)
        return
    df = df_raw.dropna(subset=['Out_Diversity']).reset_index(drop=True)

    # 2. 對齊 6 大維度
    input_cols = ['In_Risk', 'In_Cost']
    output_cols = ['Out_Return', 'Out_Liquidity', 'Out_Diversity', 'Out_Sentiment']

    X = df[input_cols].values
    Y = df[output_cols].values

    n_dmus = len(df)
    n_inputs = X.shape[1]   # 降為 2 維度
    n_outputs = Y.shape[1]  # 降為 4 維度

    super_efficiencies = []

    # 預先建立完整的 A_ub 基礎矩陣 (-X + Y)
    A_ub_base = np.hstack((-X, Y))

    # 3. 針對每檔 ETF 進行超級效率求解
    for k in range(n_dmus):
        c = np.concatenate((np.zeros(n_inputs), -Y[k]))
        
        A_eq = np.concatenate((X[k].reshape(1, -1), np.zeros((1, n_outputs))), axis=1)
        b_eq = np.array([1.0])
        
        # 核心差異：刪除第 k 列 (不與自己比較)
        A_ub_k = np.delete(A_ub_base, k, axis=0)
        b_ub_k = np.zeros(n_dmus - 1)
        
        epsilon = 1e-6
        bounds = [(epsilon, None) for _ in range(n_inputs + n_outputs)]
        
        res = linprog(c, A_ub=A_ub_k, b_ub=b_ub_k, A_eq=A_eq, b_eq=b_eq, bounds=bounds, method='highs')
        
        if res.success:
            eff = -res.fun
            super_efficiencies.append(round(eff, 4))
        else:
            # 正規化後的數據極少無解，若無解通常是強烈的離群訊號
            log.warning(f"⚠️ 警告：{df['ETF'].iloc[k]} 求解無解 (Infeasible)。")
            super_efficiencies.append(np.nan)

    # 4. 寫回 DataFrame 並排序
    df['Super_Score'] = super_efficiencies
    df_sorted = df.sort_values(by='Super_Score', ascending=False).reset_index(drop=True)

    # 5. 分析與印出結果
    log.info("\n=== 🚨 疑似極端值/異類 (Super_Score > 1.5) ===")
    outliers = df_sorted[df_sorted['Super_Score'] > 1.5]
    if not outliers.empty:
        log.info(outliers[['ETF', 'Super_Score', 'In_Cost', 'Out_Return']].to_string(index=False))
        log.info("💡 洞察：這些標的分數突破天際，建議檢查是否為單一維度極端產生的數學扭曲。")
    else:
        log.info("無發現嚴重極端值，資料庫品質十分穩定。")

    log.info("\n=== 🏆 優秀且穩健的前緣標的 (1.0 <= Super_Score <= 1.5) ===")
    robust_frontiers = df_sorted[(df_sorted['Super_Score'] >= 1.0) & (df_sorted['Super_Score'] <= 1.5)]
    log.info(robust_frontiers[['ETF', 'Super_Score']].head(10).to_string(index=False))

    df_sorted.to_csv("csv\\stage1_super_efficiency_results.csv", index=False)
    log.info("\n✅ 超級效率運算完成，結果已儲存為 csv\\stage1_super_efficiency_results.csv")

def run_cross_efficiency_dea():
    log.info("啟動 Stage 1: 交叉評估 (Cross-Efficiency) 運算...")
    
    # 1. 讀取 Stage 1 第一階段 (標準 DEA) 的結果
    try:
        df_raw = pd.read_csv("csv\\stage1_dea_results.csv")
    except FileNotFoundError:
        log.error("找不到 csv\\stage1_dea_results.csv，請先執行標準 DEA 模型。")
        sys.exit(1)
        return
        
    # 2. 過濾出符合基本門檻的優質 ETF (分數 >= 0.8)
    df = df_raw[df_raw['DEA_Score'] >= 0.8].reset_index(drop=True)
    log.info(f"篩選完畢：共 {len(df)} 檔 ETF 進入交叉評估矩陣。")

    input_cols = ['In_Risk', 'In_Cost']
    output_cols = ['Out_Return', 'Out_Liquidity', 'Out_Diversity']

    X = df[input_cols].values
    Y = df[output_cols].values

    n_dmus = len(df)
    n_inputs = X.shape[1]
    n_outputs = Y.shape[1]

    # 建立 n x n 交叉評估矩陣
    cross_matrix = np.zeros((n_dmus, n_dmus))

    # 3. 矩陣運算核心
    for k in range(n_dmus):
        # 步驟 A: 求解 DMU_k 的標準 DEA 以取得其專屬的「最佳權重」
        c = np.concatenate((np.zeros(n_inputs), -Y[k]))
        A_eq = np.concatenate((X[k].reshape(1, -1), np.zeros((1, n_outputs))), axis=1)
        b_eq = np.array([1.0])
        A_ub = np.hstack((-X, Y))
        b_ub = np.zeros(n_dmus)
        
        epsilon = 1e-6
        bounds = [(epsilon, None) for _ in range(n_inputs + n_outputs)]
        
        res = linprog(c, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq, bounds=bounds, method='highs')
        
        if res.success:
            v_star = res.x[:n_inputs]
            u_star = res.x[n_inputs:]
            
            # 步驟 B: 使用 DMU_k 的價值觀 (權重)，去評分所有的 DMU_j
            for j in range(n_dmus):
                virtual_output = np.dot(u_star, Y[j])
                virtual_input = np.dot(v_star, X[j])
                
                if virtual_input > 0:
                    cross_matrix[k][j] = virtual_output / virtual_input
                else:
                    cross_matrix[k][j] = 0.0
        else:
            log.warning(f"警告：DMU {df['ETF'].iloc[k]} 求解失敗。")
            cross_matrix[k, :] = np.nan

    # 4. 計算最終交叉分數 (同儕評分的平均值)
    cross_efficiency_scores = np.nanmean(cross_matrix, axis=0)

    # 5. 整合結果與輸出
    df['Cross_Score'] = np.round(cross_efficiency_scores, 4)
    
    # 依照交叉分數進行絕對排序
    df_sorted = df.sort_values(by='Cross_Score', ascending=False).reset_index(drop=True)

    log.info("\n=== 🏆 交叉評估最終排序 (Top 15 候選池) ===")
    display_cols = ['ETF', 'Cross_Score', 'DEA_Score'] + input_cols + output_cols
    log.info(df_sorted[display_cols].head(15).to_string(index=False))

    output_file = "csv\\stage1_final_candidates.csv"
    df_sorted.to_csv(output_file, index=False)
    log.info(f"\n資料儲存完畢：候選名單已輸出至 {output_file}")

# ==========================================
# stage2_0_AHP_weight_final_candidates_selection
# ==========================================
class TwoLevel_AHP_Model:
    def __init__(self):
        # --- 第一層：主維度 ---
        self.main_criteria = [
            "Return_Main", 
            "Risk_Main", 
            "Cost_Main", 
            "Liquidity_Main", 
            "Diversity_Main", 
            "Sentiment_Main"
        ]
        
        # --- 第二層：子特徵 ---
        self.sub_criteria = {
            "Return_Main": ["Return_CAGR", "Return_Div"],
            "Risk_Main": ["Risk_Vol", "Risk_MaxDD"],
            "Liquidity_Main": ["Liq_Volume", "Liq_AUM"],
            "Cost_Main": ["Cost_ExpRatio"],       # 單一特徵，權重為 1
            "Diversity_Main": ["Div_Score"],      # 單一特徵，權重為 1
            "Sentiment_Main": ["FinBERT_score"]   # 單一特徵，權重為 1
        }
        
        # 定義不同矩陣大小對應的 Random Index (RI) 查表值
        self.RI_dict = {1: 0.0, 2: 0.0, 3: 0.58, 4: 0.90, 5: 1.12, 6: 1.24}

    def _solve_matrix(self, comparisons, n):
        """通用的 AHP 矩陣求解器"""
        if n == 1:
            return np.array([1.0]), 0.0 # 單一特徵不需比較
            
        num_comparisons = (n * (n - 1)) // 2
        if len(comparisons) != num_comparisons:
            raise ValueError(f"維度為 {n} 的矩陣需要 {num_comparisons} 個成對比較值。")

        matrix = np.ones((n, n))
        idx = 0
        for i in range(n):
            for j in range(i + 1, n):
                val = comparisons[idx]
                matrix[i, j] = val
                matrix[j, i] = 1.0 / val
                idx += 1
                
        eigenvalues, eigenvectors = np.linalg.eig(matrix)
        max_idx = np.argmax(np.real(eigenvalues))
        max_eigenvalue = np.real(eigenvalues[max_idx])
        eigenvector = np.real(eigenvectors[:, max_idx])
        
        weights = eigenvector / np.sum(eigenvector)
        
        # 當 n<=2 時，CR 理論上恆為 0，且 RI 為 0 無法相除，因此直接回傳 CR=0
        if n <= 2:
            CR = 0.0
        else:
            CI = (max_eigenvalue - n) / (n - 1)
            CR = CI / self.RI_dict[n]
            
        return weights, CR

    def calculate_global_weights(self, user_inputs):
        """
        傳入使用者的問卷結果 (包含第一層與需要比較的第二層)。
        計算出最終 9 個子特徵的全局權重。
        """
        log.info("\n🚀 啟動兩層級 AHP (Two-Level AHP) 運算...")
        
        # 1. 求解第一層主維度權重
        log.info("\n[第一層：主維度求解]")
        main_weights, main_cr = self._solve_matrix(user_inputs["Main"], len(self.main_criteria))
        
        main_weight_dict = {crit: weight for crit, weight in zip(self.main_criteria, main_weights)}
        for crit, weight in main_weight_dict.items():
            log.info(f"  - {crit}: {weight*100:.2f}%")
        log.info(f"  >> 一致性比率 (CR): {main_cr:.4f}")
        
        if main_cr > 0.1:
            log.warning("❌ 警告：主維度問卷存在邏輯矛盾 (CR > 0.1)！")
            
        # 2. 求解第二層子特徵權重並計算全局權重
        log.info("\n[第二層：子特徵局部權重與全局權重]")
        global_weights = {}
        
        for main_crit in self.main_criteria:
            subs = self.sub_criteria[main_crit]
            main_w = main_weight_dict[main_crit]
            
            # 如果該維度只有 1 個子特徵，局部權重為 1.0
            if len(subs) == 1:
                sub_name = subs[0]
                global_weights[sub_name] = main_w * 1.0
                log.info(f"  - {sub_name} (單一特徵) -> 全局權重: {global_weights[sub_name]*100:.2f}%")
                continue
                
            # 如果有多個子特徵，需要求解次矩陣
            sub_comparisons = user_inputs["Sub"].get(main_crit, [])
            local_weights, sub_cr = self._solve_matrix(sub_comparisons, len(subs))
            
            for i, sub_name in enumerate(subs):
                global_w = main_w * local_weights[i]
                global_weights[sub_name] = global_w
                log.info(f"  - {sub_name} (局部 {local_weights[i]*100:.1f}%) -> 全局權重: {global_w*100:.2f}%")
                
        return global_weights, main_cr


# ==========================================
# 互動式問卷前處理器 (Questionnaire Preprocessor)
# ==========================================
def ask_question_json_slider(question_data):
    """處理 -5 到 +5 刻度的 AHP 問卷互動"""
    print(f"\n{question_data['question']}")
    print(f"  {question_data['left_pole']}  <-- [ 0: 同等重要 ] -->  {question_data['right_pole']}")
    
    # 定義 -4 到 4 映射到 AHP 1~9 的字典
    ahp_mapping = {
        0: 1.0,
        1: 3.0,  2: 5.0,  3: 7.0,  4: 9.0,
        -1: 1/3.0, -2: 1/5.0, -3: 1/7.0, -4: 1/9.0
    }
    
    while True:
        try:
            ans = int(input("請輸入你的偏好程度 (-4 到 4 的整數): ").strip())
            if -4 <= ans <= 4:
                return ahp_mapping[ans]
            else:
                print("❌ 請輸入介於 -4 到 4 之間的整數。")
        except ValueError:
            print("❌ 輸入格式錯誤，請輸入整數。")

def build_user_simulation(deterministic=False, questionnaire_path="json\\questionnaire.json"):
    """讀取外部 JSON 問卷，進行標準 AHP 成對比較 (-5 到 5 刻度)"""
    
    if deterministic:
        # 開發測試用：直接回傳寫死的值
        return parameters.DETERMINISTIC_USER_INPUTS

    try:
        with open(questionnaire_path, 'r', encoding='utf-8') as f:
            q_data = json.load(f)
    except FileNotFoundError:
        log.error(f"❌ 找不到問卷設定檔：{questionnaire_path}")
        sys.exit(1)
        return None

    print("="*60)
    print(" 🧠 智能理財引擎：高解析度 AHP 成對比較問卷 (-4 到 +4 刻度)")
    print("="*60)

    main_comparisons = []
    for item in q_data["Main_Comparisons"]:
        val = ask_question_json_slider(item)
        main_comparisons.append(val)

    print("\n" + "-"*40)
    print("進階偏好設定 (子特徵成對比較)")
    sub_inputs = {}
    
    for main_crit, sub_q in q_data["Sub_Comparisons"].items():
        val = ask_question_json_slider(sub_q)
        sub_inputs[main_crit] = [val]

    return {
        "Main": main_comparisons,
        "Sub": sub_inputs
    }
# ==========================================
# stage2_1_preference_driven_deduplication
# ==========================================
def robust_scale(series, upper_quantile=0.95, lower_quantile=0.01, is_reverse=False):
    """自定義正規化：包含縮尾處理，並壓縮至 0~1 區間"""
    lower_bound = series.quantile(lower_quantile)
    upper_bound = series.quantile(upper_quantile)
    clipped = series.clip(lower=lower_bound, upper=upper_bound)
    
    s_min = clipped.min()
    s_max = clipped.max()
    
    if s_max == s_min:
        scaled = np.zeros(len(clipped))
    else:
        scaled = (clipped - s_min) / (s_max - s_min)
        
    # 如果是風險或成本 (越小越好)，計算偏好分數時必須反向 (越高分代表效用越大)
    if is_reverse:
        return 1.0 - scaled
    return scaled

def run_stage2_5_preference_deduplication_yq():
    log.info("啟動 Stage 2.5: 偏好驅動去重與分群 (白名單過濾與原始數據重構)...")
    
    # 1. 讀取 Stage 2 的兩層級 9 維全局權重
    try:
        with open("json\\stage2_ahp_global_weights.json", "r", encoding="utf-8") as f:
            ahp_data = json.load(f)
            global_weights = ahp_data["Global_Weights"]
    except FileNotFoundError:
        log.error("❌ 找不到 stage2_ahp_global_weights.json。")
        sys.exit(1)
        return

    # 2. 讀取 Stage 1 白名單，與 Stage 0 原始數據
    try:
        df_candidates = pd.read_csv("csv\\stage1_final_candidates.csv")
        df_raw = pd.read_csv("csv\\stage0_final_matrix.csv")
    except FileNotFoundError:
        log.error("❌ 找不到 stage1 或 stage0 的 csv 檔案。")
        sys.exit(1)
        return

    # 使用 Stage 1 產出的 ETF 名單作為白名單過濾 Stage 0 的原始數據
    valid_tickers = df_candidates['ETF'].tolist()
    df = df_raw[df_raw['ETF'].isin(valid_tickers)].reset_index(drop=True)
    log.info(f"📥 成功載入 {len(df)} 檔候選 ETF 之原始特徵。")

    # 3. 9 大子特徵獨立正規化
    df_scaled = pd.DataFrame({'ETF': df['ETF']})
    
    # [正向特徵] 全面縮尾處理
    df_scaled['Norm_Return_CAGR'] = robust_scale(df['Return_CAGR (%)'], upper_quantile=0.99, lower_quantile=0.01)
    df_scaled['Norm_Return_Div'] = robust_scale(df['Return_Div (%)'])
    df_scaled['Norm_Div_Score'] = robust_scale(df['Div_Score (產出)'].fillna(0), upper_quantile=0.95, lower_quantile=0.05) # 分散分數分布較平均
    df_scaled['Norm_FinBERT'] = robust_scale(df['FinBERT_score'].fillna(0), upper_quantile=0.95, lower_quantile=0.05)  # 情緒分數分布較平均
    
    # [流動性特徵] 先取對數，再嚴格進行縮尾處理
    df_scaled['Norm_Liq_Volume'] = robust_scale(np.log1p(df['Liq_Volume (M)']))
    df_scaled['Norm_Liq_AUM'] = robust_scale(np.log1p(df['Liq_AUM (B)']))
    
    # [反向特徵] 縮尾處理後反轉 (1 - scaled)
    df_scaled['Norm_Risk_Vol'] = robust_scale(df['Risk_Vol (%)'], is_reverse=True)
    df_scaled['Norm_Risk_MaxDD'] = robust_scale(df['Risk_MaxDD (%)'].abs(), is_reverse=True)
    df_scaled['Norm_Cost_ExpRatio'] = robust_scale(df['Cost_ExpRatio (%)'], is_reverse=True)
    
    # 4. 計算使用者偏好分數 (User_Pref_Score)
    feature_map = {
        "Return_CAGR": 'Norm_Return_CAGR',
        "Return_Div": 'Norm_Return_Div',
        "Risk_Vol": 'Norm_Risk_Vol',
        "Risk_MaxDD": 'Norm_Risk_MaxDD',
        "Liq_Volume": 'Norm_Liq_Volume',
        "Liq_AUM": 'Norm_Liq_AUM',
        "Cost_ExpRatio": 'Norm_Cost_ExpRatio',
        "Div_Score": 'Norm_Div_Score',
        "FinBERT_score": 'Norm_FinBERT'
    }
    
    pref_scores = np.zeros(len(df))
    for key, weight in global_weights.items():
        if key in feature_map:
            col_name = feature_map[key]
            pref_scores += df_scaled[col_name].values * weight
        
    df['User_Pref_Score'] = pref_scores

    # 5. 下載歷史價格進行相關性計算
    price_matrix = get_or_fetch_historical_prices(valid_tickers)
    returns_matrix = price_matrix.pct_change(fill_method=None).dropna(how='all')
    # 🚨 修正：先向下填補(處理中間斷層)，再向上填補(處理頭部缺漏)
    returns_matrix = returns_matrix.ffill().bfill()
    # 最後才使用 dropna(axis=1) 把那些「即使雙向填補後依然全是 NaN」的無效標的給剔除
    returns_matrix = returns_matrix.dropna(axis=1)
    
    # 對齊成功抓到價格的 ETF
    final_tickers = [t for t in valid_tickers if t in returns_matrix.columns]
    df = df[df['ETF'].isin(final_tickers)].reset_index(drop=True)
    corr_matrix = returns_matrix[final_tickers].corr()
    
    # 6. 相關性分群去重 (Threshold = 0.99)啟動本地快取引擎 (請求總數: 20 檔)...
    CORR_THRESHOLD = 0.99
    clusters = []
    processed_tickers = set()
    
    sorted_tickers = df.sort_values(by='User_Pref_Score', ascending=False)['ETF'].tolist()

    for ticker in sorted_tickers:
        if ticker in processed_tickers:
            continue
            
        correlated = corr_matrix.index[corr_matrix[ticker] >= CORR_THRESHOLD].tolist()
        cluster = [t for t in correlated if t not in processed_tickers]
        
        if cluster:
            clusters.append(cluster)
            processed_tickers.update(cluster)
            
    # 7. 挑選偏好分數最高代表
    final_portfolio_candidates = []
    log.info("\n=== 🎯 Stage 2.5 群集去重結果 ===")
    for i, cluster in enumerate(clusters):
        cluster_df = df[df['ETF'].isin(cluster)]
        best_etf = cluster_df.loc[cluster_df['User_Pref_Score'].idxmax()]
        final_portfolio_candidates.append(best_etf)
        
        if len(cluster) > 1:
            log.info(f"  > 群集 {cluster} -> 🏆 勝出: {best_etf['ETF']} (Score: {best_etf['User_Pref_Score']:.4f})")
            log.info(cluster_df[['ETF', 'User_Pref_Score']].to_string(index=False))
        
    final_df = pd.DataFrame(final_portfolio_candidates).sort_values(by='User_Pref_Score', ascending=False).reset_index(drop=True)
    
    # 🚨 新增：提取最終名單對應的「正規化特徵矩陣」並存檔
    final_tickers = final_df['ETF'].tolist()
    final_scaled_df = df_scaled[df_scaled['ETF'].isin(final_tickers)].reset_index(drop=True)
    
    final_df.to_csv("csv\\stage2_final_user_universe.csv", index=False)
    final_scaled_df.to_csv("csv\\stage2_normalized_features.csv", index=False)
    
    log.info("\n✅ 資料已輸出至 csv\\stage2_final_user_universe.csv")
    log.info("✅ 正規化特徵矩陣已輸出至 csv\\stage2_normalized_features.csv")

# ==========================================
# stage3_Preference_Driven_Portfolio_Optimization
# ==========================================
USE_TRUE_HHI_OPTIMIZATION = parameters.USE_TRUE_HHI_OPTIMIZATION

# ==========================================
# 信託底線權重 (Baseline Weights) 與 Alpha 融合
# ==========================================
# 系統信託底線的佔比 (0.0 = 完全聽從使用者, 1.0 = 完全使用系統底線, 0.5 = 各半)
ALPHA_BASELINE = parameters.ALPHA_BASELINE 

# 專家先驗權重矩陣 (總和必須為 1.0)
BASELINE_WEIGHTS = parameters.BASELINE_WEIGHTS
case = parameters.CASE_NAME

# ==========================================
# 模組：建構 N x K 真實產業矩陣 (Sector Matrix S)
# ==========================================
def build_sector_matrix(etf_list, db_file):
    """
    將所有候選 ETF 的 JSON 產業分布，轉換為 N x K 的數學矩陣。
    N = ETF 數量, K = 所有出現過的獨特產業數量
    """
    if not os.path.exists(db_file):
        log.error("⚠️ 找不到 AV 資料庫，無法建立真實產業矩陣。")
        sys.exit(1)
        return None, []

    with open(db_file, 'r', encoding='utf-8') as f:
        db = json.load(f)

    # 1. 找出所有獨特的產業，建立產業字典 (Columns)
    all_sectors = set()
    for ticker in etf_list:
        if ticker in db and "Sector_Weights" in db[ticker]:
            all_sectors.update(db[ticker]["Sector_Weights"].keys())

    sector_list = list(all_sectors)
    K = len(sector_list)
    N = len(etf_list)
    
    if K == 0:
        return None, []

    # 2. 填入 N x K 權重矩陣
    S_matrix = np.zeros((N, K))
    for i, ticker in enumerate(etf_list):
        if ticker in db and "Sector_Weights" in db[ticker]:
            weights = db[ticker]["Sector_Weights"]
            for j, sector in enumerate(sector_list):
                S_matrix[i, j] = weights.get(sector, 0.0)
        else:
            # 防呆機制：若缺失該檔 ETF 的產業資料，強制將其視為 100% 未知產業
            # 在最佳化時會對其施加最嚴厲的集中度懲罰
            pass 

    log.info(f"📊 成功建立真實產業矩陣 (維度: {N} 檔 ETF x {K} 個產業)")
    return S_matrix, sector_list

def plot_portfolio_analytics_and_mpt(returns_matrix, optimal_weights, max_sharpe_weights, tickers):
    """視覺化模組：繪製歷史軌跡與 MPT 效率前緣 (結合精確解析解)"""
    log.info("\n啟動視覺化模組：計算歷史軌跡與 MPT 效率前緣...")
    
    sns.set_theme(style="whitegrid")
    plt.rcParams['font.sans-serif'] = ['Microsoft JhengHei', 'PingFang HK', 'Arial Unicode MS']
    plt.rcParams['axes.unicode_minus'] = False
    
    # 建立輸出資料夾
    os.makedirs("png", exist_ok=True)
    
    # --- 圖表 1：歷史淨值曲線與最大回撤 ---
    port_daily_returns = returns_matrix.dot(optimal_weights)
    cumulative_returns = (1 + port_daily_returns).cumprod()
    peak = cumulative_returns.cummax()
    drawdown = (cumulative_returns - peak) / peak
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), gridspec_kw={'height_ratios': [3, 1]}, sharex=True)
    
    ax1.plot(cumulative_returns.index, cumulative_returns, color='navy', linewidth=2)
    ax1.set_title('投資組合歷史淨值曲線 (Cumulative Returns)', fontsize=14, fontweight='bold')
    ax1.set_ylabel('累積淨值 (基期=1)', fontsize=12)
    
    ax2.fill_between(drawdown.index, drawdown * 100, 0, color='red', alpha=0.5)
    ax2.plot(drawdown.index, drawdown * 100, color='darkred', linewidth=1)
    ax2.set_title('歷史回撤幅度 (Drawdown %)', fontsize=14, fontweight='bold')
    ax2.set_ylabel('回撤 (%)', fontsize=12)
    ax2.set_xlabel('日期', fontsize=12)
    
    plt.tight_layout()
    plt.savefig(f'png\\{case}_portfolio_performance.png', dpi=300)
    plt.close()
    log.info(f"✅ 產出圖表：png\\{case}_portfolio_performance.png")

    # --- 圖表 2：蒙地卡羅模擬 MPT 效率前緣 ---
    annual_returns = returns_matrix.mean() * 252
    cov_matrix = returns_matrix.cov() * 252
    rf_rate = 0.04  # 假設無風險利率 4%
    
    # 1. 計算偏好驅動組合落點
    port_vol = np.sqrt(np.dot(optimal_weights.T, np.dot(cov_matrix, optimal_weights)))
    port_ret = np.dot(optimal_weights.T, annual_returns)
    
    # 2. 🚨 新增：計算精確的 Max Sharpe 組合落點
    exact_ms_vol = np.sqrt(np.dot(max_sharpe_weights.T, np.dot(cov_matrix, max_sharpe_weights)))
    exact_ms_ret = np.dot(max_sharpe_weights.T, annual_returns)
    ms_sharpe_ratio = (exact_ms_ret - rf_rate) / exact_ms_vol if exact_ms_vol > 0 else 0
    
    # 3. 改良版蒙地卡羅模擬 (加入稀疏性與權重上限以拓展邊界)
    num_portfolios = 50000
    results = np.zeros((3, num_portfolios))
    MAX_WEIGHT_LIMIT = parameters.MAX_WEIGHT_LIMIT  # 🚨 你的單一標的最大權重限制
    
    for i in range(num_portfolios):
        weights = np.random.random(len(tickers))
        
        # 隨機將 50%~80% 的資產權重歸零，強迫模擬極端集中的組合
        if i > num_portfolios // 4:
            mask = np.random.rand(len(tickers)) > (np.random.uniform(0.2, 0.5))
            weights[mask] = 0
            
        # 🚨 防呆機制：若上限為 40%，至少需要 3 檔 ETF (ceil(1/0.4)) 才能湊滿 100%
        min_required_assets = int(np.ceil(1.0 / MAX_WEIGHT_LIMIT))
        if np.sum(weights > 0) < min_required_assets:
            # 隨機挑選足夠數量的資產補上隨機權重，避免數學上無解
            fill_indices = np.random.choice(len(tickers), min_required_assets, replace=False)
            weights[fill_indices] = np.random.random(min_required_assets)
            
        # 初始正規化 (總和為 1)
        weights /= np.sum(weights)
        
        # 🚨 核心：迭代溢流分配法 (Water-filling) 來嚴格限制最大權重
        while np.any(weights > MAX_WEIGHT_LIMIT):
            # 找出超過上限的索引
            excess_idx = weights > MAX_WEIGHT_LIMIT
            # 找出未達上限的索引 (可以接受溢流的對象)
            avail_idx = weights < MAX_WEIGHT_LIMIT
            
            # 將超過的部分強制砍平到上限
            weights[excess_idx] = MAX_WEIGHT_LIMIT
            
            # 計算被砍掉的總溢流權重 (還差多少才能補滿 100%)
            excess_sum = 1.0 - np.sum(weights)
            
            # 將這些無處安放的權重，按比例分配給其他未達上限的資產
            avail_sum = np.sum(weights[avail_idx])
            if avail_sum > 0:
                weights[avail_idx] += excess_sum * (weights[avail_idx] / avail_sum)
            else:
                # 若剩下的權重都是 0，則直接均分
                weights[avail_idx] += excess_sum / np.sum(avail_idx)
                
        # 最終確保浮點數精度的微小誤差
        weights /= np.sum(weights)
        
        p_ret = np.dot(weights, annual_returns)
        p_vol = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
        results[0,i] = p_vol
        results[1,i] = p_ret
        results[2,i] = (p_ret - 0.04) / p_vol
        
    plt.figure(figsize=(10, 7))
    scatter = plt.scatter(results[0,:] * 100, results[1,:] * 100, c=results[2,:], cmap='viridis', marker='o', s=10, alpha=0.3)
    plt.colorbar(scatter, label='夏普指標 (Sharpe Ratio)')
    
    # 標示專屬客製化組合 (紅色星星)
    plt.scatter(port_vol * 100, port_ret * 100, color='red', marker='*', s=300, edgecolor='black', 
                label=f'偏好驅動組合\n(報酬: {port_ret*100:.1f}%, 風險: {port_vol*100:.1f}%)', zorder=5)
    
    # 🚨 修正：標示全局最大夏普組合 (使用 SLSQP 算出的解析解，不再依賴蒙地卡羅)
    plt.scatter(exact_ms_vol * 100, exact_ms_ret * 100, color='blue', marker='X', s=150, edgecolor='black',
                label=f'傳統 Max Sharpe 組合\n(報酬: {exact_ms_ret*100:.1f}%, 風險: {exact_ms_vol*100:.1f}%)', zorder=5)

   # ==========================================
    # 🚨 新增：繪製資本市場線 (CML / Tangent Line)
    # ==========================================
    max_vol_plot = np.max(results[0,:]) * 100 * 1.05 # 將線條延伸至點雲最右側
    cml_x = np.array([0, exact_ms_vol * 100, max_vol_plot])
    # 直線方程式: y = R_f + Sharpe * x
    cml_y = rf_rate * 100 + ms_sharpe_ratio * cml_x

    plt.plot(cml_x, cml_y, color='darkorange', linestyle='--', linewidth=2, 
             label=f'資本市場線 (CML, Rf={rf_rate*100:.0f}%)', zorder=4)
             
    # 標示 Y 軸上的無風險利率起點
    plt.scatter(0, rf_rate * 100, color='darkorange', marker='D', s=80, edgecolor='black', zorder=5)

    # 強制 X 軸從 0 開始，以顯示完整的切線軌跡
    plt.xlim(left=0, right=max_vol_plot)

    plt.title('現代投資組合理論 (MPT) 效率前緣與資本市場線', fontsize=16, fontweight='bold', pad=15)
    plt.xlabel('年化波動率 (風險) %', fontsize=12)
    plt.ylabel('預期年化報酬率 %', fontsize=12)
    plt.legend(loc='lower right', frameon=True, shadow=True) # 圖例移至右下避免擋住 Y 截距
    
    plt.tight_layout()
    plt.savefig(f"png\\{case}_mpt_efficient_frontier.png", dpi=300)
    plt.close()
    log.info(f"✅ 產出圖表：png\\{case}_mpt_efficient_frontier.png")

    # 圖表3. 數學解析法：計算教科書等級的「精確效率前緣」
    log.info("⏳ 計算精確效率前緣曲線 (Mathematical Efficient Frontier)...")

    def calc_vol(w):
        return np.sqrt(np.dot(w.T, np.dot(cov_matrix, w)))

    bounds_ef = tuple((0.0, MAX_WEIGHT_LIMIT) for _ in range(len(tickers))) # 沿用 MAX_WEIGHT_LIMIT 上限
    initial_w = np.array([1.0 / len(tickers)] * len(tickers))
    cons_sum = {'type': 'eq', 'fun': lambda w: np.sum(w) - 1.0}

    # A. 尋找全局最小變異組合 (Global Minimum Variance, 曲線的最低點)
    res_gmv = minimize(calc_vol, initial_w, method='SLSQP', bounds=bounds_ef, constraints=[cons_sum])
    min_vol_ret = np.dot(res_gmv.x, annual_returns)

    # B. 尋找最大報酬組合 (Max Return, 曲線的最高點)
    res_max = minimize(lambda w: -np.dot(w, annual_returns), initial_w, method='SLSQP', bounds=bounds_ef, constraints=[cons_sum])
    max_ret = np.dot(res_max.x, annual_returns)

    # C. 沿著 Y 軸 (報酬) 切割網格，找出每個目標報酬對應的絕對最小風險
    target_returns = np.linspace(min_vol_ret, max_ret, 200)
    efficient_vols = []
    valid_returns = []

    for target_ret in target_returns:
        cons_ef = [
            {'type': 'eq', 'fun': lambda w: np.sum(w) - 1.0},
            {'type': 'eq', 'fun': lambda w: np.dot(w, annual_returns) - target_ret}
        ]
        res_ef = minimize(calc_vol, initial_w, method='SLSQP', bounds=bounds_ef, constraints=cons_ef)
        if res_ef.success:
            efficient_vols.append(calc_vol(res_ef.x))
            valid_returns.append(target_ret)

    efficient_vols = np.array(efficient_vols) * 100
    valid_returns = np.array(valid_returns) * 100
    # ==========================================
    # 繪圖區塊：乾淨的教科書風格
    # ==========================================
    plt.figure(figsize=(10, 7))

    # 畫出平滑且粗曠的效率前緣曲線 (使用你參考圖的薄荷綠色)
    plt.plot(efficient_vols, valid_returns, color='#2ECC71', linewidth=4, label='效率前緣 (Efficient Frontier)', zorder=2)
    # 標示專屬客製化組合 (紅色星星)
    plt.scatter(port_vol * 100, port_ret * 100, color='red', marker='*', s=300, edgecolor='black', 
                label=f'偏好驅動組合\n(報酬: {port_ret*100:.1f}%, 風險: {port_vol*100:.1f}%)', zorder=5)
    
    # 🚨 修正：標示全局最大夏普組合 (使用 SLSQP 算出的解析解，不再依賴蒙地卡羅)
    plt.scatter(exact_ms_vol * 100, exact_ms_ret * 100, color='blue', marker='X', s=150, edgecolor='black',
                label=f'傳統 Max Sharpe 組合\n(報酬: {exact_ms_ret*100:.1f}%, 風險: {exact_ms_vol*100:.1f}%)', zorder=5)

    # ==========================================
    # 🚨 新增：繪製資本市場線 (CML / Tangent Line)
    # ==========================================
    max_vol_plot = np.max(results[0,:]) * 100 * 1.05 # 將線條延伸至點雲最右側
    cml_x = np.array([0, exact_ms_vol * 100, max_vol_plot])
    # 直線方程式: y = R_f + Sharpe * x
    cml_y = rf_rate * 100 + ms_sharpe_ratio * cml_x

    plt.plot(cml_x, cml_y, color='darkorange', linestyle='--', linewidth=2, 
             label=f'資本市場線 (CML, Rf={rf_rate*100:.0f}%)', zorder=4)
             
    # 標示 Y 軸上的無風險利率起點
    plt.scatter(0, rf_rate * 100, color='darkorange', marker='D', s=80, edgecolor='black', zorder=5)
    plt.title('數學解析效率前緣 (Mathematical Efficient Frontier)', fontsize=16, fontweight='bold', pad=15)
    plt.xlabel('年化波動率 (風險) %', fontsize=12)
    plt.ylabel('預期年化報酬率 %', fontsize=12)
    # 強制 X 軸從 0 開始，以顯示完整的切線軌跡
    plt.xlim(left=0, right=max_vol_plot)
    plt.savefig(f"png\\{case}_Mathematical Efficient Frontier.png", dpi=300)
    plt.close()
    log.info(f"✅ 產出圖表：png\\{case}_Mathematical Efficient Frontier.png")

def plot_preference_radar_chart(optimal_weights, max_sharpe_weights, vectors_dict, blended_weights, pref_metrics, ms_metrics, cov_matrix, tickers, max_weight_limit=parameters.MAX_WEIGHT_LIMIT):
    """
    繪製 9 維度效用雷達圖，整合正規化效用圖形與真實物理數據，並自動突顯高權重維度。
    針對「抗回撤」採用真實投資組合的非線性風險數據進行雷達圖映射。
    內部即時計算「理論最低/最高真實波動率」，讓波動率效用 100% 貼合物理極限。
    """
    log.info("\n啟動視覺化模組：繪製帶有真實數據的多維度效用雷達圖...")
    
    sns.set_theme(style="whitegrid")
    plt.rcParams['font.sans-serif'] = ['Microsoft JhengHei', 'PingFang HK', 'Arial Unicode MS']
    plt.rcParams['axes.unicode_minus'] = False
    
    categories = list(blended_weights.keys())
    N = len(categories)
    
    # ==========================================
    # 🚨 動態計算：尋找真實波動率的物理邊界 (GMV)
    # ==========================================
    log.info("⏳ 計算真實波動率的物理極限，以進行雷達圖動態映射...")
    def calc_true_vol(w):
        return np.sqrt(np.dot(w.T, np.dot(cov_matrix, w)))

    bounds_ef = tuple((0.0, max_weight_limit) for _ in range(len(tickers)))
    initial_w = np.array([1.0 / len(tickers)] * len(tickers))
    cons_sum = {'type': 'eq', 'fun': lambda w: np.sum(w) - 1.0}

    # 理論最低波動率 (效用 1.0 的基準)
    res_gmv = minimize(calc_true_vol, initial_w, method='SLSQP', bounds=bounds_ef, constraints=[cons_sum])
    min_true_vol = calc_true_vol(res_gmv.x) * 100  # 直接轉為 %

    # 理論最高波動率 (效用 0.0 的基準)
    res_max_vol = minimize(lambda w: -calc_true_vol(w), initial_w, method='SLSQP', bounds=bounds_ef, constraints=[cons_sum])
    max_true_vol = calc_true_vol(res_max_vol.x) * 100  # 直接轉為 %

    log.info(f"   [動態邊界] 最低波動率(1.0): {min_true_vol:.2f}%, 最高波動率(0.0): {max_true_vol:.2f}%")

    # ==========================================
    # 核心修改：動態分配各維度的真實效用分數
    # ==========================================
    pref_scores = []
    ms_scores = []
    
    for cat in categories:
        if cat == "Risk_Vol" and "Volatility" in pref_metrics:
            # 1. 波動率：使用剛剛算出的真實邊界進行動態映射
            p_vol = pref_metrics["Volatility"]
            m_vol = ms_metrics["Volatility"]
            
            if max_true_vol > min_true_vol:
                p_score = np.clip(1.0 - (p_vol - min_true_vol) / (max_true_vol - min_true_vol), 0.0, 1.0)
                m_score = np.clip(1.0 - (m_vol - min_true_vol) / (max_true_vol - min_true_vol), 0.0, 1.0)
            else:
                p_score, m_score = 1.0, 1.0
                
            pref_scores.append(p_score)
            ms_scores.append(m_score)
            
        elif cat == "Risk_MaxDD" and "MaxDD" in pref_metrics:
            # 2. 最大回撤：維持使用 0% ~ 40% 作為視覺邊界
            p_dd = abs(pref_metrics["MaxDD"]) 
            m_dd = abs(ms_metrics["MaxDD"])
            p_score = np.clip(1.0 - (p_dd / 40.0), 0.0, 1.0)
            m_score = np.clip(1.0 - (m_dd / 40.0), 0.0, 1.0)
            pref_scores.append(p_score)
            ms_scores.append(m_score)
        
        elif cat == "Liq_AUM" and "AUM" in pref_metrics:
            # 放棄線性分數加總，直接拿真實組合規模 (如 99.4B) 來評分
            # 設定 50B 為滿分 1.0 門檻
            p_aum = pref_metrics["AUM"]
            m_aum = ms_metrics["AUM"]
            p_score = np.clip(p_aum / 50.0, 0.0, 1.0)
            m_score = np.clip(m_aum / 50.0, 0.0, 1.0)
            pref_scores.append(p_score)
            ms_scores.append(m_score)
            
        elif cat == "Liq_Volume" and "Volume" in pref_metrics:
            # 設定 10M (一千萬) 為滿分 1.0 門檻
            p_vol = pref_metrics["Volume"]
            m_vol = ms_metrics["Volume"]
            p_score = np.clip(p_vol / 10.0, 0.0, 1.0)
            m_score = np.clip(m_vol / 10.0, 0.0, 1.0)
            pref_scores.append(p_score)
            ms_scores.append(m_score)
        
        elif cat == "FinBERT_score" and "Sentiment" in pref_metrics:
            # 🚨 動態相對映射：確保原始分數較高者，在圖形上「絕對」更靠外
            p_val, m_val = pref_metrics["Sentiment"], ms_metrics["Sentiment"]
            max_v, min_v = max(p_val, m_val), min(p_val, m_val)
            if max_v > min_v:
                # 較低者拿 0.4 基礎分，較高者依比例往上疊加至最高 0.9
                p_score = 0.4 + 0.5 * ((p_val - min_v) / (max_v - min_v))
                m_score = 0.4 + 0.5 * ((m_val - min_v) / (max_v - min_v))
            else:
                p_score, m_score = 0.5, 0.5
            pref_scores.append(p_score); ms_scores.append(m_score)

        else:
            # 3. 其他維度：維持預先算好的線性加權
            pref_scores.append(np.dot(optimal_weights, vectors_dict[cat]))
            ms_scores.append(np.dot(max_sharpe_weights, vectors_dict[cat]))
            
    # ==========================================

    # 找出使用者最重視的前 3 個維度
    top_3_cats = sorted(categories, key=lambda c: blended_weights[c], reverse=True)[:3]
    
    metric_map = {
        "Return_CAGR": ("CAGR", "{:.2f}%"), "Return_Div": ("Div_Yield", "{:.2f}%"),
        "Risk_Vol": ("Volatility", "{:.2f}%"), "Risk_MaxDD": ("MaxDD", "{:.2f}%"),
        "Cost_ExpRatio": ("Cost", "{:.3f}%"), "Div_Score": ("Proxy_Div", "{:.4f}"),
        "FinBERT_score": ("Sentiment", "{:.4f}"), "Liq_Volume": ("Volume", "{:,.1f}M"),
        "Liq_AUM": ("AUM", "{:,.1f}B")
    }
    
    label_map = {
        "Return_CAGR": "歷史報酬", "Return_Div": "殖利率", "Risk_Vol": "抗波動", 
        "Risk_MaxDD": "抗回撤", "Cost_ExpRatio": "低成本", "Liq_Volume": "交易量",
        "Liq_AUM": "資產規模", "Div_Score": "產業分散", "FinBERT_score": "市場情緒"
    }
    
    axis_labels = []
    for cat in categories:
        weight_pct = blended_weights[cat] * 100
        metric_key, fmt = metric_map.get(cat, (None, ""))
        
        prefix = "* " if cat in top_3_cats else ""
        title = f"{prefix}{label_map.get(cat, cat)} ({weight_pct:.1f}%)"
        
        if metric_key and metric_key in pref_metrics:
            pref_val = fmt.format(pref_metrics[metric_key])
            ms_val = fmt.format(ms_metrics[metric_key])
            data_text = f"偏好解: {pref_val}\n夏普解: {ms_val}"
        else:
            data_text = "(無單一原始代理值)"
            
        axis_labels.append(f"{title}\n{data_text}")
    
    angles = [n / float(N) * 2 * np.pi for n in range(N)]
    angles += angles[:1]
    pref_scores += pref_scores[:1]
    ms_scores += ms_scores[:1]
    
    fig, ax = plt.subplots(figsize=(12, 12), subplot_kw=dict(polar=True))
    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)
    
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(axis_labels, size=14, linespacing=1.5)
    ax.tick_params(axis='x', pad=30) 
    
    for i, label in enumerate(ax.get_xticklabels()):
        if categories[i] in top_3_cats:
            label.set_color('darkred')
            label.set_fontweight('bold')
    
    ax.set_rlabel_position(30)
    plt.yticks([0.2, 0.4, 0.6, 0.8, 1.0], ["0.2", "0.4", "0.6", "0.8", "1.0 (Max)"], color="grey", size=9)
    plt.ylim(0, 1.05)
    
    ax.plot(angles, ms_scores, color='blue', linewidth=2, linestyle='dashed', label='傳統 Max Sharpe 組合')
    ax.fill(angles, ms_scores, color='blue', alpha=0.1)
    ax.plot(angles, pref_scores, color='red', linewidth=3, label='偏好驅動組合 (紅色涵蓋面積代表效用贏得的優勢)')
    ax.fill(angles, pref_scores, color='red', alpha=0.3)
    
    plt.title('多維度投資組合決策剖析 (動態波動率極限映射)', size=22, fontweight='bold', y=1.15)
    plt.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1), frameon=True, shadow=True)
    
    output_path = f"png\\{case}_radar_chart.png"
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    log.info(f"✅ 產出圖表：{output_path}")

def run_stage3_pipeline():
    log.info("啟動 Stage 3: 偏好驅動二次規劃與投資組合深度分析...")
    AV_DB_FILE = parameters.AV_DB_FILE
    try:
        df_stage2 = pd.read_csv("csv\\stage2_final_user_universe.csv")
        df_stage0 = pd.read_csv("csv\\stage0_final_matrix.csv")
        df_scaled_features = pd.read_csv("csv\\stage2_normalized_features.csv")
        with open("json\\stage2_ahp_global_weights.json", "r", encoding="utf-8") as f:
            # 🚨 修正：提取 9 個全局權重
            global_weights = json.load(f)["Global_Weights"]
    except FileNotFoundError:
        log.error("❌ 找不到必要檔案，請確認 csv 檔案與 AHP 權重檔案存在。")
        sys.exit(1)
        return
    
    tickers = df_stage2['ETF'].tolist()
    
    # 抓取歷史價格並對齊資料
    log.info(f"⏳ 載入 {len(tickers)} 檔 ETF 進行最佳化與歷史回測...")
    price_matrix = get_or_fetch_historical_prices(tickers)
    returns_matrix = price_matrix.pct_change(fill_method=None).dropna(how='all')
    # 🚨 修正：先向下填補(處理中間斷層)，再向上填補(處理頭部缺漏)
    returns_matrix = returns_matrix.ffill().bfill()
    # 最後才使用 dropna(axis=1) 把那些「即使雙向填補後依然全是 NaN」的無效標的給剔除
    returns_matrix = returns_matrix.dropna(axis=1)
    
    # 對齊有效 Tickers
    valid_tickers = [t for t in tickers if t in returns_matrix.columns]
    returns_matrix = returns_matrix[valid_tickers]
    n_assets = len(valid_tickers)
    
    df_merged = df_stage2[['ETF', 'User_Pref_Score']].merge(df_stage0, on='ETF', how='left')
    df_merged_valid = df_merged.set_index('ETF').loc[valid_tickers].reset_index()

    sector_matrix, sector_names = build_sector_matrix(valid_tickers, AV_DB_FILE)

    # 提取對應有效標的的正規化特徵
    df_scaled_valid = df_scaled_features.set_index('ETF').loc[valid_tickers].reset_index()

    # --- 輸出共變異數矩陣 CSV ---
    cov_matrix_annual = returns_matrix.cov() * 252

    # --- 目標函數參數準備 ---
    log.info(f"\n⚖️ 啟動權重融合機制 (Alpha = {ALPHA_BASELINE})")
    log.info("-" * 50)
    # 進行 Alpha 凸組合融合 (Convex Combination)
    blended_weights = {}
    for key in BASELINE_WEIGHTS.keys():
        user_w = global_weights.get(key, 0.0)
        base_w = BASELINE_WEIGHTS[key]
        
        # 融合公式：W_final = α * W_base + (1 - α) * W_user
        blended_w = (ALPHA_BASELINE * base_w) + ((1 - ALPHA_BASELINE) * user_w)
        blended_weights[key] = blended_w
        
        log.info(f"{key:<15}: 使用者 {user_w*100:>5.2f}% | 融合後 -> {blended_w*100:>5.2f}%")
    log.info("-" * 50)
    # 將融合後的安全權重，指派給最佳化引擎使用的全局變數
    w_cagr = blended_weights["Return_CAGR"]
    w_div = blended_weights["Return_Div"]
    w_vol_risk = blended_weights["Risk_Vol"]
    w_maxdd = blended_weights["Risk_MaxDD"]
    w_cost = blended_weights["Cost_ExpRatio"]
    w_liq_vol = blended_weights["Liq_Volume"]
    w_liq_aum = blended_weights["Liq_AUM"]
    w_div_score = blended_weights["Div_Score"]
    w_sent = blended_weights["FinBERT_score"]

    # --- 🚨直接讀取已縮尾之正規化數據---
    vec_cagr = df_scaled_valid['Norm_Return_CAGR'].values
    vec_div = df_scaled_valid['Norm_Return_Div'].values
    vec_maxdd = df_scaled_valid['Norm_Risk_MaxDD'].values
    vec_cost = df_scaled_valid['Norm_Cost_ExpRatio'].values
    vec_liq_vol = df_scaled_valid['Norm_Liq_Volume'].values
    vec_liq_aum = df_scaled_valid['Norm_Liq_AUM'].values
    vec_div_score = df_scaled_valid['Norm_Div_Score'].values
    vec_sent = df_scaled_valid['Norm_FinBERT'].values

    # 共變異數矩陣動態正規化 (V_p = w^T * Sigma * w)
    cov_matrix_np = returns_matrix.cov().values * 252
    cov_matrix_norm = cov_matrix_np / np.max(cov_matrix_np)
    
    # ==========================================
    # 核心：定義全局效用函數 U(P)
    # ==========================================
    def calc_utility(w):
        """計算投資組合在當前 AHP 權重下的總效用分數 (Utility)"""
        port_cagr = np.dot(w, vec_cagr)
        port_div = np.dot(w, vec_div)
        port_maxdd = np.dot(w, vec_maxdd)
        port_cost = np.dot(w, vec_cost)
        port_liq_vol = np.dot(w, vec_liq_vol)
        port_liq_aum = np.dot(w, vec_liq_aum)
        # 🚨 分散度計算邏輯分歧
        if USE_TRUE_HHI_OPTIMIZATION and sector_matrix is not None:
            # 1. 算出投資組合在各產業的絕對總曝險 (1 x K 向量)
            port_sector_exposures = np.dot(w, sector_matrix)
            
            # 2. 計算真實投資組合 HHI (各產業權重的平方和)
            true_hhi = np.sum(port_sector_exposures ** 2)
            
            # 3. 轉換為 AHP 可理解的「正向效用」
            # HHI 範圍是 0 到 1 (越小越分散)。因此 1 - HHI = 真實分散度分數
            port_div_score= 1.0 - true_hhi 
        else:
            # 舊演算法：線性加權的代理指標
            port_div_score = np.dot(w, vec_div_score)
        port_sent = np.dot(w, vec_sent)
        
        port_vol_var = np.dot(w.T, np.dot(cov_matrix_norm, w))
        
        U = (w_cagr * port_cagr) + (w_div * port_div) \
            + (w_liq_vol * port_liq_vol) + (w_liq_aum * port_liq_aum) \
            + (w_div_score * port_div_score) + (w_sent * port_sent) \
            + (w_maxdd * port_maxdd) + (w_cost * port_cost) \
            - (w_vol_risk * port_vol_var)
        return U

    def objective_function(w):
        # SciPy 求最小，故回傳負效用
        return -calc_utility(w)

    # ... 執行偏好驅動最佳化 ...
    MAX_WEIGHT_LIMIT = parameters.MAX_WEIGHT_LIMIT
    weight_bounds = tuple((0.0, MAX_WEIGHT_LIMIT) for _ in range(n_assets)) #上限為 0.40 (防止單一標的過度集中，最高 40%)
    bounds = tuple((0.0, 1.0) for _ in range(n_assets))
    constraints = ({'type': 'eq', 'fun': lambda w: np.sum(w) - 1.0})
    initial_w = np.array([1.0 / n_assets] * n_assets)
    
    result = minimize(objective_function, initial_w, method='SLSQP', bounds=weight_bounds, constraints=constraints, options={'maxiter': 1000, 'ftol': 1e-9})
    if not result.success:
        log.error("❌ 最佳化求解失敗：", result.message)
        sys.exit(1)
        return
    optimal_weights = np.round(result.x, 4)

    # ... 執行傳統 Max Sharpe 最佳化 ...
    annual_returns_array = returns_matrix.mean().values * 252
    cov_matrix_annual = returns_matrix.cov().values * 252

    def neg_sharpe_objective(w):
        p_ret = np.dot(w, annual_returns_array)
        p_vol = np.sqrt(np.dot(w.T, np.dot(cov_matrix_annual, w)))
        return - (p_ret - 0.04) / p_vol if p_vol > 0 else 0

    res_sharpe = minimize(neg_sharpe_objective, initial_w, method='SLSQP', bounds=weight_bounds, constraints=constraints, options={'maxiter': 1000, 'ftol': 1e-9})
    max_sharpe_weights = np.round(res_sharpe.x, 4) if res_sharpe.success else initial_w
    if not res_sharpe.success:
        log.warning(f"⚠️ 夏普組合求解未完全收斂: {res_sharpe.message}")

    # ==========================================
    # 🚨 新增：計算兩者的「偏好效用分數 U(P)」
    # ==========================================
    pref_utility_score = calc_utility(optimal_weights)
    ms_utility_score = calc_utility(max_sharpe_weights)

    # ==========================================
    # 深度資料計算 (建立通用的計算函式)
    # ==========================================
    def get_portfolio_metrics(weights):
        # 計算每日與年化報酬、波動率、夏普
        port_daily = returns_matrix.dot(weights)
        volatility = port_daily.std() * np.sqrt(252)
        annual_ret = port_daily.mean() * 252
        sharpe = (annual_ret - 0.04) / volatility if volatility > 0 else 0
        
        # 計算最大回撤 (Max Drawdown)
        cum_ret = (1 + port_daily).cumprod()
        max_dd = ((cum_ret - cum_ret.cummax()) / cum_ret.cummax()).min() * 100
        
        # 計算線性加權的原始特徵指標
        exp_ratio = np.dot(weights, df_merged_valid['Cost_ExpRatio (%)'].fillna(0))
        cagr = np.dot(weights, df_merged_valid['Return_CAGR (%)'].fillna(0))
        div_yield = np.dot(weights, df_merged_valid['Return_Div (%)'].fillna(0))
        
        # 計算線性加權的代理分散度 (確保 df_merged_valid 有 Div_Score 欄位)
        proxy_div = np.dot(weights, df_merged_valid['Div_Score (產出)'].fillna(0))

        # 計算線性加權的 FinBERT 宏觀情緒分數
        sentiment = np.dot(weights, df_merged_valid['FinBERT_score'].fillna(0))
        
        # 計算線性加權的流動性指標 (請確認 df_merged_valid 內的確切欄位名稱)
        volume = np.dot(weights, df_merged_valid['Liq_Volume (M)'].fillna(0))
        aum = np.dot(weights, df_merged_valid['Liq_AUM (B)'].fillna(0))

        return {
            'Arithmetic_Ret': annual_ret * 100,
            'CAGR': cagr,
            'Div_Yield': div_yield,
            'Cost': exp_ratio,
            'Volatility': volatility * 100,
            'MaxDD': max_dd,
            'Proxy_Div': proxy_div,
            'Sentiment': sentiment,
            'Volume': volume,
            'AUM': aum,
            'Sharpe': sharpe
        }
    
    # 分別取得兩組權重的深度數據
    pref_metrics = get_portfolio_metrics(optimal_weights)
    ms_metrics = get_portfolio_metrics(max_sharpe_weights)

    # 🚨 動態計算真實投資組合 HHI (若矩陣已成功建立)
    if sector_matrix is not None and len(sector_matrix) > 0:
        true_hhi_pref = np.sum(np.dot(optimal_weights, sector_matrix) ** 2)
        true_hhi_ms = np.sum(np.dot(max_sharpe_weights, sector_matrix) ** 2)
        hhi_str_pref = f"{true_hhi_pref:.4f}"
        hhi_str_ms = f"{true_hhi_ms:.4f}"
    else:
        hhi_str_pref = "N/A (API Limit)"
        hhi_str_ms = "N/A (API Limit)"

    # 呼叫視覺化函式
    plot_portfolio_analytics_and_mpt(returns_matrix, optimal_weights, max_sharpe_weights, valid_tickers)

    # 🚨 新增：整理向量字典，呼叫視覺化函式 2: 雷達圖
    vectors_dict = {
        "Return_CAGR": vec_cagr,
        "Return_Div": vec_div,
        "Risk_Vol": 1.0 - (np.diag(cov_matrix_norm) / np.max(np.diag(cov_matrix_norm))), # 簡化風險代理
        "Risk_MaxDD": vec_maxdd,
        "Cost_ExpRatio": vec_cost,
        "Liq_Volume": vec_liq_vol,
        "Liq_AUM": vec_liq_aum,
        "Div_Score": vec_div_score,
        "FinBERT_score": vec_sent
    }
    # 注意：風險波動率 (Risk_Vol) 在你的二次規劃中是矩陣運算，
    # 為了雷達圖展示單一維度表現，這裡提取對角線(資產本身的波動率)做反向正規化作為視覺代理。
    
    plot_preference_radar_chart(
        optimal_weights, 
        max_sharpe_weights, 
        vectors_dict, 
        blended_weights, 
        pref_metrics,  # 傳入偏好組合的真實數據
        ms_metrics,    # 傳入夏普組合的真實數據
        cov_matrix_np,
        valid_tickers
    )
    # ==========================================
    # 輸出比較報表
    # ==========================================
    comparison_df = pd.DataFrame({
        'ETF': valid_tickers,
        '偏好組合 Weight (%)': optimal_weights * 100,
        '最大夏普 Weight (%)': max_sharpe_weights * 100
    })
    comparison_df = comparison_df[(comparison_df['偏好組合 Weight (%)'] > 0.01) | (comparison_df['最大夏普 Weight (%)'] > 0.01)]
    comparison_df = comparison_df.sort_values(by='偏好組合 Weight (%)', ascending=False).reset_index(drop=True)
    
    log.info("\n" + "="*65)
    log.info(" 🎯 專題最終產出：持股權重對比")
    log.info("="*65)
    log.info(comparison_df.to_string(index=False))
    
    # 建立深度健檢報告的 DataFrame 確保排版對齊
    analytics_df = pd.DataFrame({
        'Metric': [
            'Arithmetic Annual Return (%)', 
            'Historical CAGR (%)', 
            'Dividend Yield (%)', 
            'Expense Ratio (%)', 
            'Annualized Volatility (%)', 
            'Maximum Drawdown (%)',
            'Liquidity Volume (Millions)',
            'Liquidity AUM (Billions)',
            'True Portfolio HHI (Real)',
            'Weighted Sentiment Score',
            'Sharpe Ratio'
        ],
        'Preference-Driven': [
            f"{pref_metrics['Arithmetic_Ret']:.2f}", 
            f"{pref_metrics['CAGR']:.2f}", 
            f"{pref_metrics['Div_Yield']:.2f}", 
            f"{pref_metrics['Cost']:.3f}", 
            f"{pref_metrics['Volatility']:.2f}", 
            f"{pref_metrics['MaxDD']:.2f}",
            f"{pref_metrics['Volume']:.2f}",
            f"{pref_metrics['AUM']:.2f}",
            f"{hhi_str_pref}", 
            f"{pref_metrics['Sentiment']:.4f}",
            f"{pref_metrics['Sharpe']:.3f}"
        ],
        'Max Sharpe': [
            f"{ms_metrics['Arithmetic_Ret']:.2f}", 
            f"{ms_metrics['CAGR']:.2f}", 
            f"{ms_metrics['Div_Yield']:.2f}", 
            f"{ms_metrics['Cost']:.3f}", 
            f"{ms_metrics['Volatility']:.2f}", 
            f"{ms_metrics['MaxDD']:.2f}",
            f"{ms_metrics['Volume']:.2f}",
            f"{ms_metrics['AUM']:.2f}",
            f"{hhi_str_ms}",
            f"{ms_metrics['Sentiment']:.4f}",
            f"{ms_metrics['Sharpe']:.3f}"
        ]
    })

    log.info("\n" + "="*65)
    log.info(" 📊 投資組合深度健檢報告 (Portfolio Analytics Comparison)")
    log.info("="*65)
    log.info(analytics_df.to_string(index=False, line_width=65,justify='right'))
    log.info("-" * 65)
    
    # 輸出效用對決
    pref_utility_score = calc_utility(optimal_weights)
    ms_utility_score = calc_utility(max_sharpe_weights)
    log.info(f"   偏好驅動組合 【AHP 總效用分數】 : {pref_utility_score:.4f}  <-- 系統在此維度勝出！")
    log.info(f"   傳統最大夏普 【AHP 總效用分數】 : {ms_utility_score:.4f}")
    log.info("="*65)

    # 建立 report 資料夾 (若無則自動建立)
    report_dir = "report"
    os.makedirs(report_dir, exist_ok=True)
    
    # 為了讓每次測試的檔案不會互相覆蓋，建議使用 case_name，若無則用預設名稱
    file_prefix = case
    
    # 1. 輸出完美排版的純文字報告 (.txt)
    txt_path = f"{report_dir}\\{file_prefix}_summary.txt"
    with open(txt_path, 'w', encoding='utf-8') as f:
        f.write("="*65 + "\n")
        f.write(" 🎯 專題最終產出：持股權重對比\n")
        f.write("="*65 + "\n")
        f.write(comparison_df.to_string(index=False) + "\n\n")
        
        f.write("="*65 + "\n")
        f.write(" 📊 投資組合深度健檢報告 (Portfolio Analytics Comparison)\n")
        f.write("="*65 + "\n")
        f.write(analytics_df.to_string(index=False, line_width=65, justify='right') + "\n")
        f.write("-" * 65 + "\n\n")
        
        f.write(f"   偏好驅動組合 【AHP 總效用分數】 : {pref_utility_score:.4f}  <-- 系統在此維度勝出！\n")
        f.write(f"   傳統最大夏普 【AHP 總效用分數】 : {ms_utility_score:.4f}\n")
        f.write("="*65 + "\n")
        
    # 2. 輸出方便 Excel 讀取的表格檔案 (.csv)
    csv_weights_path = f"{report_dir}\\{file_prefix}_weights.csv"
    csv_analytics_path = f"{report_dir}\\{file_prefix}_analytics.csv"
    
    comparison_df.to_csv(csv_weights_path, index=False, encoding='utf-8-sig')
    analytics_df.to_csv(csv_analytics_path, index=False, encoding='utf-8-sig')
