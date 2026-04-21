# ==========================================
# 參數設定區
# ==========================================
CSV_UNIVERSE_FILE = "csv\\all_etfs.csv"       # 步驟一產生的全市場排序清單
YQ_OUTPUT_FILE = "csv\\stage0_yq_features.csv" # YQ 財務特徵的輸出檔案
AV_DB_FILE = "json\\etf_database.json"         # AV 分散度特徵的本地資料庫
AV_API_KEY = "TY6TXNC6W7D2YT9N"          # Alpha Vantage API Key TY6TXNC6W7D2YT9N 
AV_MAX_CALLS_PER_DAY = 25                # AV 每日安全呼叫上限
TOP_N_ETFS = 500                         # 我們要送入 DEA 模型的候選數量
FINNHUB_API_KEY = "d7g00o9r01qqb8rhrkvgd7g00o9r01qqb8rhrl00"  # 請填入你的 Finnhub API Key
OUTPUT_FILE = "csv\\stage0_final_matrix.csv" # 最終多維度特徵矩陣的輸出檔案
USE_TRUE_HHI_OPTIMIZATION = True # 是否使用真正的 HHI 最小化優化（會增加計算時間），否則使用近似方法
ALPHA_BASELINE = 0.0 # 基線模型的 alpha 預期值 (可以根據歷史數據調整)
BASELINE_WEIGHTS = { # 專家先驗權重矩陣 (總和必須為 1.0)
    "Return_CAGR": 0.23,
    "Return_Div": 0.10,
    "Risk_Vol": 0.10,
    "Risk_MaxDD": 0.12,
    "Cost_ExpRatio": 0.15,
    "Liq_Volume": 0.05,
    "Liq_AUM": 0.05,
    "Div_Score": 0.15,
    "FinBERT_score": 0.0
}
MAX_WEIGHT_LIMIT = 0.40 # 單一標的最大權重限制 (40%)
CASE_NAME = "Neutral_user" # 預設情境名稱
VERBOSE = False # 是否輸出詳細的運算過程訊息 True (Verbose 模式) -> 顯示普通資訊與除錯訊息，False (Silence 模式) -> 只顯示警告與錯誤
DETERMINISTIC_AHP_WEIGHTS = True # 這裡可以切換是否使用確定性模擬結果, 若想讓教授體驗互動問卷，可以改成 DETERMINISTIC_AHP_WEIGHTS=False

# AHP 尺度提醒：
# 1.0 = 兩者同等重要
# 3.0 = 前者稍微重要 | 1/3 = 後者稍微重要
# 5.0 = 前者明顯重要 | 1/5 = 後者明顯重要
# 7.0 = 前者強烈重要 | 1/7 = 後者強烈重要
# 9.0 = 前者極端重要 | 1/9 = 後者極端重要
DETERMINISTIC_USER_INPUTS = {
    "Main": [0.75, 3.0, 6.0, 3.0, 6.0, 4.0, 8.0, 4.0, 8.0, 2.0, 1.0, 2.0, 0.5, 1.0, 2.0],
    "Sub": {'Return_Main': [2.3333], 'Risk_Main': [4.0], 'Liquidity_Main': [1.0]}
}
'''
DETERMINISTIC_USER_INPUTS = {
    "Main": [
        # --- [0] 報酬 (Return) vs 其他 ---
        1.0,  # 0:  報酬 vs 風險 (Risk)
        1.0,  # 1:  報酬 vs 成本 (Cost)
        1.0,  # 2:  報酬 vs 流動性 (Liquidity)
        1.0,  # 3:  報酬 vs 產業分散 (Diversity)
        1.0,  # 4:  報酬 vs 市場情緒 (Sentiment)

        # --- [1] 風險 (Risk) vs 其他 ---
        1.0,  # 5:  風險 vs 成本
        1.0,  # 6:  風險 vs 流動性
        1.0,  # 7:  風險 vs 產業分散
        1.0,  # 8:  風險 vs 市場情緒

        # --- [2] 成本 (Cost) vs 其他 ---
        1.0,  # 9:  成本 vs 流動性
        1.0,  # 10: 成本 vs 產業分散
        1.0,  # 11: 成本 vs 市場情緒

        # --- [3] 流動性 (Liquidity) vs 其他 ---
        1.0,  # 12: 流動性 vs 產業分散
        1.0,  # 13: 流動性 vs 市場情緒

        # --- [4] 產業分散 (Diversity) vs 其他 ---
        1.0   # 14: 產業分散 vs 市場情緒
    ],
    
    "Sub": {
        # 歷史報酬 (CAGR) vs 殖利率 (Div)
        "Return_Main": [1.0],     
        
        # 抗波動 (Vol) vs 抗回撤 (MaxDD)
        "Risk_Main": [1.0],       
        
        # 交易量 (Volume) vs 資產規模 (AUM)
        "Liquidity_Main": [1.0]   
    }
}
'''