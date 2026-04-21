# FinTech-Project
Preference-driven ETF portfolio optimization program
# 偏好驅動之智能 ETF 資產配置系統 
**(Preference-Driven Smart ETF Portfolio Optimization)**

本專案結合**作業研究 (OR)** 理論與金融科技 (FinTech)，旨在打造一套具備「信託責任防線」的客製化機器人理財系統。系統不僅追求傳統的財務效率，更透過量化模型融合投資人的主觀偏好。

## 🌟 核心系統架構 (System Pipeline)

本系統分為四個核心運算階段：
1. **Stage 0 - 數據獲取與 NLP 情緒萃取 (Data ETL & FinBERT):** - 串接 YahooQuery、Alpha Vantage、Finnhub API 獲取財務基本面與產業分散度。
   - 部署本地端 FinBERT 模型，將 180 天歷史財經新聞標題轉換為「時間衰減加權」的宏觀情緒分數。
2. **Stage 1 - DEA 效率前緣過濾 (Data Envelopment Analysis):** - 以風險與成本為 Input，報酬、流動性與分散度為 Output，剔除相對無效率的 ETF 標的，確保進入決策池的資產具備紮實的財務體質。
3. **Stage 2 - AHP 偏好量化 (Analytic Hierarchy Process):** - 透過層級分析問卷，將使用者的模糊主觀偏好，轉化為 9 大維度的數學權重矩陣。
   - 將相關性高於0.99的ETF分群，依照使用者偏好(來自AHP權重)保留各分群中最優者。
4. **Stage 3 - 具備邊界防護的二次規劃 (Quadratic Programming with Fiduciary Bounds):** - 將 $\alpha$ 信託底線權重與使用者權重融合，防止極端偏好。
   - 使用 `scipy.optimize` 進行二次規劃，並透過單一標的 40% 上限與真實 HHI 產業矩陣，確保最終投資組合的結構安全性。

## 🛠️ 安裝與環境設定 (Installation)

請確保您的電腦已安裝 Python 3.8 或以上版本。

1. Clone 此專案至本地端：
   ```bash
   git clone [https://github.com/your-username/your-repo-name.git](https://github.com/your-username/your-repo-name.git)
   cd your-repo-name
2. 載入需要的模組與套件
   ```bash
   pip install -r requirements.txt

3. 檔案管理:
   - parameters.py 參數設定
   - main.py 主程式架構
   - functions.py 各階段所使用函式彙整，可使用ctrl^f(階段名稱)查詢各階段函式位置
      - 階段名稱:
         - stage0_0_get_sorted_ETFsorted
         - stage0_1_ETF_data_input
         - stage0_2_EDA_and_Visualization
         - stage0_3_regularization_and_dimensionality_reduction
         - stage1_DEA_efficiency_calculation
         - stage2_0_AHP_weight_final_candidates_selection
         - stage2_1_preference_driven_deduplication
         - stage3_Preference_Driven_Portfolio_Optimization
   - AHP_weights_setting_script.py 測試權重時使用，輸入預期權重能夠反推出AHP矩陣，再貼上parameters.py中的DETERMINISTIC_USER_INPUTS字典即可
   - Directory:
      - csv: 儲存各階段矩陣與表格
      - json: 
         - etf_database.json: 儲存各ETF產業分布與前三持股占比
         - questionnaire.json: 儲存AHP問卷
         - stage2_ahp_global_weights.json: 儲存使用者偏好權重與該次填答之CR值
      - local_finbert: 儲存本地FinBERT模型，加速計算情緒分數
      - png: 儲存特徵視覺化圖表與最佳化分析結果圖表
      - report: 儲存最佳化偏好投資組合與最佳化夏普值投資組合，與其深度分析
