# FinTech-Project
Preference-driven ETF portfolio optimization program

# 偏好驅動之智能 ETF 資產配置系統
**(Preference-Driven Smart ETF Portfolio Optimization)**

本專案結合**作業研究 (OR)** 理論與金融科技 (FinTech)，旨在打造一套具備「效率篩選、偏好理解、風險防護與數學最佳化」的客製化 ETF 機器人理財系統。系統不僅追求傳統的財務效率，也透過 AHP 或自然語言貝式推論，將投資人的主觀偏好轉換為可計算的投資組合效用函數。

## 🌟 核心系統架構 (System Pipeline)

本系統目前分為五個主要階段：

1. **Stage 0 - 市場資料擷取與特徵處理 (Market Data Preparation):**
   - 串接 YahooQuery、Alpha Vantage、Finnhub 等資料來源，取得 ETF 財務特徵、歷史價格、產業分散度與新聞資料。
   - 使用本地端 FinBERT 模型，將財經新聞轉換為時間衰減加權的情緒分數。
   - 進行 EDA、特徵合併、AUM 校正、DEA 前置正規化與客觀降維。

2. **Stage 1 - DEA 效率篩選 (DEA Screening):**
   - 以風險與成本作為 Input，報酬、流動性、分散度與情緒分數作為 Output。
   - 執行標準 DEA、超級效率 DEA 與交互效率 DEA，剔除相對無效率標的。
   - 產出具備財務效率與競爭力的 ETF 候選池。

3. **Stage 2_1 - 使用者偏好提取 (Preference Extraction):**
   - **Stage 2_1-A Static AHP:** 保留原本的靜態 AHP 問卷，將使用者偏好轉成 9 維 Global Weights，作為 baseline 與傳統方法對照組。
   - **Stage 2_1-B Active Bayesian:** 新增自然語言偏好探測流程，透過動態提問、語意萃取與階層式貝式信念更新，估計使用者偏好的 $\mu$ 與 $\sigma$。
   - 兩種方法最後都輸出至 `json/stage2_ahp_global_weights.json`，讓後續流程維持同一個介面。

4. **Stage 2_2 - 高相關 ETF 分群與偏好篩選 (Preference Cluster Selection):**
   - 將相關性過高的 ETF 分群，避免最終投資組合塞入高度重複的標的。
   - 依照使用者偏好分數，保留每個群集中最符合偏好的 ETF。

5. **Stage 3 - 偏好投資組合最佳化 (Preference Portfolio Optimization):**
   - 將使用者偏好權重放入效用函數，使用 `scipy.optimize` 的 SLSQP 求解偏好驅動投資組合。
   - 加入單一 ETF 權重上限、產業 HHI 分散度與風險尺度映射等防護機制。
   - 與傳統 Max Sharpe 投資組合比較，輸出權重、深度分析報告、MPT 效率前緣與多維度雷達圖。

## 🛠️ 安裝與環境設定 (Installation)

請確保您的電腦已安裝 Python 3.8 或以上版本。

1. Clone 此專案至本地端：
   ```bash
   git clone https://github.com/JasonRott/FinTech-Project.git
   cd FinTech-Project
   ```

2. 載入需要的模組與套件：
   ```bash
   pip install -r requirements.txt
   ```

3. 若要使用 Gemini / Active Bayesian 相關功能，可另外安裝：
   ```bash
   pip install google-generativeai
   ```

## 🚀 執行方式 (Usage)

主程式入口為：

```bash
python main.py
```

在 `main.py` 中可以切換使用者偏好提取模式：

```python
preference_mode="static_ahp"
```

或：

```python
preference_mode="active_bayesian"
```

也可以透過 `PipelineConfig` 選擇只執行部分階段，例如只跑 Stage 2_1：

```python
from pipeline_stages import PipelineConfig, run_full_pipeline

run_full_pipeline(
    PipelineConfig(
        run_stage0_fetch=False,
        run_stage0_feature_processing=False,
        run_stage1_dea=False,
        run_stage2_1_preference=True,
        run_stage2_2_cluster_selection=False,
        run_stage3_optimization=False,
        preference_mode="active_bayesian",
    )
)
```

## 📁 檔案管理 (Project Files)

- `main.py`：主程式入口，負責設定 pipeline 參數與執行階段。
- `pipeline_stages.py`：統一管理 Stage 0、1、2_1、2_2、3 的流程入口，並在每個主階段開始與結束時輸出提示文字。
- `functions.py`：各階段核心函式彙整，包含資料擷取、DEA、AHP、分群篩選、最佳化求解器與視覺化分析。
- `parameters.py`：全域參數設定，例如 API key、ETF 數量、單一標的權重上限、AHP deterministic inputs。
- `active_preference/`：自然語言偏好探測、Gemini 訪談、合成訓練資料、特徵編碼與 BNN 相關模組。
- `AHP_weights_setting_script.py`：測試 AHP 權重時使用，可由預期權重反推 AHP 成對比較矩陣。
- `ARCHITECTURE.md`：更詳細的架構說明與各 stage 對應函式。

## 📂 目錄說明 (Directories)

- `csv/`：儲存各階段矩陣與表格，例如 ETF universe、DEA 結果、最終候選池與正規化特徵矩陣。
- `json/`：儲存 ETF database、AHP 問卷、使用者偏好權重、Active Bayesian 狀態。
- `local_finbert/`：儲存本地 FinBERT 模型，加速新聞情緒分數計算。
- `png/`：儲存 EDA、DEA、MPT、投資組合績效與雷達圖等視覺化結果。
- `report/`：儲存最終偏好投資組合與 Max Sharpe 投資組合的權重表與深度分析報告。

## 🔎 主要輸出 (Outputs)

- `csv/stage0_final_matrix.csv`：ETF 多維度原始特徵矩陣。
- `csv/stage0_dea_ready_matrix.csv`：DEA 前置正規化矩陣。
- `csv/stage1_final_candidates.csv`：DEA 交互效率篩選後的候選 ETF。
- `json/stage2_ahp_global_weights.json`：使用者偏好權重，無論來源是 AHP 或 Active Bayesian 都會輸出到此介面。
- `csv/stage2_final_user_universe.csv`：經高相關分群與偏好篩選後的最終 ETF universe。
- `report/*_summary.txt`、`report/*_weights.csv`、`report/*_analytics.csv`：最終投資組合分析報告。
- `png/*_portfolio_performance.png`、`png/*_mpt_efficient_frontier.png`、`png/*_radar_chart.png`：投資組合視覺化結果。

## 📌 設計取捨 (Design Notes)

- AHP 仍保留為 baseline、靜態問卷與傳統方法對照組。
- Active Bayesian 是後續主研究方向，可處理自然語言回答、偏好不確定性與動態提問。
- Stage 3 不直接依賴偏好來源，只讀取 `json/stage2_ahp_global_weights.json`，確保求解器介面穩定。
- 若 Alpha Vantage 觸發每日 API 限制，系統仍會使用本地 `json/etf_database.json` 既有資料繼續後續流程。
