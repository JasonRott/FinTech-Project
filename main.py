import os
import json
import parameters
import sys
from functions import *

# ==========================================
# 系統初始化與防呆設定
# ==========================================
def initialize_environment():
    """確保所有必要的輸出資料夾都存在"""
    folders = ["csv", "json", "png"]
    for folder in folders:
        os.makedirs(folder, exist_ok=True)
    print("✅ 系統環境初始化完成，所需資料夾已確認。")

# ==========================================
# 補齊 Stage 2 的執行與存檔函式
# ==========================================
def run_stage2_ahp_pipeline():
    log.info("\n啟動 Stage 2: 執行 AHP 問卷並生成全局權重...")
    try:
        # 取得模擬使用者的問卷結果 (Deterministic=True 代表使用預設模擬的結果)
        deterministic = parameters.DETERMINISTIC_AHP_WEIGHTS
        user_inputs = build_user_simulation(deterministic=deterministic)
        
        # 實例化 AHP 模型並計算
        ahp_model = TwoLevel_AHP_Model()
        global_weights, cr = ahp_model.calculate_global_weights(user_inputs)
        
        # 存檔供後續 Stage 使用
        output_data = {
            "CR": cr,
            "Global_Weights": global_weights
        }
        with open("json\\stage2_ahp_global_weights.json", "w", encoding="utf-8") as f:
            json.dump(output_data, f, ensure_ascii=False, indent=4)
        log.info("✅ AHP 權重計算完畢，已儲存至 json\\stage2_ahp_global_weights.json")
        
    except Exception as e:
        log.error(f"❌ Stage 2 AHP 執行失敗: {e}")
        sys.exit(1)

# ==========================================
# 主流程編排 (Orchestration)
# ==========================================
if __name__ == "__main__":
    print("="*50)
    print("偏好驅動之智能 ETF 資產配置系統 - 主流程啟動")
    print("="*50)

    initialize_environment()

    # ---------------------------------------------------------
    # 🎛️ 執行開關 (Toggle Flags)
    # 在開發階段，你可以將已經跑完且耗時的階段設為 False
    # ---------------------------------------------------------
    RUN_STAGE_0_DATA_FETCH   = True    # 極度耗時，包含 API 呼叫與 NLP 推論 (建議跑過一次後關閉)
    RUN_STAGE_0_EDA_AND_NORM = True   # EDA 繪圖與資料降維正規化
    RUN_STAGE_1_DEA          = True   # 三階段 DEA 效率評估
    RUN_STAGE_2_AHP          = True   # AHP 權重計算與白名單過濾
    RUN_STAGE_3_OPTIMIZATION = True   # 二次規劃與深度健檢報告
    # ---------------------------------------------------------

    # ==========================================
    # Stage 0: 獲取全市場資料與特徵建立
    # ==========================================
    if RUN_STAGE_0_DATA_FETCH:
        print("="*50)
        print("▶️ 執行 Stage 0: 數據獲取與特徵工程")

        # 0.1 取得全市場清單並篩選 Top N
        get_all_etfs()
        target_tickers = get_target_tickers_from_csv(parameters.CSV_UNIVERSE_FILE, parameters.TOP_N_ETFS)
        
        if target_tickers:
            # 0.2 依序呼叫各大 API 抓取數據 (內部皆帶有快取與斷點續傳機制)
            fetch_etf_data_yq(target_tickers)
            build_etf_database_av(target_tickers)
            clean_existing_database()
            append_sentiment_to_csv() # 包含 FinBERT 執行
            
            # 0.3 最終合併與數值校正
            merge_final_features()
            patch_aum_from_csv()
            print("✅ Stage 0 數據獲取與特徵工程完成")
    else:
        log.warning("⏩ [略過] Stage 0 數據獲取 (使用既有快取資料)。")

    # ==========================================
    # Stage 0.5: EDA 與降維
    # ==========================================
    if RUN_STAGE_0_EDA_AND_NORM:
        print("="*50)
        print("▶️ 執行 Stage 0.5: 探索性分析與特徵降維")
        run_stage0_2_eda()
        run_stage0_normalization_and_reduction()
        print("✅ Stage 0.5 EDA 與降維完成")
    else:
        log.warning("\n⏩ [略過] Stage 0.5 EDA 與降維。")

    # ==========================================
    # Stage 1: DEA 效率前緣過濾
    # ==========================================
    if RUN_STAGE_1_DEA:
        print("="*50)
        print("▶️ 執行 Stage 1: DEA 效率包絡分析")
        run_stage1_normalized_dea()
        plot_dea_distribution()
        run_stage1_super_efficiency_normalized()
        run_cross_efficiency_dea()
        print("✅ Stage 1 DEA 效率分析完成")
    else:
        log.warning("\n⏩ [略過] Stage 1 DEA。")

    # ==========================================
    # Stage 2 & 2.5: AHP 偏好分析與分群去重
    # ==========================================
    if RUN_STAGE_2_AHP:
        print("="*50)
        print("▶️ 執行 Stage 2: AHP 權重計算與白名單過濾")
        run_stage2_ahp_pipeline()
        run_stage2_5_preference_deduplication_yq()
        print("✅ Stage 2 AHP 與 2.5 分群去重完成")
    else:
        log.warning("\n⏩ [略過] Stage 2 AHP。")

    # ==========================================
    # Stage 3: 二次規劃最佳化與產出報告
    # ==========================================
    if RUN_STAGE_3_OPTIMIZATION:
        print("="*50)
        print("▶️ 執行 Stage 3: 二次規劃與投資組合健檢")
        run_stage3_pipeline()
        print("✅ Stage 3 二次規劃與健檢完成")
    else:
        print("\n⏩ [略過] Stage 3 二次規劃。")

    log.warning("\n🎉 全線管線執行完畢！所有報表已更新至 png/ 與 csv/ 目錄。")
