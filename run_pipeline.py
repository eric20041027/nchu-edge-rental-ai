import subprocess
import sys
import os
import time

def run_script(script_path):
    print(f"\n" + "="*60)
    print(f"🚀 Running: {script_path}")
    print("="*60)
    
    start_time = time.time()
    # 使用當前的 Python 解譯器執行
    process = subprocess.Popen([sys.executable, script_path], stdout=None, stderr=None)
    process.wait()
    
    if process.returncode != 0:
        print(f"\n❌ Error: {script_path} failed with return code {process.returncode}")
        sys.exit(1)
        
    duration = time.time() - start_time
    print(f"✅ Finished: {script_path} (Took {duration:.1f}s)")

def main():
    # 確保在專案根目錄執行
    base_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(base_dir)

    # 定義完整流水線順序
    pipeline = [
        "pipeline/data_prep/update_commute_data.py",      # 1. 更新通勤時間 (OSRM)
        "pipeline/data_prep/precompute_embeddings.py",   # 2. 生成前端資產
        "pipeline/data_prep/generate_dataset.py",       # 3. 生成訓練數據集
        "pipeline/model_training/train_and_export_onnx.py", # 4. 訓練並導出 ONNX
        "pipeline/model_training/quantize_model.py",     # 5. 模型量化
        "pipeline/model_training/evaluate_model.py"      # 6. 最終效能評估
    ]

    print("🌟 Starting End-to-End Rental AI Pipeline 🌟")
    total_start = time.time()

    for script in pipeline:
        if os.path.exists(script):
            run_script(script)
        else:
            print(f"⚠️ Warning: Script {script} not found, skipping...")

    total_duration = time.time() - total_start
    print(f"\n" + "🎉"*20)
    print(f"全流程執行完畢！總耗時: {total_duration/60:.1f} 分鐘")
    print("現在您可以將代碼推送到 GitHub/Vercel 上線了。")
    print("🎉"*20)

if __name__ == "__main__":
    main()
