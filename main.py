import sys
import subprocess
from pathlib import Path

def run_step(script_path, description):
    """Executes a python script from a specific path."""
    print(f"\n🚀 {description}...")
    
    # Check if the file actually exists before running
    if not Path(script_path).exists():
        print(f"⚠️  Skipping: {script_path} not found.")
        return True # Continue to next step

    try:
        # Run using the current Python interpreter
        subprocess.run(
            [sys.executable, script_path],
            check=True
        )
        print(f"✅ Success: {script_path}")
        return True
    except subprocess.CalledProcessError:
        print(f"❌ Error occurred in {script_path}")
        return False

def main():
    # 1. Setup Folders
    for folder in ['data/raw', 'data/processed', 'models', 'reports/figures']:
        Path(folder).mkdir(parents=True, exist_ok=True)

    # 2. Define the execution sequence (pointing to the src/ folder)
    # We add 'src/' because that is where your .py files are located
    pipeline_steps = [
        ('src/generate_data.py', 'Generating Synthetic Data'),
        ('src/data_preprocessing.py', 'Preprocessing Data'),
        ('src/full_eda.py', 'Running Exploratory Data Analysis'),
        ('src/modeling.py', 'Training Models (RF & XGBoost)'),
        ('src/generate_top_customers.py', 'Identifying At-Risk Customers')
    ]

    print("🏁 Starting Telecom Churn Pipeline")
    print("="*40)

    for script_path, desc in pipeline_steps:
        success = run_step(script_path, desc)
        if not success:
            print("\n⛔ Pipeline stopped due to a critical error.")
            sys.exit(1)

    print("\n" + "="*40)
    print("🎉 ALL STEPS COMPLETED SUCCESSFULLY!")

if __name__ == "__main__":
    main()