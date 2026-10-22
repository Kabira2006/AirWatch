import os
import nbformat
from nbconvert.preprocessors import ExecutePreprocessor

def run_notebook():
    notebook_path = "Initial_work/AirWatch_Initial_Stage.ipynb"
    print(f"Executing {notebook_path}...")
    
    with open(notebook_path, "r", encoding="utf-8") as f:
        nb = nbformat.read(f, as_version=4)

    ep = ExecutePreprocessor(timeout=600, kernel_name="python3")

    try:
        ep.preprocess(nb, {"metadata": {"path": "Initial_work"}})
        print("Notebook executed successfully!")
    except Exception as e:
        print(f"Error executing notebook: {e}")

    with open(notebook_path, "w", encoding="utf-8") as f:
        nbformat.write(nb, f)
        
    print(f"Saved executed notebook to {notebook_path}")

if __name__ == "__main__":
    run_notebook()
