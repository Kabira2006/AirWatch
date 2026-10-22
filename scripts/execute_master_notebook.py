import os
import nbformat
from nbclient import NotebookClient

def execute_master():
    notebook_path = "AirWatch_Comprehensive_Master.ipynb"
    print(f"Reading {notebook_path}...")
    with open(notebook_path, "r", encoding="utf-8") as f:
        nb = nbformat.read(f, as_version=4)

    print("Executing notebook cells with NotebookClient (timeout=600s)...")
    client = NotebookClient(nb, timeout=600, kernel_name="python3")

    try:
        client.execute()
        print("Master notebook execution completed successfully!")
    except Exception as e:
        print(f"Exception during execution (partial execution saved): {e}")

    print("Saving executed notebook back to disk...")
    with open(notebook_path, "w", encoding="utf-8") as f:
        nbformat.write(nb, f)

    print(f"Executed master notebook saved to {notebook_path}!")

if __name__ == "__main__":
    execute_master()
