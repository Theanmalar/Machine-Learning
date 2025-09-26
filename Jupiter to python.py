import os
import nbformat

# Folder containing your notebooks and subfolders
source_folder = r"ML"

# Destination folder for Python scripts
destination_folder = r"C:\Users\User\Documents\ML"
os.makedirs(destination_folder, exist_ok=True)

# Function to extract only code cells from notebook
def convert_notebook_to_py_code_only(ipynb_path, py_output_path):
    with open(ipynb_path, 'r', encoding='utf-8') as f:
        notebook = nbformat.read(f, as_version=4)

    # Collect only code cells
    code_lines = []
    for cell in notebook.cells:
        if cell.cell_type == 'code':
            code_lines.append(cell.source)
            code_lines.append('\n\n')  # Add space between code blocks

    # Write to .py file
    with open(py_output_path, 'w', encoding='utf-8') as f:
        f.writelines(code_lines)

# Walk through all files inside the folder and subfolders
for root, dirs, files in os.walk(source_folder):
    for file in files:
        if file.endswith(".ipynb") and not file.startswith("."):
            full_input_path = os.path.join(root, file)

            # Create a unique .py filename (flatten folder names to prevent conflict)
            relative_path = os.path.relpath(full_input_path, source_folder)
            safe_filename = relative_path.replace(os.sep, "_").replace(".ipynb", ".py")
            full_output_path = os.path.join(destination_folder, safe_filename)

            # Convert and save only code
            convert_notebook_to_py_code_only(full_input_path, full_output_path)

            print(f"✅ Saved Python code: {full_output_path}")

print("\n🎉 All notebooks have been converted to Python code files (code cells only)!")

