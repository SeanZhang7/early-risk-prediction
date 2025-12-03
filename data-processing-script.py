import nbformat
from nbconvert.preprocessors import ExecutePreprocessor

def run_notebook(path):
    with open(path) as f:
        nb = nbformat.read(f, as_version=4)

    ep = ExecutePreprocessor(timeout=-1)

    ep.preprocess(nb, {'metadata': {'path': '.'}})

    with open(path, 'w', encoding='utf-8') as f:
        nbformat.write(nb, f)

# Run notebooks in order
run_notebook("data-processing.ipynb")
run_notebook("merge_data.ipynb")

print("All notebooks executed successfully!")
