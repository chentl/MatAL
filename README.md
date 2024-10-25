## MatAL

Codes for "Machine Intelligence-Accelerated Discovery of All-Natural Plastic Substitutes".

### Python environment & JupyterLab setup

```python
cd MatAL

conda env create -f environment.yml
conda activate matal-py10

jupyter lab
```

### Notebooks:

- ```design_boundary.ipynb```: Feasibility design boundary
- ```data_augmentation.ipynb```: Data augmentation for model training
- ```active_learning.ipynb```: Composition genration in active learning
- ```plot_shaps.ipynb```: SHAP analysis
- ```reverse_design.ipynb```: Reverse design based on performance requirements
- ```hpopt.py```: Hyperparameter optimization for ANN models 
