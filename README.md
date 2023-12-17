# CoDy Evaluation

This repository holds the code for the evaluation of "Counterfactual Explainer for Models on Dynamic Graphs" (CoDy) and "Greedy Explainer for Models on Dynamic Graphs using Coutnerfactuals" (GreeDyCF).

The repository is structured into different folders:
- [notebooks](./notebooks): Contains Jupyter notebooks that are used to conduct the evaluation of the explanation apporoaches. It contains one notebook for each of the datasets used in the evaluation.
- [plots](./plots): Contains plots exported from the Jupyter notebooks as tikz/pgfplots code that can be used in LaTeX. Additionally, some of the plots are also exported to SVG.
- [results](./results): Contains the raw results that are achieved from running the evalutaion of CoDy, GreeDyCF, and T-GNNExplainer.
- [tables](./tables): Tables exported from the analysis notebooks.

### Running the analysis notebooks

For everything to work correctly, please use the following versions of the dependiencies:

```
matplotlib==3.5.3
numpy==1.25.2
pandas==2.0.1
pyarrow==14.0.1
scikit-learn==1.3.2
scipy==1.11.4
seaborn==0.13.0
tikzplotlib==0.10.1
```