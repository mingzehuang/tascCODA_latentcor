import numpy as np
import pandas as pd
import re
import pickle
"""import toytree as tt
from tasccoda import tree_utils as util
from tasccoda import tree_ana as ana
from tasccoda import tree_results as tr
from tasccoda import tree_agg_model_sslasso as mod
import anndata as ad
from sccoda.util import comp_ana as ana2
from sccoda.model import other_models as om
import toyplot
import toyplot.locator"""
import statsmodels as sm
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.patches as mpatches
import lzma
"""import toyplot.svg"""
import os
import scanpy as sc
data_path = '../../../tascCODA_data/applications/smillie_UC/SCP259/expression/5cdc540d328cee7a2efc234a/'
adata = sc.read(data_path + 'gene_sorted-Imm.matrix.mtx')
expression = adata.X
expression_dense = expression.todense()
print(expression)
print(expression_dense)
print(expression_dense.shape)

"""data_path = '../../../tascCODA_data/applications/smillie_UC/SCP259/metadata/'"""


meta = pd.read_csv(data_path + 'meta_processed.csv')
meta["Cluster"] = [str.replace(x, " ", "") for x in meta["Cluster"]]
meta
print(meta)

vars = meta.groupby("Cluster").agg({
    "Major_l1": "first",
    "Major_l2": "first",
    "Major_l3": "first",
    "Major_l4": "first",
})
vars
print(vars)

with lzma.open(os.path.join(os.getcwd(), "vars.xz"), "wb", preset = 9) as f:
    pickle.dump(vars, f)