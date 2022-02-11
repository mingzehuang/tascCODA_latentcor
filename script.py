import lzma
from timeit import repeat
import pkg_resources
from latentcor import latentcor
import pickle
import numpy
import matplotlib.pyplot as plt
from scipy import stats
import pandas
import umap
a=numpy.array([0, 2, 4])
print(stats.gmean(a[a!=0]))

with lzma.open(pkg_resources.resource_stream("latentcor", "counts.xz"), "rb") as f:
     counts = pickle.load(f)
counts.to_csv(path_or_buf='counts.csv')
with lzma.open(pkg_resources.resource_stream("latentcor", "vars.xz"), "rb") as f:
     vars = pickle.load(f)
vars.to_csv(path_or_buf='vars.csv')
print(counts)
print(counts.reset_index())
print(vars)
print(vars.shape)
print(vars['Major_l1'])
print(set(vars['Major_l1']))
print(set(counts.reset_index()['Health']))
print(set(counts.reset_index()['Location']))
health = counts.reset_index()['Health']
location = counts.reset_index()['Location']
print(len(health))
health_status = numpy.repeat('b', len(health))
health_status[health == 'Non-inflamed'] = 'g'
health_status[health == 'Inflamed'] = 'r'
location_status = numpy.repeat(0, len(location))
location_status[location == 'LP'] = 1
print(health_status)
counts_mat = numpy.array(counts)
counts_prop = counts_mat / numpy.sum(counts_mat,axis=1)[ : , None]
print(counts_prop)
print(numpy.sum(counts_prop, axis=1))
"""cl+pearson"""
counts_prop_add1 = counts_prop + 1
print(counts_prop_add1)
counts_gmean_cl = stats.gmean(counts_prop_add1, axis=1)
print(counts_gmean_cl)
counts_pseudo_cl = numpy.log(counts_prop_add1 / counts_gmean_cl[ : , None])
print(counts_pseudo_cl)
counts_pseudo_cl_corr = numpy.array(pandas.DataFrame(counts_pseudo_cl).corr(method = 'pearson'))
print(counts_pseudo_cl_corr)
w, v = numpy.linalg.eig(counts_pseudo_cl_corr)
print(w)
first_eigvec = v[ : , 0]
print(first_eigvec)
second_eigvec = v[ : , 1]
print(second_eigvec)
colors = numpy.repeat('b', len(vars['Major_l1']))
colors[vars['Major_l1'] == 'Immune'] = 'g'
colors[vars['Major_l1'] == 'Stromal'] = 'r'
plt.scatter(first_eigvec[vars['Major_l1'] == 'Epithelial'], second_eigvec[vars['Major_l1'] == 'Epithelial'], s = 15, c=colors[vars['Major_l1'] == 'Epithelial'], alpha=0.5, label='Epithelial')
plt.scatter(first_eigvec[vars['Major_l1'] == 'Immune'], second_eigvec[vars['Major_l1'] == 'Immune'], s = 15, c=colors[vars['Major_l1'] == 'Immune'], alpha=0.5, label='Immune')
plt.scatter(first_eigvec[vars['Major_l1'] == 'Stromal'], second_eigvec[vars['Major_l1'] == 'Stromal'], s = 15, c=colors[vars['Major_l1'] == 'Stromal'], alpha=0.5, label='Stromal')
plt.legend()
plt.title("PCA for Pearson (pxp)")
plt.show()
plt.savefig("PCA for Pearson (pxp).png")

fit = umap.UMAP()
u = fit.fit_transform(v[ : , 0:10].real)
plt.scatter(u[:,0][vars['Major_l1'] == 'Epithelial'], u[:,1][vars['Major_l1'] == 'Epithelial'], c=colors[vars['Major_l1'] == 'Epithelial'], alpha=0.5, label='Epithelial')
plt.scatter(u[:,0][vars['Major_l1'] == 'Immune'], u[:,1][vars['Major_l1'] == 'Immune'], c=colors[vars['Major_l1'] == 'Immune'], alpha=0.5, label='Immune')
plt.scatter(u[:,0][vars['Major_l1'] == 'Stromal'], u[:,1][vars['Major_l1'] == 'Stromal'], c=colors[vars['Major_l1'] == 'Stromal'], alpha=0.5, label='Stromal')
plt.legend()
plt.title("PCA10+UMAP for Pearson (pxp)")
plt.show()
plt.savefig("PCA10+UMAP for Pearson (pxp).png")

fit = umap.UMAP()
u = fit.fit_transform(counts_pseudo_cl_corr)
plt.scatter(u[:,0][vars['Major_l1'] == 'Epithelial'], u[:,1][vars['Major_l1'] == 'Epithelial'], c=colors[vars['Major_l1'] == 'Epithelial'], alpha=0.5, label='Epithelial')
plt.scatter(u[:,0][vars['Major_l1'] == 'Immune'], u[:,1][vars['Major_l1'] == 'Immune'], c=colors[vars['Major_l1'] == 'Immune'], alpha=0.5, label='Immune')
plt.scatter(u[:,0][vars['Major_l1'] == 'Stromal'], u[:,1][vars['Major_l1'] == 'Stromal'], c=colors[vars['Major_l1'] == 'Stromal'], alpha=0.5, label='Stromal')
plt.legend()
plt.title("UMAP for Pearson (pxp)")
plt.show()
plt.savefig("UMAP for Pearson (pxp).png")

"""mcl+latentcor"""
counts_gmean_mcl = numpy.repeat(numpy.nan, counts_prop.shape[0])
for i in range(counts_prop.shape[0]):
     counts_prop_row = counts_prop[i, : ]
     counts_gmean_mcl[i] = stats.gmean(counts_prop_row[counts_prop_row != 0])
     print('%i row has %f zeros' % (i, len(counts_prop_row[counts_prop_row == 0]) / len(counts_prop_row)))
print(counts_gmean_mcl)
counts_pseudo_mcl = numpy.log(counts_prop / counts_gmean_mcl[ : , None])
print(counts_pseudo_mcl)
counts_pseudo_mcl = counts_pseudo_mcl + abs(numpy.min(counts_pseudo_mcl[counts_pseudo_mcl != -numpy.infty]))
print(counts_pseudo_mcl)
counts_pseudo_mcl[counts_pseudo_mcl == -numpy.infty] = 0
print(counts_pseudo_mcl)
print(counts_pseudo_mcl.shape)
output = latentcor(X=counts_pseudo_mcl, tps=numpy.repeat("tru", counts_pseudo_mcl.shape[1]))
print(output[1])
print('minimum pointwise estimator is %f' % (numpy.min(output[1])))

"""positive definite"""
counts_pseudo_mcl_latentcor = output[0]
print(counts_pseudo_mcl_latentcor)
w, v = numpy.linalg.eig(counts_pseudo_mcl_latentcor)
print(w)
first_eigvec = v[ : , 0]
print(first_eigvec)
second_eigvec = v[ : , 1]
print(second_eigvec)
plt.scatter(first_eigvec[vars['Major_l1'] == 'Epithelial'], second_eigvec[vars['Major_l1'] == 'Epithelial'], s = 15, c=colors[vars['Major_l1'] == 'Epithelial'], alpha=0.5, label='Epithelial')
plt.scatter(first_eigvec[vars['Major_l1'] == 'Immune'], second_eigvec[vars['Major_l1'] == 'Immune'], s = 15, c=colors[vars['Major_l1'] == 'Immune'], alpha=0.5, label='Immune')
plt.scatter(first_eigvec[vars['Major_l1'] == 'Stromal'], second_eigvec[vars['Major_l1'] == 'Stromal'], s = 15, c=colors[vars['Major_l1'] == 'Stromal'], alpha=0.5, label='Stromal')
plt.legend()
plt.title("PCA for latentcor (pxp)")
plt.show()
plt.savefig("PCA for latentcor (pxp).png")

fit = umap.UMAP()
u = fit.fit_transform(v[ : , 0:10].real)
plt.scatter(u[:,0][vars['Major_l1'] == 'Epithelial'], u[:,1][vars['Major_l1'] == 'Epithelial'], c=colors[vars['Major_l1'] == 'Epithelial'], alpha=0.5, label='Epithelial')
plt.scatter(u[:,0][vars['Major_l1'] == 'Immune'], u[:,1][vars['Major_l1'] == 'Immune'], c=colors[vars['Major_l1'] == 'Immune'], alpha=0.5, label='Immune')
plt.scatter(u[:,0][vars['Major_l1'] == 'Stromal'], u[:,1][vars['Major_l1'] == 'Stromal'], c=colors[vars['Major_l1'] == 'Stromal'], alpha=0.5, label='Stromal')
plt.legend()
plt.title("PCA10+UMAP for latentcor (pxp)")
plt.show()
plt.savefig("PCA10+UMAP for latentcor (pxp).png")

fit = umap.UMAP()
u = fit.fit_transform(counts_pseudo_mcl_latentcor)
plt.scatter(u[:,0][vars['Major_l1'] == 'Epithelial'], u[:,1][vars['Major_l1'] == 'Epithelial'], c=colors[vars['Major_l1'] == 'Epithelial'], alpha=0.5, label='Epithelial')
plt.scatter(u[:,0][vars['Major_l1'] == 'Immune'], u[:,1][vars['Major_l1'] == 'Immune'], c=colors[vars['Major_l1'] == 'Immune'], alpha=0.5, label='Immune')
plt.scatter(u[:,0][vars['Major_l1'] == 'Stromal'], u[:,1][vars['Major_l1'] == 'Stromal'], c=colors[vars['Major_l1'] == 'Stromal'], alpha=0.5, label='Stromal')
plt.legend()
plt.title("UMAP for latentcor (pxp)")
plt.show()
plt.savefig("UMAP for latentcor (pxp).png")

"""pointwise"""
counts_pseudo_mcl_latentcor = output[1]
print(counts_pseudo_mcl_latentcor)
w, v = numpy.linalg.eig(counts_pseudo_mcl_latentcor)
print(w)
first_eigvec = v[ : , 0]
print(first_eigvec)
second_eigvec = v[ : , 1]
print(second_eigvec)
plt.scatter(first_eigvec[vars['Major_l1'] == 'Epithelial'], second_eigvec[vars['Major_l1'] == 'Epithelial'], s = 15, c=colors[vars['Major_l1'] == 'Epithelial'], alpha=0.5, label='Epithelial')
plt.scatter(first_eigvec[vars['Major_l1'] == 'Immune'], second_eigvec[vars['Major_l1'] == 'Immune'], s = 15, c=colors[vars['Major_l1'] == 'Immune'], alpha=0.5, label='Immune')
plt.scatter(first_eigvec[vars['Major_l1'] == 'Stromal'], second_eigvec[vars['Major_l1'] == 'Stromal'], s = 15, c=colors[vars['Major_l1'] == 'Stromal'], alpha=0.5, label='Stromal')
plt.legend()
plt.title("PCA for latentcor pointwise (pxp)")
plt.show()
plt.savefig("PCA for latentcor pointwise (pxp).png")

fit = umap.UMAP()
u = fit.fit_transform(v[ : , 0:10].real)
plt.scatter(u[:,0][vars['Major_l1'] == 'Epithelial'], u[:,1][vars['Major_l1'] == 'Epithelial'], c=colors[vars['Major_l1'] == 'Epithelial'], alpha=0.5, label='Epithelial')
plt.scatter(u[:,0][vars['Major_l1'] == 'Immune'], u[:,1][vars['Major_l1'] == 'Immune'], c=colors[vars['Major_l1'] == 'Immune'], alpha=0.5, label='Immune')
plt.scatter(u[:,0][vars['Major_l1'] == 'Stromal'], u[:,1][vars['Major_l1'] == 'Stromal'], c=colors[vars['Major_l1'] == 'Stromal'], alpha=0.5, label='Stromal')
plt.legend()
plt.title("PCA10+UMAP for latentcor pointwise (pxp)")
plt.show()
plt.savefig("PCA10+UMAP for latentcor pointwise (pxp).png")

fit = umap.UMAP()
u = fit.fit_transform(counts_pseudo_mcl_latentcor)
plt.scatter(u[:,0][vars['Major_l1'] == 'Epithelial'], u[:,1][vars['Major_l1'] == 'Epithelial'], c=colors[vars['Major_l1'] == 'Epithelial'], alpha=0.5, label='Epithelial')
plt.scatter(u[:,0][vars['Major_l1'] == 'Immune'], u[:,1][vars['Major_l1'] == 'Immune'], c=colors[vars['Major_l1'] == 'Immune'], alpha=0.5, label='Immune')
plt.scatter(u[:,0][vars['Major_l1'] == 'Stromal'], u[:,1][vars['Major_l1'] == 'Stromal'], c=colors[vars['Major_l1'] == 'Stromal'], alpha=0.5, label='Stromal')
plt.legend()
plt.title("UMAP for latentcor pointwise (pxp)")
plt.show()
plt.savefig("UMAP for latentcor pointwise (pxp).png")



"""cl+pearson, nxn"""
counts_T_add1 = counts_mat.T + 1
print(counts_T_add1)
counts_T_gmean_cl = stats.gmean(counts_T_add1, axis=1)
print(counts_T_gmean_cl)
counts_T_pseudo_cl = numpy.log(counts_T_add1 / counts_T_gmean_cl[ : , None])
print(counts_T_pseudo_cl)
counts_T_pseudo_cl_corr = numpy.array(pandas.DataFrame(counts_T_pseudo_cl).corr(method = 'pearson'))
print(counts_T_pseudo_cl_corr)
w, v = numpy.linalg.eig(counts_T_pseudo_cl_corr)
print(w)
first_eigvec = v[ : , 0]
print(first_eigvec)
second_eigvec = v[ : , 1]
print(second_eigvec)
plt.scatter(first_eigvec[location_status == 0], second_eigvec[location_status == 0], marker='^', c=health_status[location_status == 0], alpha=0.5, label='EPi')
plt.scatter(first_eigvec[location_status == 1], second_eigvec[location_status == 1], marker='o', c=health_status[location_status == 1], alpha=0.5, label='LP')
plt.legend()
plt.title("PCA for Pearson (nxn)")
plt.show()
plt.savefig("PCA for Pearson (nxn).png")

fit = umap.UMAP()
u = fit.fit_transform(v[ : , 0:10].real)
plt.scatter(u[:,0][location_status == 0], u[:,1][location_status == 0], marker='^', c=health_status[location_status == 0], alpha=0.5, label='EPi')
plt.scatter(u[:,0][location_status == 1], u[:,1][location_status == 1], marker='o', c=health_status[location_status == 1], alpha=0.5, label='LP')
plt.legend()
plt.title("PCA10+UMAP for Pearson (nxn)")
plt.show()
plt.savefig("PCA10+UMAP for Pearson (nxn).png")

fit = umap.UMAP()
u = fit.fit_transform(counts_T_pseudo_cl_corr)
plt.scatter(u[:,0][location_status == 0], u[:,1][location_status == 0], marker='^', c=health_status[location_status == 0], alpha=0.5, label='EPi')
plt.scatter(u[:,0][location_status == 1], u[:,1][location_status == 1], marker='o', c=health_status[location_status == 1], alpha=0.5, label='LP')
plt.legend()
plt.title("UMAP for Pearson (nxn)")
plt.show()
plt.savefig("UMAP for Pearson (nxn).png")


out = latentcor(X=counts_mat.T, tps=numpy.repeat("tru", (counts_mat.T).shape[1]))
print(out[1])
print('minimum pointwise estimator is %f' % (numpy.min(out[1])))

"""latentcor"""
w, v = numpy.linalg.eig(out[0])
print(w)
first_eigvec = v[ : , 0]
print(first_eigvec)
second_eigvec = v[ : , 1]
print(second_eigvec)
plt.scatter(first_eigvec[location_status == 0], second_eigvec[location_status == 0], marker='^', c=health_status[location_status == 0], alpha=0.5, label='EPi')
plt.scatter(first_eigvec[location_status == 1], second_eigvec[location_status == 1], marker='o', c=health_status[location_status == 1], alpha=0.5, label='LP')
plt.legend()
plt.title("PCA for latentcor (nxn)")
plt.show()
plt.savefig("PCA for latentcor (nxn).png")

fit = umap.UMAP()
u = fit.fit_transform(v[ : , 0:10].real)
plt.scatter(u[:,0][location_status == 0], u[:,1][location_status == 0], marker='^', c=health_status[location_status == 0], alpha=0.5, label='EPi')
plt.scatter(u[:,0][location_status == 1], u[:,1][location_status == 1], marker='o', c=health_status[location_status == 1], alpha=0.5, label='LP')
plt.legend()
plt.title("PCA10+UMAP for latentcor (nxn)")
plt.show()
plt.savefig("PCA10+UMAP for latentcor (nxn).png")

fit = umap.UMAP()
u = fit.fit_transform(out[0])
plt.scatter(u[:,0][location_status == 0], u[:,1][location_status == 0], marker='^', c=health_status[location_status == 0], alpha=0.5, label='EPi')
plt.scatter(u[:,0][location_status == 1], u[:,1][location_status == 1], marker='o', c=health_status[location_status == 1], alpha=0.5, label='LP')
plt.legend()
plt.title("UMAP for latentcor (nxn)")
plt.show()
plt.savefig("UMAP for latentcor (nxn).png")

"""pointwise"""
w, v = numpy.linalg.eig(out[1])
print(w)
first_eigvec = v[ : , 0]
print(first_eigvec)
second_eigvec = v[ : , 1]
print(second_eigvec)
plt.scatter(first_eigvec[location_status == 0], second_eigvec[location_status == 0], marker='^', c=health_status[location_status == 0], alpha=0.5, label='EPi')
plt.scatter(first_eigvec[location_status == 1], second_eigvec[location_status == 1], marker='o', c=health_status[location_status == 1], alpha=0.5, label='LP')
plt.legend()
plt.title("PCA for latentcor pointwise (nxn)")
plt.show()
plt.savefig("PCA for latentcor pointwise (nxn).png")

fit = umap.UMAP()
u = fit.fit_transform(v[ : , 0:10].real)
plt.scatter(u[:,0][location_status == 0], u[:,1][location_status == 0], marker='^', c=health_status[location_status == 0], alpha=0.5, label='EPi')
plt.scatter(u[:,0][location_status == 1], u[:,1][location_status == 1], marker='o', c=health_status[location_status == 1], alpha=0.5, label='LP')
plt.legend()
plt.title("PCA10+UMAP for latentcor pointwise (nxn)")
plt.show()
plt.savefig("PCA10+UMAP for latentcor pointwise (nxn).png")

fit = umap.UMAP()
u = fit.fit_transform(out[1])
plt.scatter(u[:,0][location_status == 0], u[:,1][location_status == 0], marker='^', c=health_status[location_status == 0], alpha=0.5, label='EPi')
plt.scatter(u[:,0][location_status == 1], u[:,1][location_status == 1], marker='o', c=health_status[location_status == 1], alpha=0.5, label='LP')
plt.legend()
plt.title("UMAP for latentcor pointwise (nxn)")
plt.show()
plt.savefig("UMAP for latentcor pointwise (nxn).png")

"""Epi"""
fit = umap.UMAP()
u = fit.fit_transform(out[0][location_status == 0])
plt.scatter(u[:,0], u[:,1], marker='^', c=health_status[location_status == 0], alpha=0.5)
plt.legend()
plt.title("UMAP EPi for latentcor (nxn)")
plt.show()
plt.savefig("UMAP Epi for latentcor (nxn).png")

"""LP"""
fit = umap.UMAP()
u = fit.fit_transform(out[0][location_status == 1])
plt.scatter(u[:,0], u[:,1], marker='^', c=health_status[location_status == 1], alpha=0.5)
plt.legend()
plt.title("UMAP LP for latentcor (nxn)")
plt.show()
plt.savefig("UMAP LP for latentcor (nxn).png")

"""
print(counts_mat)
print(counts_mat.shape)
output = latentcor(X=counts_mat, tps=numpy.repeat("tru", counts_mat.shape[1]))
w, v = numpy.linalg.eig(output[0])
print(w)
first_eigvec = v[ : , 0]
print(first_eigvec)
second_eigvec = v[ : , 1]
print(second_eigvec)
colors = range(len(first_eigvec))
plt.scatter(first_eigvec, second_eigvec, s = 15, c=colors, alpha=0.5)
plt.show()

output2 = latentcor(X=counts_mat.T, tps=numpy.repeat("tru", (counts_mat.T).shape[1]))
w, v = numpy.linalg.eig(output2[0])
print(w)
first_eigvec = v[ : , 0]
print(first_eigvec)
second_eigvec = v[ : , 1]
print(second_eigvec)
plt.scatter(first_eigvec, second_eigvec, s = 15, c=health_status, alpha=0.5)
plt.show()"""
