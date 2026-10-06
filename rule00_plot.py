import numpy as np
from matplotlib import pyplot as plt
import matplotlib as mpl
import scipy
from numba import njit
from collections.abc import Iterable
from util import *


# RHM BP Vary Sigma Last Token
# Parameters
L = 5   # depth
v = 8   # number of symbols
s = 2   # branching factor
m = 3   # number of high probability rules per parent
rules_seed = 0
epsilon = 0.0
rules_str = "special00"
temperature = [ 0.1, 1.0, 2.0]
num_str=100000
max_distance=None
log_lin = False
fit = True
num_fit_steps = 8
save_data = False
save_fig = False
last_token_mi=True
data_sampled=True
neglect_pos_dep=False
add_title=False
return_fit = True
top_p_prob = 1.0

T2_shift_temperature=None
all_temperatures = temperature
temps = np.array(all_temperatures)
log_min = np.log(temps.min())
log_max = np.log(temps.max())
normalized_array = (np.log(temps) - log_min) / (log_max - log_min)
cmap = mpl.colormaps['gist_rainbow']
all_colors = cmap(normalized_array)
colors = [all_colors[idx] for idx in range(len(all_temperatures))]



rhm_para = {"L": L, "v": v, "s": s, "m": m, "rules_seed": rules_seed,
            "epsilon": epsilon, "rules_str": rules_str}

mask_mi=None

errors=True


path_fig="plots/"


ylim=None
xlim_exp = None
ylim_exp = None


fig, ax, title, fig_exp, ax_exp =mi_plotter_from_data(rhm_para, temperature, num_str, add_title=add_title, data_sampled=data_sampled,
                            max_distance=max_distance, save_fig=save_fig,
                            neglect_pos_dep=neglect_pos_dep, last_token_mi=last_token_mi,
                            log_lin=log_lin, fit=fit, num_fit_steps=num_fit_steps, color=colors,
                            T2_shift_temperature=T2_shift_temperature, ylim=ylim,
                            exponent_plot=True, path_data="data/", mask_mi=mask_mi,
                            errors=errors, legend=False, path_fig=path_fig, xlim_exp=xlim_exp, ylim_exp=ylim_exp)

norm = mpl.colors.LogNorm(vmin=min(temperature), vmax=max(temperature))
sm = mpl.cm.ScalarMappable(cmap=cmap, norm=norm)
sm.set_array([])
cbar = fig.colorbar(sm, ax=ax)
cbar.set_ticks(temperature)
cbar.set_ticklabels([str(t) for t in temperature])
cbar.set_label('Temperature') # Optional: Add a title to the colorbar


title="rule00_s2_epsilon01"

for line in ax[0,0].get_lines():
    line.set_rasterized(True)
for collection in ax[0,0].collections:
    collection.set_rasterized(True)

fig_exp.savefig(path_fig+title+"_exponents.jpeg",dpi=600)
fig_exp.savefig(path_fig+title+"_exponents.pdf",dpi=600)

fig.savefig(path_fig+title+".jpeg",dpi=600)
fig.savefig(path_fig+title+".pdf",dpi=600)