import numpy as np
from matplotlib import pyplot as plt
import util
from numba import njit
import argparse
from pathlib import Path


parser = argparse.ArgumentParser()
parser.add_argument('-L', type=int, default=4, dest="L")
parser.add_argument('-v', type=int, default=6, dest="v")
parser.add_argument('-s', type=int, default=2, dest="s")
parser.add_argument('-m', type=int, default=4, dest="m")
parser.add_argument('--rules_seed', type=int, default=0, dest="rules_seed")
parser.add_argument('--epsilon', type=float, default=0.1, dest="epsilon")
parser.add_argument('--temperature', type=float, nargs="+", default=[0.1], dest="temperature")
parser.add_argument('--num_str', type=int, nargs="+", default=[100], dest="num_str")
parser.add_argument('--max_distance', type=int, default=None, dest="max_distance")
parser.add_argument('--save_fig', type=bool, default=True, dest="save_fig")
parser.add_argument('--save_data', type=bool, default=True, dest="save_data")
parser.add_argument('--path_fig', type=str, default="plots/", dest="path_fig")
parser.add_argument('--path_data', type=str, default="data/", dest="path_data")
parser.add_argument('--last_token_mi', type=bool, default=True, dest="last_token_mi")
parser.add_argument('--neglect_pos_dep', type=bool, default=False, dest="neglect_pos_dep")
parser.add_argument('--sample', type=bool, default=True, dest="sample")
parser.add_argument('--log_lin', type=bool, default=True, dest="log_lin")
parser.add_argument('--fit', type=bool, default=True, dest="fit")
parser.add_argument('--num_fit_steps', type=int, default=4, dest="num_fit_steps")
parser.add_argument('--rules_str', type=str, default="special01", dest="rules_str")
parser.add_argument('--top_p_prob', type=float, default=1.0, dest="top_p_prob")
parser.add_argument('--sample_seed', type=int, default=0, dest="sample_seed")



args = parser.parse_args()
L = args.L
v = args.v
s= args.s
m = args.m
rules_seed = args.rules_seed
epsilon = args.epsilon
temperature = args.temperature
num_str = args.num_str
max_distance = args.max_distance
save_fig = args.save_fig
save_data = args.save_data
path_fig = args.path_fig
path_data = args.path_data
last_token_mi = args.last_token_mi
neglect_pos_dep = args.neglect_pos_dep
sample = args.sample
log_lin=args.log_lin
fit=args.fit
num_fit_steps=args.num_fit_steps
rules_str = args.rules_str
top_p_prob = args.top_p_prob
sample_seed = args.sample_seed


for p in [path_fig, path_data]:
    path_obj = Path(p)
    if not path_obj.exists():
        path_obj.mkdir(parents=True, exist_ok=True)

print("last_token_mi:")
print(last_token_mi)

if len(temperature) == 1:
    temperature = temperature[0]
if len(num_str) == 1:
    num_str = num_str[0]


rhm_para = {"L": L, "v": v, "s": s, "m": m, "rules_seed": rules_seed,
            "epsilon": epsilon, "rules_str": rules_str}
plot_class = util.process_and_visualize_rhm_mutual_information(**rhm_para)

if sample:
  fig, ax, title = plot_class.plot_mi_sampled_top_p(
      temperature=temperature, num_str=num_str, top_p_prob=top_p_prob, max_distance=max_distance, dot_size=3.0, verbose=True,
      save_fig=save_fig, save_data=save_data, last_token_mi=last_token_mi, neglect_pos_dep=neglect_pos_dep, log_lin=log_lin, fit=fit, num_fit_steps=num_fit_steps, seed=sample_seed,
      )
else:
    fig, ax, title = plot_class.plot_mi_exact_no_temperature(
        dot_size=3.0, verbose=True, save_fig=save_fig, save_data=save_data, last_token_mi=last_token_mi, neglect_pos_dep=neglect_pos_dep, log_lin=log_lin, fit=fit, num_fit_steps=num_fit_steps,
        )