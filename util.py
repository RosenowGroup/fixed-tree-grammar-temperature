# import dependencies
import numpy as np
from numba import njit
from collections.abc import Iterable
from matplotlib import pyplot as plt
import matplotlib as mpl
import scipy
from itertools import product


def linebreak_title(title, max_len=50):
    """
    Inserts line breaks into a title if it's longer than max_len characters.
    Splits at the latest '_' before the limit.
    
    Args:
        title (str): The input title.
        max_len (int): Maximum allowed characters per line.
    
    Returns:
        str: Title with line breaks.
    """
    result_lines = []
    remaining = title

    while len(remaining) > max_len:
        # Find last underscore before max_len
        break_pos = remaining.rfind("_", 0, max_len + 1)
        if break_pos == -1:
            # No underscore found, just break at max_len
            break_pos = max_len
            result_lines.append(remaining[:break_pos])
            remaining = remaining[break_pos:] 
        else:
            result_lines.append(remaining[:break_pos])
            remaining = remaining[break_pos + 1:]  # skip underscore

    result_lines.append(remaining)  # add final chunk
    return "\n".join(result_lines)


def generate_old_rules_unambiguous(L, v, s, m, rules_seed=0):
    """
    Generates a ruleset for a hierarchical model where each layer has `m` distinct rules
    for each parent value, and each rule is a set of `s` outputs (symbols) from `v` values.

    Parameters:
        L (int): Number of layers in the model.
        v (int): Number of possible symbol values.
        s (int): Number of symbols (width) per rule.
        m (int): Number of distinct rules per parent value.

    Returns:
        rules_array (np.ndarray): Array of shape (L-1, v, m, s) containing the rule set.
    """
    
    np.random.seed(rules_seed)
    rng = np.random.default_rng(rules_seed)
    
    max_distinct = v**(s-1)
    if m > max_distinct:
        raise ValueError(f"Cannot generate {m} distinct rules with v={v}, s={s}. "
                         f"Maximum possible is {max_distinct}.")
    
    rules_array = np.zeros((L-1, v, m, s), dtype=int)

    tuples = list(product(*[range(v) for _ in range(s)]))

    for layer in range(L-1):
        rules_array[layer] = np.array(rng.choice(tuples, v*m, replace=False)).reshape(v,m,-1)
    return rules_array



def generate_rules_unambiguous_epsilon(L, v, s, m, epsilon, rules_seed=0):
  shape = [L-1,v]+[v for idx in range(s)]
  rules_array_old = generate_old_rules_unambiguous(L, v, s, m, rules_seed=rules_seed)
  rules_array = np.full(shape,epsilon/(v**s-m), dtype="float")
  for l in range(L-1):
    for _m in range(m):
      for _v in range(v):
        rules_array[l,_v][tuple(rules_array_old[l,_v,_m])] = (1-epsilon)*1/m
  return rules_array


# def rules_special01(L, v, s, rules_seed=0):
#     np.random.seed(rules_seed)
#     probabilities = 1/np.array(range(1,v**s+1))
#     probabilities = probabilities/np.sum(probabilities)

#     shape = [L-1,v]+[v for idx in range(s)]
#     rules_array = np.full(shape,0.1, dtype="float")

#     grid = np.indices((v,)*s).reshape(s, -1).T
#     for l in range(L-1):
#       for idx_tuple, tuple_of_idx in enumerate(grid):
#           for parent_idx in range(v):
#             rules_array[l,parent_idx][tuple((tuple_of_idx+parent_idx)%v)] = probabilities[idx_tuple]
#     return rules_array

# def rules_special02(L, v, s, rules_seed=0):
#     np.random.seed(rules_seed)
#     probabilities = (1/np.array(range(1,v**s+1)))**2
#     probabilities = probabilities/np.sum(probabilities)

#     shape = [L-1,v]+[v for idx in range(s)]
#     rules_array = np.full(shape,0.1, dtype="float")

#     grid = np.indices((v,)*s).reshape(s, -1).T
#     for l in range(L-1):
#       for idx_tuple, tuple_of_idx in enumerate(grid):
#           for parent_idx in range(v):
#             rules_array[l,parent_idx][tuple((tuple_of_idx+parent_idx)%v)] = probabilities[idx_tuple]
#     return rules_array
    
# def rules_special03(L, v, s, rules_seed=0):
#     if v!=3 or s!=3:
#       raise ValueError("v and s must be 3")
#     np.random.seed(rules_seed)
#     probabilities = (1/np.array(range(1,v**s+1)))**2
#     probabilities = probabilities/np.sum(probabilities)

#     shape = [L-1,v]+[v for idx in range(s)]
#     rules_array = np.full(shape,0.1, dtype="float")

#     rules_array_small = np.full(shape[1:],0, dtype="float")
#     rules_array_small[0,0,0,0] = 2/3
#     rules_array_small[0,0,1,2] = 1/3

#     rules_array_small[1,1,1,1] = 2/3
#     rules_array_small[1,1,2,0] = 1/3

#     rules_array_small[2,2,2,2] = 2/3
#     rules_array_small[2,2,0,1] = 1/3

#     for l in range(L-1):
#       rules_array[l] = rules_array_small
#     return rules_array

# def rules_special04(L, v, s, rules_seed=0):
#     if v!=3 or s!=4:
#       raise ValueError("v must be 3 and s must be 4")
#     np.random.seed(rules_seed)
#     probabilities = (1/np.array(range(1,v**s+1)))**2
#     probabilities = probabilities/np.sum(probabilities)

#     shape = [L-1,v]+[v for idx in range(s)]
#     rules_array = np.full(shape,0.1, dtype="float")

#     rules_array_small = np.full(shape[1:],0, dtype="float")
#     rules_array_small[0,0,0,0,0] = 2/3
#     rules_array_small[0,0,1,2,0] = 1/3

#     rules_array_small[1,1,1,1,1] = 2/3
#     rules_array_small[1,1,2,0,1] = 1/3

#     rules_array_small[2,2,2,2,2] = 2/3
#     rules_array_small[2,2,0,1,1] = 1/3

#     for l in range(L-1):
#       rules_array[l] = rules_array_small
#     return rules_array

# def rules_special05(L, v, s, rules_seed=0):
#     if v!=4 or s!=4:
#       raise ValueError("v and s must be 4")
#     np.random.seed(rules_seed)
#     probabilities = (1/np.array(range(1,v**s+1)))**2
#     probabilities = probabilities/np.sum(probabilities)

#     shape = [L-1,v]+[v for idx in range(s)]
#     rules_array = np.full(shape,0.1, dtype="float")

#     rules_array_small = np.full(shape[1:],0, dtype="float")
#     rules_array_small[0,0,0,0,0] = 2/3
#     rules_array_small[0,0,1,2,3] = 1/3

#     rules_array_small[1,1,1,1,1] = 2/3
#     rules_array_small[1,1,2,3,1] = 1/3

#     rules_array_small[2,2,2,2,2] = 2/3
#     rules_array_small[2,2,3,0,1] = 1/3

#     rules_array_small[3,3,3,3,3] = 2/3
#     rules_array_small[3,3,0,1,2] = 1/3

#     for l in range(L-1):
#       rules_array[l] = rules_array_small
#     return rules_array
    
# def rules_special06(L, v, s, rules_seed=0, rule_std=1.0):
#     rng = np.random.default_rng(rules_seed)
#     shape = [L-1,v]+[v for idx in range(s)]
#     rules_array = np.full(shape,0.1, dtype="float")

#     probabilities = rng.normal(loc=0, scale=1, size=v**s)
#     probabilities = probabilities-np.mean(probabilities)
#     probabilities = rule_std*probabilities/np.std(probabilities)
#     probabilities = np.exp(probabilities)
#     probabilities = probabilities/np.sum(probabilities)

#     grid = np.indices((v,)*s).reshape(s, -1).T
#     for l in range(L-1):
#       for idx_tuple, tuple_of_idx in enumerate(grid):
#           for parent_idx in range(v):
#             rules_array[l,parent_idx][tuple((tuple_of_idx+parent_idx)%v)] = probabilities[idx_tuple]
#     return rules_array

# def rules_special07(L, v, s, rules_seed=0, rule_std=10.1):
#     rng = np.random.default_rng(rules_seed)
#     shape = [L-1,v]+[v for idx in range(s)]
#     rules_array = np.full(shape,0.1, dtype="float")

#     grid = np.indices((v,)*s).reshape(s, -1).T
#     for l in range(L-1):
#       for parent_idx in range(v):
#         probabilities = rng.normal(loc=0, scale=1, size=v**s)
#         probabilities = probabilities-np.mean(probabilities)
#         probabilities = rule_std*probabilities/np.std(probabilities)
#         probabilities = np.exp(probabilities)
#         probabilities = probabilities/np.sum(probabilities)
#         for idx_tuple, tuple_of_idx in enumerate(grid):
#               rules_array[l,parent_idx][tuple((tuple_of_idx+parent_idx)%v)] = probabilities[idx_tuple]
#     return rules_array

# def rules_special08(L, v, s, m, epsilon, rules_seed=0):
#     max_distinct = v**(s-1)
#     if m > max_distinct:
#         raise ValueError(f"Cannot generate {m} distinct rules with v={v}, s={s}. "
#                           f"Maximum possible is {max_distinct}.")

#     rng = np.random.default_rng(rules_seed)

#     rules_array_old = np.zeros((L-1, v, m, s), dtype=int)

#     for layer in range(L-1):
#       if m==max_distinct:
#         grid = np.indices((v,)*s).reshape(s, -1).T
#         distinct_rules = grid[rng.permutation(len(grid))]
#         rules_array_old[layer] = np.reshape(np.array(distinct_rules), (v, m, s))
#       else:
#         distinct_rules = []
#         while len(distinct_rules) < v*m:
#           new_rule = rng.integers(0, v, s)
#           is_distinct = True
#           for existing_rule in distinct_rules:
#             if np.array_equal(new_rule, existing_rule):
#               is_distinct = False
#               break  # If a match is found, the new rule is not distinct
#           if is_distinct:
#             distinct_rules.append(new_rule)

#         rules_array_old[layer] = np.reshape(np.array(distinct_rules), (v, m, s))

#     shape = [L-1,v]+[v for idx in range(s)]
#     rules_array = np.full(shape,epsilon/(v**s-m), dtype="float")
#     prefactor = np.array(np.arange(1,m+1), dtype=float)**-1
#     prefactor = prefactor/np.sum(prefactor)
#     for l in range(L-1):
#       for _m in range(m):
#         for _v in range(v):
#           rules_array[l,_v][tuple(rules_array_old[l,_v,_m])] = (1-epsilon)*prefactor[_m]
#     return rules_array


# def rules_special09(L, v, s, m, epsilon, rules_seed=0):
#     max_distinct = v**(s-1)
#     if m > max_distinct:
#         raise ValueError(f"Cannot generate {m} distinct rules with v={v}, s={s}. "
#                           f"Maximum possible is {max_distinct}.")

#     rng = np.random.default_rng(rules_seed)

#     rules_array_old = np.zeros((L-1, v, m, s), dtype=int)

#     for layer in range(L-1):
#       distinct_rules = []
#       while len(distinct_rules) < v*m:
#         new_rule = rng.integers(0, v, s)
#         is_distinct = True
#         for existing_rule in distinct_rules:
#           for i in range(v):  # Iterate over the v subarrays in new_rule
#             for j in range(v): # Iterate over the v subarrays in existing_rule
#               if np.array_equal(new_rule, existing_rule):
#                 is_distinct = False
#                 break  # If a match is found, the new rule is not distinct
#             if not is_distinct:
#               break
#           if not is_distinct:
#             break

#         if is_distinct:
#           distinct_rules.append(new_rule)

#       rules_array_old[layer] = np.reshape(np.array(distinct_rules), (v, m, s))

#     shape = [L-1,v]+[v for idx in range(s)]
#     rules_array = np.full(shape,epsilon/(v**s-m), dtype="float")
#     for l in range(L-1):
#       for _m in range(m):
#         for _v in range(v):
#           rules_array[l,_v][tuple(rules_array_old[l,_v,_m])] = (1-epsilon)/m
#     return rules_array
    
# def rules_special10(L, v, s, m, epsilon, rules_seed=0):
#     max_distinct = v**(s-1)
#     if m > max_distinct:
#         raise ValueError(f"Cannot generate {m} distinct rules with v={v}, s={s}. "
#                           f"Maximum possible is {max_distinct}.")

#     rng = np.random.default_rng(rules_seed)
#     rules_array_old = generate_old_rules_unambiguous_even(L, v, s, m, rules_seed)

#     shape = [L-1,v]+[v for idx in range(s)]
#     rules_array = np.full(shape,epsilon/(v**s-m), dtype="float")
#     prefactor = np.array(np.arange(1,m+1), dtype=float)**-1
#     prefactor = prefactor/np.sum(prefactor)
#     for l in range(L-1):
#       for _m in range(m):
#         for _v in range(v):
#           rules_array[l,_v][tuple(rules_array_old[l,_v,_m])] = (1-epsilon)*prefactor[_m]
#     return rules_array

# def rules_special11(L, v, s, rules_seed=0, epsilon=1.0):
#     if v!=2 or s!=2:
#       raise ValueError("v and s must be 2")
#     np.random.seed(rules_seed)

#     shape = [L-1,v]+[v for idx in range(s)]
#     rules_array = np.full(shape,0.1, dtype="float")

#     prob_sum = 1+2*np.exp(-1/epsilon)+np.exp(-2/epsilon)
#     rules_array_small = np.full(shape[1:],0, dtype="float")
#     rules_array_small[0,0,0] = 1/prob_sum 
#     rules_array_small[0,0,1] = np.exp(-1/epsilon)/prob_sum 
#     rules_array_small[0,1,0] = np.exp(-1/epsilon)/prob_sum 
#     rules_array_small[0,1,1] = np.exp(-2/epsilon)/prob_sum 
#     rules_array_small[1,1,1] = 1/prob_sum 
#     rules_array_small[1,0,1] = np.exp(-1/epsilon)/prob_sum 
#     rules_array_small[1,1,0] = np.exp(-1/epsilon)/prob_sum 
#     rules_array_small[1,0,0] = np.exp(-2/epsilon)/prob_sum 

#     for l in range(L-1):
#       rules_array[l] = rules_array_small
#     return rules_array

# def rules_special12(L, v, s, m, epsilon, rules_seed=0):
#     max_distinct = v**(s-1)
#     if m > max_distinct:
#         raise ValueError(f"Cannot generate {m} distinct rules with v={v}, s={s}. "
#                           f"Maximum possible is {max_distinct}.")

#     rng = np.random.default_rng(rules_seed)
#     rules_array_old = generate_old_rules_unambiguous_even(L, v, s, m, rules_seed)

#     shape = [L-1,v]+[v for idx in range(s)]
#     rules_array = np.full(shape,epsilon/(v**s-m), dtype="float")
#     prefactor = np.array(np.arange(1,m+1), dtype=float)**-2
#     prefactor = prefactor/np.sum(prefactor)
#     for l in range(L-1):
#       for _m in range(m):
#         for _v in range(v):
#           rules_array[l,_v][tuple(rules_array_old[l,_v,_m])] = (1-epsilon)*prefactor[_m]
#     return rules_array

# def rules_special13(L, v, s, m, epsilon, rules_seed=0):
#     max_distinct = v**(s-1)
#     if m > max_distinct:
#         raise ValueError(f"Cannot generate {m} distinct rules with v={v}, s={s}. "
#                           f"Maximum possible is {max_distinct}.")

#     rng = np.random.default_rng(rules_seed)
#     rules_array_old = generate_old_rules_unambiguous_even(L, v, s, m, rules_seed)

#     shape = [L-1,v]+[v for idx in range(s)]
#     rules_array = np.full(shape,0, dtype="float")
#     prefactor = np.linspace(0,1.00,m+2)[1:-1]
#     prefactor = np.exp(epsilon*np.sqrt(2)*scipy.special.erfinv(2*prefactor-1))
#     prefactor = prefactor / np.sum(prefactor)
#     for l in range(L-1):
#       for _m in range(m):
#         for _v in range(v):
#           rules_array[l,_v][tuple(rules_array_old[l,_v,_m])] = prefactor[_m]
#     return rules_array

# def rules_special14(L, v, s, m, epsilon, rules_seed=0):
#     max_distinct = v**(s-1)
#     if m > max_distinct:
#         raise ValueError(f"Cannot generate {m} distinct rules with v={v}, s={s}. "
#                           f"Maximum possible is {max_distinct}.")

#     rng = np.random.default_rng(rules_seed)
#     rules_array_old = generate_old_rules_unambiguous_even(L, v, s, m, rules_seed)

#     prefactor = np.linspace(0,1.00,m+2)[1:-1]
#     prefactor = np.exp(epsilon*np.sqrt(2)*scipy.special.erfinv(2*prefactor-1))
#     prefactor = prefactor / np.sum(prefactor)
#     shape = [L-1,v]+[v for idx in range(s)]
#     rules_array = np.full(shape,0, dtype="float")
#     for l in range(L-1):
#       for _v in range(v):
#         prefactor = rng.permutation(prefactor)
#         for _m in range(m):
#           rules_array[l,_v][tuple(rules_array_old[l,_v,_m])] = prefactor[_m]
#     return rules_array

def rules_special01(L, v, s, m, epsilon, rules_seed=0):
    if s!=2:
      raise ValueError("s must be 2")
    max_distinct = v**(s-1)
    if m > max_distinct:
        raise ValueError(f"Cannot generate {m} distinct rules with v={v}, s={s}. "
                          f"Maximum possible is {max_distinct}.")

    rng = np.random.default_rng(rules_seed)
    rules_array_old = generate_old_rules_unambiguous(L, v, s, m, rules_seed)

    rules_array_temp = np.zeros(shape=[v for idx in range(s+1)])
    rules_array_temp = np.log(rules_array_temp) #make them logits
    for _v in range(v):
        prefactor = epsilon*rng.normal(size=m)
        for _m in range(m):
          rules_array_temp[_v][tuple(rules_array_old[0,_v,_m])] = prefactor[_m]
        rules_array_temp[_v] = scipy.special.softmax(rules_array_temp[_v])

    shape = [L-1,v]+[v for idx in range(s)]
    rules_array = np.full(shape,0, dtype="float")
    for l in range(L-1):
          rules_array[l] = rules_array_temp
    return rules_array

# def rules_special16(L, v, s, m, epsilon, rules_seed=0):
#     if s!=2:
#       raise ValueError("s must be 2")
#     max_distinct = v**(s-1)
#     if m > max_distinct:
#         raise ValueError(f"Cannot generate {m} distinct rules with v={v}, s={s}. "
#                           f"Maximum possible is {max_distinct}.")

#     rng = np.random.default_rng(rules_seed)
#     rules_array_old = generate_old_rules_unambiguous(L, v, s, m, rules_seed)

#     rules_array_temp = np.zeros(shape=[v for idx in range(s+1)])
#     #rules_array_temp = np.log(rules_array_temp) #make them logits
#     for _v in range(v):
#         prefactor = np.linspace(0,1.00,m+2)[1:-1]
#         prefactor = np.exp(epsilon*np.sqrt(2)*scipy.special.erfinv(2*prefactor-1))
#         prefactor = prefactor / np.sum(prefactor)
#         for _m in range(m):
#           rules_array_temp[_v][tuple(rules_array_old[0,_v,_m])] = prefactor[_m]
#         #rules_array_temp[_v] = scipy.special.softmax(rules_array_temp[_v])

#     shape = [L-1,v]+[v for idx in range(s)]
#     rules_array = np.full(shape,0, dtype="float")
#     for l in range(L-1):
#           rules_array[l] = rules_array_temp
#     return rules_array

def generate_rules_str(
    L, v, s, m, epsilon, rules_seed=0, rules_str=""):
  if rules_str == "special00":
     return generate_rules_unambiguous_epsilon(L,v,s,m,epsilon=epsilon, rules_seed=rules_seed)
  elif rules_str == "special01":
    return rules_special01(L, v, s, m, epsilon, rules_seed=rules_seed)
  else:
    raise ValueError("Invalid rules_str")


# def generate_string(rules_array, return_full=False):
#     """
#     Simulates the forward generation of a string from the hierarchical rule array.

#     Parameters:
#         rules_array (np.ndarray): The rule set of shape (L-1, v, m, s).
#         return_full (bool): If True, return the full hierarchy; otherwise return only final string.

#     Returns:
#         final_string (np.ndarray): The output string at the final layer.
#         full_hierarchy (list of np.ndarray, optional): List of layer outputs from input to final string.
#     """
#     L_minus_1, v, m, s = rules_array.shape
#     L = L_minus_1 + 1  # Recover L

#     full_string = [np.array([np.random.randint(0, v)])]
#     for l in range(L - 1):
#         prev = full_string[-1]
#         rand_m = np.random.randint(0, m, size=len(prev))
#         next_layer = rules_array[l, prev, rand_m].reshape(-1)
#         full_string.append(next_layer)
    
#     if return_full:
#         return full_string
#     return full_string[-1]



@njit
def compute_downward_belief_numba(rules_array, upward_belief_array, s, v, L, leaf_index):
    """
    Compute downward belief from root to the specified leaf node.

    Parameters
    ----------
    rules_array : np.ndarray
        Rules array with shape (L-1, v, v, ..., v) (2+s dims).
    upward_belief_array : np.ndarray
        Upward beliefs with shape (L, layer_size, v).
    s : int
        Number of children per node.
    v : int
        Number of states.
    L : int
        Number of layers in the tree.
    leaf_index : int
        Index of the leaf node.

    Returns
    -------
    np.ndarray
        Downward belief vector of length v or sentinel array (-1) on error.
    """
    parent_down = np.full(v, 1 / v)

    # Build path from leaf to root
    path = np.empty((L - 1, 4), dtype=np.int32)
    idx = leaf_index
    for l in range(L - 1, 0, -1):
        parent_idx = idx // s
        pos = idx % s
        path[l - 1] = np.array([l, idx, parent_idx, pos])
        idx = parent_idx

    sibling_up = np.empty((s, v), dtype=np.float64)

    for i in range(L - 1):
        l, idx, parent_idx, pos = path[i]
        for j in range(s):
            sibling_up[j] = upward_belief_array[l, parent_idx * s + j]

        # Compute product over all children except the current position
        # Start with parent_down copy
        belief_product = np.empty_like(parent_down)
        for k in range(v):
            belief_product[k] = parent_down[k]

        # Multiply in place for each sibling (skip pos)
        for j in range(s):
            if j != pos:
                # Outer product along new axis
                old_size = belief_product.size
                new_size = old_size * v
                temp = np.empty(new_size, dtype=np.float64)
                idx_out = 0
                for a in range(old_size):
                    for b in range(v):
                        temp[idx_out] = belief_product[a] * sibling_up[j, b]
                        idx_out += 1
                belief_product = temp

        # Apply rules
        rules_swapped = np.swapaxes(rules_array[l - 1], pos, pos+1)
        for temp_idx in range(1, pos+1):
            rules_swapped = np.swapaxes(rules_swapped, pos-temp_idx, pos-temp_idx+1)
        flat_rules = rules_swapped.copy()  # copy to ensure contiguous
        flat_rules = flat_rules.reshape(v, -1)
        belief = np.empty(v, dtype=np.float64)
        for r in range(v):
            s_val = 0.0
            for c in range(flat_rules.shape[1]):
                s_val += flat_rules[r, c] * belief_product[c]
            belief[r] = s_val

        total = belief.sum()
        if total > 0:
            parent_down = belief / total
        else:
            return np.full(v, -1.0)  # signal error

    return parent_down


@njit
def update_upward_from_leaf_numba(rules_array, upward_belief_array, s, v, L, leaf_index, new_leaf_belief):
    """
    Update upward beliefs starting from the specified leaf node upward through the tree.

    Parameters
    ----------
    rules_array : np.ndarray
        Rules array with shape (L-1, v, v, ..., v) (2+s dims).
    upward_belief_array : np.ndarray
        Current upward beliefs with shape (L, layer_size, v).
    s : int
        Number of children per node.
    v : int
        Number of states.
    L : int
        Number of layers in the tree.
    leaf_index : int
        Index of the leaf node.
    new_leaf_belief : np.ndarray
        New belief vector at the leaf node.

    Returns
    -------
    np.ndarray
        Updated upward_belief_array or sentinel values (-1) on error.
    """
    upward_belief_array[L - 1, leaf_index] = new_leaf_belief
    idx = leaf_index

    for l in range(L - 2, -1, -1):
        parent_idx = idx // s

        child_beliefs = np.empty((s, v))
        for i in range(s):
            child_idx = parent_idx * s + i
            child_beliefs[i] = upward_belief_array[l + 1, child_idx]

        # Compute product over all children
        belief_product = np.empty_like(child_beliefs[0])
        for k in range(v):
            belief_product[k] = child_beliefs[0, k]

        # Multiply in place for each sibling (skip pos)
        for j in range(1, s):
            # Outer product along new axis
            old_size = belief_product.size
            new_size = old_size * v
            temp = np.empty(new_size, dtype=np.float64)
            idx_out = 0
            for a in range(old_size):
                for b in range(v):
                    temp[idx_out] = belief_product[a] * child_beliefs[j, b]
                    idx_out += 1
            belief_product = temp


        flat_rules_matrix = rules_array[l].reshape(v, -1)
        belief = np.dot(flat_rules_matrix, belief_product)

        total = belief.sum()
        if total > 0:
            upward_belief_array[l, parent_idx] = belief / total
        else:
            # print(str(l)+" "+str(parent_idx)+" "+str(leaf_index))
            # print(new_leaf_belief)
            upward_belief_array[l, parent_idx] = np.full(v, -1.0)  # signal error

        idx = parent_idx

    return upward_belief_array


@njit
def compute_all_marginals_numba(rules_array, upward_belief_array, s, v, L):
    """
    Compute downward beliefs for all nodes and then marginals by combining upward and downward beliefs.

    Parameters
    ----------
    rules_array : np.ndarray
        Rules array with shape (L-1, v, v, ..., v) (2+s dims).
    upward_belief_array : np.ndarray
        Current upward beliefs with shape (L, layer_size, v).
    s : int
        Number of children per node.
    v : int
        Number of states.
    L : int
        Number of layers in the tree.

    Returns
    -------
    np.ndarray
        Marginals with shape (L, layer_size, v), normalized or sentinel (-1) on error.
    """
    layer_sizes = np.empty(L, dtype=np.int32)
    for l in range(L):
        layer_sizes[l] = s ** l

    downward_belief_array = np.zeros((L, layer_sizes[L - 1], v))
    downward_belief_array[0, 0] = np.full(v, 1 / v)  # root uniform

    sibling_up = np.empty((s, v), dtype=np.float64)

    for l in range(1, L):
        for parent_idx in range(layer_sizes[l - 1]):
            parent_down = downward_belief_array[l - 1, parent_idx]

            for j in range(s):
                sibling_up[j] = upward_belief_array[l, parent_idx * s + j]

            for pos in range(s):
                idx = parent_idx * s + pos

                # Compute product over all children except the current position
                # Start with parent_down copy
                belief_product = np.empty_like(parent_down)
                for k in range(v):
                    belief_product[k] = parent_down[k]

                # Multiply in place for each sibling (skip pos)
                for j in range(s):
                    if j != pos:
                        # Outer product along new axis
                        old_size = belief_product.size
                        new_size = old_size * v
                        temp = np.empty(new_size, dtype=np.float64)
                        idx_out = 0
                        for a in range(old_size):
                            for b in range(v):
                                temp[idx_out] = belief_product[a] * sibling_up[j, b]
                                idx_out += 1
                        belief_product = temp

                rules_swapped = np.swapaxes(rules_array[l - 1], pos, pos+1)
                for temp_idx in range(1, pos+1):
                    rules_swapped = np.swapaxes(rules_swapped, pos-temp_idx, pos-temp_idx+1)
                flat_rules = rules_swapped.copy()  # copy to ensure contiguous
                flat_rules = flat_rules.reshape(v, -1)
                belief = np.empty(v, dtype=np.float64)
                for r in range(v):
                    s_val = 0.0
                    for c in range(flat_rules.shape[1]):
                        s_val += flat_rules[r, c] * belief_product[c]
                    belief[r] = s_val

                total = belief.sum()
                if total > 0:
                    downward_belief_array[l, idx] = belief / total
                else:
                    downward_belief_array[l, idx] = np.full(v, -1.0)  # signal error

    marginals = upward_belief_array * downward_belief_array
    for l in range(L):
        for i in range(layer_sizes[l]):
            norm = marginals[l, i].sum()
            if norm > 0:
                marginals[l, i] /= norm
            else:
                marginals[l, i] = np.full(v, -1.0)  # signal error

    return marginals


class bp_autoreg:
    """
    Belief Propagation autoregressive model class with Numba-accelerated belief computations.

    Attributes
    ----------
    rules_array : np.ndarray
        Rules array with shape (L-1, v, v, ..., v) (2+s dims).
    upward_belief_array : np.ndarray
        Upward belief arrays with shape (L, layer_size, v).
    L : int
        Number of layers.
    v : int
        Number of states.
    s : int
        Number of children per node.
    layer_sizes : list of int
        Number of nodes per layer.
    """

    def __init__(self, rules_array, upward_belief_list=None):
        """
        Initialize the BP autoregressive model.

        Parameters
        ----------
        rules_array : np.ndarray
            Rules array with shape (L-1, v, v, ..., v) (2+s dims).
        upward_belief_list : list of np.ndarray, optional
            Optional list of upward beliefs per layer.
        """
        self.rules_array = rules_array
        self.L = rules_array.shape[0] + 1
        self.v = rules_array.shape[1]
        self.s = rules_array.ndim - 2
        self.layer_sizes = [self.s**l for l in range(self.L)]

        if upward_belief_list is None:
            self.upward_belief_array = np.full(
                (self.L, self.layer_sizes[-1], self.v), 1 / self.v
            )
        else:
            self.upward_belief_array = np.zeros(
                (self.L, self.layer_sizes[-1], self.v)
            )
            for l in range(self.L):
                self.upward_belief_array[l, : self.layer_sizes[l], :] = upward_belief_list[
                    l
                ]

    def _check_belief_validity(self, belief):
        """
        Check if a belief vector or array contains error sentinel values (-1).

        Parameters
        ----------
        belief : np.ndarray
            Belief vector or array.

        Raises
        ------
        ValueError
            If belief contains sentinel error values (-1).
        """
        if np.any(belief == -1):
            raise ValueError(
                "Encountered invalid belief (all zero or error state) during computation."
            )

    def compute_downward_belief(self, leaf_index):
        """
        Compute downward belief vector for a specified leaf index.

        Parameters
        ----------
        leaf_index : int
            Index of the leaf node.

        Returns
        -------
        np.ndarray
            Downward belief vector of length v.

        Raises
        ------
        ValueError
            If computation produces invalid beliefs.
        """
        belief = compute_downward_belief_numba(
            self.rules_array,
            self.upward_belief_array,
            self.s,
            self.v,
            self.L,
            leaf_index,
        )
        self._check_belief_validity(belief)
        return belief

    def update_upward_from_leaf(self, leaf_index, new_leaf_belief):
        """
        Update upward beliefs after changing the belief at a leaf node.

        Parameters
        ----------
        leaf_index : int
            Index of the leaf node.
        new_leaf_belief : np.ndarray
            New belief vector at the leaf node.

        Raises
        ------
        ValueError
            If computation produces invalid beliefs.
        """
        self.upward_belief_array = update_upward_from_leaf_numba(
            self.rules_array,
            self.upward_belief_array,
            self.s,
            self.v,
            self.L,
            leaf_index,
            new_leaf_belief,
        )
        self._check_belief_validity(self.upward_belief_array)

    def compute_all_marginals(self):
        """
        Compute all marginals by combining upward and downward beliefs.

        Returns
        -------
        np.ndarray
            Marginals array with shape (L, layer_size, v).

        Raises
        ------
        ValueError
            If computation produces invalid beliefs.
        """
        marginals = compute_all_marginals_numba(
            self.rules_array, self.upward_belief_array, self.s, self.v, self.L
        )
        self._check_belief_validity(marginals)
        return marginals

    def update_leaf_and_compute_all_marginals(self, leaf_index, new_leaf_belief, all_layers=False):
        """
        Update leaf belief and compute marginals.

        By default, returns only marginals at the leaf layer for efficiency.
        Set all_layers=True to return marginals for all layers.

        Parameters
        ----------
        leaf_index : int
            Index of the leaf node.
        new_leaf_belief : np.ndarray
            New belief vector at the leaf node.
        all_layers : bool, optional
            Whether to return marginals for all layers (default False).

        Returns
        -------
        np.ndarray
            Marginals array for leaf layer if all_layers=False,
            otherwise for all layers.

        Raises
        ------
        ValueError
            If computation produces invalid beliefs.
        """
        self.update_upward_from_leaf(leaf_index, new_leaf_belief)
        marginals = self.compute_all_marginals()

        if all_layers:
            return marginals
        else:
            # Return marginals for leaf layer only
            return marginals[self.L - 1, : self.layer_sizes[-1]]

    def update_leaf_and_compute_next_marginal(self, leaf_index, new_leaf_belief):
        """
        Update leaf belief and compute updated marginals for the next element.

        This method computes downward beliefs toward the next node only, then combines
        with upward beliefs for the marginal at that node.

        Parameters
        ----------
        leaf_index : int
            Index of the leaf node.
        new_leaf_belief : np.ndarray
            New belief vector at the leaf node.

        Returns
        -------
        np.ndarray
            Marginal belief vector for the next node in the tree.

        Raises
        ------
        ValueError
            If computation produces invalid beliefs.
        """
        # Update upward beliefs first
        self.update_upward_from_leaf(leaf_index, new_leaf_belief)

        # Compute downward belief at leaf node
        downward_belief = self.compute_downward_belief(leaf_index+1)
        self._check_belief_validity(downward_belief)

        # Check if the updated leaf was the last one
        if leaf_index+1 >= self.layer_sizes[-1]:
            raise IndexError("Next element out of bounds.")

        # Compute marginal for the next node
        upward_belief_next = self.upward_belief_array[-1, leaf_index+1]
        marginal_next = downward_belief * upward_belief_next
        norm = marginal_next.sum()
        if norm > 0:
            marginal_next /= norm
        else:
            raise ValueError("Normalized marginal next has zero sum.")

        return marginal_next


# was updated to class above which was speed up with numba
# class bp_autoreg:
#     """
#     Belief Propagation (BP) class for autoregressive hierarchical models.

#     Supports upward and downward belief computation over a tree-like hierarchy
#     defined by a `rules_array`.

#     Attributes:
#         rules_array (np.ndarray): Array of shape (L-1, v, m, s) representing the grammar rules.
#         L (int): Number of layers.
#         v (int): Vocabulary size.
#         m (int): Number of rules per parent.
#         s (int): Branching factor per node.
#         upward_belief_list (list): List of arrays with shape (num_nodes_at_layer, v) for each layer.
#     """

#     def __init__(self, rules_array, upward_belief_list=None):
#         """
#         Initializes the BP_autoreg object.

#         Args:
#             rules_array (np.ndarray): Shape (L-1, v, m, s) representing rules.
#             upward_belief_list (list[np.ndarray], optional): If not provided, initializes uniform beliefs.
#         """
#         self.rules_array = rules_array
#         self.L, self.v, self.m, self.s = rules_array.shape[0] + 1, rules_array.shape[1], rules_array.shape[2], rules_array.shape[3]

#         # Number of nodes at each layer: grows by factor s
#         self.layer_sizes = [self.s**l for l in range(self.L)]

#         # Initialize beliefs
#         if upward_belief_list is None:
#             self.upward_belief_list = [
#                 np.full((size, self.v), 1 / self.v) for size in self.layer_sizes
#             ]
#         else:
#             self.upward_belief_list = upward_belief_list

#     def compute_downward_belief(self, index):
#         """
#         Computes the downward belief at a leaf node.

#         Args:
#             index (int): Index at the leaf layer (L-1).

#         Returns:
#             np.ndarray: Downward belief at the specified leaf index.
#         """
#         parent_down = np.full(self.v, 1 / self.v)  # root starts with uniform belief

#         # Build path from root to leaf
#         path = []
#         idx = index
#         for l in reversed(range(1, self.L)):
#             parent_idx = idx // self.s
#             pos = idx % self.s
#             path.append((l, idx, parent_idx, pos))
#             idx = parent_idx
#         path.reverse()

#         for l, idx, parent_idx, pos in path:
#             sibling_indices = [parent_idx * self.s + i for i in range(self.s)]
#             sibling_up = np.array([self.upward_belief_list[l][i] for i in sibling_indices])  # shape: (s, v)

#             belief = np.zeros(self.v)
#             for y in range(self.v):  # parent value
#                 down_y = parent_down[y]
#                 for mi in range(self.m):
#                     rule = self.rules_array[l - 1, y, mi]  # shape: (s,)
#                     prod = down_y
#                     for i in range(self.s):
#                         if i == pos:
#                             continue
#                         prod *= sibling_up[i][rule[i]]
#                     belief[rule[pos]] += prod

#             if belief.sum() > 0:
#                 parent_down = belief / belief.sum()
#             else:
#                 raise Exception(f"All-zero downward belief at layer {l}, node {idx}")

#         return parent_down

#     def update_upward_from_leaf(self, leaf_index, new_leaf_belief):
#         """
#         Updates the upward belief tree from a new belief vector at the leaf.

#         Args:
#             leaf_index (int): Index at layer L-1.
#             new_leaf_belief (np.ndarray): New belief vector of shape (v,) for the leaf.
#         """
#         self.upward_belief_list[self.L - 1][leaf_index] = new_leaf_belief
#         idx = leaf_index

#         for l in reversed(range(self.L - 1)):
#             parent_idx = idx // self.s
#             child_indices = [parent_idx * self.s + i for i in range(self.s)]
#             child_beliefs = np.array([self.upward_belief_list[l + 1][ci] for ci in child_indices])

#             belief = np.zeros(self.v)
#             for y in range(self.v):
#                 for mi in range(self.m):
#                     rule = self.rules_array[l, y, mi]  # shape: (s,)
#                     prod = 1.0
#                     for i in range(self.s):
#                         prod *= child_beliefs[i][rule[i]]
#                     belief[y] += prod

#             if belief.sum() > 0:
#                 self.upward_belief_list[l][parent_idx] = belief / belief.sum()
#             else:
#                 raise Exception(f"All-zero upward belief at layer {l}, index {parent_idx}")

#             idx = parent_idx

#     def update_leaf_and_compute_marginal(self, leaf_index, new_leaf_belief):
#         """
#         Updates the belief at a leaf and returns the marginal distribution for its right sibling.

#         Args:
#             leaf_index (int): Index of the updated leaf.
#             new_leaf_belief (np.ndarray): New belief vector of shape (v,) at `leaf_index`.

#         Returns:
#             np.ndarray: Marginal probability vector at leaf_index + 1.
#         """
#         self.update_upward_from_leaf(leaf_index, new_leaf_belief)
#         if leaf_index + 1 >= self.layer_sizes[-1]:
#             raise IndexError(f"No right sibling exists at index {leaf_index + 1}")
        
#         upward = self.upward_belief_list[self.L - 1][leaf_index + 1]
#         downward = self.compute_downward_belief(leaf_index + 1)
#         return upward * downward / np.sum(upward * downward)

#     def update_leaf_and_compute_all_marginals(self, leaf_index, new_leaf_belief, all_layers=False):
#         """
#         Updates the belief at a leaf and returns the marginal probabilities for all leaves (or all nodes).

#         Args:
#             leaf_index (int): Index of the updated leaf node.
#             new_leaf_belief (np.ndarray): New belief vector at the specified leaf.
#             all_layers (bool): If True, returns marginals for all layers, otherwise only leaf layer.

#         Returns:
#             list[np.ndarray] | np.ndarray: Marginals for all layers (list) or only the leaves (array).
#         """
#         self.update_upward_from_leaf(leaf_index, new_leaf_belief)

#         # Initialize downward belief list
#         downward_belief_list = [np.zeros((size, self.v)) for size in self.layer_sizes]
#         downward_belief_list[0][0] = np.full(self.v, 1 / self.v)  # root is uniform

#         for l in range(1, self.L):
#             for parent_idx in range(self.layer_sizes[l - 1]):
#                 parent_down = downward_belief_list[l - 1][parent_idx]
#                 sibling_indices = [parent_idx * self.s + i for i in range(self.s)]
#                 sibling_up = np.array([self.upward_belief_list[l][i] for i in sibling_indices])

#                 for pos, idx in enumerate(sibling_indices):
#                     belief = np.zeros(self.v)
#                     for y in range(self.v):
#                         down_y = parent_down[y]
#                         for mi in range(self.m):
#                             rule = self.rules_array[l - 1, y, mi]
#                             prod = down_y
#                             for i in range(self.s):
#                                 if i == pos:
#                                     continue
#                                 prod *= sibling_up[i][rule[i]]
#                             belief[rule[pos]] += prod
#                     if belief.sum() > 0:
#                         downward_belief_list[l][idx] = belief / belief.sum()
#                     else:
#                         raise Exception(f"All-zero downward belief at layer {l}, node {idx}")

#         # Compute marginals = upward * downward (normalize)
#         if all_layers:
#             marginals = []
#             for l in range(self.L):
#                 u = self.upward_belief_list[l]
#                 d = downward_belief_list[l]
#                 m = u * d
#                 marginals.append(m / m.sum(axis=1, keepdims=True))
#             return marginals
#         else:
#             u = self.upward_belief_list[-1]
#             d = downward_belief_list[-1]
#             m = u * d
#             return m / m.sum(axis=1, keepdims=True)



def compute_joint_probabilities(rules_array, max_distance=None, verbose=True):
    """
    Computes joint probability p(a, b, d) over leaf vocab symbols a, b
    at leaf-node distance d using belief propagation.
    
    Now correctly weights the accumulation by the unconditional probability 
    of finding 'a' at the source position.

    Args:
        rules_array : np.ndarray
            Rules array with shape (L-1, v, v, ..., v) (2+s dims).
        max_distance (int): Max leaf distance to compute p(a, b, d)

    Returns:
        np.ndarray: Joint probability array of shape (v, v, max_distance)
                    where result[a, b, d] = p(a, b, d)
    """
    L = rules_array.shape[0] + 1
    v = rules_array.shape[1]
    s = rules_array.ndim - 2
    n_leaves = s ** (L - 1)

    if max_distance is None:
        max_distance = n_leaves
    elif max_distance > n_leaves:
        raise ValueError("max_distance must be less than or equal to n_leaves")

    # 1. Pre-compute unconditional probabilities for all leaves
    upward_belief_array = np.full((L, n_leaves, v), 1.0 / v)
    unconditional_marginals_all = compute_all_marginals_numba(rules_array, upward_belief_array, s, v, L)
    unconditional_leaf_probs = unconditional_marginals_all[-1]  # Shape: (n_leaves, v)

    # 2. Compute Joint Probabilities
    joint_counts = np.zeros((v, v, max_distance))
    counts = np.zeros(max_distance)

    for i in range(n_leaves):
        if verbose and i % 10 == 0:  # reduce print frequency slightly
            print(f"{i}/{n_leaves}")
        
        # Determine the valid range of d for this position i such that j < n_leaves
        # j = i + d  =>  d < n_leaves - i
        valid_dist_limit = min(max_distance, n_leaves - i)
        
        # Update the normalization counts for valid distances
        counts[:valid_dist_limit] += 1

        for a in range(v):
            # Get the unconditional probability of a at i: P(leaf_i = a)
            p_a_at_i = unconditional_leaf_probs[i, a]

            # If p_a_at_i is 0, we can skip to save time
            if p_a_at_i == 0:
                continue

            # Set constraint: leaf_i = a
            beliefs = np.zeros(v)
            beliefs[a] = 1.0

            # Run BP to get conditional marginals: P(leaf_j = b | leaf_i = a)
            bp = bp_autoreg(rules_array)
            conditional_marginals = bp.update_leaf_and_compute_all_marginals(i, beliefs)

            # 3. Accumulate joint counts
            # We want: Sum [ P(leaf_j=b | leaf_i=a) * P(leaf_i=a) ]
            
            # Slice only the valid forward distances (j from i to i + limit)
            # Shape of slice: (valid_dist_limit, v)
            # Note: conditional_marginals index corresponds to leaf index j
            valid_window_marginals = conditional_marginals[i : i + valid_dist_limit]
            
            # Vectorized update:
            # joint_counts[a, b, d] += conditional_marginals[i+d][b] * p_a_at_i
            # We transpose valid_window_marginals to (v, valid_dist_limit) to match (b, d)
            joint_counts[a, :, :valid_dist_limit] += valid_window_marginals.T * p_a_at_i

    # 4. Normalize to get probabilities
    with np.errstate(divide='ignore', invalid='ignore'):
        p_abd = np.divide(joint_counts, counts[None, None, :])
        p_abd = np.nan_to_num(p_abd)

    return p_abd



def compute_mi_full_no_temperature(rules_array, max_distance=None, verbose=True):
    """
    Computes mutual information I(d) via rules_array)

    Args:
        rules_array : np.ndarray
            Rules array with shape (L-1, v, v, ..., v) (2+s dims).
        max_distance (int): Max leaf distance to compute p(a, b, d)

    Returns:
        np.ndarray: Mutual information for each distance d, shape (max_distance+1,)
    """
    
    p_abd = compute_joint_probabilities(rules_array, max_distance=max_distance, verbose=verbose)

    v, _, max_d_plus_1 = p_abd.shape
    mi = np.zeros(max_d_plus_1)

    for d in range(max_d_plus_1):
        joint = p_abd[:, :, d]  # shape (v, v)
        p_a = np.sum(joint, axis=1, keepdims=True)  # shape (v, 1)
        p_b = np.sum(joint, axis=0, keepdims=True)  # shape (1, v)
        #p_a = 1/v*np.ones((v,1))
        #p_b = 1/v*np.ones((1,v))

        with np.errstate(divide='ignore', invalid='ignore'):
            ratio = np.divide(joint, p_a @ p_b)  # shape (v, v)
            log_term = np.log2(ratio, where=(joint > 0))
            log_term = np.nan_to_num(log_term)
            mi[d] = np.sum(joint * log_term)

    return mi

def compute_mi_full_no_temperature_per_position(rules_array, max_distance=None, verbose=True):
    """
    Computes mutual information I(d) for each position pair individually
    using belief propagation marginals, then averages over all pairs
    with the same distance.

    Args:
        rules_array : np.ndarray
            Rules array with shape (L-1, v, v, ..., v) (2+s dims).
        max_distance (int, optional): Max leaf distance to consider.
                                       If None, defaults to n_leaves - 1.
        verbose (bool): Whether to print progress.

    Returns:
        np.ndarray: MI averaged over position pairs for each distance.
                    Shape (max_distance+1,)
    """
    L = rules_array.shape[0] + 1
    v = rules_array.shape[1]
    s = rules_array.ndim - 2
    n_leaves = s ** (L - 1)

    if max_distance is None:
        max_distance = n_leaves - 1
    elif max_distance >= n_leaves:
        raise ValueError("max_distance must be less than n_leaves")

    mi_sum = np.zeros(max_distance + 1, dtype=np.float64)
    mi_count = np.zeros(max_distance + 1, dtype=np.int64)

    # Loop over each starting position i
    for i in range(n_leaves):
        if verbose and i % 50 == 0:
            print(f"Processing leaf {i}/{n_leaves}")

        # For each possible token at position i
        distance = np.min((max_distance+1, n_leaves - i))
        p_abd = np.zeros((v, v, distance))
        for a in range(v):
            beliefs = np.zeros(v)
            beliefs[a] = 1.0

            # Run belief propagation from this fixed a
            bp = bp_autoreg(rules_array)
            marginals = bp.update_leaf_and_compute_all_marginals(i, beliefs)
            p_abd[a] = marginals[i:i+distance].T*1/v

        
        for d in range(distance):
            # Compute MI for this single-position pair
            mi_val = 0.0
            for a in range(v):
                for b in range(v):
                    if p_abd[a, b, d] > 0.0:
                        mi_val += p_abd[a, b, d] * np.log2(p_abd[a, b, d] / (1/v**2))
            mi_sum[d] += mi_val
            mi_count[d] += 1


    # Average MI for each distance
    mi_avg = np.zeros(max_distance + 1)
    for d in range(max_distance + 1):
        if mi_count[d] > 0:
            mi_avg[d] = mi_sum[d] / mi_count[d]

    return mi_avg

def compute_mi_full_no_temperature_last_token(rules_array, max_distance=None, verbose=True):
    """
    Computes mutual information I(d) for each position pair individually
    using belief propagation marginals, then averages over all pairs
    with the same distance.

    Args:
        rules_array : np.ndarray
            Rules array with shape (L-1, v, v, ..., v) (2+s dims).
        max_distance (int, optional): Max leaf distance to consider.
                                       If None, defaults to n_leaves - 1.
        verbose (bool): Whether to print progress.

    Returns:
        np.ndarray: MI averaged over position pairs for each distance.
                    Shape (max_distance+1,)
    """
    L = rules_array.shape[0] + 1
    v = rules_array.shape[1]
    s = rules_array.ndim - 2
    n_leaves = s ** (L - 1)

    if max_distance is None:
        max_distance = n_leaves - 1
    elif max_distance >= n_leaves:
        raise ValueError("max_distance must be less than n_leaves")

    mi_final = np.zeros(max_distance + 1, dtype=np.float64)

    beliefs = np.ones(v)/v
    bp = bp_autoreg(rules_array)
    independent_marginals = bp.update_leaf_and_compute_all_marginals(0, beliefs)
    #print(independent_marginals)

    # For each possible token at position i
    p_abd = np.zeros((v, v, max_distance+1))
    for a in range(v):
        beliefs = np.zeros(v)
        beliefs[a] = 1.0

        # Run belief propagation from this fixed a
        bp = bp_autoreg(rules_array)
        marginals = bp.update_leaf_and_compute_all_marginals(n_leaves-1, beliefs)
        p_abd[a] = marginals[n_leaves-1-max_distance:n_leaves].T*independent_marginals[n_leaves-1][a]
    p_abd = np.flip(p_abd, axis=2)
    
    for d in range(max_distance+1):
        # Compute MI for this single-position pair
        mi_val = 0.0
        p_a = np.sum(p_abd[:,:,d], axis=1)
        p_b = np.sum(p_abd[:,:,d], axis=0)
        for a in range(v):
            for b in range(v):
                if p_abd[a, b, d] > 0.0:
                    mi_val += p_abd[a, b, d] * np.log2(p_abd[a, b, d] / (p_a[a]*p_b[b]))
        mi_final[d] = mi_val

    return mi_final

@njit
def string_generator_numba(rules_array, num_str, temperature, seed=0, verbose=False):
    """
    Generates valid strings based on given rules and temperature.

    Parameters
    ----------
    rules_array : np.ndarray
        Rules array with shape (L-1, v, v, ..., v) (2+s dims).
    num_str : int
        Number of strings to generate.
    temperature : float
        Temperature for tempered sampling (>0).
    seed : int
        Random seed for reproducibility.

    Returns
    -------
    np.ndarray
        Sampled strings of shape (num_str, len_str) and dtype np.int8.
    """

    L = rules_array.shape[0] + 1
    v = rules_array.shape[1]
    s = rules_array.ndim - 2

    layer_sizes = np.empty(L, dtype=np.int32)
    for l in range(L):
        layer_sizes[l] = s ** l

    sampled_strings = np.zeros((num_str, layer_sizes[-1]), dtype=np.int8)

    np.random.seed(seed)

    for _num_str in range(num_str):
      if verbose:
        if _num_str % 100 == 0:
          print(f"sampled string: {_num_str}/{num_str}")
      upward_belief_array = np.full((L, layer_sizes[-1], v), 1 / v)
      upward_belief_array = update_upward_from_leaf_numba(rules_array, upward_belief_array, s, v, L, 0, np.ones(v)/v)
      probs = compute_downward_belief_numba(rules_array, upward_belief_array, s, v, L, 0)
      if temperature != 1.0:
          probs = np.power(probs, 1.0 / temperature)
          probs /= np.sum(probs)
      for idx in range(layer_sizes[-1]):
        cum_probs = np.cumsum(probs)
        r = np.random.random()
        state = np.searchsorted(cum_probs, r)
        sampled_strings[_num_str, idx] = state

        new_leaf_belief = np.zeros(v)
        new_leaf_belief[state] = 1.0
        upward_belief_array = update_upward_from_leaf_numba(rules_array, upward_belief_array, s, v, L, idx, new_leaf_belief)
        probs = compute_downward_belief_numba(rules_array, upward_belief_array, s, v, L, idx+1)

        if temperature != 1.0:
          probs = np.power(probs, 1.0 / temperature)
          probs /= np.sum(probs)

      if np.any(upward_belief_array == -1.0):
        raise ValueError("Encountered invalid belief (all zero or error state) during computation.")

    return sampled_strings

@njit
def string_generator_top_p_numba(rules_array, num_str, temperature, top_p_prob, seed=0, verbose=False):
    """
    Generates valid strings based on given rules, temperature, and top-p sampling.

    Parameters
    ----------
    rules_array : np.ndarray
        Rules array with shape (L-1, v, v, ..., v) (2+s dims).
    num_str : int
        Number of strings to generate.
    temperature : float
        Temperature for tempered sampling (>0).
    top_p_prob : float
        Cumulative probability threshold for nucleus sampling (0 < top_p <= 1.0).
    seed : int
        Random seed for reproducibility.

    Returns
    -------
    np.ndarray
        Sampled strings of shape (num_str, len_str) and dtype np.int8.
    """

    L = rules_array.shape[0] + 1
    v = rules_array.shape[1]
    s = rules_array.ndim - 2

    layer_sizes = np.empty(L, dtype=np.int32)
    for l in range(L):
        layer_sizes[l] = s ** l

    sampled_strings = np.zeros((num_str, layer_sizes[-1]), dtype=np.int8)

    np.random.seed(seed)

    for _num_str in range(num_str):
        if verbose:
            if _num_str % 100 == 0:
                print(f"sampled string: {_num_str}/{num_str}")
        upward_belief_array = np.full((L, layer_sizes[-1], v), 1 / v)
        upward_belief_array = update_upward_from_leaf_numba(rules_array, upward_belief_array, s, v, L, 0, np.ones(v)/v)
        probs = compute_downward_belief_numba(rules_array, upward_belief_array, s, v, L, 0)
        #probs = np.ones(v)/v
    
        if temperature != 1.0:
            probs = np.power(probs, 1.0 / temperature)
            probs /= np.sum(probs)
            
        if top_p_prob < 1.0:
            sorted_indices = np.argsort(-probs)
            cum_sum = 0.0
            for i in range(len(sorted_indices)):
                cum_sum += probs[sorted_indices[i]]
                if cum_sum >= top_p_prob:
                    for j in range(i + 1, len(sorted_indices)):
                        probs[sorted_indices[j]] = 0.0
                    break
            probs /= np.sum(probs)

        for idx in range(layer_sizes[-1]):
            cum_probs = np.cumsum(probs)
            r = np.random.random()
            state = np.searchsorted(cum_probs, r)
            sampled_strings[_num_str, idx] = state

            new_leaf_belief = np.zeros(v)
            new_leaf_belief[state] = 1.0
            upward_belief_array = update_upward_from_leaf_numba(rules_array, upward_belief_array, s, v, L, idx, new_leaf_belief)
            probs = compute_downward_belief_numba(rules_array, upward_belief_array, s, v, L, idx+1)

            if temperature != 1.0:
                probs = np.power(probs, 1.0 / temperature)
                probs /= np.sum(probs)
            
            if top_p_prob < 1.0:
                sorted_indices = np.argsort(-probs)
                cum_sum = 0.0
                for i in range(len(sorted_indices)):
                    cum_sum += probs[sorted_indices[i]]
                    if cum_sum >= top_p_prob:
                        for j in range(i + 1, len(sorted_indices)):
                            probs[sorted_indices[j]] = 0.0
                        break
            probs /= np.sum(probs)



        if np.any(upward_belief_array == -1.0):
            raise ValueError("Encountered invalid belief (all zero or error state) during computation.")

    return sampled_strings

@njit
def compute_mutual_information_from_samples_numba(samples, max_distance=-1):
    """
    Computes mutual information I(d) from sampled strings using Numba-compatible
    counting (vectorized slicing, minimal Python loops).

    Args:
        samples (np.ndarray): Shape (num_str, len_str), dtype=np.int8
            Each row is a sequence of token IDs (0-indexed ints).
        max_distance (int, optional): Maximum distance to consider.
            If -1, defaults to len_str.

    Returns:
        np.ndarray: Mutual information for each distance d, shape (max_distance+1,)
    """
    num_str, len_str = samples.shape
    v = int(samples.max()) + 1

    if max_distance == -1:
        max_distance = len_str

    p_abd = np.zeros((v, v, max_distance + 1), dtype=np.float64)

    # Count co-occurrences
    for d in range(max_distance + 1):
        if d >= len_str:
            break
        a_vals = samples[:, :len_str - d].ravel()
        b_vals = samples[:, d:].ravel()
        n_pairs = a_vals.shape[0]
        for k in range(n_pairs):
            p_abd[a_vals[k], b_vals[k], d] += 1.0

    # Normalize counts
    for d in range(max_distance + 1):
        total_pairs = num_str * (len_str - d)
        if total_pairs > 0:
            p_abd[:, :, d] /= total_pairs

    # Compute MI
    mi = np.zeros(max_distance + 1, dtype=np.float64)
    for d in range(max_distance + 1):
        joint = p_abd[:, :, d]
        if np.all(joint == 0.0):
            continue

        p_a = np.sum(joint, axis=1).reshape((v, 1))
        p_b = np.sum(joint, axis=0).reshape((1, v))

        for a in range(v):
            for b in range(v):
                if joint[a, b] > 0.0:
                    mi[d] += joint[a, b] * np.log2(joint[a, b] / (p_a[a, 0] * p_b[0, b]))

    return mi

@njit
def compute_mutual_information_from_samples_per_position_numba(samples, max_distance=-1):
    """
    Computes mutual information I(d) from sampled strings, taking into account
    that probabilities may vary per position. MI is computed for each position
    pair (i, j) individually and then averaged for each distance d.

    Args:
        samples (np.ndarray): Shape (num_str, len_str), dtype=np.int8
            Each row is a sequence of token IDs (0-indexed ints).
        max_distance (int, optional): Maximum distance to consider.
            If -1, defaults to len_str - 1.

    Returns:
        np.ndarray: Mutual information for each distance d, shape (max_distance+1,)
    """
    num_str, len_str = samples.shape
    v = int(samples.max()) + 1

    if max_distance == -1 or max_distance > len_str - 1:
        max_distance = len_str - 1

    # Output array and counter for averaging
    mi_sum = np.zeros(max_distance + 1, dtype=np.float64)
    mi_count = np.zeros(max_distance + 1, dtype=np.int64)

    # Loop over all position pairs (i, j)
    for i in range(len_str):
        for j in range(i, len_str):
            d = j - i
            if d > max_distance:
                continue

            # Count joint distribution for this position pair
            joint = np.zeros((v, v), dtype=np.float64)
            for s in range(num_str):
                a = samples[s, i]
                b = samples[s, j]
                joint[a, b] += 1.0

            # Normalize to probabilities
            joint /= num_str

            # Marginals for this exact position pair
            p_a = np.sum(joint, axis=1)
            p_b = np.sum(joint, axis=0)

            # Compute MI for this pair
            mi_val = 0.0
            for a in range(v):
                for b in range(v):
                    if joint[a, b] > 0.0:
                        mi_val += joint[a, b] * np.log2(joint[a, b] / (p_a[a] * p_b[b]))

            mi_sum[d] += mi_val
            mi_count[d] += 1

    # Average over all position pairs for each distance
    mi_avg = np.zeros(max_distance + 1, dtype=np.float64)
    for d in range(max_distance + 1):
        if mi_count[d] > 0:
            mi_avg[d] = mi_sum[d] / mi_count[d]

    return mi_avg

@njit
def compute_mutual_information_from_samples_last_token_numba_p_abd(samples, max_distance=-1):
    """
    Computes mutual information I(d) from sampled strings, taking into account
    that probabilities may vary per position. MI is computed for each position
    pair (i, j) individually and then averaged for each distance d.

    Args:
        samples (np.ndarray): Shape (num_str, len_str), dtype=np.int8
            Each row is a sequence of token IDs (0-indexed ints).
        max_distance (int, optional): Maximum distance to consider.
            If -1, defaults to len_str - 1.

    Returns:
        np.ndarray: Mutual information for each distance d, shape (max_distance+1,)
    """
    num_str, len_str = samples.shape
    v = int(samples.max()) + 1

    if max_distance == -1 or max_distance > len_str - 1:
        max_distance = len_str - 1

    # Output array and counter for averaging
    mi_sum = np.zeros(max_distance + 1, dtype=np.float64)
    mi_count = np.zeros(max_distance + 1, dtype=np.int64)

    p_abd = np.zeros((v, v, max_distance + 1), dtype=np.float64)

    # Loop over all position pairs (i, j)
    for i in range(len_str):
        j = len_str-1
        d = j - i
        if d > max_distance:
            continue

        # Count joint distribution for this position pair
        joint = np.zeros((v, v), dtype=np.float64)
        for s in range(num_str):
            a = samples[s, i]
            b = samples[s, j]
            joint[a, b] += 1.0

        # Normalize to probabilities
        joint /= num_str

        # Marginals for this exact position pair
        p_a = np.sum(joint, axis=1)
        p_b = np.sum(joint, axis=0)

        # Compute MI for this pair
        mi_val = 0.0
        for a in range(v):
            for b in range(v):
                if joint[a, b] > 0.0:
                    mi_val += joint[a, b] * np.log2(joint[a, b] / (p_a[a] * p_b[b]))

        mi_sum[d] += mi_val
        mi_count[d] += 1

        p_abd[:,:,d] = joint

    # Average over all position pairs for each distance
    mi_avg = np.zeros(max_distance + 1, dtype=np.float64)
    for d in range(max_distance + 1):
        if mi_count[d] > 0:
            mi_avg[d] = mi_sum[d] / mi_count[d]

    return (mi_avg, p_abd)

@njit
def compute_mutual_information_from_p_abd_last_token_numba(p_abd, max_distance=-1):
    """
    Computes mutual information I(d) from sampled strings, taking into account
    that probabilities may vary per position. MI is computed for each position
    pair (i, j) individually and then averaged for each distance d.

    Args:
        samples (np.ndarray): Shape (num_str, len_str), dtype=np.int8
            Each row is a sequence of token IDs (0-indexed ints).
        max_distance (int, optional): Maximum distance to consider.
            If -1, defaults to len_str - 1.

    Returns:
        np.ndarray: Mutual information for each distance d, shape (max_distance+1,)
    """
    v, temp, len_str = p_abd.shape

    if max_distance == -1 or max_distance > len_str - 1:
        max_distance = len_str - 1

    # Output array and counter for averaging
    mi_sum = np.zeros(max_distance + 1, dtype=np.float64)
    mi_count = np.zeros(max_distance + 1, dtype=np.int64)

    # Loop over all position pairs (i, j)
    for i in range(len_str):
        j = len_str-1
        d = j - i
        if d > max_distance:
            continue

        # Normalize to probabilities
        joint = p_abd[:,:,d]

        # Marginals for this exact position pair
        p_a = np.sum(joint, axis=1)
        p_b = np.sum(joint, axis=0)

        # Compute MI for this pair
        mi_val = 0.0
        for a in range(v):
            for b in range(v):
                if joint[a, b] > 0.0:
                    mi_val += joint[a, b] * np.log2(joint[a, b] / (p_a[a] * p_b[b]))

        mi_sum[d] += mi_val
        mi_count[d] += 1

    # Average over all position pairs for each distance
    mi_avg = np.zeros(max_distance + 1, dtype=np.float64)
    for d in range(max_distance + 1):
        if mi_count[d] > 0:
            mi_avg[d] = mi_sum[d] / mi_count[d]

    return (mi_avg)

@njit
def compute_mutual_information_from_samples_last_token_numba(samples, max_distance=-1):
    """
    Computes mutual information I(d) from sampled strings, taking into account
    that probabilities may vary per position. MI is computed for each position
    pair (i, j) individually and then averaged for each distance d.

    Args:
        samples (np.ndarray): Shape (num_str, len_str), dtype=np.int8
            Each row is a sequence of token IDs (0-indexed ints).
        max_distance (int, optional): Maximum distance to consider.
            If -1, defaults to len_str - 1.

    Returns:
        np.ndarray: Mutual information for each distance d, shape (max_distance+1,)
    """
    num_str, len_str = samples.shape
    v = int(samples.max()) + 1

    if max_distance == -1 or max_distance > len_str - 1:
        max_distance = len_str - 1

    # Output array and counter for averaging
    mi_sum = np.zeros(max_distance + 1, dtype=np.float64)
    mi_count = np.zeros(max_distance + 1, dtype=np.int64)


    # Loop over all position pairs (i, j)
    for i in range(len_str):
        j = len_str-1
        d = j - i
        if d > max_distance:
            continue

        # Count joint distribution for this position pair
        joint = np.zeros((v, v), dtype=np.float64)
        for s in range(num_str):
            a = samples[s, i]
            b = samples[s, j]
            joint[a, b] += 1.0

        # Normalize to probabilities
        joint /= num_str

        # Marginals for this exact position pair
        p_a = np.sum(joint, axis=1)
        p_b = np.sum(joint, axis=0)

        # Compute MI for this pair
        mi_val = 0.0
        for a in range(v):
            for b in range(v):
                if joint[a, b] > 0.0:
                    mi_val += joint[a, b] * np.log2(joint[a, b] / (p_a[a] * p_b[b]))

        mi_sum[d] += mi_val
        mi_count[d] += 1

    # Average over all position pairs for each distance
    mi_avg = np.zeros(max_distance + 1, dtype=np.float64)
    for d in range(max_distance + 1):
        if mi_count[d] > 0:
            mi_avg[d] = mi_sum[d] / mi_count[d]
    
    return mi_avg

class process_and_visualize_rhm_mutual_information:

  def __init__(self, L=4, v=8, s=4, m=3, rules_seed=0, epsilon=0.01, rules_str=""):
    self.L = L
    self.v = v
    self.s = s
    self.m = m
    self.rules_seed = rules_seed
    self.epsilon = epsilon
    self.layer_sizes = [self.s**l for l in range(self.L)]
    self.rules_str = rules_str
    if self.rules_str == "":
      self.rules_array = generate_rules_unambiguous_epsilon(
        L, v, s, m, epsilon, rules_seed=self.rules_seed)
    else:
      self.rules_array = generate_rules_str(
        L, v, s, m, epsilon, rules_seed=self.rules_seed, rules_str=rules_str)
    

  def plot_mi_exact_no_temperature(self, add_title=True, max_distance=None, verbose=False, 
                                   save_fig=False, save_data=False, path_fig="plots/", 
                                   path_data="data/", neglect_pos_dep=False, 
                                   last_token_mi=False, dot_size=3.0, fit=False, **kwargs):
    if last_token_mi:
      mi = compute_mi_full_no_temperature_last_token(self.rules_array, max_distance=max_distance, verbose=verbose)
    elif neglect_pos_dep:
      mi = compute_mi_full_no_temperature(self.rules_array, max_distance=max_distance, verbose=verbose)
    else:
      mi = compute_mi_full_no_temperature_per_position(self.rules_array, max_distance=max_distance, verbose=verbose)

    fig, ax, title = self.mi_plotter(mi, 1, 0, add_title=add_title, data_sampled=False, 
                                     max_distance=max_distance, save_fig=save_fig, 
                                     path_fig=path_fig, neglect_pos_dep=neglect_pos_dep, last_token_mi=last_token_mi,
                                     dot_size=dot_size, fit=fit, **kwargs)

    if save_data:
      np.save(path_data+title+".npy", mi)

    return fig, ax, title

  def plot_mi_sampled(self, temperature, num_str, seed=0, add_title=True, max_distance=None, 
                      save_fig=False, save_data=False, path_fig="plots/", 
                      path_data="data/", verbose=False, neglect_pos_dep=False,
                      last_token_mi=False, dot_size=3.0, fit=False, **kwargs):

    if max_distance == None:
      numba_max_distance = -1
    else:
      numba_max_distance = max_distance

    if isinstance(temperature, Iterable):
      mi = []
      for _temperature in temperature:
        sampled_strings = string_generator_numba(self.rules_array, num_str, _temperature, seed=seed, verbose=verbose)
        if last_token_mi:
          mi.append(compute_mutual_information_from_samples_last_token_numba(sampled_strings, max_distance=numba_max_distance ))
        elif neglect_pos_dep:
          mi.append(compute_mutual_information_from_samples_numba(sampled_strings, max_distance=numba_max_distance ))
        else:
          mi.append(compute_mutual_information_from_samples_per_position_numba(sampled_strings, max_distance=numba_max_distance))
      mi = np.array(mi)

    elif isinstance(num_str, Iterable):
      mi = []
      for _num_str in num_str:
        sampled_strings = string_generator_numba(self.rules_array, _num_str, temperature, seed=seed, verbose=verbose)
        if last_token_mi:
          mi.append(compute_mutual_information_from_samples_last_token_numba(sampled_strings, max_distance=numba_max_distance ))
        elif neglect_pos_dep:
          mi.append(compute_mutual_information_from_samples_numba(sampled_strings, max_distance=numba_max_distance ))
        else:
          mi.append(compute_mutual_information_from_samples_per_position_numba(sampled_strings, max_distance=numba_max_distance))
      mi = np.array(mi)

    else:
      sampled_strings = string_generator_numba(self.rules_array, num_str, temperature, seed=seed, verbose=verbose)
      if last_token_mi:
        mi = compute_mutual_information_from_samples_last_token_numba(sampled_strings, max_distance=numba_max_distance )
      elif neglect_pos_dep:
        mi = compute_mutual_information_from_samples_numba(sampled_strings, max_distance=numba_max_distance )
      else:
        mi = compute_mutual_information_from_samples_per_position_numba(sampled_strings, max_distance=numba_max_distance)
    fig, ax, title = self.mi_plotter(mi, temperature, num_str, add_title=add_title, data_sampled=True, 
                                     max_distance=max_distance, save_fig=save_fig, path_fig=path_fig, 
                                     neglect_pos_dep=neglect_pos_dep, last_token_mi=last_token_mi, 
                                     dot_size=dot_size, fit=fit, **kwargs)

    if save_data:
      np.save(path_data+title+".npy", mi)

    return fig, ax, title
  
  def plot_mi_sampled_top_p(self, temperature, num_str, top_p_prob=1.0, seed=0, add_title=True, max_distance=None, 
                      save_fig=False, save_data=False, path_fig="plots/", 
                      path_data="data/", verbose=False, neglect_pos_dep=False,
                      last_token_mi=False, dot_size=3.0, fit=False, **kwargs):

    if max_distance == None:
      numba_max_distance = -1
    else:
      numba_max_distance = max_distance

    if isinstance(temperature, Iterable):
       if len(temperature) == 1:
          temperature = temperature[0]

    if isinstance(temperature, Iterable):
      mi = []
      for _temperature in temperature:
        sampled_strings = string_generator_top_p_numba(self.rules_array, num_str, _temperature, top_p_prob, seed=seed, verbose=verbose)
        if last_token_mi:
          mi.append(compute_mutual_information_from_samples_last_token_numba(sampled_strings, max_distance=numba_max_distance))
        elif neglect_pos_dep:
          mi.append(compute_mutual_information_from_samples_numba(sampled_strings, max_distance=numba_max_distance ))
        else:
          mi.append(compute_mutual_information_from_samples_per_position_numba(sampled_strings, max_distance=numba_max_distance))
      mi = np.array(mi)

    elif isinstance(num_str, Iterable):
      mi = []
      for _num_str in num_str:
        sampled_strings = string_generator_top_p_numba(self.rules_array, _num_str, temperature, top_p_prob, seed=seed, verbose=verbose)
        if last_token_mi:
          mi.append(compute_mutual_information_from_samples_last_token_numba(sampled_strings, max_distance=numba_max_distance))
        elif neglect_pos_dep:
          mi.append(compute_mutual_information_from_samples_numba(sampled_strings, max_distance=numba_max_distance ))
        else:
          mi.append(compute_mutual_information_from_samples_per_position_numba(sampled_strings, max_distance=numba_max_distance))
      mi = np.array(mi)

    else:
      sampled_strings = string_generator_top_p_numba(self.rules_array, num_str, temperature, top_p_prob, seed=seed, verbose=verbose)
      if last_token_mi:
        mi, p_abd = compute_mutual_information_from_samples_last_token_numba_p_abd(sampled_strings, max_distance=numba_max_distance)
      elif neglect_pos_dep:
        mi = compute_mutual_information_from_samples_numba(sampled_strings, max_distance=numba_max_distance )
      else:
        mi = compute_mutual_information_from_samples_per_position_numba(sampled_strings, max_distance=numba_max_distance)
        
    fig, ax, title = self.mi_plotter(mi, temperature, num_str, add_title=add_title, data_sampled=True, 
                                     max_distance=max_distance, save_fig=save_fig, path_fig=path_fig, 
                                     neglect_pos_dep=neglect_pos_dep, last_token_mi=last_token_mi, 
                                     dot_size=dot_size, fit=fit, top_p_prob=top_p_prob, **kwargs)

    if save_data:
      np.save(path_data+title+".npy", mi)
      np.save(path_data+title+f"_sampleSeed={seed}.npy", p_abd)

    return fig, ax, title

  def mi_plotter(self, mi, temperature, num_str, add_title=True, data_sampled=True, 
                  max_distance=None, save_fig=False, path_fig="plots/",
                  neglect_pos_dep=False, last_token_mi=False, log_lin=False, fit=False, num_fit_steps=3, 
                  dot_size=3.0, color=None, return_fit=False, errors=False, top_p_prob=1.0,
                  legend=True,**kwargs):
    
    plt.style.use('plt_style.pstyle')
    plt.rcParams["font.size"] = 7.8		#tick label
    plt.rcParams["figure.titlesize"] = 9	#unkown
    plt.rcParams["legend.fontsize"] = 7	#legend
    plt.rcParams["axes.titlesize"] = 9 	#title (eigentlich *9)
    plt.rcParams["axes.labelsize"] = 8.8  #(eigentlich *9)
    
    if log_lin:
      plt.rcParams['figure.figsize'] = [2*3.1, 1.7]
      fig, ax = plt.subplots(1,2, squeeze=False)
      fig.subplots_adjust(left=0.15, right=0.93, bottom=0.2, top=0.95)
    else:
      plt.rcParams['figure.figsize'] = [3.1, 1.7]
      fig, ax = plt.subplots(squeeze=False)
      fig.subplots_adjust(left=0.15, right=0.93, bottom=0.2, top=0.95)

    if data_sampled:
      title = f"mi_sampled_"
      if top_p_prob<1.0:
         title+=f"p={top_p_prob:.1e}_"
    else:
      title = f"mi_exact_"

    if not(self.rules_str == ""):
      title += f"rulesstr="+self.rules_str+"_"

    title += f"epsilon={self.epsilon:.1e}_"

    if isinstance(temperature, Iterable):
      fit_coefficients = []
      temp_str = ""
      for idx, _temperature in enumerate(temperature):
        temp_str += f"{_temperature:.1e}"
        if idx != len(temperature)-1:
          temp_str += "--"
        if color is not None:
          ax[0,0].scatter(range(1,len(mi[idx])),mi[idx][1:], label=r"$T$"+f"={_temperature}", 
                          color=color[idx], s=dot_size, **kwargs)
        else:  
          ax[0,0].scatter(range(1,len(mi[idx])),mi[idx][1:], label=r"$T$"+f"={_temperature}", s=dot_size, **kwargs)
        if legend:
            ax[0,0].legend()
        if fit:
          try:
            fit_coefficients.append(self.linear_fit_plot_addition(mi[idx], ax[0,0], 
                                  num_fit_steps=num_fit_steps, log_lin=False, errors=errors, **kwargs))
          except:
             fit_coefficients.append((self.linear_fit_plot_addition(mi[idx], ax[0,0], 
                                  num_fit_steps=num_fit_steps, log_lin=False, errors=False, **kwargs), 0*fit_coefficients[-1][1]))
        if log_lin:
          if color is not None:
            ax[0,1].scatter(range(1,len(mi[idx])),mi[idx][1:], label=r"$T$"+f"={_temperature}", 
                            color=color[idx], s=dot_size, **kwargs)
          else:
            ax[0,1].scatter(range(1,len(mi[idx])),mi[idx][1:], label=r"$T$"+f"={_temperature}", s=dot_size, **kwargs)
          if legend:
            ax[0,1].legend()
          if fit:
            fit_coefficients.append(self.linear_fit_plot_addition(mi[idx], ax[0,1], 
                                    num_fit_steps=num_fit_steps, log_lin=True, errors=errors, **kwargs))
      title += f"rules_seed={self.rules_seed}_L={self.L}_v={self.v}_s={self.s}_m={self.m}_temperature="+temp_str
      if data_sampled:
        title += f"_num_str={num_str}"

    elif isinstance(num_str, Iterable) and data_sampled:
      fit_coefficients = []
      temp_str = ""
      for idx, _num_str in enumerate(num_str):
        temp_str += f"{_num_str:.1e}"
        if idx != len(num_str)-1:
          temp_str += "--"
        if color is not None:
          ax[0,0].scatter(range(1,len(mi[idx])),mi[idx][1:], label=f"#str={_num_str}", 
                          color=color[idx], s=dot_size, **kwargs)
        else:
          ax[0,0].scatter(range(1,len(mi[idx])),mi[idx][1:], label=f"#str={_num_str}", s=dot_size, **kwargs)
        if legend:
            ax[0,0].legend()
        if fit:
          fit_coefficients.append(self.linear_fit_plot_addition(mi[idx], ax[0,0], 
                              num_fit_steps=num_fit_steps, log_lin=False, errors=errors, **kwargs))
        if log_lin:
          if color is not None:
            ax[0,1].scatter(range(1,len(mi[idx])),mi[idx][1:], label=f"#str={_num_str}", 
                            color=color[idx], s=dot_size, **kwargs)
          else:
            ax[0,1].scatter(range(1,len(mi[idx])),mi[idx][1:], label=f"#str={_num_str}", s=dot_size, **kwargs)
          if legend:
            ax[0,1].legend()
          if fit:
            fit_coefficients.append(self.linear_fit_plot_addition(mi[idx], ax[0,1], 
                                    num_fit_steps=num_fit_steps, log_lin=True, errors=errors, **kwargs))
          
      title += f"rules_seed={self.rules_seed}_L={self.L}_v={self.v}_s={self.s}_m={self.m}_temperature={temperature:.1e}_num_str="+temp_str
    
    else:
      if color is not None:
        ax[0,0].scatter(range(1,len(mi)),mi[1:], s=dot_size, color=color, **kwargs)
      else:
        ax[0,0].scatter(range(1,len(mi)),mi[1:], s=dot_size, **kwargs)
      if fit:
        fit_coefficients = self.linear_fit_plot_addition(mi, ax[0,0], 
                              num_fit_steps=num_fit_steps, log_lin=False, errors=errors, **kwargs)
      if log_lin:
        if color is not None:
          ax[0,1].scatter(range(1,len(mi)),mi[1:], color=color, s=dot_size, **kwargs)
        else:
          ax[0,1].scatter(range(1,len(mi)),mi[1:], s=dot_size, **kwargs)
        if fit:
            fit_coefficients = [fit_coefficients, 
              self.linear_fit_plot_addition(mi, ax[0,1],
                             num_fit_steps=num_fit_steps, log_lin=True, errors=errors, **kwargs)]
      title += f"rules_seed={self.rules_seed}_L={self.L}_v={self.v}_s={self.s}_m={self.m}_temperature={temperature:.1e}"
      if data_sampled:
        title += f"_num_str={num_str}"

    if max_distance is not None:
      title += f"_max_distance={max_distance}"
    if last_token_mi:
      title += "_last_token_mi"
    elif neglect_pos_dep:
      title += "_neglect_pos_dep"
    if log_lin:
      title += "_log_lin"
    if fit:
      title += "_fit"

    ax[0,0].set_xlabel(r"Distance $r$")
    ax[0,0].set_ylabel(r"Mutual Information $I(r)$")
    ax[0,0].set_yscale("log")
    ax[0,0].set_xscale("log")
    if log_lin:
      ax[0,1].set_xlabel(r"Distance $r$")
      ax[0,1].set_ylabel(r"Mutual Information $I(r)$")
      ax[0,1].set_yscale("log")
    if add_title:
      temp_title = linebreak_title(title, max_len=50)
      ax[0,0].set_title(temp_title)

    if save_fig:
      fig.savefig(path_fig+title+".jpeg",dpi=600, bbox_inches='tight')
      fig.savefig(path_fig+title+".pdf")

    if return_fit:
      return (fig, ax, title, fit_coefficients)
    else:
      return (fig, ax, title)

  def linear_fit_plot_addition(self, mi, ax, num_fit_steps=3, log_lin=False, errors=False, **kwargs):
    num_fit_steps = int(np.min([num_fit_steps, int(np.log(len(mi)-1)/np.log(self.s))+1]))
    split_indices = [self.s**idx -1 for idx in range(num_fit_steps+1)]

    split_indices_pairs = zip(split_indices[:-1], split_indices[1:])
    x_data = [np.mean(np.array(range(1,len(mi)))[start:end]) for start, end in split_indices_pairs]
    if log_lin:
      x_fitdata = np.array(x_data)
    else:
      x_fitdata = np.log10(x_data)
    split_indices_pairs = zip(split_indices[:-1], split_indices[1:])
    y_data = [np.mean(mi[start+1:end+1]) for start, end in split_indices_pairs]
    y_fitdata = np.log10(y_data)
       

    if errors:
        fit_coefficients, V = np.polyfit(x_fitdata, y_fitdata, 1, full = False, cov=True)
    else:
        fit_coefficients = np.polyfit(x_fitdata, y_fitdata, 1, full = False, cov=False)
    print(fit_coefficients[0])
    fit_function = np.poly1d(fit_coefficients)
    y_fit_result = fit_function(x_fitdata)

    print("RMSE")
    print(np.sqrt(np.mean((np.log10(y_data) - y_fit_result)**2)))

    last_color = ax.collections[-1].get_facecolor()[0]
    #ax.collections[-1].set_alpha(0.5)
    lighter_color = lighten_color(last_color, amount=0.6)
    ax.collections[-1].set_facecolor(lighter_color)
    ax.collections[-1].set_edgecolor(lighter_color)
    ax.scatter(x_data, 10**y_fitdata, marker="x",color=last_color,  alpha=1.0, **kwargs)
    ax.plot(x_data, 10**y_fit_result, alpha=0.5, ls="--",color=last_color,**kwargs)
    if errors:
       return (fit_coefficients, V)
    else:
        return fit_coefficients










def mi_plotter_from_data(rhm_para, temperature, num_str, add_title=True, data_sampled=True,
                  max_distance=None, save_fig=False, path_fig="plots/from_data/",
                  path_data="data/", neglect_pos_dep=False, last_token_mi=False,
                  log_lin=False, fit=False, num_fit_steps=3,
                  dot_size=3.0, color=None, T2_shift_temperature=None, ylim=None,
                  mask_mi = None, top_p_prob=1.0, exponent_plot = False, errors=False,
                  ylim_exp=None, xlim_exp=None, legend=True, fit_load=None,):

  #T2_shift_temperature list of indices, first one indicating which temperature to start from

  plot_class = process_and_visualize_rhm_mutual_information(**rhm_para)

  if fit_load is None:
     fit_load =fit

  if data_sampled == False:
    if isinstance(temperature, Iterable):
        _rhm_para= rhm_para.copy()
        mi = []
        for _temperature in temperature:
            _rhm_para["epsilon"] = rhm_para["epsilon"]/_temperature
            _plot_class = process_and_visualize_rhm_mutual_information(**_rhm_para)
            mi_len = plot_class.s**(plot_class.L - 1) -1
            if max_distance is not None:
                if max_distance < mi_len:
                    mi_len = max_distance
            mi_placeholder = np.ones(mi_len)
            try:
                fig, ax, title = _plot_class.mi_plotter(mi_placeholder, 1.0, num_str, data_sampled=data_sampled,
                            max_distance=max_distance, save_fig=False,
                            neglect_pos_dep=neglect_pos_dep, last_token_mi=last_token_mi,
                            log_lin=False, fit=fit_load , num_fit_steps=num_fit_steps)
                plt.close()
                mi.append(np.load(path_data+title+".npy"))
            except FileNotFoundError:
                fig, ax, title = _plot_class.mi_plotter(mi_placeholder, 1.0, num_str, data_sampled=data_sampled,
                            max_distance=max_distance, save_fig=False,
                            neglect_pos_dep=neglect_pos_dep, last_token_mi=last_token_mi,
                            log_lin=True, fit=fit_load , num_fit_steps=num_fit_steps)
                plt.close()
                mi.append(np.load(path_data+title+".npy"))
        mi = np.array(mi)
  elif isinstance(temperature, Iterable):
    mi = []
    for idx, _temperature in enumerate(temperature):
      mi_len = plot_class.s**(plot_class.L - 1) -1
      if max_distance is not None:
        if max_distance < mi_len:
          mi_len = max_distance
      mi_placeholder = np.ones(mi_len)
      try:
        fig, ax, title = plot_class.mi_plotter(mi_placeholder, _temperature, num_str, data_sampled=data_sampled,
                    max_distance=max_distance, save_fig=False,
                    neglect_pos_dep=neglect_pos_dep, last_token_mi=last_token_mi,
                    log_lin=False, fit=fit_load , num_fit_steps=num_fit_steps, top_p_prob=top_p_prob)
        plt.close()
        mi_data = np.load(path_data+title+".npy")
        if mask_mi is not None:
           power = np.min([plot_class.L-1, mask_mi[idx]])
           mi_data = mi_data[:plot_class.s**power]
        mi.append(mi_data)
      except FileNotFoundError:
        fig, ax, title = plot_class.mi_plotter(mi_placeholder, _temperature, num_str, data_sampled=data_sampled,
                    max_distance=max_distance, save_fig=False,
                    neglect_pos_dep=neglect_pos_dep, last_token_mi=last_token_mi,
                    log_lin=True, fit=fit_load , num_fit_steps=num_fit_steps, top_p_prob=top_p_prob)
        plt.close()
        mi_data = np.load(path_data+title+".npy")
        if mask_mi is not None:
           power = np.min([plot_class.L-1, mask_mi[idx]])
           mi_data = mi_data[:plot_class.s**power]
        mi.append(mi_data)
    #mi = np.array(mi)
  else:
    mi_len = plot_class.s**(plot_class.L - 1) -1
    if max_distance is not None:
      if max_distance < mi_len:
        mi_len = max_distance
    mi_placeholder = np.ones(mi_len)
    try:
      fig, ax, title = plot_class.mi_plotter(mi_placeholder, _temperature, num_str, data_sampled=data_sampled,
                  max_distance=max_distance, save_fig=False,
                  neglect_pos_dep=neglect_pos_dep, last_token_mi=last_token_mi,
                  log_lin=False, fit=fit_load , num_fit_steps=num_fit_steps, top_p_prob=top_p_prob)
      plt.close()
      mi = np.load(path_data+title+".npy")
    except FileNotFoundError:
      fig, ax, title = plot_class.mi_plotter(mi_placeholder, _temperature, num_str, data_sampled=data_sampled,
                  max_distance=max_distance, save_fig=False,
                  neglect_pos_dep=neglect_pos_dep, last_token_mi=last_token_mi,
                  log_lin=True, fit=fit_load , num_fit_steps=num_fit_steps, top_p_prob=top_p_prob)
      plt.close()
      mi = np.load(path_data+title+".npy")

  fig, ax, title, fit_coefficients = plot_class.mi_plotter(mi, temperature, num_str,
                  add_title=add_title, data_sampled=data_sampled,
                  max_distance=max_distance, save_fig=False,
                  neglect_pos_dep=neglect_pos_dep, last_token_mi=last_token_mi,
                  log_lin=log_lin, fit=fit, num_fit_steps=num_fit_steps,
                  dot_size=dot_size, color=color, return_fit=True, top_p_prob=top_p_prob,
                  errors=errors, legend=legend)

  if T2_shift_temperature is not None:
    if log_lin:
      raise Exception("T**2 shifts work only for log_log plots")
    if not isinstance(temperature, Iterable):
      raise Exception("T**2 shifts work only for multiple temperatures")

    x_data = np.array([1,np.min([plot_class.s**num_fit_steps - 1,len(mi[0])])])
    x_fitdata = np.log10(x_data)
    fit_function = np.poly1d(fit_coefficients[T2_shift_temperature[0]])
    y_fit_result = fit_function(x_fitdata)
    ax[0,0].plot(x_data, 10**y_fit_result, alpha=0.7, ls="--",color="red")

    T_orig = temperature[T2_shift_temperature[0]]
    for T2_shift_idx in T2_shift_temperature[1:]:
      T_current = temperature[T2_shift_idx]
      ax[0,0].plot(x_data, (T_orig/T_current)**2*10**y_fit_result, alpha=0.7, ls="--",color="red")

  if ylim is not None:
    ax[0,0].set_ylim(ylim)

  if legend:
    ax[0,0].legend(handletextpad=-0.2, markerscale=1.5, labelspacing=0.1)

  if save_fig:
      fig.savefig(path_fig+title+".jpeg",dpi=600, bbox_inches='tight')
      fig.savefig(path_fig+title+".pdf")


  if exponent_plot:
    if log_lin:
        raise Exception("exponent plot for log_lin double plot not implemented")
    if not isinstance(temperature, Iterable):
        raise Exception("exponent plot works only for multiple temperatures")
    
    plt.style.use('plt_style.pstyle')
    plt.rcParams["font.size"] = 7.8		#tick label
    plt.rcParams["figure.titlesize"] = 9	#unkown
    plt.rcParams["legend.fontsize"] = 7	#legend
    plt.rcParams["axes.titlesize"] = 9 	#title (eigentlich *9)
    plt.rcParams["axes.labelsize"] = 8.8  #(eigentlich *9)

    plt.rcParams['figure.figsize'] = [2.35, 1.7]
    fig_exp, ax_exp = plt.subplots(squeeze=False)
    fig_exp.subplots_adjust(left=0.2, right=0.93, bottom=0.2, top=0.95)
    if  errors:
        exponents = [fit_coefficients[idx][0][0] for idx in range(len(temperature))]
        variance = [fit_coefficients[idx][1][0,0] for idx in range(len(temperature))]
    else:
        exponents = [fit_coefficients[idx][0] for idx in range(len(temperature))]

    if errors:
        #plt.errorbar(temperatures, -np.array(exponents), yerr=2*np.sqrt(variance), capsize=2.0, marker="x", color="#FF2C00", linewidth=0.5)
        ax_exp[0,0].errorbar(temperature, -np.array(exponents), yerr=2*np.sqrt(variance), capsize=2.0, marker="x", color="#00B945", linewidth=0.5, ls="None")
    else:
        ax_exp[0,0].plot(temperature, -np.array(exponents), marker="x", color="#00B945", linewidth=0.5)
    ax_exp[0,0].loglog()
    ax_exp[0,0].set_xlabel(r"Temperature $T$")
    ax_exp[0,0].set_ylabel(r"Exponent $\alpha$")
    temp_title = linebreak_title(title, max_len=50)
    if ylim_exp is not None:
        ax_exp[0,0].set_ylim(ylim_exp)
    if xlim_exp is not None:
        ax_exp[0,0].set_xlim(xlim_exp)
    if add_title:
        ax_exp[0,0].set_title(temp_title)

    if save_fig:
        fig_exp.savefig(path_fig+title+"_exponents.jpeg",dpi=600, bbox_inches='tight')
        fig_exp.savefig(path_fig+title+"_exponents.pdf")

  if exponent_plot:
     return (fig, ax, title, fig_exp, ax_exp)
  else:
    return (fig, ax, title)


def lighten_color(color, amount=0.5):
    """
    Lightens the given color by mixing it with white.
    
    amount=0 -> original color
    amount=1 -> white
    """
    c = np.array(mpl.colors.to_rgb(color))
    white = np.array([1, 1, 1])
    return tuple(c + (white - c) * amount)


def mi_unify_p_abd_from_data(rhm_para, temperature, num_str, sample_seeds = [0],
                  max_distance=None, path_data="data/", neglect_pos_dep=False, 
                  last_token_mi=False, fit=False, top_p_prob=1.0, save_data=False):

    plot_class = process_and_visualize_rhm_mutual_information(**rhm_para)

    if isinstance(temperature, Iterable):
        if len(temperature)==1:
           temperature=temperature[0]
        else:
            raise("single temperature only")

    mi_len = plot_class.s**(plot_class.L - 1) -1
    if max_distance is not None:
        if max_distance < mi_len:
            mi_len = max_distance
    mi_placeholder = np.ones(mi_len)
    p_abd_list = []
    for seed in sample_seeds:
        try:
            fig, ax, title = plot_class.mi_plotter(mi_placeholder,temperature, num_str, data_sampled=True,
                        max_distance=max_distance, save_fig=False,
                        neglect_pos_dep=neglect_pos_dep, last_token_mi=last_token_mi,
                        log_lin=False, fit=fit, num_fit_steps=3, top_p_prob=top_p_prob)
            plt.close()
            p_abd = np.load(path_data+title+f"_sampleSeed={seed}.npy")

        except FileNotFoundError:
            fig, ax, title = plot_class.mi_plotter(mi_placeholder, temperature, num_str, data_sampled=True,
                        max_distance=max_distance, save_fig=False,
                        neglect_pos_dep=neglect_pos_dep, last_token_mi=last_token_mi,
                        log_lin=True, fit=fit, num_fit_steps=3, top_p_prob=top_p_prob)
            plt.close()
            p_abd = np.load(path_data+title+f"_sampleSeed={seed}.npy")
        p_abd_list.append(p_abd)

    p_abd = np.mean(np.array(p_abd_list), axis = 0)
    mi = compute_mutual_information_from_p_abd_last_token_numba(p_abd)
    if save_data:
        fig, ax, title = plot_class.mi_plotter(mi_placeholder,temperature, num_str*len(sample_seeds), data_sampled=True,
                        max_distance=max_distance, save_fig=False,
                        neglect_pos_dep=neglect_pos_dep, last_token_mi=last_token_mi,
                        log_lin=False, fit=fit, num_fit_steps=3, top_p_prob=top_p_prob)
        plt.close()
        np.save(path_data+title+".npy", mi)

    return mi