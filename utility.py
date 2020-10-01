import pandas as pd
import numpy as np
import datetime

def search_prior_indices(lst, adjacent_lst):
    """
    For every object in lst, this function returns the next smaller
    element in adjadent_lst.
    """
    prior = []

    iter_adjacent_lst = iter(adjacent_lst)

    adj_index = next(iter_adjacent_lst) # current non_na_index
    prior_adj_index = -1

    for i in lst:
        if adj_index > i:
            prior += [prior_adj_index]
        else:
            while adj_index < i:
                prior_adj_index = adj_index
                try:
                    adj_index = next(iter_adjacent_lst)
                except:
                    adj_index = i+1

            prior += [prior_adj_index]

    return pd.Series(prior, index=lst)

def search_posterior_indices(lst: list, adjacent_lst: list):
    """
    For every object in lst, this function returns the next bigger
    element in adjadent_lst.
    """
    posterior = []

    iter_adjacent_lst = iter(adjacent_lst)

    adj_index = next(iter_adjacent_lst) # current non_na_index
    posterior_adj_index = -1
    last_index = max(adjacent_lst)

    for i in lst:
        if adj_index > i:
            posterior += [adj_index]
        else:
            while adj_index < i:
                try:
                    adj_index = next(iter_adjacent_lst)
                except:
                    adj_index = i+1

            if adj_index > last_index:
                adj_index = -1
            posterior += [adj_index]

    return pd.Series(posterior, index=lst)
    