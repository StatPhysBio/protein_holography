"""Module for zernikegram convention"""

from typing import Iterable,List,Dict

import numpy as np

channel_ordering = ['C','N','O','S','H','SASA','charge']

def channel_freq_to_ind(
    channel: str,
    freq: int,
    channels: Iterable[str]=channel_ordering,
    freqs: Iterable[int]=np.arange(21)
) -> int:
    """Get index for a given channel and frequency combination"""
    num_channels = len(channels)
    num_freqs = len(freqs)
    channels = np.array(channels)
    freqs = np.sort(freqs)

    ind = (
        np.squeeze(np.argwhere(channels == channel)) * num_freqs
        + np.squeeze(np.argwhere(freqs == freq))
    )

    return ind

def channel_freq_to_inds(
    channel_freq_dict: Dict[str,Iterable[int]]
) -> np.ndarray:
    """Get indices for many channel frequency pairs"""
    num_channels = len(channel_ordering)
    channel_freq_pairs = []
    for channel,freqs in channel_freq_dict.items():
        for freq in freqs:
            channel_freq_pairs.append((channel,freq))
    return np.sort([channel_freq_to_ind(x,y) for x,y in channel_freq_pairs])

def get_channel_dict(channels,freqs):
    if channels == "el":
        return el(np.max(freqs))
    if channels == "el_noH":
        return el_noH(np.max(freqs))
    if channels == "el_SASA":
        return el_SASA(np.max(freqs))
    if channels == "el_charge":
        return el_charge(np.max(freqs))
    if channels == "test":
        return test(np.max(freqs))

def test(max_freq):
    return {
        'C': [0],
    }

def el(max_freq):
    freqs = np.arange(max_freq+1)
    return {
        'C': freqs,
        'N': freqs,
        'O': freqs,
        'S': freqs,
        'H': freqs,
    }

def el_noH(max_freq):
    return {
        'C': freqs,
        'N': freqs,
        'O': freqs,
        'S': freqs
    }

def el_SASA(max_freq):
    return {
        'C': freqs,
        'N': freqs,
        'O': freqs,
        'S': freqs,
        'H': freqs,
        'SASA': freqs
    }

def el_charge(max_freq):
    return {
        'C': freqs,
        'N': freqs,
        'O': freqs,
        'S': freqs,
        'H': freqs,
        'charge': freqs
    }
