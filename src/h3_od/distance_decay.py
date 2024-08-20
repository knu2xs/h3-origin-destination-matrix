#!/usr/bin/env python
# coding: utf-8
"""
Functions enabling calculation of distance decay, applying a S-shaped sigmoid curve to model distance decay.
"""
from typing import Union

import numpy as np

__all__ = [
    "get_sigmoid_distance_decay_index",
    "get_bus_stop_distance_decay_index",
    "get_light_rail_stop_distance_decay_index",
]


def get_sigmoid_distance_decay_index(
    distance: Union[float, int], steepness: Union[float, int], offset: Union[float, int]
) -> float:
    """
    Get sigmoid distance decay index.

    Args:
        distance: Distance to calculate decay for.
        steepness:
        offset:
    """
    distance_index = 1 / (1 + np.exp(steepness * (distance - offset)))

    return distance_index


def get_bus_stop_distance_decay_index(distance: Union[float, int]) -> float:
    """
    Get the distance decay coefficient for a bus stop.

    Args:
        distance: Walking distance in miles to the bus stop.
    """
    distance_index = get_sigmoid_distance_decay_index(distance, 5.8, 0.65)
    return distance_index


def get_light_rail_stop_distance_decay_index(distance: Union[float, int]) -> float:
    """
    Get the distance decay coefficient for a light rail stop.

    Args:
        distance: Walking distance in miles to the light rail stop or station.
    """
    distance_index = get_sigmoid_distance_decay_index(distance, 4.8, 1.3) * 0.98
    return distance_index
