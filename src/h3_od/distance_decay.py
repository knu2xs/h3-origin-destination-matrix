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
        steepness: Controls how sharply the decay curve transitions from 1 to 0.
            Higher values produce a steeper drop-off.
        offset: Distance value at which the decay index equals 0.5, shifting
            the midpoint of the sigmoid curve along the distance axis.

    Returns:
        Sigmoid distance decay index between 0.0 and 1.0.
    """
    distance_index = 1 / (1 + np.exp(steepness * (distance - offset)))

    return distance_index


def get_bus_stop_distance_decay_index(distance: Union[float, int]) -> float:
    """
    Get the distance decay coefficient for a bus stop.

    Args:
        distance: Walking distance in miles to the bus stop.

    Returns:
        Sigmoid distance decay index between 0.0 and 1.0 for the given bus stop distance.
    """
    distance_index = get_sigmoid_distance_decay_index(distance, 5.8, 0.65)
    return distance_index


def get_light_rail_stop_distance_decay_index(distance: Union[float, int]) -> float:
    """
    Get the distance decay coefficient for a light rail stop.

    Args:
        distance: Walking distance in miles to the light rail stop or station.

    Returns:
        Sigmoid distance decay index between 0.0 and ~0.98 for the given light rail stop distance.
    """
    distance_index = get_sigmoid_distance_decay_index(distance, 4.8, 1.3) * 0.98
    return distance_index
