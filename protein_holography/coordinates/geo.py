#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""This module converts Cartesian coordinate arrays to spherical coordinate arrays.
    Copyright (C) 2019 Pun, Michael
    This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.
    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.
    You should have received a copy of the GNU General Public License
    along with this program.  If not, see <https://www.gnu.org/licenses/>.
"""
from typing import Tuple

import numpy as np

def cartesian_to_spherical(r: np.ndarray) -> Tuple[float]:
    """
    Convert Cartesian coordinates to spherical coorindates.

    Parameters
    ----------
    r : numpy.ndarray
        Array of Cartesian coordinates x, y, z.
    Returns
    -------
    r_mag : float
        Radial distance.
    theta : float
        Polar angle.
    phi : float
        Azimuthal angle.
    """
    x, y, z = *r,

    r_mag = np.sqrt(np.sum(r**2))
    theta = np.arccos(z / r_mag)
    phi = np.arctan2(y, x)

    return r_mag, theta , phi
