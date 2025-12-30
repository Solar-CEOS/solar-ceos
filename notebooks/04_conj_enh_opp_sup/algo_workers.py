# Filename: algo_workers.py
import numpy as np

# =============================================================================
# 1. Core Geometric Calculation Functions (Must be here for subprocess calls)
# =============================================================================

def count_events_vectorized(sunspot_lons, planet_lons, w, event_type):
    """
    Core geometric calculation function (Algo 1)
    sunspot_lons: (N,)
    planet_lons: (N, 8)
    """
    if len(sunspot_lons) == 0: return 0
    
    # Vectorized calculation of angle difference (sun - planet) and normalized to [-180, 180]
    delta = np.mod(sunspot_lons[:, np.newaxis] - planet_lons + 180, 360) - 180
    
    if event_type == 'Conjunction':
        # Conjunction: delta close to 0
        is_event = np.abs(delta) <= w
    elif event_type == 'Opposition':
        # Opposition: delta close to 180 or -180
        is_event = np.abs(np.abs(delta) - 180) <= w
    else:
        return 0
    
    return np.sum(is_event)

def count_events_at_least_once(sunspot_lons, planet_lons, w, event_type):
    """Algo 2 Core: At least once"""
    if len(sunspot_lons) == 0: return 0
    
    delta = np.mod(sunspot_lons[:, np.newaxis] - planet_lons + 180, 360) - 180
    
    if event_type == 'Conjunction':
        is_evt_mat = np.abs(delta) <= w
    elif event_type == 'Opposition':
        is_evt_mat = np.abs(np.abs(delta) - 180) <= w
    else:
        return 0
        
    # Check if any True exists along the planet dimension (axis=1)
    return np.sum(np.any(is_evt_mat, axis=1))

# =============================================================================
# 2. Worker Functions (Parallel Worker Functions)
# =============================================================================

def cts_worker_algo1(seed, sunspot_lons, ephem_matrix, sunspot_indices, w, event_type):
    """Algo 1 Worker: Cyclic Time Shift (CTS)"""
    np.random.seed(seed)
    T = ephem_matrix.shape[0]
    # CTS random shift
    shift = np.random.randint(0, T)
    shifted_indices = (sunspot_indices + shift) % T
    shifted_planets = ephem_matrix[shifted_indices]
    
    return count_events_vectorized(sunspot_lons, shifted_planets, w, event_type)

def cts_worker_algo2(seed, sunspot_lons, ephem_matrix, sunspot_indices, w, event_type):
    """Algo 2 Worker: Cyclic Time Shift (CTS)"""
    np.random.seed(seed)
    T = ephem_matrix.shape[0]
    shift = np.random.randint(0, T)
    
    shifted_indices = (sunspot_indices + shift) % T
    shifted_planets = ephem_matrix[shifted_indices]
    
    return count_events_at_least_once(sunspot_lons, shifted_planets, w, event_type)