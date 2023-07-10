import numpy as np

def find_nearest_value(array, x):
    """
    Perform binary search to find the nearest value to x in a sorted array.

    Parameters
    ----------
    array : np.ndarray
        Sorted array in which to search.
    x : int or float
        The value to find the nearest match for in the array.

    Returns
    -------
    nearest : int or float
        The value in the array closest to x.

    Examples
    --------
    >>> find_nearest_value(np.array([1, 2, 4, 5]), 3)
    2
    """
    index = np.searchsorted(array, x, side='left')
    if index != 0 and (index == array.shape[0] or np.abs(x - array[index-1]) <= np.abs(x - array[index])):
        return array[index-1]
    else:
        return array[index]

def nearest_value_in_arrays(array_from, array_to):
    """
    For each element in array_from, find the nearest value in array_to.

    Parameters
    ----------
    array_from : np.ndarray
        Array from which to take elements.
    array_to : np.ndarray
        Sorted array in which to find the nearest values.

    Returns
    -------
    nearest_values : np.ndarray
        Array of nearest values in array_to for each element in array_from.

    Examples
    --------
    >>> nearest_value_in_arrays(np.array([1, 3, 5]), np.array([2, 4, 6]))
    array([2, 2, 4])
    """
    array_to.sort()  # Sort the array for binary search
    nearest_values = np.empty_like(array_from)
    for i, element in enumerate(array_from):
        nearest_values[i] = find_nearest_value(array_to, element)
    return nearest_values


def distance_to_nearest_value(array, x):
    """
    Perform binary search to find the distance to the nearest value to x in a sorted array.

    Parameters
    ----------
    array : np.ndarray
        Sorted array in which to search.
    x : int or float
        The value to find the nearest match for in the array.

    Returns
    -------
    distance : int or float
        The distance to the value in the array closest to x.

    Examples
    --------
    >>> distance_to_nearest_value(np.array([1, 2, 4, 5]), 3)
    1
    """
    index = np.searchsorted(array, x, side='left')
    if index != 0 and (index == array.shape[0] or np.abs(x - array[index-1]) <= np.abs(x - array[index])):
        return np.abs(x - array[index-1])
    else:
        return np.abs(x - array[index])

def distance_to_nearest_value_in_arrays(array_from, array_to):
    """
    For each element in array_from, find the distance to the nearest value in array_to.

    Parameters
    ----------
    array_from : np.ndarray
        Array from which to take elements.
    array_to : np.ndarray
        Sorted array in which to find the nearest values.

    Returns
    -------
    distances : np.ndarray
        Array of distances to nearest values in array_to for each element in array_from.

    Examples
    --------
    >>> distance_to_nearest_value_in_arrays(np.array([1, 3, 8]), np.array([2, 4, 6]))
    array([1, 1, 2])
    """
    array_to.sort()  # Sort the array for binary search
    distances = np.empty_like(array_from)
    for i, element in enumerate(array_from):
        distances[i] = distance_to_nearest_value(array_to, element)
    return distances
