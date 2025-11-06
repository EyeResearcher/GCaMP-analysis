import numpy as np
import networkx as nx
from sklearn.cluster import AgglomerativeClustering

def calculate_average_sttc_for_groups(groups, sttc_matrix):
    average_sttc_per_group = []

    for group in groups:
        # Extract all pairs of neurons within the group
        sttc_values = []
        for i in range(len(group)):
            for j in range(i + 1, len(group)):
                neuron_1 = group[i]
                neuron_2 = group[j]
                sttc_values.append(sttc_matrix[neuron_1, neuron_2])

        # Calculate the average STTC value for the group
        if sttc_values:
            average_sttc = np.mean(sttc_values)
        else:
            average_sttc = 1  # If there are no pairs, default to 0
        average_sttc_per_group.append(average_sttc)

    return average_sttc_per_group

def neuronGrouping(sttc, params = {"threshold_for_neuron_groups": .7}):
    graph_thresh = params["threshold_for_neuron_groups"]
    adjacency_matrix = sttc > graph_thresh

    G_graph_085 = nx.from_numpy_array(adjacency_matrix)

    connected_components = list(nx.connected_components(G_graph_085))

    graph_neuron_groups_085 = [list(component) for component in connected_components]

    real_groups = [group for group in graph_neuron_groups_085 if len(group) > 1]
    return real_groups

def split_groups(groups, average_sttc, stats, sttc, threshold=0.8):
    new_groups = []
    for group_index, group in enumerate(groups):
        # If the average STTC is below the threshold, split the group into two
        if average_sttc[group_index] < threshold:
            # Extract the coordinates of the neurons in the group
            group_x_coords = [stats[neuron_index]['xpix'].mean() for neuron_index in group]
            group_y_coords = [stats[neuron_index]['ypix'].mean() for neuron_index in group]
            coordinates = np.array(list(zip(group_x_coords, group_y_coords)))

            # Apply Agglomerative Clustering to split the group into two subgroups
            clustering = AgglomerativeClustering(n_clusters=2)
            subgroup_labels = clustering.fit_predict(coordinates)

            # Assign neurons to subgroups
            subgroup_1 = [group[i] for i in range(len(group)) if subgroup_labels[i] == 0]
            subgroup_2 = [group[i] for i in range(len(group)) if subgroup_labels[i] == 1]
            def calculate_group_sttc(subgroup):
                sttc_values = [
                    sttc[neuron_1, neuron_2]
                    for i, neuron_1 in enumerate(subgroup)
                    for j, neuron_2 in enumerate(subgroup)
                    if i < j
                ]
                return np.mean(sttc_values) if sttc_values else 1

            subgroup_1_sttc = calculate_group_sttc(subgroup_1)
            subgroup_2_sttc = calculate_group_sttc(subgroup_2)
            if subgroup_1_sttc > average_sttc[group_index] and subgroup_2_sttc > average_sttc[group_index]:
                if len(subgroup_1) > 1:
                    new_groups.append(subgroup_1)
                if len(subgroup_2) > 1:
                    new_groups.append(subgroup_2)
                # Keep the original group if the split condition is not met
            # Append the new subgroups to the new_groups list
        else:
            # If the average STTC is above the threshold, keep the group as is
            new_groups.append(group)
    return new_groups


def extract_coordinates(stat, indices=None):
    """
    Extract the centroid coordinates for specific neurons or all neurons if no indices provided.

    Args:
        stat (list of dict): Contains neuron data, including 'med' for coordinates.
        indices (list of int, optional): Indices of neurons to extract. If None, extract all neurons.

    Returns:
        np.array: Coordinates of the neurons as an array of shape (N, 2).
    """
    if indices is None:
        return np.array([neuron['med'] for neuron in stat])
    return np.array([stat[idx]['med'] for idx in indices])

def mean_pairwise_distance(coords):
    """
    Calculate the mean pairwise distance for a set of coordinates.

    Args:
        coords (np.array): A numpy array of shape (N, 2) representing N (x, y) coordinates.

    Returns:
        float: The mean pairwise distance.
    """
    coords = np.asarray(coords)
    if coords.size == 0:
        return np.nan
    coords = np.atleast_2d(coords)
    if coords.shape[0] < 2:
        return 0.0
    pairwise_distances = np.linalg.norm(coords[:, np.newaxis, :] - coords[np.newaxis, :, :], axis=-1)
    return np.mean(pairwise_distances)

def compute_group_dispersion(stat, new_groups):
    """
    Compute and compare the spatial dispersion of all cells in `stat` versus grouped neurons.

    Args:
        stat (list of dict): Contains neuron data, including 'med' for coordinates.
        new_groups (list of list): Each group is a list of neuron indices.

    Returns:
        dict: A dictionary containing:
            - MPD_All_Cells: Mean Pairwise Distance of all cells in `stat`.
            - MPD_Groups: List of Mean Pairwise Distance for each group.
            - MPD_Groups_Mean: Mean of the group's MPDs.
            - Relative_Dispersion_Ratio: MPD_All_Cells / MPD_Groups_Mean
    """
    # Compute coordinates for all cells in `stat`
    all_coords = extract_coordinates(stat)

    # Compute MPD for all cells
    mpd_all_cells = mean_pairwise_distance(all_coords)

    # Compute MPD for each group
    mpd_groups = []
    for group in new_groups:
        if len(group) < 2:
            mpd_groups.append(0.0)
            continue
        group_coords = extract_coordinates(stat, group)
        mpd_groups.append(mean_pairwise_distance(group_coords))

    if mpd_groups:
        mpd_groups_mean = float(np.mean(mpd_groups))
        if np.isfinite(mpd_groups_mean) and mpd_groups_mean > 0:
            relative_dispersion_ratio = mpd_all_cells / mpd_groups_mean
        else:
            relative_dispersion_ratio = np.nan
    else:
        mpd_groups_mean = np.nan
        relative_dispersion_ratio = np.nan

    return {
        "MPD_All_Cells": mpd_all_cells,
        "MPD_Groups": mpd_groups,
        "MPD_Groups_Mean": mpd_groups_mean,
        "Relative_Dispersion_Ratio": relative_dispersion_ratio
    }
def main_grouping(sttc, stat):
    """
    Perform neuron grouping based on STTC matrix, then compute group dispersion.

    Returns:
        new_groups (list of list of int): indices into your filtered neuron list
        distances_dict (dict): dispersion statistics for each group
        new_avg (list of float): average STTC per group
    """
    # 1) Build initial groups from STTC threshold
    groups = neuronGrouping(sttc)
    avg_sttc = calculate_average_sttc_for_groups(groups, sttc)

    # 2) Optionally split any low-STTC groups
    new_groups = split_groups(groups, avg_sttc, stat, sttc)
    new_avg    = calculate_average_sttc_for_groups(new_groups, sttc)

    # 3) Compute spatial dispersion
    distances_dict = compute_group_dispersion(stat, new_groups)

    # Now return the index‐based groups directly, no “restore” step
    return new_groups, distances_dict, new_avg