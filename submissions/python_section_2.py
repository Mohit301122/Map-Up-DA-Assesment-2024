import pandas as pd
import numpy as np


def calculate_distance_matrix(df)->pd.DataFrame():
    """
    Calculate a distance matrix based on the dataframe, df.

    Args:
        df (pandas.DataFrame)

    Returns:
        pandas.DataFrame: Distance matrix
    """
    df = pd.read_csv('dataset-2.csv')
    df.columns = ['id_1', 'id_2', 'distance']

    toll_locations = pd.concat([df['id_1'], df['id_2']]).unique()

    dist_matrix = pd.DataFrame(np.inf, index=toll_locations, columns=toll_locations)
    np.fill_diagonal(dist_matrix.values, 0)

    for _, row in df.iterrows():
        id_1, id_2, distance = row['id_1'], row['id_2'], row['distance']
        dist_matrix.at[id_1, id_2] = distance
        dist_matrix.at[id_2, id_1] = distance

    for k in toll_locations:
        for i in toll_locations:
            for j in toll_locations:
                dist_matrix.at[i, j] = min(dist_matrix.at[i, j], dist_matrix.at[i, k] + dist_matrix.at[k, j])

    return dist_matrix

distance_matrix = calculate_distance_matrix()
print(distance_matrix)

    return distance_matrix
    return df









def unroll_distance_matrix(df)->pd.DataFrame():
    """
    Unroll a distance matrix to a DataFrame in the style of the initial dataset.

    Args:
        df (pandas.DataFrame)

    Returns:
        pandas.DataFrame: Unrolled DataFrame containing columns 'id_start', 'id_end', and 'distance'.
    """
    unrolled_data = []

    for id_start in distance_matrix.index:
        for id_end in distance_matrix.columns:
            if id_start != id_end:
                unrolled_data.append({
                    'id_start': id_start,
                    'id_end': id_end,
                    'distance': distance_matrix.at[id_start, id_end]
                })

    unrolled_df = pd.DataFrame(unrolled_data)
    return unrolled_df

# Usage example
unrolled_df = unroll_distance_matrix(distance_matrix)
print(unrolled_df)

    return unrolled_df
    return df










def find_ids_within_ten_percentage_threshold(df, reference_id)->pd.DataFrame():
    """
    Find all IDs whose average distance lies within 10% of the average distance of the reference ID.

    Args:
        df (pandas.DataFrame)
        reference_id (int)

    Returns:
        pandas.DataFrame: DataFrame with IDs whose average distance is within the specified percentage threshold
                          of the reference ID's average distance.
    """
     reference_distances = df[df['id_start'] == reference_id]
    average_distance = reference_distances['distance'].mean()
    lower_bound = average_distance * 0.9
    upper_bound = average_distance * 1.1

    ids_within_threshold = reference_distances[
        (reference_distances['distance'] >= lower_bound) & 
        (reference_distances['distance'] <= upper_bound)
    ][['id_start', 'id_end', 'distance']]

    return ids_within_threshold

    return df










def calculate_toll_rate(df)->pd.DataFrame():
    """
    Calculate toll rates for each vehicle type based on the unrolled DataFrame.

    Args:
        df (pandas.DataFrame)

    Returns:
        pandas.DataFrame
    """
    df['moto'] = df['distance'] * 0.8
    df['car'] = df['distance'] * 1.2
    df['rv'] = df['distance'] * 1.5
    df['bus'] = df['distance'] * 2.2
    df['truck'] = df['distance'] * 3.6

    return df








def calculate_time_based_toll_rates(df)->pd.DataFrame():
    """
    Calculate time-based toll rates for different time intervals within a day.

    Args:
        df (pandas.DataFrame)

    Returns:
        pandas.DataFrame
    """
    days_of_week = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    time_ranges = [
        (datetime.time(0, 0, 0), datetime.time(10, 0, 0), 0.8),  # 00:00 to 10:00, factor 0.8
        (datetime.time(10, 0, 0), datetime.time(18, 0, 0), 1.2),  # 10:00 to 18:00, factor 1.2
        (datetime.time(18, 0, 0), datetime.time(23, 59, 59), 0.8)  # 18:00 to 23:59, factor 0.8
    ]
    
    toll_df = pd.DataFrame()

    for _, row in df.iterrows():
        for day in days_of_week:
            for start_time, end_time, weekday_factor in time_ranges:
                modified_row = row.copy()
                modified_row['start_day'] = day
                modified_row['end_day'] = day
                modified_row['start_time'] = start_time
                modified_row['end_time'] = end_time

                if day in ['Saturday', 'Sunday']:
                    discount_factor = 0.7
                else:
                    discount_factor = weekday_factor

                modified_row['moto'] = row['moto'] * discount_factor
                modified_row['car'] = row['car'] * discount_factor
                modified_row['rv'] = row['rv'] * discount_factor
                modified_row['bus'] = row['bus'] * discount_factor
                modified_row['truck'] = row['truck'] * discount_factor

                toll_df = toll_df.append(modified_row, ignore_index=True)
    
    return toll_df

    return df
