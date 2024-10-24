from typing import Dict, List

import pandas as pd
import polyline
import math


def reverse_by_n_elements(lst: List[int], n: int) -> List[int]:
    """
    Reverses the input list by groups of n elements.
    """
     for i in range(0, len(lst), n):
        group_end = min(i + n, len(lst))
        for j in range((group_end - i) // 2):
            lst[i + j], lst[group_end - j - 1] = lst[group_end - j - 1], lst[i + j]
    return lst





def group_by_length(lst: List[str]) -> Dict[int, List[str]]:
    """
    Groups the strings by their length and returns a dictionary.
    """
    length_dict = {}
    
    for string in strings:
        length = len(string)
        if length not in length_dict:
            length_dict[length] = []
        length_dict[length].append(string)
    
    return dict(sorted(length_dict.items()))




def flatten_dict(nested_dict: Dict, sep: str = '.') -> Dict:
    """
    Flattens a nested dictionary into a single-level dictionary with dot notation for keys.
    
    :param nested_dict: The dictionary object to flatten
    :param sep: The separator to use between parent and child keys (defaults to '.')
    :return: A flattened dictionary
    """
    def _flatten(d, parent_key=''):
        items = []
        for k, v in d.items():
            new_key = f"{parent_key}{sep}{k}" if parent_key else k
            if isinstance(v, dict):
                items.extend(_flatten(v, new_key).items())
            elif isinstance(v, list):
                for i, item in enumerate(v):
                    list_key = f"{new_key}[{i}]"
                    if isinstance(item, dict):
                        items.extend(_flatten(item, list_key).items())
                    else:
                        items.append((list_key, item))
            else:
                items.append((new_key, v))
        return dict(items)

    return _flatten(nested_dict)





def unique_permutations(nums: List[int]) -> List[List[int]]:
    """
    Generate all unique permutations of a list that may contain duplicates.
    
    :param nums: List of integers (may contain duplicates)
    :return: List of unique permutations
    """
    def backtrack(start=0):
        if start == len(nums):
            result.append(nums[:])  # Add a copy of the current permutation
        for i in range(start, len(nums)):
            if i != start and nums[i] == nums[start]:
                continue  # Skip duplicates
            nums[start], nums[i] = nums[i], nums[start]  # Swap
            backtrack(start + 1)
            nums[start], nums[i] = nums[i], nums[start]  # Swap back
    
    nums.sort()  # Sorting helps to easily skip duplicates
    result = []
    backtrack()
    return result
    pass







def find_all_dates(text: str) -> List[str]:
    """
    This function takes a string as input and returns a list of valid dates
    in 'dd-mm-yyyy', 'mm/dd/yyyy', or 'yyyy.mm.dd' format found in the string.
    
    Parameters:
    text (str): A string containing the dates in various formats.

    Returns:
    List[str]: A list of valid dates in the formats specified.
    """
    patterns = [
        r'\b\d{2}-\d{2}-\d{4}\b',  # dd-mm-yyyy
        r'\b\d{2}/\d{2}/\d{4}\b',  # mm/dd/yyyy
        r'\b\d{4}\.\d{2}\.\d{2}\b'  # yyyy.mm.dd
    ]
    
    # Combine patterns into a single regex pattern
    combined_pattern = '|'.join(patterns)
    
    # Find all matches in the text
    dates = re.findall(combined_pattern, text)
    
    return dates
    pass





def polyline_to_dataframe(polyline_str: str) -> pd.DataFrame:
    """
    Converts a polyline string into a DataFrame with latitude, longitude, and distance between consecutive points.
    
    Args:
        polyline_str (str): The encoded polyline string.

    Returns:
        pd.DataFrame: A DataFrame containing latitude, longitude, and distance in meters.
    """def haversine(lat1, lon1, lat2, lon2):
        """
        Calculate the great-circle distance in meters between two points on the Earth
        specified in decimal degrees (latitude and longitude).
        """
        
        lat1, lon1, lat2, lon2 = map(math.radians, [lat1, lon1, lat2, lon2])
        
        # Haversine formula
        dlon = lon2 - lon1 
        dlat = lat2 - lat1 
        a = math.sin(dlat/2)**2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon/2)**2
        c = 2 * math.asin(math.sqrt(a)) 
        r = 6371000  
        return c * r

    
    coordinates = polyline.decode(polyline_str)

    
    latitudes = []
    longitudes = []
    distances = [0]  


    for i, (lat, lon) in enumerate(coordinates):
        latitudes.append(lat)
        longitudes.append(lon)

        
        if i > 0:
            distance = haversine(latitudes[i-1], longitudes[i-1], lat, lon)
            distances.append(distance)

    
    df = pd.DataFrame({
        'latitude': latitudes,
        'longitude': longitudes,
        'distance': distances
    })

    return df
    



    
    


def rotate_and_multiply_matrix(matrix: List[List[int]]) -> List[List[int]]:
    """
    Rotate the given matrix by 90 degrees clockwise, then multiply each element 
    by the sum of its original row and column index before rotation.
    
    Args:
    - matrix (List[List[int]]): 2D list representing the matrix to be transformed.
    
    Returns:
    - List[List[int]]: A new 2D list representing the transformed matrix.
    """
     n = len(matrix)
    rotated_matrix = [[0] * n for _ in range(n)]
    
    for i in range(n):
        for j in range(n):
            rotated_matrix[j][n - 1 - i] = matrix[i][j]

    print("Rotated Matrix:")
    for row in rotated_matrix:
        print(row)

    final_matrix = [[0] * n for _ in range(n)]
    
    for i in range(n):
        for j in range(n):
            row_sum = sum(rotated_matrix[i]) - rotated_matrix[i][j]
            col_sum = sum(rotated_matrix[k][j] for k in range(n)) -rotated_matrix[i][j]
            final_matrix[i][j] = row_sum + col_sum
    return final_matrix
    








def time_check(df) -> pd.Series:
    """
    Use shared dataset-2 to verify the completeness of the data by checking whether the timestamps for each unique (`id`, `id_2`) pair cover a full 24-hour and 7 days period

    Args:
        df (pandas.DataFrame)

    Returns:
        pd.Series: return a boolean series
    """

    df['start'] = pd.to_datetime(df['startDay'] + ' ' + df['startTime'])
    df['end'] = pd.to_datetime(df['endDay'] + ' ' + df['endTime'])

    grouped = df.groupby(['id', 'id_2'])

    results = {}

    for (id_val, id_2_val), group in grouped:
        full_day = (group['end'].max() - group['start'].min() >= pd.Timedelta(days=1))
        days_covered = group['start'].dt.day_name().nunique() == 7
        
        results[(id_val, id_2_val)] = not (full_day and days_covered)

    boolean_series = pd.Series(results)
    boolean_series.index = pd.MultiIndex.from_tuples(boolean_series.index, names=['id', 'id_2'])
    
    return boolean_series
    
