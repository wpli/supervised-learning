import datetime
import collections
def get_train_validation_landmark_dicts( landmark_pickup_dict, pickups, train_test_split ):
    for landmark, pickup_indices in landmark_pickup_dict.items():
        pickup_time_windows = []
        for p_idx in pickup_indices:
            p = pickups[p_idx]
            x = p[0].date()
            y = datetime.time( hour = p[0].hour / 2 * 2 )
            z = datetime.datetime.combine( date=x, time=y )
            pickup_time_windows.append( z )
        
        count_by_time_window_dict = collections.Counter( pickup_time_windows )
        count_by_time_window = count_by_time_window_dict.items()

        random.shuffle( count_by_time_window )
    
        split_idx = int( train_test_split * len( count_by_time_window ) )
        landmark_train_dict[landmark] = count_by_time_window[:split_idx]
        landmark_test_dict[landmark] = count_by_time_window[split_idx:]
    
    return landmark_train_dict, landmark_test_dict
