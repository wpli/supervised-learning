import datetime
import collections
import random
import utils_general
import os
import cPickle

def get_landmark_pickup_dict( landmarks, pickups, sq_dist_threshold, filename ):

    if os.path.exists( filename ):
        with open( filename ) as f:
            landmark_pickup_dict = cPickle.load( f ) 
    else:
        landmark_pickup_dict = {}
        for landmark in list( landmarks ):
            landmark_pickup_dict[landmark] = [ idx for idx, i in enumerate( pickups ) \
                                                   if utils_general.calculate_square_dist( landmark[0], landmark[1], i[1], i[2] ) \
                                                   <= sq_dist_threshold ]

        with open( filename, 'w' ) as f:
            cPickle.dump( landmark_pickup_dict, f )

    return landmark_pickup_dict
                                              
def get_background_taxi_activity( landmark_pickup_dict ):
    background_pickup_indices = [ set( i ) for i in landmark_pickup_dict.values() ]
    background_pickup_indices = set.union( *background_pickup_indices )
    return background_pickup_indices 


def get_train_validation_landmark_dicts( landmark_pickup_dict, pickups, train_test_split, even_intervals_only=True ):
    landmark_train_dict = {}
    landmark_test_dict = {}
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
