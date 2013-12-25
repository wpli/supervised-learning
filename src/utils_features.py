import datetime

# master feature bundler

def featurize( dt, feature_function_list, **kwargs ):

    #features = day_time_binary_features + day_of_week_binary_features +time_quadrant_binary_feature + \
    #    corrected_hour_feature + day_binary  + [ dt.isocalendar()[1] ] #+ time_quadrant_binary_feature 

    features = []
    

    for feature_function in feature_function_list:
        features += feature_function( dt, **kwargs )

    return features

### INDIVIDUAL FEATURE FUNCTIONS
# Each feature should return a Python list (not a numpy array) of numerical features (they can be ints or floats)
# If the feature function only returns one feature, put it in a list, i.e. "return [ feature_val ]"
def feature_day_of_week_binary( dt, **kwargs ):
    day_of_week_binary_features = []
    for day in range(7):
        if dt.weekday() == day:
            day_of_week_binary_features.append( 1 )
        else:
            day_of_week_binary_features.append( 0 )  

    return day_of_week_binary_features

def feature_time_quadrant_binary( dt, **kwargs ):
    #time quadrants: 4-10, 10-4, 4-10, 10-4
    time_quadrant_binary_feature = [0,0,0,0]
    #idx_to_activate = ( dt.hour - 4 ) / ( 6 )
    offsets = [ 2,4 ]
    #offsets = [0,1,2,3,4,5]
    for offset in offsets:
        if dt.hour >= offset and dt.hour< offset+6:
            idx_to_activate = 0
        elif dt.hour >= offset+6 and dt.hour < offset+12:
            idx_to_activate = 1
        elif dt.hour >= offset+12 and dt.hour < offset+18:
            idx_to_activate = 2
        else:
            idx_to_activate = 3

        time_quadrant_binary_feature[idx_to_activate] = 1

    return time_quadrant_binary_feature

def feature_even_hour_binary( dt, **kwargs ):
    binary_features = [ 0 ] * 12

    
    if dt.hour % 2 == 0:
        binary_features[ dt.hour / 2 ] = 1
    else:
        binary_features[ dt.hour / 2 ] = 0.5
        next_idx = ( dt.hour / 2 + 1 ) % 12
        binary_features[ next_idx ] = 0.5
    return binary_features

def feature_hour_binary( dt, **kwargs ):
    binary_features = [ 0 ] * 24
    binary_features[ dt.hour ] = 1
    return binary_features


def feature_day_time_product_binary( dt, **kwargs ):
    day_time_binary_features = []
    day_of_week_binary_features = feature_day_of_week_binary( dt )
    time_quadrant_binary_feature = feature_time_quadrant_binary( dt )

    for i in day_of_week_binary_features:
        for j in time_quadrant_binary_feature:
            day_time_binary_features.append( i*j )
    return day_time_binary_features

def feature_corrected_hour( dt, **kwargs ):    
    # make 2am the cutoff time
    corrected_hour = dt.hour - 3
    if corrected_hour < 0:
        corrected_hour += 24
    corrected_hour_feature = [ corrected_hour ]
    return corrected_hour_feature

def feature_other_pickups( dt, **kwargs ):
    if "background_pickups_counter" in kwargs:
        background_pickups_counter = kwargs["background_pickups_counter"]

    total_pickups = 0
    total_pickups += background_pickups_counter[ ( dt.date(), dt.hour ) ] 
    total_pickups += background_pickups_counter[ ( dt.date(), dt.hour+1 ) ]

    return [ total_pickups ]

def feature_lon_lat_rounded( dt, **kwargs ):
    if "lon_lat" in kwargs:
        lon_lat = kwargs["lon_lat"]

    lon_rounded = lon_lat[0] + 72
    lon_rounded = lon_rounded * 100.0
    lat_rounded = lon_lat[1] - 42
    lat_rounded = lat_rounded * 100.0

    return [ lon_rounded, lat_rounded ]


        

def feature_nearby_pickups( dt, **kwargs ):
    features = []
    for nearby_pickups_counter in ( kwargs['nearby_pickups_counter'], kwargs['km_nearby_pickups_counter'] ):
        total_nearby_pickups = 0
        total_nearby_pickups += nearby_pickups_counter.get( ( dt.date(), dt.hour ), 0 )
        total_nearby_pickups += nearby_pickups_counter.get( ( dt.date(), dt.hour+1 ), 0 )
        next_day_pickups = nearby_pickups_counter.get( ( dt.date()+datetime.timedelta(days=1), dt.hour ), 0 ) + nearby_pickups_counter.get( ( dt.date()+datetime.timedelta(days=1), dt.hour+1 ), 0 )
        previous_day_pickups = nearby_pickups_counter.get( ( dt.date()-datetime.timedelta(days=1), dt.hour ), 0 ) + nearby_pickups_counter.get( ( dt.date()-datetime.timedelta(days=1), dt.hour+1 ), 0 )
        next_week_pickups = nearby_pickups_counter.get( ( dt.date()+datetime.timedelta(days=7), dt.hour ), 0 ) + nearby_pickups_counter.get( ( dt.date()+datetime.timedelta(days=7), dt.hour+1 ), 0 )
        previous_week_pickups = nearby_pickups_counter.get( ( dt.date()-datetime.timedelta(days=7), dt.hour ), 0 ) + nearby_pickups_counter.get( ( dt.date()-datetime.timedelta(days=7), dt.hour+1 ), 0 )
        features += [ total_nearby_pickups ] #, previous_day_pickups, next_week_pickups, previous_week_pickups, next_week_pickups ]

    return features


def feature_day( dt, **kwargs ):
    if "possible_days" in kwargs:
        possible_days = kwargs["possible_days"]
    else:
        possible_days = []

    if "test_days_dict" in kwargs:
        test_days_dict = kwargs["test_days_dict"]
    else:
        test_days_dict = {}

    if possible_days == None:
        possible_days = range(200)
        day_binary = [ 0 ] * len( possible_days )
        day_of_year = dt.timetuple().tm_yday
        day_binary[day_of_year-150] = 1    
    else:
        day_binary = [ 0 ] * len( possible_days )
        day_of_year = dt.date()
        if day_of_year in test_days_dict:
            day_binary[test_days_dict[day_of_year]] = 1

    return day_binary
        
def feature_week_number( dt, **kwargs ):
    return [ dt.isocalendar()[1] ]

def feature_landmark_taxi_popularity( dt, **kwargs ):
    if "all_pickups" in kwargs:
        all_pickups = kwargs["all_pickups"]

    else:
        all_pickups = 0

    return [ all_pickups ]



# unit test
def unit_test():
    feature_function_list = [ feature_day_of_week_binary , feature_time_quadrant_binary, feature_day_time_product_binary, \
                                  feature_corrected_hour, feature_day, feature_week_number ]
    feats = featurize( datetime.datetime( 2012, 3, 30, 16, 40 ), feature_function_list, possible_days=[ datetime.date( 2012,5,20) ] )
    

    print feats 
    print len(feats)
    feats = featurize( datetime.datetime( 2012, 5, 25, 12, 10 ), feature_function_list )

    print len( feats )

if __name__ == '__main__':
    unit_test()
    
