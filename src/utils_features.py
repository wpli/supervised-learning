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
    if dt.hour >= 4 and dt.hour< 10:
        idx_to_activate = 0
    elif dt.hour >= 10 and dt.hour < 16:
        idx_to_activate = 1
    elif dt.hour >= 16 and dt.hour < 22:
        idx_to_activate = 2
    else:
        idx_to_activate = 3

    time_quadrant_binary_feature[idx_to_activate] = 1

    return time_quadrant_binary_feature

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
    
