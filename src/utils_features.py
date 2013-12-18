import datetime

def featurize( dt, possible_days = None, test_days_dict = None, num_features = None ):
    features = [ dt.isoweekday() ] #, dt.isocalendar()[1] ]
    day_of_week_binary_features = []
    
    for day in range(7):
        if dt.weekday() == day:
            day_of_week_binary_features.append( 1 )
        else:
            day_of_week_binary_features.append( 0 )  
            
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
        
    day_time_binary_features = []
    for i in day_of_week_binary_features:
        for j in time_quadrant_binary_feature:
            day_time_binary_features.append( i*j )
            
    # make 2am the cutoff time
    corrected_hour = dt.hour - 3
    if corrected_hour < 0:
        corrected_hour += 24
    corrected_hour_feature = [ corrected_hour ]

    
    if possible_days == None:
        possible_days = range(200)
        day_binary = [ 0 ] * len( possible_days )
        day_of_year = dt.timetuple().tm_yday
        day_binary[day_of_year-150] = 1    
    else:
        day_binary = [ 0 ] * len( possible_days )
        day_of_year = dt.date()
        if day_of_year in test_days_dict:
            day_binary[days_dict[day_of_year]] = 1
        
    features = day_time_binary_features + day_of_week_binary_features +time_quadrant_binary_feature + \
        corrected_hour_feature + day_binary  + [ dt.isocalendar()[1] ] #+ time_quadrant_binary_feature 


    if num_features != None:
        features = features[:num_features]
    #features = [ 0 ]
    return features

# unit test
def unit_test():
    feats = featurize( datetime.datetime( 2012, 3, 30, 16, 40 ) )
    print feats 
    print len( feats )

if __name__ == '__main__':
    unit_test()

