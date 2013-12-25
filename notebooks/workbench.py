# -*- coding: utf-8 -*-
# <nbformat>3.0</nbformat>

# <markdowncell>

# Run this first cell in order to pre-load the data.

# <codecell>

sys.path.append( '../src' )
import utils_raw
reload( utils_raw )
import utils_general
reload( utils_general )
import utils_preprocess
reload( utils_preprocess )
import utils_features
reload( utils_features )

import time
import ipdb
import sys
import collections
import numpy
from sklearn import linear_model

# <codecell>

sys.path.append( '../src' )
import utils_raw
reload( utils_raw )
import utils_general
reload( utils_general )
import utils_preprocess
reload( utils_preprocess )
import utils_features
reload( utils_features )

import time
import ipdb
import sys
import collections
import numpy
from sklearn import linear_model
DIST_THRESHOLD = 0.00224946357
SQ_DIST_THRESHOLD = DIST_THRESHOLD**2

TRAIN_TEST_SPLIT = 0.8

if 1:
    # ---
    # get raw data
    sys.stderr.write( "Getting raw data..." )

    # raw data: pickups
    # pickups schema: [ datetime, lon, lat ]


    pickups = utils_raw.get_pickups( '../data/pickups_train.csv' )
    
    # raw data: times/locations that need to be predicted
    # times_locations schema: [ prediction_index, start_datetime, end_datetime, lon, lat ] 
    times_locations = utils_raw.get_times_locations( '../data/test1.txt' )

    # set of lon/lats and times that we want pickups for
    landmarks = set( [ (i[3], i[4]) for i in times_locations ] )
    times = set( i[1].hour for i in times_locations )
    sys.stderr.write( "done.\n" )


    # ---
    # get pre-processed data
    sys.stderr.write( "Pre-processing data..." )
    sys.stderr.write( "counting pickups by landmarks..." )
    landmark_pickup_dict = utils_preprocess.get_landmark_pickup_dict( landmarks, pickups, SQ_DIST_THRESHOLD )

    landmark_time_location_dict = collections.defaultdict( list )
    sys.stderr.write( "counting times by landmarks..." )
    for idx, i in enumerate( times_locations ):
        landmark_time_location_dict[(i[3],i[4])].append( idx )
        
    sys.stderr.write( "done.\n" )

    possible_days = list( set( [ i[1].date() for i in times_locations ] ) )
    test_days_dict = dict( zip( possible_days, range(len(possible_days) ) ) )

    # ---
    # create "raw" training set and validation set
    #landmark_{train|val}_dict[(lon,lat)] = [ ( window_start_datetime0, count0 ), ( window_start_datetime1, count1 ) ... ]
    ( landmark_train_dict, landmark_val_dict ) = utils_preprocess.get_train_validation_landmark_dicts( landmark_pickup_dict, pickups, train_test_split = TRAIN_TEST_SPLIT )

    

# <codecell>

landmark_pickup_indices = utils_preprocess.get_background_taxi_activity( landmark_pickup_dict )
all_excluded_pickups = []
for idx, i in enumerate( pickups ):
    if idx not in landmark_pickup_indices:
        all_excluded_pickups.append( ( i[0].date(), i[0].hour ) )

background_pickups_counter = collections.Counter( all_excluded_pickups )

# <codecell>

background_pickups_counter.iteritems().next()

# <codecell>

pickups[0]

# <codecell>

multiplier_landmark_dict = {}
for multiplier in ( 2, 4 ):
    multiplier_landmark_dict[multiplier] = {}
    dist = DIST_THRESHOLD * 2
    sq_dist = dist**2
    #landmark_other_trips_dict = {}
    landmark_other_trips_counter = {}
    for landmark in list( landmarks ):
        other_trips = []
        for idx, i in enumerate( pickups ):
            if idx not in landmark_pickup_indices:
                if utils_general.calculate_square_dist( landmark[0], landmark[1], pickups[idx][1], pickups[idx][2] ) < sq_dist:
                    other_trips.append( ( i[0].date(), i[0].hour ) )
                

        #landmark_other_trips_dict[landmark] = other_trips
        landmark_other_trips_counter[landmark] = collections.Counter( other_trips )
        
    multiplier_landmark_dict[multiplier] = landmark_other_trips_counter

# <codecell>

x = list(landmarks)[1]
len( landmark_other_trips_counter[x] )

# <codecell>

print len( pickups )

# <codecell>

query = [ i for i in pickups if i[0] < datetime.datetime( 2012, 5, 11, 22, 0 ) and i[0] > datetime.datetime(2012,5,11,20)]

# <codecell>

query = [ i for i in pickups if i[0] < datetime.datetime( 2012, 5, 12, 1, 0 ) and i[0] > datetime.datetime(2012,5,11,23)]
X = [ i[1] for i in query ]
Y = [ i[2] for i in query ]
p = plt.Figure()
figsize(10,10)
scatter(X,Y,s=2,alpha=.2)
landmarks_lon = [ i[0] for i in list( landmarks ) ]
landmarks_lat = [ i[1] for i in list( landmarks ) ]
scatter( landmarks_lon, landmarks_lat )
xlim( -71.15, -71.01 )
ylim( 42.33, 42.37 )

# <codecell>

for key, val in landmark_pickup_dict.items():
    print key, len(val)

# <codecell>

if 1:
    # ---
    # sample the training set and validation set (possibly to be more reflective of the test set)
    sampled_landmark_train_dict = {}
    sampled_landmark_val_dict = {}
    
    for landmark, entries in landmark_train_dict.items():
        test_start_hours = [ ( times_locations[idx][1].weekday(), times_locations[idx][1].hour / 2 * 2 ) \
                            for idx in landmark_time_location_dict[landmark] ]
        #start_time_counts = collections.Counter( test_start_dts )
        #start_time_counts = start_time_counts.keys()
        #start_time_counts = start_time_counts + [ i+2 for i in start_time_counts ]
        temp_list = list( set( test_start_hours ) )
        start_times = temp_list + [ ( i[0], i[1]+2) for i in temp_list ]
        start_times_set = set( start_times )
        #train_data = [ i for i in landmark_train_dict[landmark] if i[0].hour in start_time_counts ]
        sampled_landmark_train_dict[landmark] = [ i for i in landmark_train_dict[landmark] if ( i[0].weekday(), i[0].hour ) \
                                                in start_times_set ]
        sampled_landmark_val_dict[landmark] = [ i for i in landmark_val_dict[landmark] if ( i[0].weekday(), i[0].hour ) \
                                                in start_times_set ]
        
        
        
        start_time_hours = set( [ i[1] for i in list(start_times_set) ] )
        
        
        sampled_landmark_train_dict[landmark] = [ i for i in landmark_train_dict[landmark] if i[0].hour \
                                                in start_time_hours ]
        sampled_landmark_val_dict[landmark] = [ i for i in landmark_val_dict[landmark] if i[0].hour \
                                                in start_time_hours ]


# <codecell>

for landmark, item in sampled_landmark_train_dict.items():
    print landmark, len( item )
print sum( [ len(i) for i in sampled_landmark_train_dict.values() ] )

# <markdowncell>

# Run this cell to add/play around with different functions and write submission files.

# <codecell>

from sklearn import ensemble

# <codecell>

runs_rms_vals = []

# <codecell>

def get_nearest_landmarks( target_landmark, landmarks, radius_km ):
    nearby_landmarks = []
    for landm in landmarks:
        if utils_general.calculate_square_dist( target_landmark[0], target_landmark[1], landm[0], landm[1] ) < radius_km**2:
            nearby_landmarks.append( landm )
            
    return nearby_landmarks

# <codecell>

import utils_features
reload( utils_features )
feature_function_list = [ \
                              utils_features.feature_day_of_week_binary , \
                              utils_features.feature_time_quadrant_binary, \
                              utils_features.feature_day_time_product_binary, \
                              utils_features.feature_corrected_hour, \
                              utils_features.feature_day, \
                              utils_features.feature_week_number, \
                              utils_features.feature_landmark_taxi_popularity, \
                              utils_features.feature_even_hour_binary, \
                              utils_features.feature_other_pickups, \
                              utils_features.feature_nearby_pickups, \
                              utils_features.feature_lon_lat_rounded ]
    

    
#regression_model = linear_model.Ridge()
regression_model = ensemble.RandomForestRegressor( n_estimators=30  )
    
if 1:
    
    all_train_prediction_rms_errors = []
    all_val_predictions_rms_errors = []
    final_predictions = []

    val_prediction_actual_tuples = []
    
    # compute the features on the training set, validation set, and test set
    # train model on training set and evaluate performance on training and validation set
    # train model on training+validation set and compute predictions on test set                                
    
    train_val_performance = []
    for landm in list( landmarks ):
        
        train_data = []
        train_features = []
        train_y = []
        nearest_landmarks = get_nearest_landmarks( landm, list( landmarks ), radius_km = DIST_THRESHOLD * 2 )
        print len( nearest_landmarks )
        for landmark in nearest_landmarks:
            train_data = [ i for i in sampled_landmark_train_dict[landmark] ]
            train_features += [ utils_features.featurize( \
                                                                 i[0], feature_function_list, \
                                                                 lon_lat = landmark, \
                                                                 possible_days=possible_days, \
                                                                 test_days_dict=test_days_dict, \
                                                                 all_pickups=total_pickups, \
                                                                 background_pickups_counter=background_pickups_counter, \
                                                                 nearby_pickups_counter=multiplier_landmark_dict[2][landmark], \
                                                                 km_nearby_pickups_counter=multiplier_landmark_dict[4][landmark], \
                                                                 ) for i in \
                                                                train_data ]
            train_y += [i[1] for i in train_data ]
            
            
            
            

        val_data = []

        landmark = landm
        val_data += sampled_landmark_val_dict[landmark]
        
        test_data = [ times_locations[idx][1] for idx in landmark_time_location_dict[landmark] ]
        test_data_indices = [ times_locations[idx][0] for idx in landmark_time_location_dict[landmark] ]
        total_pickups = len( landmark_pickup_dict[landmark] )
        #total_pickups = 0

        val_features = numpy.array( [ utils_features.featurize( \
                                                                 i[0], feature_function_list, \
                                                                 lon_lat = landmark, \
                                                                 possible_days=possible_days, \
                                                                 test_days_dict=test_days_dict, \
                                                                 all_pickups=total_pickups, \
                                                                 background_pickups_counter=background_pickups_counter, \
                                                                 nearby_pickups_counter=multiplier_landmark_dict[2][landmark], \
                                                                 km_nearby_pickups_counter=multiplier_landmark_dict[4][landmark], \
                                                                 ) for i in \
                                                                 val_data ] )
        
        # note that test_data is in a slightly different format than train_data and val_data because it does not have predictions
        test_features = numpy.array( [ utils_features.featurize( i, \
                                                                 feature_function_list, \
                                                                 lon_lat = landmark, \
                                                                 possible_days=possible_days, \
                                                                 test_days_dict=test_days_dict, \
                                                                 all_pickups=total_pickups, \
                                                                 background_pickups_counter=background_pickups_counter, \
                                                                 nearby_pickups_counter=multiplier_landmark_dict[2][landmark], \
                                                                 km_nearby_pickups_counter=multiplier_landmark_dict[4][landmark], \
                                                                 ) for i in \
                                                                test_data ] )
        
        val_y = numpy.array( [i[1] for i in val_data ] )                                                    
                                                                
        if len( train_features ) == 0:
            test_predictions = [ 0 for i in test_features ]
            
            
            
            
            
        else:
            sys.stderr.write( "Fitting regression model on training data..." )
            clf = regression_model
            clf.fit( train_features, train_y )
            # compute performance on training data
            predictions = clf.predict( train_features )
            
            
            
            
            predictions = [ max( i,0) for i in predictions ]
            landmark_train_prediction_rms_errors = [ numpy.sqrt((predictions[idx] - train_y[idx])**2) for idx in range(len(predictions)) ]
            all_train_prediction_rms_errors += landmark_train_prediction_rms_errors
            
            # compute performance on validation data
            predictions = clf.predict( val_features )
            predictions = [ max( round(i,0), 0 ) for i in predictions ]
            landmark_val_prediction_rms_errors = \
                [ numpy.sqrt((predictions[idx] - val_y[idx])**2) for idx in range(len(predictions)) ]
                
            val_prediction_actual_tuples += [ ( predictions[idx], val_y[idx] ) for idx in range(len(predictions))]
                
            all_val_predictions_rms_errors += landmark_val_prediction_rms_errors

            sys.stderr.write( "\n" )
            train_val_performance.append( ( landmark, numpy.average( landmark_train_prediction_rms_errors ), \
                numpy.average( landmark_val_prediction_rms_errors ) ) )
            sys.stderr.write( "%s, %s, %s, %s\n" % ( \
                landmark, numpy.average( landmark_train_prediction_rms_errors ), \
                numpy.average( landmark_val_prediction_rms_errors ), \
                len( test_features ) ) )
            
            test_hours = [ i.hour for i in test_data ]
            train_hours = [ i[0].hour for i in train_data ]
            #plt.subplot(2, 1, 1)
            #plt.hist( test_hours )

            #plt.subplot(2, 1, 2)
            #plt.hist( train_hours )
            #plt.show()
            
            
            #plt.hist( train_hours )
            #print test_data
            
            # train on all of the data
            
            sys.stderr.write( "done.\n" )
                             
                    
            
            full_train_features = numpy.vstack( ( train_features, val_features ) )
            full_train_y = numpy.hstack( ( train_y, val_y ) )          
            clf = regression_model
            
            clf.fit( full_train_features, full_train_y )
            test_predictions = clf.predict( test_features )
            
            test_predictions = [ max( int( round(i,0) ), 0 ) for i in test_predictions ]
            
        final_predictions += zip( test_data_indices, test_predictions )
        #print clf.coef_[-1], 
    
    sys.stderr.write( "Average Train RMS Error: " ) 
    train_rms = numpy.average( all_train_prediction_rms_errors )
    sys.stderr.write( "%s\n" % train_rms )

    sys.stderr.write( "Average Validation RMS Error: " )
    val_rms = numpy.average( all_val_predictions_rms_errors )
    sys.stderr.write( "%s\n" % val_rms )
    pickups_guess = dict( final_predictions )

    # write to file
    extension = str( int( time.time() ) )
    final_predictions = pickups_guess.items()
    final_predictions.sort( key=lambda x:int(x[0]) )
    output_filename = '../submissions/submission.%s.txt' % extension
    with open( output_filename, 'w' ) as f:
        for i in final_predictions:
            f.write( "%s %s\n" %( i[0], str( i[1] ) ) )
    
    sys.stderr.write( "Wrote %s\n" % output_filename )
    runs_rms_vals.append( ( output_filename, train_rms, val_rms ) )


# <codecell>

train_val_performance.sort( key=lambda x:x[2], reverse=True )
scatter( *zip( *[ x[0] for x in train_val_performance if x[2] > 9.0 ] ), marker='x' )
scatter( *zip( *[ x[0] for x in train_val_performance if x[2] < 6.0 ] ), c='r' )


# <codecell>

nearby = []
for v, y in zip( val_features, val_y ):
    nearby.append( ( numpy.average( v[-10] ), y  ))
    
scatter( [ v[-2] for v in val_features ], val_y )

# <codecell>

hist( pickups_guess.values() )

# <codecell>

train_vals = []
for i in sampled_landmark_train_dict.itervalues():
    train_vals += [ j[1] for j in i ]
    
hist(train_vals)
    

# <codecell>

hist( pickups_guess.values() )

# <codecell>

def frange(x, y, jump):
  while x < y:
    yield x
    x += jump

# <codecell>

#factors = frange(0,5,0.1):
for factor in frange( 0,5,0.1):
    print factor, numpy.average( [ numpy.sqrt( (i[1]*factor-i[0])**2 ) for i in val_prediction_actual_tuples ] )

# <codecell>

val_prediction_actual_tuples.sort( key=lambda x:x[0])

# <codecell>

pickups = [ i[1] for i in sampled_landmark_train_dict[landmark] ]
hist(pickups)

# <codecell>

diffs = [ (idx,( i[0] - i[1])) for (idx,i) in enumerate( val_prediction_actual_tuples )]
diffs.sort( key=lambda x:x[1] )

for i in diffs[:5] + diffs[-5:]:
    print i[0], val_prediction_actual_tuples[i[0]]

# <codecell>

print runs_rms_vals

# <codecell>

lonlats = landmark_pickup_dict.keys()
lons = [ i[0] for i in lonlats ]
lats = [ i[1] for i in lonlats ]
scatter( lons, lats )

# <codecell>

pickup_lons = [i[1] for i in pickups ]
pickup_lats = [i[2] for i in pickups ]
scatter( pickup_lons[:1000], pickup_lats[:1000] )

