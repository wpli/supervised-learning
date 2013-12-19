
import utils_raw
import utils_general
import utils_preprocess
import utils_features

import time
import ipdb
import sys
import collections
import numpy
from sklearn import linear_model



DIST_THRESHOLD = 0.00224946357
SQ_DIST_THRESHOLD = DIST_THRESHOLD**2

TRAIN_TEST_SPLIT = 0.8


feature_function_list = [ utils_features.feature_day_of_week_binary , \
                              utils_features.feature_time_quadrant_binary, \
                              utils_features.feature_day_time_product_binary, \
                              utils_features.feature_corrected_hour, \
                              utils_features.feature_day, \
                              utils_features.feature_week_number ]




def main():
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

    # ---
    # sample the training set and validation set (possibly to be more reflective of the test set)
    sampled_landmark_train_dict = {}
    sampled_landmark_val_dict = {}
    for landmark, entries in landmark_train_dict.items():
        test_start_dts = [ times_locations[idx][1].hour / 2 * 2 for idx in landmark_time_location_dict[landmark] ]
        start_time_counts = collections.Counter( test_start_dts )
        train_data = [ i for i in landmark_train_dict[landmark] if i[0].hour in start_time_counts ]
        sampled_landmark_train_dict[landmark] = [ i for i in landmark_train_dict[landmark] if i[0].hour in start_time_counts ]
        sampled_landmark_val_dict[landmark] = [ i for i in landmark_val_dict[landmark] if i[0].hour in start_time_counts ]

    all_train_prediction_rms_errors = []
    all_val_predictions_rms_errors = []
    final_predictions = []

    # compute the features on the training set, validation set, and test set
    # train model on training set and evaluate performance on training and validation set
    # train model on training+validation set and compute predictions on test set                                
    for landmark in list( landmarks ):
        train_data = sampled_landmark_train_dict[landmark]
        val_data = sampled_landmark_val_dict[landmark]
        test_data = [ times_locations[idx][1] for idx in landmark_time_location_dict[landmark] ]
        test_data_indices = [ times_locations[idx][0] for idx in landmark_time_location_dict[landmark] ]
        train_features = numpy.array( [ utils_features.featurize( i[0], feature_function_list, possible_days=possible_days, test_days_dict=test_days_dict ) for i in train_data ] )
        train_y = numpy.array( [i[1] for i in train_data ] )
        val_features = numpy.array( [ utils_features.featurize( i[0], feature_function_list, possible_days=possible_days, test_days_dict=test_days_dict ) for i in val_data ] )
        val_y = numpy.array( [i[1] for i in val_data ] )
        
        # note that test_data is in a slightly different format than train_data nad val_data because it does not have predictions
        test_features = numpy.array( [ utils_features.featurize( i, feature_function_list, possible_days=possible_days, test_days_dict=test_days_dict ) for i in test_data ] )

        if len( train_features ) == 0:
            test_predictions = [ 0 for i in test_features ]
        else:
            clf = linear_model.Ridge(alpha = .5)
            clf.fit( train_features, train_y )
            # compute performance on training data
            predictions = clf.predict( train_features )
            all_train_prediction_rms_errors += [ numpy.sqrt((predictions[idx] - train_y[idx])**2) for idx in range(len(predictions)) ]

            # compute performance on validation data
            predictions = clf.predict( val_features )
            all_val_predictions_rms_errors += [ numpy.sqrt((predictions[idx] - val_y[idx])**2) for idx in range(len(predictions)) ]

            # train on all of the data
            full_train_features = numpy.vstack( ( train_features, val_features ) )
            full_train_y = numpy.hstack( ( train_y, val_y ) )
           
            clf = linear_model.Ridge(alpha = .5)
            clf.fit( full_train_features, full_train_y )
            test_predictions = clf.predict( test_features )

        final_predictions += zip( test_data_indices, test_predictions )
        #print clf.coef_[-1], 
    
    sys.stderr.write( "Average Train RMS Error: " ) 
    sys.stderr.write( "%s\n" % numpy.average( all_train_prediction_rms_errors ) )

    sys.stderr.write( "Average Validation RMS Error: " ) 
    sys.stderr.write( "%s\n" % numpy.average( all_val_predictions_rms_errors ) )
    pickups_guess = dict( final_predictions )

    # write to file
    extension = str( int( time.time() ) )
    final_predictions = pickups_guess.items()
    final_predictions.sort( key=lambda x:int(x[0]) )
    output_filename = '../submissions/submission.%s.txt' % extension
    with open( output_filename, 'w' ) as f:
        for i in final_predictions:
            f.write( "%s %s\n" %( i[0], str(max( int(round(i[1],0) ), 0 ) ) ) )
    
    sys.stderr.write( "Wrote %s.\n" % output_filename )

    ipdb.set_trace()


if __name__ == '__main__':
    main()


