
import utils_raw
import utils_general
import utils_preprocess

import ipdb
import sys
import collections


DIST_THRESHOLD = 0.00224946357
SQ_DIST_THRESHOLD = DIST_THRESHOLD**2

TRAIN_TEST_SPLIT = 0.8

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
    landmark_pickup_dict = {}
    for landmark in list( landmarks ):
        print landmark
        landmark_pickup_dict[landmark] = [ idx for idx, i in enumerate( pickups ) \
                                      if utils_general.calculate_square_dist( landmark[0], landmark[1], i[1], i[2] ) <= SQ_DIST_THRESHOLD ] 

    sys.stderr.write( "done.\n" )

    possible_days = list( set( [ i[1].date() for i in times_locations ] ) )
    days_dict = dict( zip( possible_days, range(len(possible_days) ) ) )

    # ---
    # create "raw" training set and validation set
    #landmark_{train|val}_dict[(lon,lat)] = [ ( window_start_datetime0, count0 ), ( window_start_datetime1, count1 ) ... ]
    ( landmark_train_dict, landmark_val_dict ) = utils_preprocess.get_train_validation_landmark_dicts( landmark_pickup_dict, pickups, train_test_split = TRAIN_TEST_SPLIT )
    
    # sample the training set and validation set (possibly to be more reflective of the test set)
    ipdb.set_trace()

    for landmark, entries in landmark_train_dict.items():
        test_start_dts = [ times_locations[idx][1].hour / 2 * 2 for idx in landmark_prediction_idx_dict[landmark] ]
        start_time_counts = collections.Counter( test_start_dts )
        total_start_time_counts = float( sum( start_time_counts.values() ) )
        num_training_points = len( landmark_train_dict[landmark] ) 
        current_training_points = 0
        train_data = [ i for i in landmark_train_dict[landmark] if i[0].hour in start_time_counts ]


        sampled_landmark_train_dict[landmark] = landmark_train_dict[landmark]
        sampled_landmark_val_dict = landmark_val_dict[landmark]

    # compute the features on the training set, validation set, and test set
    for landmark_idx, landmark in enumerate( landmarks_list ):
        #sys.stderr.write( "%s " % landmark_idx )
        #train_features = numpy.array( [ featurize( i[0], feature_limit ) for i in landmark_train_dict[landmark] ] )
        #train_y = numpy.array( [i[1] for i in landmark_train_dict[landmark] ] )
        
        # for balanced training set
        test_start_dts = [ times_locations[idx][1].hour / 2 * 2 for idx in landmark_prediction_idx_dict[landmark] ]
        start_time_counts = collections.Counter( test_start_dts )
        total_start_time_counts = float( sum( start_time_counts.values() ) )
        num_training_points = len( landmark_train_dict[landmark] ) 
        
        current_training_points = 0
        
        train_data = [ i for i in landmark_train_dict[landmark] if i[0].hour in start_time_counts ]
        train_features = numpy.array( [ featurize( i[0], possible_days, test_days_dict, feature_limit ) for i in train_data ] )
        train_y = numpy.array( [i[1] for i in train_data ] )
        
        #for i in range( num_training_points ):
            
        
        if len( train_features ) == 0:
            prediction_indices = [ times_locations[idx][0] for idx in landmark_prediction_idx_dict[landmark] ]
            test_predictions = [ 0 for  idx in landmark_prediction_idx_dict[landmark] ]
            #print prediction_indices
        else:
            clf = linear_model.Ridge(alpha = .5)
            #print landmark
            #print len( train_features )
            clf.fit( train_features, train_y )
            #print collections.Counter( train_y )
            
            val_data = [ i for i in landmark_test_dict[landmark] if i[0].hour in start_time_counts ]
            val_features = numpy.array( [ featurize( i[0], feature_limit ) for i in val_data ] )
            
            validation_features = numpy.array( [ featurize(i[0], feature_limit ) for i in val_data ] )
            validation_y = numpy.array( [i[1] for i in val_data ] )
            predictions = clf.predict( train_features )
            #print clf.coef_
            #print landmark 
            #print "TRAIN", 
            
            all_train_prediction_rms_errors += [ numpy.sqrt((predictions[idx] - train_y[idx])**2) for idx in range(len(predictions)) ]
            predictions = clf.predict( validation_features )
            #print "TEST",
            #print landmark, 
            all_test_predictions_rms_errors += [ numpy.sqrt((predictions[idx] - validation_y[idx])**2) for idx in range(len(predictions)) ]
            #print numpy.sum( [ numpy.sqrt((predictions[idx] - validation_y[idx] )**2) for idx in range(len(predictions)) ] ) / len( predictions )
            #print numpy.sum( [ numpy.sqrt((predictions[idx] - landmark_test_dict[landmark][idx][1])**2) for i in range(len(predictions)) ] ) / len( predictions )
            #print "\n"
            
            # train on all of the data
            full_train_features = numpy.vstack( ( train_features, validation_features ) )
            full_train_y = numpy.hstack( ( train_y, validation_y ) )
            
            clf = linear_model.Ridge(alpha = .5)
            clf.fit( full_train_features, full_train_y )
            
            prediction_indices = [ times_locations[idx][0] for idx in landmark_prediction_idx_dict[landmark] ]
            start_dts = [ times_locations[idx][1] for idx in landmark_prediction_idx_dict[landmark] ]
            test_features = numpy.array( [ featurize(i, feature_limit ) for i in start_dts ] )
            test_predictions = clf.predict( test_features )
        final_predictions += zip( prediction_indices, test_predictions )
        #print clf.coef_[-1], 
    print feature_limit
    print "Average RMS Train Error:",
    print numpy.average( all_train_prediction_rms_errors )
    print "Average RMS Test Error:", 
    print numpy.average( all_test_predictions_rms_errors )
    print "\n"    
    pickups_guess = dict( final_predictions )






    # build regression model on training data, evaluate on validation data

    # build regression model on training+validation data

    # write the predictions to a file

    ipdb.set_trace()

if __name__ == '__main__':
    main()


