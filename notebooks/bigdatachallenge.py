# -*- coding: utf-8 -*-
# <nbformat>3.0</nbformat>

# <codecell>

import time

# <codecell>

t0 = time.time()
with open( '../data/pickups_train.csv' ) as f:
    x = f.read().split('\n')
    if x[-1] == '':
        del x[-1]
t1 = time.time()  
print t1 - t0
pickups = []
for j in x:
    i = j.split(',')
    pickups.append( [ datetime.datetime.strptime( i[1], "%Y-%m-%d %H:%M:%S" ), float( i[3] ), float( i[4] ) ] )


print time.time() - t1

# <codecell>

import datetime
def convert_date( date_string ):
    return datetime.datetime.strptime( date_string, "%Y-%m-%d %H:%M" )

# <codecell>

import datetime
with open( '../data/test1.txt' ) as f:
    x = f.read().split('\n')
    if x[-1] == '':
        del x[-1]

times_locations = []
for j in x:
    i = j.split(',')
    times_locations.append( [ i[0], convert_date( i[1] ), convert_date( i[2] ), float( i[3] ), float( i[4] ) ] )

# <codecell>

landmarks = set( [ (i[3], i[4]) for i in times_locations ] )
times = set( i[1].hour for i in times_locations )

# <codecell>

landmark_pickup_dict = {}
for landmark in list( landmarks ):
    print landmark
    landmark_pickup_dict[landmark] = [ idx for idx, i in enumerate( pickups ) \
                                      if calculate_square_dist( landmark[0], landmark[1], i[2], i[1] ) <= SQ_DIST_THRESHOLD ] 
        

# <codecell>

import cPickle
with open( '../processed_data/landmark_pickup_dict.pkl', 'w' ) as f:
    cPickle.dump( landmark_pickup_dict, f )

# <codecell>

def f( thing1 ):
    print thing1
    
def print_everything(*args):
    for count, thing in enumerate(args):
        print '{0}. {1}'.format(count, thing)

    f( args[2] )
        
        
print_everything('apple', 'banana', 'cabbage')

# <codecell>

# create a training set and a test set of two hour blocks
import collections
TRAIN_TEST_SPLIT = 0.8

landmark_train_dict = {}
landmark_test_dict = {}

t0 = time.time()
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
    #print count_by_time_window
    split_idx = int( TRAIN_TEST_SPLIT * len( count_by_time_window ) )
    
    landmark_train_dict[landmark] = count_by_time_window[:split_idx]
    landmark_test_dict[landmark] = count_by_time_window[split_idx:]
    
print time.time() - t0

# common to both: day of week, start hour, month, indicator of equality, distance to the landmark


# <codecell>


# <codecell>

possible_days = list( set( [ i[1].date() for i in times_locations ] ) )
days_dict = dict( zip( possible_days, range(len(possible_days) ) ) )
def featurize( dt, num_features = None ):
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

    
    
    #possible_days = range(200)
    
    day_binary = [ 0 ] * len( possible_days )
    #day_of_year = dt.timetuple().tm_yday
    #day_binary[day_of_year-150] = 1
    day_of_year = dt.date()
    if day_of_year in days_dict:
        day_binary[days_dict[day_of_year]] = 1
    
    #features += day_of_week_binary_features
    features = day_time_binary_features + day_of_week_binary_features +time_quadrant_binary_feature + \
        corrected_hour_feature + day_binary  + [ dt.isocalendar()[1] ] #+ time_quadrant_binary_feature 
    #features = day_time_binary_features
    if num_features != None:
        features = features[:num_features]
    #features = [ 0 ]
    return features

# unit test
feats = featurize( datetime.datetime( 2012, 3, 30, 16, 40 ) )
print feats 
print len( feats )

# <codecell>





# <codecell>

from sklearn import linear_model
clf = linear_model.Ridge (alpha = .5)
clf.fit ([[0, 0], [0, 0], [1, 1]], [0, .1, 1]) 
print clf.coef_
print clf.intercept_

# <codecell>

import collections
landmark_prediction_idx_dict = collections.defaultdict( list )
for idx, i in enumerate( times_locations ):
    landmark_prediction_idx_dict[ (i[3], i[4] ) ].append( idx )

# <codecell>


# <codecell>

x = [1,2,3]
y = [2,3,4]
zip(x,y)

# <codecell>

landmarks_list = list( landmarks )
from sklearn import linear_model

final_predictions = []

#for feature_limit in range( 1, 61, 5 ):


feature_limit = None
if 1:
    all_train_prediction_rms_errors = []
    all_test_predictions_rms_errors = []
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
        train_features = numpy.array( [ featurize( i[0], feature_limit ) for i in train_data ] )
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
    
    

# <codecell>

t_ids = [ int(i) for i in pickups_guess.keys() ]
t_ids.sort()

write = True
output_file = '../submissions/026.txt' 
import os

if write: 
    pass
else:
    output_file = 'tmp.txt'

with open( output_file, 'w' ) as f:
    for t_id in t_ids:
        output_string = "%s %s" % ( t_id, max(  int( round( pickups_guess[str(t_id)], 0 )), 0 ) )
        print output_string
        if write:
            f.write( output_string + "\n" )
        else:
            pass

# <codecell>

print [ i for i in landmark if ( i[-2], i[-1] ) == (42.358086, -71.127762) ]

# <codecell>

print landmark_prediction_idx_dict[(42.358086, -71.127762)]

# <codecell>

print numpy.array( features )[:5]

# <codecell>

pickups_guess = {}
for entry in times_locations: 
    sys.stderr.write( entry[0] + " " )
    landmark_pickups = landmark_pickup_dict[ ( entry[3], entry[4] ) ]
    
    hours = [ pickups[i][0].hour for i in landmark_pickups ]
    hours_count = collections.Counter( hours )
    days = [ pickups[i][0].date() for i in landmark_pickups ]
    num_days = len( set( days ) )
    
    start_hour = entry[1].hour
    end_hour = entry[2].hour - 1
    total_pickups_in_interval = hours_count[start_hour] + hours_count[end_hour]
    if len( hours ) != 0:
    
        fraction_of_pickups = float( total_pickups_in_interval ) / len( hours ) / num_days
    else: 
        fraction_of_pickups = 1.0
    
    float_estimate = len(landmark_pickups) * fraction_of_pickups
    pickups_guess[entry[0]] = int( round( float_estimate, 0 ) )

# <codecell>

print times_locations[0]
print pickups[0]

# <codecell>

def calculate_square_dist( lon1, lat1, lon2, lat2 ):
    return ( lon1 - lon2 )**2 + ( lat1 - lat2 )**2 

#((pickup.lat - location.lat)^2 + (pickup.lon - location.lon)^2)^0.5 <  and
#     (pickup.time >= timespan.start and pickup.time <= timespan.end)
    

# <codecell>

DIST_THRESHOLD = 0.00224946357
SQ_DIST_THRESHOLD = DIST_THRESHOLD**2

# <codecell>

# get day
import collections
pickups_dict = {}

pickups_guess = {}
for entry in times_locations: 
    day = entry[1].date()
    if day in pickups_dict:
        pass
    else:
        pickups_dict[day] = [ i for i in pickups if i[0].date() == day ]
    
    
    hours = [ i[0].hour for i in pickups_dict[day] ]
    hours_count = collections.Counter( hours )
    
    
    close_pickups = [ i for i in pickups_dict[day] if calculate_square_dist( i[1], i[2], entry[4], entry[3] ) <= SQ_DIST_THRESHOLD ]

    start_hour = entry[1].hour
    end_hour = entry[2].hour - 1
    total_pickups_in_interval = hours_count[start_hour] + hours_count[end_hour]
    fraction_of_pickups = float( total_pickups_in_interval ) / len( hours )
    
    float_estimate = len(close_pickups) * fraction_of_pickups
    pickups_guess[entry[0]] = int( round( float_estimate, 0 ) )

# <codecell>

# get day of week
import sys
import collections
pickups_dict = {}

pickups_guess = {}
for entry in times_locations: 
    sys.stderr.write( "%s " % entry[0] )
    target_property = entry[1].date() + datetime.timedelta( days=7 )
    if target_property in pickups_dict:
        pass
    else:
        satisfying_pickups = [ i for i in pickups if i[0].date() == target_property ]
        if len( satisfying_pickups ) > 100000:
            pickups_dict[target_property] = random.sample( satisfying_pickups, 100000 )
        else:
            pickups_dict[target_property] = satisfying_pickups
    
    hours = [ i[0].hour for i in pickups_dict[target_property] ]
    hours_count = collections.Counter( hours )
    
    
    close_pickups = [ i for i in pickups_dict[target_property] if calculate_square_dist( i[1], i[2], entry[4], entry[3] ) <= SQ_DIST_THRESHOLD ]

    start_hour = entry[1].hour
    end_hour = entry[2].hour - 1
    total_pickups_in_interval = hours_count[start_hour] + hours_count[end_hour]
    fraction_of_pickups = float( total_pickups_in_interval ) / len( hours )
    
    float_estimate = len(close_pickups) * fraction_of_pickups
    pickups_guess[entry[0]] = int( round( float_estimate, 0 ) )

# <codecell>

val = ( 1/0.001918 - 1 )**2 / 678
print val

# <codecell>

t_ids = [ int(i) for i in pickups_guess.keys() ]
t_ids.sort()

write = True
output_file = '../submissions/012.txt' 
import os

if write: 
    pass
else:
    output_file = 'tmp.txt'

with open( output_file, 'w' ) as f:
    for t_id in t_ids:
        output_string = "%s %s" % ( t_id, max(  int( round( pickups_guess[str(t_id)]*2, 0 )), 0 ) )
        print output_string
        if write:
            f.write( output_string + "\n" )
        else:
            pass

# <codecell>

print sum( pickups_guess.values() )

# <codecell>

import random
downsampled_pickups = random.sample( pickups, 10000 )
lons = [ i[1] for i in downsampled_pickups ]
lats = [ i[2] for i in downsampled_pickups ]

# <codecell>

plt.Figure()
figsize( 8, 8 )
scatter( lons, lats, s=2, alpha=0.2 )
plt.show()

# <codecell>

plt.Figure()
figsize( 8, 8 )
scatter( lons, lats, s=2, alpha=0.2 )
plt.show()

# <codecell>

x = pickups_dict.iteritems().next()
times = [ i[0].hour for i in x[1] ]

