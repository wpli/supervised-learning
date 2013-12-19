import datetime
import cPickle
import os

def convert_date( date_string ):
    return datetime.datetime.strptime( date_string, "%Y-%m-%d %H:%M" )


def get_pickups( filename ):
    pickle_file = filename + ".pkl"
    if os.path.exists( pickle_file ):
        with open( pickle_file ) as f:
            pickups = cPickle.load( f )
    else:

        with open( filename ) as f:
            x = f.read().split('\n')
            if x[-1] == '':
                del x[-1]
        pickups = []
        for j in x:
            i = j.split(',')
            pickups.append( [ datetime.datetime.strptime( i[1], "%Y-%m-%d %H:%M:%S" ), float( i[3] ), float( i[4] ) ] )

        with open( pickle_file, 'w' ) as f:
            cPickle.dump( pickups, f )

    return pickups

def get_times_locations( filename ):
    pickle_file = filename + ".pkl"
    if os.path.exists( pickle_file ):
        with open( pickle_file ) as f:
            times_locations = cPickle.load( f )
    else:
        with open( filename ) as f:
            x = f.read().split('\n')
            if x[-1] == '':
                del x[-1]

        times_locations = []
        for j in x:
            i = j.split(',')
            times_locations.append( [ i[0], convert_date( i[1] ), convert_date( i[2] ), float( i[4] ), float( i[3] ) ] )


        with open( pickle_file, 'w' ) as f:
            cPickle.dump( times_locations, f )

    return times_locations
