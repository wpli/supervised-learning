import datetime

def convert_date( date_string ):
    return datetime.datetime.strptime( date_string, "%Y-%m-%d %H:%M" )


def get_pickups( filename ):
    with open( filename ) as f:
        x = f.read().split('\n')
        if x[-1] == '':
            del x[-1]
    pickups = []
    for j in x:
        i = j.split(',')
        pickups.append( [ datetime.datetime.strptime( i[1], "%Y-%m-%d %H:%M:%S" ), float( i[3] ), float( i[4] ) ] )

    return pickups

def get_times_locations( filename ):
    with open( filename ) as f:
        x = f.read().split('\n')
        if x[-1] == '':
            del x[-1]

    times_locations = []
    for j in x:
        i = j.split(',')
        times_locations.append( [ i[0], convert_date( i[1] ), convert_date( i[2] ), float( i[4] ), float( i[3] ) ] )

    return times_locations
