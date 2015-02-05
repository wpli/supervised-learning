# -*- coding: utf-8 -*-
# <nbformat>3.0</nbformat>

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

# <codecell>

with open( '../data/dropoffs.csv' ) as f:
    x = f.read().split('\n')
    if x[-1] == '':
        del x[-1]
        
del x[0]
        

    
    

# <codecell>

print x[1]

# <codecell>

import datetime
from dateutil import parser

dropoffs = []
for i in x:
    entry = i.split(',')
    try:
        dt = parser.parse( entry[1] )
        lon = float( entry[3] )
        lat = float( entry[4] )
        dropoffs.append( ( dt, lon, lat ) )
        
    except ValueError:
        print entry[1]
        dt = None

    
    #entry = i.split(',')
    #date_string = entry[1]
    #[ d, t ] = date_string.split()
    #print d.split('/')
    
    #datetime = datetime.datetime.strptime( entry[1], "%m/%d/%Y %H:%m" ) 
    
    

# <codecell>


# <codecell>

def feature_dropoffs( dt, **kwargs ):
    

