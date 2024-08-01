'''

    Name:           DFT Renderer 

    Description:    This is the script called from blenderBuilder.py in the DFT directory. 
                    This python script is run directly in Blender's python environment

    Date:           27 February 2024

    Author:         John Ferrier, NEU Physics

'''


## Color
# rgb(0, 178, 255)
# Hex: #00b2ff
# pylint: disable=fixme, import-error
import os
import re
import sys
import bpy
import json
import numpy as np
import pyopenvdb as vdb


#### Constants
# Desired colors based off of wavelength. i.e. 488nm = ( H/300, S/100, V/100 ) Setting for Blender
shell_colorsHSV = {
    '472':    np.array( [ 198./300., 1., 1. ] ),
    '488':    np.array( [ 182./300., 1., 1. ] ),
    '676':    np.array( [ 0., 1., 1. ] )
}
# Desired colors based off of wavelength. i.e. 488nm = ( R/255., G/255., B/255. ) Setting for Blender
shell_colorsRGB = {
    '472':    np.array( [ 0., 178./255., 255./255. ] ),
    '488':    np.array( [ 0., 247./255., 255./255. ] ),
    '676':    np.array( [ 255./255., 0., 0. ] )
}

# Directory names and file types. 
# Defined here to save the user from having to search for them later.
sorted_dir_name     = 'SortedVDB-Blender'
sorted_VDB_name     = 'VDB'
sorted_VDB_type     = '.vdb'
sorted_atms_name    = 'Atoms'
sorted_atms_type    = '.json'
base_name           = 'sorted'
selected_nm         = '472'
atom_mat_names      = {}
electron_mat        = 'electron_cloud'
radius_scale        = 0.5
max_edge_length     = 10.

#### HUGE constants from ASE and JMOL
atomic_values   = { "0": { "name": "", "radius": 0.2, "color": [1.0, 0.0, 0.0] }, "1": { "name": "Hydrogen", "radius": 0.31, "color": [1.0, 1.0, 1.0] }, 
                   "2": { "name": "Helium", "radius": 0.28, "color": [0.851, 1.0, 1.0] }, "3": { "name": "Lithium", "radius": 1.28, "color": [0.8, 0.502, 1.0] }, 
                   "4": {"name": "Beryllium", "radius": 0.96, "color": [0.761, 1.0, 0.0]}, "5": {"name": "Boron", "radius": 0.84, "color": [1.0, 0.71, 0.71]}, 
                   "6": {"name": "Carbon", "radius": 0.76, "color": [0.565, 0.565, 0.565]}, "7": {"name": "Nitrogen", "radius": 0.71, "color": [0.188, 0.314, 0.973]}, 
                   "8": {"name": "Oxygen", "radius": 0.66, "color": [1.0, 0.051, 0.051]}, "9": {"name": "Fluorine", "radius": 0.57, "color": [0.565, 0.878, 0.314]}, 
                   "10": {"name": "Neon", "radius": 0.58, "color": [0.702, 0.89, 0.961]}, "11": {"name": "Sodium", "radius": 1.66, "color": [0.671, 0.361, 0.949]}, 
                   "12": {"name": "Magnesium", "radius": 1.41, "color": [0.541, 1.0, 0.0]}, "13": {"name": "Aluminium", "radius": 1.21, "color": [0.749, 0.651, 0.651]}, 
                   "14": {"name": "Silicon", "radius": 1.11, "color": [0.941, 0.784, 0.627]}, "15": {"name": "Phosphorus", "radius": 1.07, "color": [1.0, 0.502, 0.0]}, 
                   "16": {"name": "Sulfur", "radius": 1.05, "color": [1.0, 1.0, 0.188]}, "17": {"name": "Chlorine", "radius": 1.02, "color": [0.122, 0.941, 0.122]}, 
                   "18": {"name": "Argon", "radius": 1.06, "color": [0.502, 0.82, 0.89]}, "19": {"name": "Potassium", "radius": 2.03, "color": [0.561, 0.251, 0.831]}, 
                   "20": {"name": "Calcium", "radius": 1.76, "color": [0.239, 1.0, 0.0]}, "21": {"name": "Scandium", "radius": 1.7, "color": [0.902, 0.902, 0.902]}, 
                   "22": {"name": "Titanium", "radius": 1.6, "color": [0.749, 0.761, 0.78]}, "23": {"name": "Vanadium", "radius": 1.53, "color": [0.651, 0.651, 0.671]}, 
                   "24": {"name": "Chromium", "radius": 1.39, "color": [0.541, 0.6, 0.78]}, "25": {"name": "Manganese", "radius": 1.39, "color": [0.612, 0.478, 0.78]}, 
                   "26": {"name": "Iron", "radius": 1.32, "color": [0.878, 0.4, 0.2]}, "27": {"name": "Cobalt", "radius": 1.26, "color": [0.941, 0.565, 0.627]}, 
                   "28": {"name": "Nickel", "radius": 1.24, "color": [0.314, 0.816, 0.314]}, "29": {"name": "Copper", "radius": 1.32, "color": [0.784, 0.502, 0.2]}, 
                   "30": {"name": "Zinc", "radius": 1.22, "color": [0.49, 0.502, 0.69]}, "31": {"name": "Gallium", "radius": 1.22, "color": [0.761, 0.561, 0.561]}, 
                   "32": {"name": "Germanium", "radius": 1.2, "color": [0.4, 0.561, 0.561]}, "33": {"name": "Arsenic", "radius": 1.19, "color": [0.741, 0.502, 0.89]}, 
                   "34": {"name": "Selenium", "radius": 1.2, "color": [1.0, 0.631, 0.0]}, "35": {"name": "Bromine", "radius": 1.2, "color": [0.651, 0.161, 0.161]}, 
                   "36": {"name": "Krypton", "radius": 1.16, "color": [0.361, 0.722, 0.82]}, "37": {"name": "Rubidium", "radius": 2.2, "color": [0.439, 0.18, 0.69]}, 
                   "38": {"name": "Strontium", "radius": 1.95, "color": [0.0, 1.0, 0.0]}, "39": {"name": "Yttrium", "radius": 1.9, "color": [0.58, 1.0, 1.0]}, 
                   "40": {"name": "Zirconium", "radius": 1.75, "color": [0.58, 0.878, 0.878]}, "41": {"name": "Niobium", "radius": 1.64, "color": [0.451, 0.761, 0.788]}, 
                   "42": {"name": "Molybdenum", "radius": 1.54, "color": [0.329, 0.71, 0.71]}, "43": {"name": "Technetium", "radius": 1.47, "color": [0.231, 0.62, 0.62]}, 
                   "44": {"name": "Ruthenium", "radius": 1.46, "color": [0.141, 0.561, 0.561]}, "45": {"name": "Rhodium", "radius": 1.42, "color": [0.039, 0.49, 0.549]}, 
                   "46": {"name": "Palladium", "radius": 1.39, "color": [0.0, 0.412, 0.522]}, "47": {"name": "Silver", "radius": 1.45, "color": [0.753, 0.753, 0.753]}, 
                   "48": {"name": "Cadmium", "radius": 1.44, "color": [1.0, 0.851, 0.561]}, "49": {"name": "Indium", "radius": 1.42, "color": [0.651, 0.459, 0.451]}, 
                   "50": {"name": "Tin", "radius": 1.39, "color": [0.4, 0.502, 0.502]}, "51": {"name": "Antimony", "radius": 1.39, "color": [0.62, 0.388, 0.71]}, 
                   "52": {"name": "Tellurium", "radius": 1.38, "color": [0.831, 0.478, 0.0]}, "53": {"name": "Iodine", "radius": 1.39, "color": [0.58, 0.0, 0.58]}, 
                   "54": {"name": "Xenon", "radius": 1.4, "color": [0.259, 0.62, 0.69]}, "55": {"name": "Caesium", "radius": 2.44, "color": [0.341, 0.09, 0.561]}, 
                   "56": {"name": "Barium", "radius": 2.15, "color": [0.0, 0.788, 0.0]}, "57": {"name": "Lanthanum", "radius": 2.07, "color": [0.439, 0.831, 1.0]}, 
                   "58": {"name": "Cerium", "radius": 2.04, "color": [1.0, 1.0, 0.78]}, "59": {"name": "Praseodymium", "radius": 2.03, "color": [0.851, 1.0, 0.78]}, 
                   "60": {"name": "Neodymium", "radius": 2.01, "color": [0.78, 1.0, 0.78]}, "61": {"name": "Promethium", "radius": 1.99, "color": [0.639, 1.0, 0.78]}, 
                   "62": {"name": "Samarium", "radius": 1.98, "color": [0.561, 1.0, 0.78]}, "63": {"name": "Europium", "radius": 1.98, "color": [0.38, 1.0, 0.78]}, 
                   "64": {"name": "Gadolinium", "radius": 1.96, "color": [0.271, 1.0, 0.78]}, "65": {"name": "Terbium", "radius": 1.94, "color": [0.188, 1.0, 0.78]}, 
                   "66": {"name": "Dysprosium", "radius": 1.92, "color": [0.122, 1.0, 0.78]}, "67": {"name": "Holmium", "radius": 1.92, "color": [0.0, 1.0, 0.612]}, 
                   "68": {"name": "Erbium", "radius": 1.89, "color": [0.0, 0.902, 0.459]}, "69": {"name": "Thulium", "radius": 1.9, "color": [0.0, 0.831, 0.322]}, 
                   "70": {"name": "Ytterbium", "radius": 1.87, "color": [0.0, 0.749, 0.22]}, "71": {"name": "Lutetium", "radius": 1.87, "color": [0.0, 0.671, 0.141]}, 
                   "72": {"name": "Hafnium", "radius": 1.75, "color": [0.302, 0.761, 1.0]}, "73": {"name": "Tantalum", "radius": 1.7, "color": [0.302, 0.651, 1.0]}, 
                   "74": {"name": "Tungsten", "radius": 1.62, "color": [0.129, 0.58, 0.839]}, "75": {"name": "Rhenium", "radius": 1.51, "color": [0.149, 0.49, 0.671]}, 
                   "76": {"name": "Osmium", "radius": 1.44, "color": [0.149, 0.4, 0.588]}, "77": {"name": "Iridium", "radius": 1.41, "color": [0.09, 0.329, 0.529]}, 
                   "78": {"name": "Platinum", "radius": 1.36, "color": [0.816, 0.816, 0.878]}, "79": {"name": "Gold", "radius": 1.36, "color": [1.0, 0.82, 0.137]}, 
                   "80": {"name": "Mercury", "radius": 1.32, "color": [0.722, 0.722, 0.816]}, "81": {"name": "Thallium", "radius": 1.45, "color": [0.651, 0.329, 0.302]}, 
                   "82": {"name": "Lead", "radius": 1.46, "color": [0.341, 0.349, 0.38]}, "83": {"name": "Bismuth", "radius": 1.48, "color": [0.62, 0.31, 0.71]}, 
                   "84": {"name": "Polonium", "radius": 1.4, "color": [0.671, 0.361, 0.0]}, "85": {"name": "Astatine", "radius": 1.5, "color": [0.459, 0.31, 0.271]}, 
                   "86": {"name": "Radon", "radius": 1.5, "color": [0.259, 0.51, 0.588]}, "87": {"name": "Francium", "radius": 2.6, "color": [0.259, 0.0, 0.4]}, 
                   "88": {"name": "Radium", "radius": 2.21, "color": [0.0, 0.49, 0.0]}, "89": {"name": "Actinium", "radius": 2.15, "color": [0.439, 0.671, 0.98]}, 
                   "90": {"name": "Thorium", "radius": 2.06, "color": [0.0, 0.729, 1.0]}, "91": {"name": "Protactinium", "radius": 2.0, "color": [0.0, 0.631, 1.0]}, 
                   "92": {"name": "Uranium", "radius": 1.96, "color": [0.0, 0.561, 1.0]}, "93": {"name": "Neptunium", "radius": 1.9, "color": [0.0, 0.502, 1.0]}, 
                   "94": {"name": "Plutonium", "radius": 1.87, "color": [0.0, 0.42, 1.0]}, "95": {"name": "Americium", "radius": 1.8, "color": [0.329, 0.361, 0.949]}, 
                   "96": {"name": "Curium", "radius": 1.69, "color": [0.471, 0.361, 0.89]}, "97": {"name": "Berkelium", "radius": 0.2, "color": [0.541, 0.31, 0.89]}, 
                   "98": {"name": "Californium", "radius": 0.2, "color": [0.631, 0.212, 0.831]}, "99": {"name": "Einsteinium", "radius": 0.2, "color": [0.702, 0.122, 0.831]}, 
                   "100": {"name": "Fermium", "radius": 0.2, "color": [0.702, 0.122, 0.729]}, "101": {"name": "Mendelevium", "radius": 0.2, "color": [0.702, 0.051, 0.651]}, 
                   "102": {"name": "Nobelium", "radius": 0.2, "color": [0.741, 0.051, 0.529]}, "103": {"name": "Lawrencium", "radius": 0.2, "color": [0.78, 0.0, 0.4]}, 
                   "104": {"name": "Rutherfordium", "radius": 0.2, "color": [0.8, 0.0, 0.349]}, "105": {"name": "Dubnium", "radius": 0.2, "color": [0.82, 0.0, 0.31]}, 
                   "106": {"name": "Seaborgium", "radius": 0.2, "color": [0.851, 0.0, 0.271]}, "107": {"name": "Bohrium", "radius": 0.2, "color": [0.878, 0.0, 0.22]}, 
                   "108": {"name": "Hassium", "radius": 0.2, "color": [0.902, 0.0, 0.18]}, "109": {"name": "Meitnerium", "radius": 0.2, "color": [0.922, 0.0, 0.149]}
}


#### Functions

# Returns whether the .blend file exists or not
def blendExists( blend_directory:str = None, override:bool = False ) -> bool:

    blend_file      = os.path.join( blend_directory, f"{base_name}.blend" )
    blend_exists    = False

    # Check if the file exists
    if os.path.isfile( blend_file ):
        # Override means to delete this file and return False
        if override:
            os.remove( blend_file )
        else:
            blend_exists = True

    return blend_exists

# Removes the VDB files
def removeVDBFiles( dir = "" ):
    # Open the directory
    files_in_dir    = os.listdir( dir )
    for f in files_in_dir:
        if f.endswith( sorted_VDB_type ):
            os.remove( os.path.join( dir, f ) )

# Returns whether the calculated values already exist or not, saving us time from recalculating everying
def alreadyCalculated( iteration_directory:str = None, override:bool = False ) -> bool:

    sorted_dir      = os.path.join( iteration_directory, sorted_dir_name )
    prev_sorted     = False
    blend_dir       = sorted_dir
    blend_file      = os.path.join( blend_dir, f"{base_name}.blend" )
    sorted_VDB_dir  = os.path.join( sorted_dir, sorted_VDB_name )
    sorted_atms_dir = os.path.join( sorted_dir, sorted_atms_name )
    atms_file       = os.path.join( sorted_atms_dir, f"{base_name}{sorted_atms_type}")

    # First, check if the override is set
    if override:
        # Override the previous stuff
        os.remove( blend_file )
        removeVDBFiles( dir = sorted_VDB_dir )
        os.remove( atms_file )
    
    # Check if these files have already been sorted or not and open them if they have
    if os.path.exists( sorted_dir ):

        # Check if the files actually exist in here
        if os.path.exists( sorted_atms_dir ) and os.path.exists( sorted_VDB_dir ):

            # Check for the files inside and that their size is not 0kb.
            if os.path.isfile( atms_file ):  

                # Check file size
                if os.stat( atms_file ).st_size > 0.:

                    # Check vdb file
                    vdb_file = os.path.join( sorted_VDB_dir, os.listdir( sorted_VDB_dir )[0] )
                    if os.path.isfile( vdb_file ):

                        # Check file size
                        if os.stat( vdb_file ).st_size > 0.:
                            prev_sorted = True
                        else:
                            # Delete the failed files
                            removeVDBFiles( dir = sorted_VDB_dir )
                            os.remove( atms_file )
                    else:
                        # Delete the failed files
                        removeVDBFiles( dir = sorted_VDB_dir )
                        os.remove( atms_file )

                else:
                    # Delete the files.
                    removeVDBFiles( dir = sorted_VDB_dir )
                    os.remove( atms_file )
    
    return prev_sorted

# Extracts the number from a string
def extract_number(file_name):
    match = re.findall(r'\d+', file_name)
    return int(''.join(match)) if match else 0

# Returns the iteration data in a sorted manner
def GetIterationSortedData( iteration_directory = None ):
    assert iteration_directory is not None, "No Iteratation directory provided"
    
    # Initialize the output variables
    atoms       = []
    densities   = []
    unitcell    = [ 1, 1, 1 ]

    # Check how many iterations exist in the directory
    folder_base = 'Iteration - '        # This is the name of folders in the directory, followed by a number. i.e. Iteration - 1
    dumb_base   = 'Interation - '       # This is because I was dumb early on and had an 'n' in the name...

    iter_dirs   = [ pdir for pdir in os.listdir( iteration_directory ) if pdir.startswith( folder_base ) ]
    if len( iter_dirs ) == 0:

        # Then it may have my dumb mistake
        iter_dirs   = [ pdir for pdir in os.listdir( iteration_directory ) if pdir.startswith( dumb_base ) ]
        
        # Recheck that they exist. If not, stop the script
        if len( iter_dirs ) == 0:
            print( f"No Iteration directorys found in {iteration_directory}" )
            sys.exit()
    
    # Sort the files by iteration number
    iter_dirs   = sorted( iter_dirs, key = extract_number )
    
    
    # Intialize some constants used in the loop
    number_of_atoms = 0
    x_grid_size     = 0
    y_grid_size     = 0
    z_grid_size     = 0

    x_step_size     = 0.
    y_step_size     = 0.
    z_step_size     = 0.

    x_real_size     = 0.
    y_real_size     = 0.
    z_real_size     = 0.

    side_scale      = 1.
    

    # Cycle through the iteration folders
    for j, it_dir in enumerate( iter_dirs ):
        
        # Build the full path
        it_dir  = os.path.join( iteration_directory, it_dir )

        # Get the band files
        bands = sorted( [ bnd for bnd in os.listdir( it_dir ) if bnd.endswith( '.cube' ) ],  key = extract_number )

        # List of all bands for this iteration
        band_dens       = []

        # Cycle through the bands
        for i, band_file in enumerate( bands ):
            
            # Build the full path
            band_file   = os.path.join( it_dir, band_file )
            
            # Load the .cube file into a numpy array
            with open( band_file, 'r') as f:
                # Skip the first two lines and Check for Sick case
                next( f )
                next( f )

                # I only care about reading these on the first band and first iteration, as they're the same throughout
                if i == 0:
                    number_of_atoms = int( f.readline().split()[0] )

                    # Read the grid dimensions from the second line
                    x_vals      = f.readline().split()
                    y_vals      = f.readline().split()
                    z_vals      = f.readline().split()

                    x_grid_size = int( x_vals[0] )
                    y_grid_size = int( y_vals[0] )
                    z_grid_size = int( z_vals[0] )

                    x_step_size = float( x_vals[1] )
                    y_step_size = float( x_vals[2] )
                    z_step_size = float( x_vals[3] )

                    x_real_size = float( x_grid_size )*x_step_size
                    y_real_size = float( y_grid_size )*y_step_size
                    z_real_size = float( z_grid_size )*z_step_size

                    # Get the largest value of x, y, or z and scale them to the max edge length set
                    # max_edge_length
                    dim_vars    = {'x': x_real_size, 'y': y_real_size, 'z': z_real_size}
                    # Find the variable with the largest value
                    lgst_side   = max( dim_vars, key = dim_vars.get )

                    side_scale  = 1.

                    if dim_vars[lgst_side] > max_edge_length:
                        side_scale  = max_edge_length/dim_vars[lgst_side]

                        # Scale the sides
                        x_real_size *= side_scale
                        y_real_size *= side_scale
                        z_real_size *= side_scale

                    unitcell = [ x_real_size, y_real_size, z_real_size ]
                    
                    atom_iter = []
                    for _ in range( number_of_atoms ):
                        #atom1: [ np.array([ x,y,z ]), 6 ] 
                        at      = f.readline().split()
                        point   = [ float( at[2] )*side_scale, float( at[3] )*side_scale, float( at[4] )*side_scale ]
                        atom_iter.append( [ np.array( point ), int( at[0] ) ] )

                    # Append these values to the atoms object
                    atoms.append( atom_iter )

                else:
                    # Skip the lines read above. They're the same
                    for _ in range( 4+number_of_atoms ):
                        next( f )
            
                #### Add the Densities TODO

                #Create a 1D numpy array containing Volumetrics data
                #data = np.fromfile( f, count = -1, sep = ' ', dtype = np.float64 )
                
                # Store the values in a 1D array (to be sorted later)
                data = np.array( [ float(l.strip()) for l in f ] )
                
                # Reshape the data array to the correct dimensions
                data = data.reshape( ( x_grid_size, y_grid_size, z_grid_size ) )

                band_dens.append( data )

        densities.append( band_dens )

    # Returns atoms, densities, unit_cell
    return atoms, densities, unitcell

# TODO:UnitCell - Returns the iterated atoms information (from JSON) and the iterated VDB files 
def GetFileInfo( iteration_directory = None ):
    assert iteration_directory is not None, "No iteration_directory received in GetFileInfo()"

    # Build the save directory
    sorted_dir      = os.path.join( iteration_directory, sorted_dir_name )
    sorted_VDB_dir  = os.path.join( sorted_dir, sorted_VDB_name )
    sorted_atms_dir = os.path.join( sorted_dir, sorted_atms_name )
    atms_file       = os.path.join( sorted_atms_dir, f"{base_name}{sorted_atms_type}")


    #### Import the atoms
    # Open the Atoms JSON file
    with open( atms_file ) as f:
        atoms = json.load(f)

    # Convert the atomic positions in the list to numpy arrays
    for atom_list in atoms:
        for atom in atom_list:
            atom[0] = np.array( atom[0] )


    #### Import the density file names
    # Check the directory
    vdb_listed  = os.listdir( sorted_VDB_dir )

    # Remove all files that don't end with the correct file type
    vdb_listed  = [ os.path.join( sorted_VDB_dir, f ) for f in vdb_listed if f.endswith( sorted_VDB_type ) ]



    #### Import the unit cell information
    # TODO. Temp return
    unt_cell_file   = os.path.join( sorted_dir, "unitcell.npy" )
    unitcell        = np.load( unt_cell_file )

    # Return atoms list and numerically sorted vdb files list
    return atoms, sorted( vdb_listed ), unitcell

# Saves the atom files in a json file
def SaveAtomFiles( atoms = None, iteration_directory = None ):
    assert atoms is not None, "No atoms received in SaveAtomFiles()"
    assert iteration_directory is not None, "No iteration_directory received in SaveAtomFiles()"

    # the atoms file should have the format of a dictionary such that
    # atom1: [ np.array( [ x,y,z ] ), 6 ]
    # The json file then has all atom objects for each iteration of the frames
    # atoms = [ [ atom1, atom2, ... atomN ], ..., [ atom1, atom2, ... atomN ] ]

    # Build the save directory
    sorted_dir      = os.path.join( iteration_directory, sorted_dir_name )
    sorted_atms_dir = os.path.join( sorted_dir, sorted_atms_name )
    atms_file       = os.path.join( sorted_atms_dir, f"{base_name}{sorted_atms_type}")

    # Cycle through each item provided
    for atom_list in atoms:

        # Positions are in numpy arrays. So, they must be converted to lists for saving in JSON
        for atom in atom_list:
            atom[0] = atom[0].tolist()

    # Check to ensure the directory exists
    if not os.path.exists( sorted_atms_dir ):
        os.mkdir( sorted_atms_dir )

    # Save the JSON file
    with open( atms_file, 'w', encoding = 'utf-8' ) as f:
        json.dump( atoms, f, ensure_ascii = False, indent = 4 )

# Saves the VDB files
def SaveVDBFiles( densities = None, iteration_directory = None ):

    # Define necessary directories
    sorted_dir      = os.path.join( iteration_directory, sorted_dir_name )
    sorted_VDB_dir  = os.path.join( sorted_dir, sorted_VDB_name )
    VDB_files       = []

    # Create a directory for the VDB files
    if not os.path.exists( sorted_VDB_dir ):
        os.makedirs( sorted_VDB_dir )

    # Calculated values for the iso surface
    delta       = 0.01
    median      = np.median( densities[0] )
    max_val     = median + delta
    min_val     = median - delta
    file_len    = len( densities )

    for i, d in enumerate( densities ):
        # Build the VDB objects
        grid = vdb.FloatGrid()
            
        # Copies image volume from numpy to VDB grid
        grid.copyFromArray( d )

        # Blender needs grid name to be "density" or "velocity" to be colorful (need data to be Vector Float)
        grid.name = 'density'

        # Set the density ring. This should be around the median value
        vector_data         = np.zeros( d.shape + (3,) )
        shell_values        = np.where( ( d > min_val ) & ( d < max_val ), d, 0. )
        zero_idces          = np.where( shell_values == 0. )
        shell_idces         = np.where( shell_values > 0. )

        # Now, renormalize the values to 1, for the sake of making the colors pop
        d_vals              = shell_values

        # Ensure values exist at all
        if d_vals.size > 0:

            min_value = np.min( d_vals )
            max_value = np.max( d_vals )
            
            # Ensure these aren't the save value
            if min_value != max_value:
                normed_val                  = ( d_vals - min_value )/( max_value - min_value )
                arr_expanded = np.expand_dims(normed_val, axis=-1)
                arr_expanded = np.repeat(arr_expanded, 3, axis=-1)
                vector_data[ shell_idces ]  = shell_colorsHSV['472']*arr_expanded[ shell_idces ]

        # Assign zeros for zero_values
        vector_data[ zero_idces ] = np.zeros( ( 3, ) )

        # Create a Blender-compatible VDB grid
        vdb_grid    = vdb.Vec3SGrid()
        vdb_grid.copyFromArray( np.array( vector_data ).reshape( d.shape + (3,) ) )

        # Set the grid name to "velocity" for color display
        vdb_grid.name = 'color_density'
       
        # Save the VDB Files
        vdb_file    = os.path.join( sorted_VDB_dir, f'{base_name}_frame-{str( i ).zfill( len(str(file_len)) )}{sorted_VDB_type}' )

        vdb.write( vdb_file, [ grid, vdb_grid ] )

        VDB_files.append( vdb_file )

    # Return the VDB file names in a list
    return VDB_files

# Interpolates the Atoms objects over a defined animation
def InterpolateAtoms( atoms = None, duration:int = 60, fps:int = 60, interpolate_style = 'linear' ):
    assert atoms is not None, "No Atoms submitted!"
    
    # First, calculate the frames needed
    frames              = fps*duration

    # Next, get how many iterations we have
    iterations          = len( atoms )

    # Initialize the return variable
    interpolated_atoms  = []

    # If we have less iterations than the requested frames, then let's interpolate this bish
    if iterations < frames and iterations > 1:
        in_between_steps    = ( frames - iterations )/( iterations - 1 )
        in_between_mod      = ( frames - iterations )%( iterations - 1 )

        # Add another step if not even. Better to have too many frames than too little.
        if in_between_mod > 0:
            in_between_steps += 1

        # Linearly interpolate only between for now
        #if interpolate_style == 'linear':

        # Cycle through the iterations, excluding the last one
        for i, iter_atom in enumerate( atoms[:-1] ):

            # Next iteration of the atoms
            next_a  = atoms[i+1]
            
            # Get the slope from all atoms in an iteration

            # Cycle through the atoms, since dictionaries can't be sliced
            # Atoms given as iter_atom[i] = [ np.array( [x,y,z] ), atomic_number ]
            slope   = []
            for j, atm in enumerate( iter_atom ):
                slope.append( ( next_a[j][0] - atm[0] )/( in_between_steps+1 ) )
        
            # Save the first value
            interpolated_atoms.append( iter_atom )

            for j in range( int( in_between_steps ) ):
                # Append the linear fit
                
                # Cycle through each atom in the iteration
                iter_atoms = []
                for k in range( len( iter_atom ) ):
                    
                    iter_atoms.append( [ slope[k]*(j+1)+iter_atom[k][0], iter_atom[k][1] ] )
                    
                interpolated_atoms.append( iter_atoms )    # 1 added because j=0 gives d (i.e. x = 0 in y = mx+b)

        # Pop on that last frame
        interpolated_atoms.append( atoms[-1] )
    else:
        # This is not likely to happen
        interpolated_atoms = atoms

    return interpolated_atoms

# Interpolates density data. This will be RAM intensive
def InterpolateDensities( densities:list = None, duration:int = 60, fps:int = 60, interpolate_style = 'linear' ):
    assert densities is not None, "No densities submitted!"

    # First, calculate the frames needed
    frames                  = fps*duration

    # Next, get how many iterations we have
    iterations              = len( densities )

    # Initialize the return variable
    interpolated_densities  = []

    # If we have less iterations than the requested frames, then let's interpolate this bish
    if iterations < frames:

        # First, let's calulate the steps in between iterations.
        # Example 1: iterations = [ IT1, IT2, IT3, IT4 ]
        #           it        = 4
        #           frames    = 10
        # frames/it     = 2
        # frames%it     = 2
        # steps         = (frames - it)/( it-1 ) = 6/3 = 2
        # -> it_frames = [ IT1, i1, i2, IT2, i3, i4, IT3, i5, i6, IT4  ]

        in_between_steps    = ( frames - iterations )/( iterations - 1 )
        in_between_mod      = ( frames - iterations )%( iterations - 1 )

        # Add another step if not even. Better to have too many frames than too little.
        if in_between_mod > 0:
            in_between_steps += 1

        # Linearly interpolate only between for now
        #if interpolate_style == 'linear':

        # Cycle through the iterations, excluding the last one
        for i, d in enumerate( densities[:-1] ):

            next_d  = densities[i+1]
            slope   = (next_d - d)/(in_between_steps+1) # 1 added because the next_d is steps+1 away in x

            # Save the first value
            interpolated_densities.append( d )

            for j in range( int(in_between_steps) ):
                # Append the linear fit
                interpolated_densities.append( (slope*(j+1))+d )    # 1 added because j=0 gives d (i.e. x = 0 in y = mx+b)

        # Pop on that last frame
        interpolated_densities.append( densities[-1] )


    else:
        # This is not likely to happen
        interpolated_densities = densities

    return interpolated_densities

# Just saves the blender file
def saveBlenderFile( blend_directory = None ):
    assert blend_directory is not None, "No Blender directory provided"
    bpy.ops.wm.save_as_mainfile( filepath = os.path.join( blend_directory, f"{base_name}.blend" ) )

# Generates the desired materials
def createMaterials( atoms = None, nm = '488' ):

    assert atoms is not None, "atoms not set in createMaterials()"

    # First, let's create the material necessary for the electron cloud
    mat = bpy.data.materials.new( electron_mat )
    mat.use_nodes = True

    # Get a reference to the material node tree
    tree = mat.node_tree

    # Clear existing nodes
    for node in tree.nodes:
        tree.nodes.remove(node)

    # Create nodes
    attr_node                   = tree.nodes.new(type="ShaderNodeAttribute")
    attr_node.attribute_name    = 'density'
    multiply_node               = tree.nodes.new(type="ShaderNodeMath")
    multiply_node.operation     = "MULTIPLY"
    multiply_node.location      = (200, 0)
    ramp_node                   = tree.nodes.new(type="ShaderNodeValToRGB")
    ramp_node.location          = (400, 0)
    principled_node             = tree.nodes.new(type="ShaderNodeVolumePrincipled")
    principled_node.location    = (700, 0)
    output_node                 = tree.nodes.new(type="ShaderNodeOutputMaterial")
    output_node.location        = (1000, 0)

    # Set values on nodes
    multiply_node.inputs[1].default_value   = 1
    ramp_node.color_ramp.elements[0].color  = (0, 0, 0, 0) # Set first color to transparent
    
    # set 2nd color
    ramp_node.color_ramp.elements[1].color  = ( shell_colorsRGB[nm][0], shell_colorsRGB[nm][1], shell_colorsRGB[nm][2], 1 ) # Set to color chosen
    mat.diffuse_color                       = ( 1, 1, 1, 1 )

    # Link nodes together
    links = tree.links
    links.new( attr_node.outputs[2], multiply_node.inputs[0] )
    links.new( multiply_node.outputs[0],ramp_node.inputs[0] )
    links.new( ramp_node.outputs[0], principled_node.inputs[7] )
    links.new( principled_node.outputs[0], output_node.inputs[1] )

    principled_node.inputs[6].default_value = 1.
    
    #add the material to the object to be able to check it faster
    #bpy.context.object.data.materials.append( mat )


    # Next, let's create the atom sphere materials based on atomic number
    # Get the first iteration
    atom_iter   = atoms[0]
    for atom in atom_iter:
        # If the atomic number is not already in the dictionary, create it
        if atom[1] not in atom_mat_names.keys():
            # Create the material
            mat_name        = atomic_values[str(atom[1])]['name']
            mat_color       = ( atomic_values[str(atom[1])]['color'][0], atomic_values[str(atom[1])]['color'][1], atomic_values[str(atom[1])]['color'][2], 1 )

            print( f"{mat_name} color = {mat_color}" )

            mat             = bpy.data.materials.new( mat_name )
            mat.use_nodes   = True

            # Get a reference to the material node tree
            tree = mat.node_tree

            # Clear existing nodes
            for node in tree.nodes:
                tree.nodes.remove(node)

            BSDF_node               = tree.nodes.new( type = "ShaderNodeBsdfPrincipled" )
            BSDF_node.location      = ( 200, 0 )
            output_node             = tree.nodes.new(type="ShaderNodeOutputMaterial")
            output_node.location    = ( 400, 0 )
            
            BSDF_node.inputs['Base Color'].default_value    = mat_color  # Base color
            BSDF_node.inputs['Roughness'].default_value     = 0.2  # Roughness

            # Link nodes together
            tree.links.new( BSDF_node.outputs[0], output_node.inputs[0] )

# TODO:Finish Imports the densities from the VDB files
def createDensities( VDB_files = None, repeat = 1, unitcell = [ 1, 1, 1 ] ):
    assert VDB_files is not None, "No VDB_files received!"

    # Get the length of VDB_files
    i   = len( VDB_files )

    # Create the scene for the wave functions
    EDcollection    = bpy.data.collections.new("Electron Densities")
    bpy.context.scene.collection.children.link( EDcollection )

    bpy.ops.object.volume_import( filepath = VDB_files[0], files = [ { 'name': f } for f in VDB_files ], scale = ( 1./unitcell[0], 1./unitcell[1], 1./unitcell[2] ) )
    obj     = bpy.context.object.data
    #EDcollection.objects.link( obj )

    # As long as there is more than 1 file, upload as sequence
    if i != 1:
        obj.is_sequence = True

    obj.frame_duration          = i-1
    obj.frame_start             = 0
    obj.sequence_mode           = 'EXTEND'
    #bpy.context.object.scale    = ( unitcell[0], unitcell[1], unitcell[2] )

# Creates the Atoms in the scene along with their keyframes
def createAtoms( atoms = None, fps = 60, repeat = 1, unitcell = [ 1, 1, 1 ] ):

    assert atoms is not None, "No atoms recieved in createAtoms()"

    # the atoms file should have the format of a dictionary such that
    # atom1: {  position: [ x,y,z ],
    #           atomic_number: 6 }
    # The json file then has all atom objects for each iteration of the frames
    # atoms = [ [ atom1, atom2, ... atomN ], ..., [ atom1, atom2, ... atomN ] ]

    frames      = len( atoms )
    atom_init   = atoms[0]

    # Set the frame rate
    bpy.context.scene.render.fps = fps

    # Create the atomic collections
    NUCcollection   = bpy.data.collections.new("Nuclei")

    # Link the collections to the scene
    bpy.context.scene.collection.children.link( NUCcollection )

    # Cycle through each atom, to build the objects and their keyframes
    for i, atom in enumerate( atom_init ):

        # Create the object
        bpy.ops.mesh.primitive_uv_sphere_add( radius = atomic_values[str(atom[1])]['radius']*radius_scale, location = ( atom[0][0], atom[0][1], atom[0][2] ) )
        
        # Smooth it for looks
        bpy.ops.object.shade_smooth()

        # Get the created sphere
        atomic_sphere = bpy.context.object
        NUCcollection.objects.link( atomic_sphere )

        # Set the defined material 
        mat = bpy.data.materials.get( atomic_values[str(atom[1])]['name'] )
        atomic_sphere.data.materials[0] = mat

        # Insert first keyframe at position
        atomic_sphere.keyframe_insert( data_path = "location", frame = 1 )

        # Cycle through the frames and update the position and keyframe
        for j in range( frames-1 ):
            pos      = atoms[j+1][i][0]

            # Move the sphere
            atomic_sphere.location = ( pos[0], pos[1], pos[2] )

            # Set a keyframe for the new position
            atomic_sphere.keyframe_insert( data_path = "location", frame = j+2 )



#### Main (This will run directly, since it's just called in Blender)


# Important inputs 
#   Iterations directory    : str
#   Repeat along xy count   : int

# iter_dir    = str( sys.argv[1] )        # The directory with the iterated wave function files. i.e. /path/to/files
# repeate_int = int( sys.argv[2] )        # How many times to repeat the domain in xy. i.e. 3 -> 3x3x1 repeat
# anim_dur    = int( sys.argv[3] )        # How long in seconds that the animation should exist for. i.e. 60 -> 1 minute
# anim_fps    = float( sys.argv[4] )      # The Frames Per Second of the animation. i.e. 24. -> 24 fps
# sliced      = int( sys.argv[5] )        # Whether or not to slice the wave functions along a calculated axis. i.e. 0->False, 1->True

# Windows            
#iter_dir    = "C:\\Users\\jpfer\\OneDrive - Northeastern University\\General\\Data Storage\\John Ferrier\\Code\\Python\\DFT\\molecules\\Naphthalene\\pw\\dzp\\PBE\\540\\h_0.19\\vac_6.0\\Wavefunctions"
# MacOS
iter_dir    = "/Users/johnferrier/Library/CloudStorage/OneDrive-NortheasternUniversity/General/Data Storage/John Ferrier/Code/Python/DFT/molecules/Naphthalene/pw/dzp/PBE/540/h_0.19/vac_6.0/Wavefunctions"
# WSL
#iter_dir    = "/mnt/c/Users/jpfer/OneDrive-NortheasternUniversity/General/Data Storage/John Ferrier/Code/Python/DFT/molecules/Naphthalene/pw/dzp/PBE/540/h_0.19/vac_6.0/Wavefunctions"

repeate_int = 1
anim_dur    = 60
anim_fps    = 60
sliced      = 0

# Set the Blend directory
blnd_dir    = os.path.join( iter_dir, sorted_dir_name )

# First, let's check to make sure these values aren't already calculated and saved.
# If they are, this saves some time. We can just import them.
# Let's check if the blender file exists already
if not blendExists( blend_directory = blnd_dir ):
    if not alreadyCalculated( iteration_directory = iter_dir, override = False ):

        # Let's sort the data for the sake of using it for animations
        # The directory shape is Iteration -> Band. But, we'll want Band -> Iteration, if there is more than one iteration.
        # Read in the sorted atoms and clouds as numpy arrays in a nested list format, indicating iteration number
        # i.e. atoms        = [ [ atm1, atm2, ..., atmN ], [ atm1, atm2, ..., atmN ] ]
        #   where atmX is a dictionary containg the position and atomic number
        # i.e. densities    = [ [ band0, band1, ..., bandN ], [ band0, band1, ..., bandN ] ] ==== RAW, un-altered data from the CUBE files
        #   where bandX is a 3D numpy array containing the relative densities
        atoms, densities, unitcell  = GetIterationSortedData( iteration_directory = iter_dir )

        # Save the unitcell
        unt_cell_file   = os.path.join( blnd_dir, "unitcell.npy" )
        np.save( unt_cell_file, np.array( unitcell ) )
        
        
        # First, let's interpolate the atoms
        anim_atoms      = InterpolateAtoms( atoms = atoms, duration = anim_dur, fps = anim_fps )

        # Save the animated Atoms
        SaveAtomFiles( atoms = atoms, iteration_directory = iter_dir )

        
        # Build the VDB files from the densities
        # We're going to combine bands into one file by adding the 3D arrays and normalizing the data.
        # This will allow for easier visualization in Blender through the use of Mapped ramps
        summed_densities    = []
        # Cycle through iterations
        for dens_list in densities:
            total_d = np.zeros_like( dens_list[0] )
            # Add all bands
            for d in dens_list:
                total_d += d

            # Normalize the values to 1 for consistency in Blender
            max_val     = np.max( total_d )
            min_val     = np.min( total_d )
            scale       = max_val - min_val

            total_d     -= min_val          # Zeros out the values
            total_d     /= scale            # Scales the magnitudes to 1.

            # Append to the summed list
            summed_densities.append( total_d )

        # Now, we need to consider the time step evolutions and interpolate the density changes
        anim_densities  = InterpolateDensities( densities = summed_densities, duration = anim_dur, fps = anim_fps )

        # Save these values to .vdb files as frames. Keep the vdbFiles for opening in blender
        vdbFiles        = SaveVDBFiles( densities = anim_densities, iteration_directory = iter_dir )
    

    else:
        anim_atoms, vdbFiles, unitcell  = GetFileInfo( iteration_directory = iter_dir )


    
    #### We have the atom information and the vdb files. Let's build the blender environment
        
    # Create the materials to be applied later
    # Creates the electron colors from nm, to assign to the cloud densities
    # Creates the atomic nucleus colors associated with each element present
    createMaterials( atoms = anim_atoms, nm = selected_nm )

    # Import the atomic nuclei, assign their materials, and generate keyframes
    createAtoms( atoms = anim_atoms, fps = anim_fps, repeat = repeate_int, unitcell = unitcell )

    # Import the VDB files, assign the color, and generate keyframes
    # This function will also create the geometry nodes necessary for controlling specific visuals
    print( f"Length of vdbFiles = {len(vdbFiles)}" )
    createDensities( VDB_files = vdbFiles, repeat = repeate_int, unitcell = unitcell )

    # Set the render
    bpy.context.scene.render.engine = 'CYCLES'
    bpy.context.scene.cycles.device = 'GPU'


    # Save the blender file to be used later
    saveBlenderFile( blend_directory = blnd_dir )

    print( f"Blender file created in {blnd_dir}" )
    

else:
    print( "Blender file already exists!" )




#### BPY functions to know for repeating indices
# Instancing an object (Saves a TON on RAM)
#bpy.ops.object.duplicate_move_linked( 
#     OBJECT_OT_duplicate     = { "linked" : True, "mode" : 'TRANSLATION' }, 
#     TRANSFORM_OT_translate  = { "value" : ( 1.62406, 4.22557, 0.445127 ), 
#                                "orient_axis_ortho" : 'X', 
#                                "orient_type" : 'GLOBAL', 
#                                "orient_matrix" : ( ( 1, 0, 0), ( 0, 1, 0 ), ( 0, 0, 1 ) ), 
#                                "orient_matrix_type" : 'GLOBAL', 
#                                "constraint_axis" : (False, False, False), 
#                                "mirror" : False,  
#                                "use_proportional_edit" : False, 
#                                "proportional_edit_falloff" : 'SMOOTH', 
#                                "proportional_size" : 1, 
#                                "use_proportional_connected" : False, 
#                                "use_proportional_projected" : False, 
#                                "snap" : False, 
#                                "snap_elements" : { 'INCREMENT' }, 
#                                "use_snap_project" : False, 
#                                "snap_target" : 'CLOSEST', 
#                                "use_snap_self" : True, 
#                                "use_snap_edit" : True, 
#                                "use_snap_nonedit" : True, 
#                                "use_snap_selectable" : False, 
#                                "snap_point" : ( 0, 0, 0 ), 
#                                "snap_align" : False, 
#                                "snap_normal" : ( 0, 0, 0 ), 
#                                "gpencil_strokes" : False, 
#                                "cursor_transform" : False, 
#                                "texture_space" : False, 
#                                "remove_on_cancel" : False, 
#                                "view2d_edge_pan" : False, 
#                                "release_confirm" : False, 
#                                "use_accurate" : False, 
#                                "use_automerge_and_split" : False
#                             }
# )
