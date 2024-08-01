'''
    Name:           Vibrational Data Converter
        
    Description:    This script opens all trajectory files in a given directory and converts them into a stack of XYZ files that can be imported to Blender.
        
    Author:         John Ferrier, NEU Physics

    Date:           27 July 2024
'''
# Import
import os
import sys
import argparse
from ase.io import read, write

#Build Class
class TrajConverter:

    def __init__( self, dir:str = None ) -> None:

        print( "Starting TrajConverter" )

        # Set constants
        self.dir    = dir
        self.files  = []
        self.sv_dir = os.path.join( self.dir, 'XYZ_VibData' )

        # Set directory if None
        if self.dir is None:
            self.dir = os.getcwd()
        
        print( self.dir )

        # Ensure directory exists
        if not os.path.exists( self.dir ):
            self.endScript( f"Directory '{self.dir}' not found!" )

        # Check that vibrational data exists
        if not self.files_exist():
            self.endScript( "No vibrational trajectory files were found!" )

        self.create_dir( self.sv_dir )

    # Ends the script with a message for the user
    def endScript( self, message = "" ) -> None:
        print( f"{message}" )
        sys.exit()

    # Checks to see if vibrational trajectory files exist or not
    def files_exist( self ) -> bool:
        
        # Looking for format 'Vibrations.x.traj'
        for file in os.listdir( self.dir ):
            f = file.split(".")
            if f[0] == "Vibrations" and f[-1] == 'traj':
                self.files.append( os.path.join( self.dir, file ) )
        return ( len( self.files ) > 0 )

    # Creates a directory
    def create_dir( self, dir = "" ) -> None:
        if not os.path.exists( dir ):
            os.mkdir( dir )

    # Creates the .XYZ files and corresponding directory
    def convert( self ) -> None:

        print( "Converting Files..." )

        # Cycle through the files
        for f in self.files:

            print( f"{f =}" )

            # Get the file name
            fn          = os.path.basename( f )

            # Get the mode number
            fn_splt     = fn.split( "." )
            mode_nm     = fn_splt[1]

            # Create save directory
            save_dir    = os.path.join( self.sv_dir, f"Vibrations-{mode_nm}" )

            # Create the directory so that it can save
            self.create_dir( save_dir )

            # Load the trajectory file
            trj = read( f, index = ':' )

            # Loop through each frame in the trajectory
            for i, atoms in enumerate( trj ):

                # Save each frame as a separate XYZ file
                write( os.path.join( save_dir, f'Vibrations.{mode_nm}_{i}.xyz'), atoms )



if __name__ == "__main__":

    # Take input argument for directory
    parser  = argparse.ArgumentParser( description = "Trajectory Converter" )

    # Add arguments
    parser.add_argument( '--d', type = str, help = 'Directory containing vibrational trajectory files', default = None )

    # Parse the arguments
    args    = parser.parse_args()

    TC      = TrajConverter( dir = args.d )

    # Assuming we made it this for
    TC.convert()






