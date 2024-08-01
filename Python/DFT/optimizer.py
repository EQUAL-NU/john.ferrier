'''

    Name:           Optimizer 

    Description:    This file solves for the optimized DFT conditions given a specified molecule or crystal and XC.
                    This minimization of the energy is necessary to ensure the most accurate calculations possible 
                    (by removing mirror interactions and boundary condition squeezing of the wave functions)

    Info:           This code optimizes the DFT settings for energy cutoff, vacuum, and grid-spacing (k-points if crystal).
                    The optimized values are saved to a text file for quick reference for the user but the actual values are
                    saved to a database file and the "Atoms" objects are pickled. Files for each molecule (crystal) are purely 
                    dictated by the name provided by the user. Saved molecules (crystals) can be viewed by typing in terminal:

                        python3 optimizer.py --list

    Date:           14 February 2024

    Author:         John Ferrier, NEU Physics

'''


import os
import sys
from typing import Union
import numpy as np
from ase.units import kB, Bohr
from ase.io import read, write
import matplotlib.pyplot as plt
from ase.optimize import QuasiNewton
from gpaw import GPAW, restart, PW, FermiDirac
from ase.calculators.emt import EMT

#from .General.database import DB
from molecules import molecules


class OptimizerCalculator:

    # Initialize settings
    def __init__( self, 
                  save_dir:str    = "", 
                  mol_name:str    = "", 
                  dbFile          = "", 
                  mol_type:int    = 0, 
                  xc_type:int     = 1, 
                  mode:int        = 0, 
                  basis:str       = 'dzp',
                  adsorbate:bool  = False,
                  UseEMT:bool     = False ):

        self.calc           = None
        self.mol            = None
        self.is_crystal     = False
        self.ENCUT          = 0.
        self.KPTS           = 0.
        self.H              = 0.
        self.saveWFs        = True
        self.iter           = 0
        self.mode           = [ 'pw', 'lcao', 'fd' ][mode]
        self.basis          = basis
        self.xc             = [ 'LDA', 'PBE', 'revPBE', 'RPBE', 'PBE0', 'B3LYP' ][xc_type]
        self.bandpath       = None
        self.mol_type       = [ 'molecules', 'crystals', 'substrates' ][mol_type]
        self.save_dir       = save_dir
        self.is_adsorbate   = adsorbate
        self.UseEMT         = UseEMT

        self.mol_name       = mol_name
        self.dbFile         = dbFile
        #self.db             = DB( self.dbFile )
        self.pot_en         = 0.

        ######### CONSTANTS
        self.vacuum         = 5.
        self.fmax           = 0.01
        self.crys_size      = ( 1, 1, 1 )
        self.subs_size      = ( 1, 1, 1 )
        self.org_cell       = None

        self.GPAW_LOC       = None
        self.WF_LOC         = None

        # This just sets the self.mol object if the mol_name is in the molecules.py file.
        self.get_molecule()

        ######### MAIN

        if self.save_dir == "":
            # This ensures the files saved are used instead of creating new ones
            self.save_dir = os.path.dirname( os.path.abspath( __file__ ) )

        # Molecule File Locations
        self.mol_base        = os.path.join( self.save_dir, self.mol_type, self.mol_name )
        self.dir_base        = os.path.join( self.mol_base, self.mode, self.basis, self.xc )

        # Optimize the ENCUT
        self.is_crystal      = False
        if self.mol_type == 'crystals' or self.mol_type == 'substrates':
            self.is_crystal  = True

    # Returns a GPAW calculator with the input settings
    def get_calculator( self,
                        ENCUT       = 500,
                        h           = 0.2,
                        width       = kB*300.,
                        kpoints     = ( 3, 3, 3 ) ):
        if self.UseEMT:
            self.calc = EMT()
        else:
            # Initialize this
            hund = False
            if self.mode == 'pw':
                mode = PW(ENCUT)
            else:
                mode = self.mode
            
            if self.is_crystal:

                self.calc   = GPAW( mode    = mode,
                                basis       = self.basis,
                                xc          = self.xc,
                                hund        = hund,
                                kpts        = kpoints,
                                occupations = { 'name': 'fermi-dirac',
                                                'width': width },
                                symmetry    = { 'point_group': False,
                                                'time_reversal': False },
                                parallel    = dict( augment_grids   = True,  # use all cores for XC/Poisson
                                                    sl_auto         = True ) )
            else:
                # For molecules
                # Check the size of self.mol. If only one atom, Hund = True
                hund        = ( len( self.mol ) == 1 )
                self.calc   = GPAW( mode    = mode,
                                basis       = self.basis,
                                xc          = self.xc,
                                hund        = hund,
                                h           = h,
                                occupations = { 'name': 'fermi-dirac',
                                                'width': width },
                                symmetry    = { 'point_group': False,
                                                'time_reversal': False },
                                parallel    = dict( augment_grids   = True,  # use all cores for XC/Poisson
                                                    sl_auto         = True ) )
            
        self.mol.set_calculator( self.calc )
        self.set_vacuum()

    # Checks to see if the normalized slope is below the given threshold
    def check_normalized_slope( self, X, Y, thresh, flip = False, minsteps = 2, perc_cutoff = 0.1 ):

        # Normalize X to 1
        x       = X.copy()
        max_x   = np.max( x )
        min_x   = np.min( x )
        mag_x   = max_x - min_x
        x       = (x - min_x)/mag_x
        if flip:
            x   = x[::-1]

        # Normalize Y to 1
        y       = Y.copy()
        max_y   = np.max( y )
        min_y   = np.min( y )
        mag_y   = max_y - min_y
        y       = ( y - min_y )/mag_y

        # Check slope
        m       = np.abs( y[-1] - y[-2] )/np.abs( x[-1] - x[-2] )

        RET     = False

        # Also check averaging, as it could be fluctuating around a minima. Run on last 3 points
        if len( y ) > minsteps:

            RET     = ( m <= thresh )
            if not RET:
                y_slice = y[-3:]
                ave_y   = np.average( y_slice )
                std     = np.std( y_slice )
                RET     = np.logical_or( RET, np.logical_and( y[-1] >= ( ave_y-std ), y[-1] <= ( ave_y+std )  ) )

            if not RET:
                perc_cutoff /= 100.
                perc    = np.abs( (y[-2] - y[-1])/( y[-2] ) )
                RET     = ( perc <= perc_cutoff )

        return RET
        
    # Finds the optimum cutoff energy for the plane wave mode.
    def optimize_ENCUT( self, perc_cutoff = 0.01, dirpath = '', encut_range: Union[np.array,None] = None ):

        self.ENCUT  = 0.

        # Check if Opt-ENCUT.npy exists
        if os.path.isfile( os.path.join( dirpath, 'Opt-ENCUT.npy' ) ):

            print( "Opt-ENCUT found! Opening File..." )
            self.ENCUT   = int( np.load( os.path.join( dirpath, 'Opt-ENCUT.npy' ) ) )

        else:
            assert not self.mol is None, "No Atoms object was recieved in optimize_ENCUT"
            
            ENS         = encut_range
            if ENS is None:
                ENS     = np.arange( 240, 800, 20 )
            curr_pos    = 0
            pot_ens     = []
            EN_list     = []
            
            kpts        = ( 3, 3, 3 )
            if self.mol_type == 'crystals':
                kptsl       = list( kpts )
                kptsl[2]    = 1
                kpts        = tuple( kptsl )

            # Cycle through the energies
            for i, E in enumerate( ENS ):
                
                curr_pos = i
                self.get_calculator( ENCUT = E, kpoints = kpts )

                pot_ens.append( self.mol.get_potential_energy() )
                EN_list.append( E )
                if i > 0:
                    if self.check_normalized_slope( EN_list, pot_ens, perc_cutoff, minsteps = 4 ):
                        break

            # Do 2nd to last, since not much variance. Will save time
            self.ENCUT = ENS[curr_pos-1]
            print( f"potential energy = {pot_ens[-1]}eV" )
            # Save the final Value in dirpath
            np.save( os.path.join( dirpath, 'Opt-ENCUT.npy' ), self.ENCUT )

            # Save images
            print( f"Saving image..." )
            plt.plot( EN_list, pot_ens )
            plt.savefig( fname = os.path.join( dirpath, 'Opt-ENCUT.png' ), dpi = 300 )
            plt.clf()

        print( f"Optimized cutoff energy = {self.ENCUT}eV" )

    # Finds the optimum cutoff energy for the plane wave mode.
    def optimize_vacuum( self, perc_cutoff = 0.05, dirpath = '' ):

        self.vacuum  = 0.

        # Check if Opt-ENCUT.npy exists
        if os.path.isfile( os.path.join( dirpath, 'Opt-Vacuum.npy' ) ):

            print( "Opt-Vacuum found! Opening File..." )
            self.vacuum   = round( float( np.load( os.path.join( dirpath, 'Opt-Vacuum.npy' ) ) ), 1 )

        else:
            assert not self.mol is None, "No Atoms object was recieved in optimize_vacuum"
            
            VAC         = np.arange( 5., 20., 1. )
            curr_pos    = 0
            pot_ens     = []
            VC_list     = []

            if self.is_crystal:
                kpts        = ( self.KPTS, self.KPTS, 1 )

            # Cycle through the energies
            for i, V in enumerate( VAC ):
                
                curr_pos = i
                if self.is_crystal:
                    self.get_calculator( ENCUT = self.ENCUT, kpoints = kpts )
                else:
                    self.get_calculator( ENCUT = self.ENCUT, h = self.H )

                self.set_vacuum( V = V )

                pot_ens.append( self.mol.get_potential_energy() )
                VC_list.append( V )
                if i > 0:
                    if self.check_normalized_slope( VC_list, pot_ens, perc_cutoff ):
                        break

            self.vacuum = VAC[curr_pos-1]
            print( f"potential energy = {pot_ens[-1]}eV" )
            # Save the final Value in dirpath
            np.save( os.path.join( dirpath, 'Opt-Vacuum.npy' ), self.vacuum )

            # Save images
            print( f"Saving image..." )
            plt.plot( VC_list, pot_ens )
            plt.savefig( fname = os.path.join( dirpath, 'Opt-Vacuum.png' ), dpi = 300 )
            plt.clf()

        print( f"Optimized vacuum = {self.vacuum}" )

    # Finds the optimum cutoff energy for the plane wave mode.
    def optimize_KPoint( self, perc_cutoff = 0.05, dirpath = '' ):

        self.KPTS   = 0

        # Check if Opt-KPTS.npy exists
        if os.path.isfile( os.path.join( dirpath, 'Opt-KPTS.npy' ) ):

            print( "Opt-KPTS found! Opening File..." )
            self.KPTS   = int( np.load( os.path.join( dirpath, 'Opt-KPTS.npy' ) ) )
            print( f"self.KPT = {self.KPTS}" )

        else:
            assert not self.mol is None, "No Atoms object was recieved in optimize_KPoint"
            
            ks          = [ 3, 4, 5, 6, 7, 8, 9 ]
            curr_pos    = 0
            pot_ens     = []
            K_list      = []

            # Cycle through the energies
            for i, k in enumerate( ks ):

                kpts    = ( k, k, k )
                if self.mol_type  == 'crystals':   # i.e., it's a 2D material and not a substrate
                    kptsl       = list( kpts )
                    kptsl[2]    = 1
                    kpts        = tuple( kptsl )

                curr_pos = i
                self.get_calculator( ENCUT = self.ENCUT, kpoints = kpts )

                # Reset the vacuum just in case
                if self.mol_type == 'substrates':
                    self.set_vacuum( V = 0. )
                else:
                    self.set_vacuum( V = self.vacuum )

                pot_ens.append( self.mol.get_potential_energy() )
                K_list.append( k )
                if i > 0:
                    if self.check_normalized_slope( K_list, pot_ens, perc_cutoff ):
                        break

            self.KPTS     = ks[curr_pos-1]
            print( f"potential energy = {pot_ens[-1]}eV" )
            # Save the final Value in dirpath
            np.save( os.path.join( dirpath, 'Opt-KPTS.npy' ), self.KPTS )

            # Save images
            print( f"Saving image..." )
            plt.plot( K_list, pot_ens )
            plt.savefig( fname = os.path.join( dirpath, 'Opt-KPTS.png' ), dpi = 300 )
            plt.clf()

        print( f"Optimized k-points = ( {self.KPTS}, {self.KPTS}, {self.KPTS} )" )

    # Finds the optimum cutoff energy for the plane wave mode.
    def optimize_h( self, perc_cutoff = 0.05, dirpath = '', cust_h_range = None ):

        self.H   = 0

        # Check if Opt-h.npy exists
        if os.path.isfile( os.path.join( dirpath, 'Opt-h.npy' ) ):

            print( "Opt-h found! Opening File..." )
            self.H   = round( float( np.load( os.path.join( dirpath, 'Opt-h.npy' ) ) ), 2 )

        else:
            assert not self.mol is None, "No Atoms object was recieved in optimize_KPoint"
            if cust_h_range is None:
                hs  = np.arange( 0.12, 0.2, 0.01 )
            else:
                hs  = cust_h_range
            hs          = hs[::-1]
            curr_pos    = 0
            pot_ens     = []
            h_list      = []

            # Cycle through the energies
            for i, h in enumerate( hs ):
                curr_pos = i
                self.get_calculator( ENCUT = self.ENCUT, h = h )

                # Only reset if not an adsorbate
                if not self.is_adsorbate:
                    # Reset the vacuum just in case
                    self.set_vacuum( V = self.vacuum )

                pot_ens.append( self.mol.get_potential_energy() )
                h_list.append( h )
                if i > 0:
                    if self.check_normalized_slope( h_list, pot_ens, perc_cutoff ):
                        break

            self.H     = hs[curr_pos-1]
            print( f"potential energy = {pot_ens[-1]}eV" )
            # Save the final Value in dirpath
            np.save( os.path.join( dirpath, 'Opt-h.npy' ), self.H )

            # Save images
            print( f"Saving image..." )
            plt.plot( h_list, pot_ens )
            plt.savefig( fname = os.path.join( dirpath, 'Opt-h.png' ), dpi = 300 )
            plt.clf()

        print( f"Optimized h = {self.H}" )

    # Sets the vacuum of the object
    def set_vacuum( self, V = None):

        # Only set if the item is not an adsorbate
        if not self.is_adsorbate:
            v = V
            if v == None:
                v = self.vacuum

            if self.mol_type == 'substrates':
                v = 0.

            # Reset the vacuum just in case
            if self.mol_type == 'crystals':
                self.mol.center( vacuum = v, axis = 2 )
            elif self.mol_type == 'molecules':
                self.mol.center( vacuum = v )

            elif self.mol_type == 'substrates':
                # Gotta reset the cell for substrates, since setting the vacuum to zero here
                # zeros out the lattice vectors. 
                #self.org_cell = self.mol.get_cell()
                self.vacuum = v
                # Set Vacuum of both to zero
                self.mol.center( vacuum = v )
                self.mol.set_cell( self.org_cell )

        
    # Checks if directory exists and builds it if it doesn't exist
    def check_directory( self, d ):
        
        assert isinstance( d, str ), f'Directory must be string format, not type {type(dir)}'

        # Create list from split
        splt    = d.split( '/' )

        # Remove empty points
        splt    = [ i for i in splt if i ]

        # Build each step of the directory and test
        for i in range( len(splt) ):
            loop_dir    = os.path.join( "/"+"/".join(splt[:i+1]) )
            if not os.path.exists( loop_dir ):
                os.makedirs( loop_dir )

        return d

    # Finds the molecule listed and returns the values associated
    def find_molecule( self, name, mol_type, directory = os.getcwd() ):

        # SERVER setting
        path    = os.path.join( directory, mol_type, name )

        # Laptop Setting
        #path    = os.path.join( directory, mol_type, name )

        mol     = None

        # Read the settings file to find the molecule
        # Check that the path exists
        if not os.path.isdir( path ):
            print( f"Molecule '{name}' in directory '{path}' was not found!" )
            print( f"Creating Directory!" )
            os.mkdir( path )

        # Check that settings file exists
        settings_file   = os.path.join( path, 'settings.txt' )
        print( f"Opening file: {settings_file}" )
        if not os.path.isfile( settings_file ):
            print( f"'settings.txt' for molecule '{name}' in directory '{mol_type}' was not found!" )
            print( "Optimization needed!" )
            return None
        else:
            settings    = []
            with open( settings_file, 'r' ) as file:
                # Read each line
                for line in file:
                    sp_line     = line.split( ":" )             # Split by :
                    nl_line     = sp_line[1].strip()            # Remove \n
                    rp_line     = nl_line.replace( " ", "" )    # Remove white space
                    settings.append( rp_line )

            mol_data_path = os.path.join( path, settings[0], settings[1], settings[3] )

            if mol_type == "molecules":
                mol_data_path = os.path.join( mol_data_path, f"h_{settings[4]}" )
            else:
                mol_data_path = os.path.join( mol_data_path, f"kpts_{settings[4]}" )

            mol_data_path = os.path.join( mol_data_path, f"vac_{settings[5]}" )
            if not mol_type == "molecules":
                mol_data_path = os.path.join( mol_data_path, f"1x1x1" )
            print( f"Opening directory {mol_data_path}" )

            # Grab the Trajectory files last image
            TRJ_FILE    = os.path.join( mol_data_path, "Trajectories", f"{name}_relaxed.traj" )
            if not os.path.isfile( TRJ_FILE ):
                print( f"'{name}_relaxed.traj' for molecule '{name}' in directory '{mol_data_path}/Trajectories' was not found!" )
                sys.exit()
            mol         = read( TRJ_FILE, -1 )
        return mol

    # Returns the proper molecule for the given input argument for the python file, along with the initial dir variables
    def get_molecule( self ):

        # see if molecule exists in the molcules dictionary
        if self.mol_name in molecules:
            self.mol        = molecules[self.mol_name]['object']
            self.mol_type   = molecules[self.mol_name]['type']
            if self.mol_type == 'substrates':
                self.org_cell = self.mol.get_cell().copy()
                print( f"{self.org_cell = }" )
            self.set_vacuum()
        else:
            # Build the object
            print( "Object not found in molecules.py" )
            pass

    # Writes the wave functions, if set to
    def GPAWwriter( self, gpaw_file, mode ):

        if self.saveWFs:
            # Save the wave function
            print( f"Saving Wavefunction files for {self.mol_name} on iteration {self.iter}" )
            nbands = self.mol.calc.get_number_of_bands()

            iter_dir    = self.check_directory( os.path.join( self.WF_LOC, f"Iteration - {self.iter}" ) )

            for band in range( nbands ):
                wf      = self.mol.calc.get_pseudo_wave_function( band = band )
                fname   = f'{self.mol_name}_Band-{band}_WF.cube'
                write(  os.path.join( iter_dir, fname ), self.mol, data = np.power( wf*Bohr**1.5, 2. ) )

            # Increase interator for each folder
            self.iter += 1

        # Save the current molecules gpaw file
        self.mol.calc.write( filename = gpaw_file, mode = mode )

    # Optimze the structure
    def optimize( self, cust_h_range = None, custom_encut_range: Union[np.array,None] = None ):

        self.settings = [ self.mode, self.basis, self.xc ]
        save_file_pth = self.dir_base
        if self.mode == 'pw':
            ### Energy cutoff only done for plane wave mode, since E_C < (|G+k|^2)/2 
            # Find the optimum energy cutoff
            self.optimize_ENCUT( dirpath = self.check_directory( self.dir_base ), encut_range = custom_encut_range )
            self.settings.append( self.ENCUT )
            dir_base    = self.check_directory( os.path.join( self.dir_base, f"{self.ENCUT}" ) )
        else:
            dir_base    = self.check_directory( self.dir_base )

        # Optimize the grid spacing
        if self.is_crystal:

            # Optimize the kpoints (sets self.KPTS)
            self.optimize_KPoint( dirpath = dir_base )
            self.settings.append( self.KPTS )
            dir_base    = self.check_directory( os.path.join( dir_base, f"kpts_{self.KPTS}" ) )

            # Optimize Vacuum (sets self.vacuum). Don't do for substrates, as they're periodic in all directions
            if self.mol_type == 'crystals':
                self.optimize_vacuum( dirpath = self.check_directory( dir_base ) )
                self.settings.append( self.vacuum )
                dir_base    = self.check_directory( os.path.join( dir_base, f"vac_{self.vacuum}" ) )

        else:
            # Optimize the h grid (sets self.H)
            self.optimize_h( dirpath = dir_base, cust_h_range = cust_h_range )
            self.settings.append( np.round(self.H, 2) )
            dir_base    = self.check_directory( os.path.join( dir_base, f"h_{np.round(self.H,2)}" ) )

            # Need to ensure we're not adding unnecessary vacuum to adsorbate measurements.
            if not self.is_adsorbate:
                # Optimize Vacuum
                self.optimize_vacuum( dirpath = self.check_directory( dir_base ) )
                self.settings.append( self.vacuum )
                dir_base    = self.check_directory( os.path.join( dir_base, f"vac_{self.vacuum}" ) )


        # After optimizing settings, save these settings for easy reference later
        with open( os.path.join( save_file_pth, 'settings.txt' ), 'w') as f:
            
            sets  = [ 'mode', 'basis', 'XC' ]

            if self.is_crystal:
                if self.mode == 'pw':
                    sets.append( 'Cutoff Energy' )
                sets.append('k-points')
                if self.mol_type == 'crystals':
                    sets.append( 'vacuum' )
            else:
                if self.mode == 'pw':
                    sets.append( 'Cutoff Energy' )
                sets.append('h-value')
                sets.append('vacuum')

            for i, item in enumerate(self.settings):
                f.write( f'{sets[i]}: {item}' + '\n' )

        # Close the file
        f.close()

        # Build the directory path
        vib_type        = 'Vibrations'

        if self.is_crystal:
            vib_type    = 'Phonons'

        if self.mol_type == 'crystals':
            dir_base    = os.path.join( dir_base, f"{self.crys_size[0]}x{self.crys_size[1]}x{self.crys_size[2]}" )

        if self.mol_type == 'substrates':
            dir_base    = os.path.join( dir_base, f"{self.subs_size[0]}x{self.subs_size[1]}x{self.subs_size[2]}" )


        self.GPAW_LOC   = self.check_directory( os.path.join( dir_base, 'GPAW' ) )
        self.TRAJ_LOC   = self.check_directory( os.path.join( dir_base, 'Trajectories' ) )
        self.WF_LOC     = self.check_directory( os.path.join( dir_base, 'Wavefunctions' ) )
        self.VIB_LOC    = self.check_directory( os.path.join( dir_base, vib_type ) )




        ########### OPTIMIZATION
        # Optimize the structure
        print( f"Checking if GPAW & relaxed Trajectory file for {self.mol_name} already exists..." )
        # Check if GPAW exists already
        name            = self.mol_name

        if os.path.exists( os.path.join( self.GPAW_LOC, f'{name}_optimized.gpw' ) ) and os.path.exists( os.path.join( self.TRAJ_LOC, f'{name}_relaxed.traj' ) ):
            print( f'GPAW file {name}_optimized.gpw & Trajectory file {name}_relaxed.traj found!' )
            self.mol        = read( os.path.join( self.TRAJ_LOC, f'{name}_relaxed.traj' ) )
            self.mol.calc   = GPAW( os.path.join( self.GPAW_LOC, f'{name}_optimized.gpw' ) )
            #self.mol.calc   = calc

            # If a substrate, set the vacuum to 0. This is to allow for boundary conditions in relaxtion
            if self.mol_type == 'substrates':
                self.set_vacuum( V = 0. )

        else:
            print( f'GPAW file {name}_optimized.gpw not found.' )
            print( f'Building GPAW file {name}_optimization.gpw' )
            hund        = (len( self.mol )==1)
            if self.is_crystal:

                calc_kpt        = ( self.KPTS, self.KPTS, self.KPTS )
                if self.mol_type == 'crystals':
                    kptsl       = list( calc_kpt )
                    kptsl[2]    = 1
                    calc_kpt    = tuple( kptsl )
                self.get_calculator( ENCUT = self.ENCUT, kpoints = calc_kpt )

                # Reset the vacuum just in case
                if self.mol_type == 'substrates':
                    self.set_vacuum( V = 0. )
                else:
                    self.set_vacuum( V = self.vacuum )

            else:
                self.get_calculator( ENCUT = self.ENCUT, h = self.H )
                if not self.is_adsorbate:
                    self.set_vacuum( V = self.vacuum )
            

            #self.mol.calc.attach( self.GPAWwriter, 1, os.path.join( self.GPAW_LOC, f'{name}_optimization.gpw' ), mode = 'all' )
            #self.mol.calc.attach( self.mol.calc.write, 1, os.path.join( self.GPAW_LOC, f'{name}.gpw' ), mode = 'all' )

            if len( self.mol ) > 1 or self.is_crystal: # Only need to relax molecules greater than 1 atom
                self.iter   = 0
                print( f'Starting relaxation of molecule {name} to fmax = {self.fmax}' )
                print( f'Saving location set to {self.TRAJ_LOC}' )

                dyn         = QuasiNewton( self.mol, trajectory = os.path.join( self.TRAJ_LOC, f'{name}_relaxation.traj' ), restart = os.path.join( self.TRAJ_LOC, f'{name}_relaxation.pckl' ) ) 
                dyn.attach( self.GPAWwriter, 1, os.path.join( self.GPAW_LOC, f'{name}_optimization.gpw' ), mode = 'all' )
                dyn.run( fmax = self.fmax )

            # Write this regardless, to be able to access properties later.
            print( f'Saving relaxed Trajectory file to {self.TRAJ_LOC}' )
            write( os.path.join( self.TRAJ_LOC, f'{name}_relaxed.traj' ), self.mol )
        




        ########### POTENTIAL ENERGY
        # Check if potential energy file exists
        if os.path.isfile( os.path.join( dir_base, 'relaxed_potential_energy.npy' ) ):
            print( 'relaxed_potential_energy.npy found!' )
            print( 'Opening file...' )
            en  = np.load( os.path.join( dir_base, 'relaxed_potential_energy.npy' ) )

            print( 'To ensure consistency in building the object, running potential energy calculation' )
            print( f'Found Potential Energy = {en}eV' )
            if self.mol_type == 'substrates':
                self.set_vacuum( V = 0. )
            else:
                if not self.is_adsorbate:
                    self.set_vacuum( V = self.vacuum )

            en  =  self.mol.get_potential_energy()

        else:
            # SAVE the potential energy for use later
            print( 'relaxed_potential_energy.npy not found!' )
            print( f'Calculating Potential energy of molecule {name}' )

            # If a substrate, set the vacuum to 0. This is to allow for boundary conditions in relaxtion
            if self.mol_type == 'substrates':
                self.set_vacuum( V = 0. )
            else:
                if not self.is_adsorbate:
                    self.set_vacuum( V = self.vacuum )

            en  =  self.mol.get_potential_energy()
            self.mol.calc.write( os.path.join( self.GPAW_LOC, f'{name}_optimized.gpw' ), mode = 'all'  )
            print( 'relaxed_potential_energy.npy being created' )
            np.save( os.path.join( dir_base, 'relaxed_potential_energy.npy' ), np.array(en) )
        
        print( f'E_{name} = {en}eV' )
        self.pot_en = en

    # Checks if the input molecular name already exists in the database
    def mol_exists( self, mol_name ) -> bool:
        return True

    # Checks if the input structure file is readable
    def check_struct( self, struct_file ) -> bool:
        return True

    # Just prints out the terminal commands for when the file is used directly
    def showHelp( self ) -> None:

        
        """
            Prints a list of the possible commands for a sole use of the optimizer file in terminal

            Parameter
            ---------
            None

            Returns
            -------
            None
        """


        options         = [ "-h", "-m", "-s", "-l", "-O" ]
        long_options    = [ "--help", "--molecule_name", "--molecular_struct", "--list", "--override_struct" ]
        explanation     = [ "Shows the possible options when running the script independently.",
                            "This is string value the represents the name of the molecule/crystal that is desired.\ni.e. python3 optimize.py -m CH4",
                            "This value is an input for the molecular structure.\nThis is only needed if the molecule does not already exist in the database.\ni.e. python3 optimize.py -m CH4 -s /path/to/file.traj",
                            "This command lists the names of all available molecular structures that have been previously relaxed.",
                            "The override command forces the script to utilize the input script over the saved one.\nWARNING - This will overrite the older structure file." ]
        none            = ""
        print( f"Parameters" )
        print( f"{none:#>38}" )

        shrt            = "Short"
        lng             = "Long"
        dscr            = "Description"
        sep             = ""

        # Print out all of the options
        for i, sh in enumerate( options ):
            print( f"{shrt:<11}: {sh}" )
            print( f"{lng:<11}: {long_options[i]}" )
            print()
            print( f"{dscr:<11}: {explanation[i]}" )
            print( f"{sep:_<25}" )

    # Shows a list of molecules saved to the database by name
    def showList( self ) -> None:
        """
            Prints a list of the currently saved optimized values in the database

            Parameter
            ---------
            None

            Returns
            -------
            None
        """
        pass

    # Sets the molecule name in the class and checks if it exists
    def setMolName( self, mol_name ) -> bool:
        """
            Sets the molecule name in the class

            Parameter
            ---------
            mol_name : str
                molecular name in string format.

            Returns
            -------
            bool
                Whether the molecule already exists in the database or not
        """
        self.mol_name   = mol_name
        return self.mol_exists( mol_name )



if __name__ == "__main__":

    '''

        This file can be called directly but it shouldn't be.
        Try to use it as an imported class only.
    
    '''
    # Temporary test
    mol_name         = str(sys.argv[1])

    # Initiate the class to prepare the molecule
    OC              = OptimizerCalculator( mol_name = mol_name )
    OC.optimize()

    '''
    # list of command line arguments
    argList         = sys.argv[1:]
    
    # Options
    options         = "hmslO:"
    
    # Long options
    long_options    = ["help", "molecule_name=", "molecular_struct=", "list", "override_struct" ]

    # Setting this first allows for multiple orders to be considered by the user.
    opts_set        = [ False for _ in long_options ]
    
    try:
        # Parsing argument
        arguments, values   = getopt.getopt( argList, options, long_options )
        mol_name            = None
        mol_struct          = None
        override_struct     = False
        
        # checking each argument
        for currentArgument, currentValue in arguments:

            if currentArgument in ("-h", "--help"):
                opts_set[0] = True
                
            elif currentArgument in ("-m", "--molecule_name"):
                opts_set[1] = True
                mol_name    = str( currentValue )
                txt         = f"Molecule name set to {mol_name}"
                print( f"{txt:_^30}" )
                if OC.setMolName( mol_name ):
                    print( "Molecular structure found! Starting Optimization..." )
                else:
                    print( "Molecular structure not found. " )

            elif currentArgument in ("-s", "--molecular_struct"):
                opts_set[2] = True
                mol_struct    = os.path.join( str(currentValue) )

            elif currentArgument in ("-l", "--list"):
                opts_set[3] = True

            elif currentArgument in ("-O", "--override_struct"):
                override_struct = True
                
        # Show the help list
        if opts_set[0]:
            txt     = "Optimizer Help"
            print( f"{txt:_^30}" )
            OC.showHelp()
        # List all molecules in the database
        elif opts_set[3]:
            txt     = "Opening list of molecules"
            print( f"{txt:_^30}" )
            OC.showList()
        # Molecule name is set! Let's check it out
        elif opts_set[2]:
            # Check if the name exists in the database
            if OC.mol_exists( mol_name ) and not override_struct:
                # This is great, since we don't need to load a new structure
                OC.optimize()

            # Molecular structure not found in database. Check for user input of the structure
            else:
                # Ensure they actually input a file path
                if mol_struct is not None:
                    # Ensure that the input file can be used
                    if OC.check_struct( mol_struct ):
                        OC.optimize()
                    else:
                        print( f"The input molecular structure file {mol_struct} is not a valid file type or it can not be read!" )
                else:
                    print( "Molecular structure not found in database!" )
                    print( "You must input the molecular structure with the command -s or --molecular_struct" )
                    print( "For more information, run the script with -h or --help" )
    
    except getopt.error as err:
        # output error, and return with an error code
        print( str( err ) )

    # Optmize the settings!
    OC.optimize()'''