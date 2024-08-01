import os
import sys
import numpy as np
from ase.units import kB
from ase.phonons import Phonons
from ase.vibrations import Vibrations
from pymatgen.io.ase import AseAtomsAdaptor
from gpaw import GPAW, restart, PW, FermiDirac
from pymatgen.symmetry.analyzer import PointGroupAnalyzer
from ase.thermochemistry import IdealGasThermo, CrystalThermo



class ThermoCalculator:

    def __init__( self, mol_obj, save_dir, mol_type, mol_name, mode, basis, xc, is_crystal, poten, 
                  vacuum        = 0.,
                  cryst_type    = 'molecules',
                  sub_size      = (1,1,1),
                  cryst_size    = (1,1,1),
                  N             = 3,
                  bandpath      = None ) -> None:

        
        # THESE VALUES CANNOT BE BLANK, AS THEY'RE USED TO SEARCH FOR OPTIMIZED VALUES!
        self.save_dir       = save_dir
        self.mol_type       = mol_type
        self.mol_name       = mol_name
        self.mode           = mode
        self.basis          = basis
        self.xc             = xc
        self.is_crystal     = is_crystal
        self.poten          = poten
        self.vacuum         = vacuum
        self.bandpath       = None

        # These values can be left blank
        self.cryst_type     = cryst_type
        self.subs_size      = sub_size
        self.crys_size      = cryst_size
        self.N              = N


        self.mol_base       = os.path.join( self.save_dir, self.mol_type, self.mol_name )
        self.dir_base       = os.path.join( self.mol_base, self.mode, self.basis, self.xc )

        # This should already have a calculator attached to it!
        self.mol            = mol_obj
        self.org_cell       = self.mol.get_cell().copy()
        
        # The main usage of this class
        self.therm          = None
        

        ##### Check the optimization has been completed already.

        # Check for energy cutoff
        opt_encut_file  = os.path.join( self.dir_base, 'Opt-ENCUT.npy' )
        if os.path.isfile( opt_encut_file ):
            print( "Opt-ENCUT found! Opening File..." )
            self.ENCUT   = int( np.load( opt_encut_file ) )
        else:
            print( f"Opt-ENCUT not found at {opt_encut_file}!" )
            self.stopThermoCalc()

        # Update the path
        self.dir_base = os.path.join( self.dir_base, f"{self.ENCUT}" )


        # Check grid spacing calcs
        if self.is_crystal:

            # check k-points
            opt_kpts_file = os.path.join( self.dir_base, 'Opt-KPTS.npy' )
            if os.path.isfile( opt_kpts_file ):
                print( "Opt-KPTS found! Opening File..." )
                self.KPTS   = int( np.load( opt_kpts_file ) )
            else:
                print( f"Opt-KPTS not found at {opt_kpts_file}!" )
                self.stopThermoCalc()

            # Update the path
            self.dir_base   = os.path.join( self.dir_base, f"kpts_{self.KPTS}" )

            # check vacuum
            if self.cryst_type == 'crystals':
                
                opt_vac_file    = os.path.join( self.dir_base, 'Opt-Vacuum.npy' )
                if os.path.isfile( opt_vac_file ):
                    print( "Opt-Vacuum found! Opening File..." )
                    self.vacuum   = round( float( np.load( opt_vac_file ) ), 1 )
                else:
                    print( f"Opt-Vacuum not found at {opt_vac_file}!" )
                    self.stopThermoCalc()

                self.dir_base    = os.path.join( self.dir_base, f"vac_{self.vacuum}" )

        else:

            # chech h values
            opt_h_file  = os.path.join( self.dir_base, 'Opt-h.npy' )
            if os.path.isfile( opt_h_file ):
                print( "Opt-h found! Opening File..." )
                self.H   = round( float( np.load( opt_h_file ) ), 2 )
            else:
                print( f"Opt-h not found at {opt_h_file}!" )
                self.stopThermoCalc()
            
            # Update path
            self.dir_base    = os.path.join( self.dir_base, f"h_{np.round(self.H,2)}" )

            # Check vacuum
            opt_vac_file    = os.path.join( self.dir_base, 'Opt-Vacuum.npy' )
            if os.path.isfile( opt_vac_file ):
                print( "Opt-Vacuum found! Opening File..." )
                self.vacuum   = round( float( np.load( opt_vac_file ) ), 1 )
            else:
                print( f"Opt-Vacuum not found at {opt_vac_file}!" )
                self.stopThermoCalc()

            self.dir_base    = os.path.join( self.dir_base, f"vac_{self.vacuum}" )
        
    # Stops the program
    def stopThermoCalc( self ):
        print( f"This molecular system ({self.mol_name}) cannot be found!" )
        print( "Please ensure you're optimizing the structure before attempting to run this class!" )
        sys.exit()

    # Just sets the calculator for the molecule
    def get_calculator( self,
                        ENCUT       = 500,
                        hund        = False,
                        h           = 0.2,
                        width       = kB*300.,
                        kpoints     = ( 3, 3, 3 ) ):
        
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

    # sets the vacuum of the molecular object
    def set_vacuum( self, V = None):

        v = V
        if v == None:
            v = self.vacuum

        if self.mol_type == 'substrates' and v > 0.:
            v = 0.

        # Reset the vacuum just in case
        if self.mol_type == 'crystals':
            self.mol.center( vacuum = v, axis = 2 )
        else:
            self.mol.center( vacuum = v )

        if self.mol_type == 'substrates':
            # Gotta reset the cell for substrates, since setting the vacuum to zero here
            # zeros out the lattice vectors. 
            self.vacuum = 0.
            self.mol.set_cell( self.org_cell )

    # Starts the vibrational analysis
    def vibrate( self ):

        # Build the directory path
        vib_type        = 'Vibrations'

        if self.is_crystal:
            vib_type    = 'Phonons'

        if self.cryst_type == 'crystals':
            self.dir_base    = os.path.join( self.dir_base, f"{self.crys_size[0]}x{self.crys_size[1]}x{self.crys_size[2]}" )

        if self.cryst_type == 'substrates':
            self.dir_base    = os.path.join( self.dir_base, f"{self.subs_size[0]}x{self.subs_size[1]}x{self.subs_size[2]}" )

        self.GPAW_LOC   = os.path.join( self.dir_base, 'GPAW' )
        self.TRAJ_LOC   = os.path.join( self.dir_base, 'Trajectories' )
        self.VIB_LOC    = os.path.join( self.dir_base, vib_type )



        ########### GIBBS
        #### Calculate the Gibbs( T, P )
        # Check if Gibbs/Helmholtz files already exist
        #print( "Looking for thermo files..." )
        #if not self.is_crystal:
        #    thermo_en_file = os.path.join(self.dir_base, 'Gibbs.npy' )
        #else:
        #   thermo_en_file = os.path.join(self.dir_base, f'Helmholtz_SUPERCELL_{self.N}x{self.N}x1.npy' )


    
        #print( f"Thermo files not found for {self.mol_name}!" )
        print( "Starting thermodynamic simulation" )

        # If the thermo_en_file doesn't exist but the file has been run before, 
        # there's a strong chance that the Vibration calculation didn't finish. 
        # Check for files in the VIB location and then remove the newest to allow it to start from where it left off.
        

        # Build the Thermos
        if not self.is_crystal: # molecule
            print( f"Calculating {self.mol_name} Vibrations" )
            self.get_calculator( ENCUT = self.ENCUT, hund = (len( self.mol )==1), h = self.H )

            # Even though this is already known, run poten again since there's an apparent bug with saving and reading the wave functions
            self.poten  = self.mol.get_potential_energy()

            # Set vacuum just in case
            self.set_vacuum( V = self.vacuum )

            #self.mol.calc.attach( self.mol.calc.write, 1, os.path.join( self.GPAW_LOC, f'{self.mol_name}_Vibrations.gpw' ), mode = 'all' )
            vib          = Vibrations( self.mol, name = self.VIB_LOC )
            #vib.clean(empty_files=True )
            vib.run()
            self.mol.calc.write( os.path.join( self.GPAW_LOC, f'{self.mol_name}_Vibrations.gpw' ), mode = 'all' )
            vib_en       = vib.get_energies()

            # Save the vibrational files for imagery
            vib.write_mode()

            # Save the JMOL for viewing there too, just in case
            vib.write_jmol()

            print( f"Calculating {self.mol_name} IdealGasThermo" )

            print( vib.summary() )
            print( vib_en )

            # Only keep real values
            real_vib_en     = np.real( vib_en )
            real_vib_en     = real_vib_en[ real_vib_en != 0. ]

            # Get symmetry
            if len( self.mol ) > 1:
                print( f"Calculating {self.mol_name} Symmetry" )
                PMMol   = AseAtomsAdaptor().get_molecule( self.mol, charge_spin_check = False )
                PGA     = PointGroupAnalyzer( PMMol )
                PG      = PGA.get_pointgroup()
                SS      = PGA.get_rotational_symmetry_number()
                print( f"{self.mol_name} Point Group = {PG}" )
            else:
                SS    = np.inf  # Symmetry of monatomic atoms is infinity
            print( f"{self.mol_name} Symmetry Number = {SS}" )

            geometry = 'monatomic'

            if len( self.mol ) == 2:
                geometry    = 'linear'
            elif len( self.mol ) > 2:
                geometry    = 'nonlinear'

            total_spin  = self.mol.calc.get_number_of_spins()/2.        # Gives 0,1,2. We need 0,0.5,1 for the calc. So, div by 2

            self.therm       = IdealGasThermo( vib_energies     = real_vib_en,
                                        potentialenergy    = self.poten,
                                        atoms              = self.mol,
                                        geometry           = geometry,
                                        symmetrynumber     = SS,            # D_∞h group, Value from Table 10.1 and Appendix B of C. Cramer “Essentials of Computational Chemistry”, 2nd Ed.
                                        spin               = total_spin )
        else:   # Crystal

            # Phonon analysis
            print( f'Running ({self.N},{self.N},x) supercell phonon calculation for {self.mol_name}' )

            # Since the supercell is a different size than the original size, we need a different calculator
            calc_kpt        = ( self.KPTS, self.KPTS, self.KPTS )
            if self.mol_type == 'crystals':
                kptsl       = list( calc_kpt )
                kptsl[2]    = 1
                calc_kpt    = tuple( kptsl )
            self.get_calculator( ENCUT = self.ENCUT, hund = False, kpoints = calc_kpt )
            
            # Reset the vacuum just in case
            if self.mol_type == 'substrates':

                # Despite calculating an optmimum vacuum, we need the vacuum here to be 0., as this is a bulk material
                self.set_vacuum( V = 0. )
            else:
                self.set_vacuum( V = self.vacuum )

            # Even though this is already known, run poten again since there's an apparent bug with saving and reading the wave functions
            self.poten  = self.mol.get_potential_energy()

            #self.mol.calc.attach( self.mol.calc.write, 1, os.path.join( self.GPAW_LOC, f'{self.mol_name}_PHONON_SUPERCELL_{self.N}x{self.N}x1.gpw' ), mode = 'all' )
            self.phonon_loc = ""
            if self.mol_type == 'crystals':
                self.phonon_loc      = os.path.join( self.VIB_LOC, f'SUPERCELL_{self.N}x{self.N}x1' )
                ph = Phonons( self.mol, self.mol.calc, supercell=(self.N, self.N, 1), delta = 0.05, name = self.phonon_loc, center_refcell = True )
            else:
                self.phonon_loc      = os.path.join( self.VIB_LOC, f'SUPERCELL_{self.N}x{self.N}x{self.N}' )
                ph = Phonons( self.mol, self.mol.calc, supercell=(self.N, self.N, self.N), delta = 0.05, name = self.phonon_loc, center_refcell = True )

            print( "Running Phonon calculations" )
            ph.run()
            self.mol.calc.write( os.path.join( self.GPAW_LOC, f'{self.mol_name}_PHONON_SUPERCELL_{self.N}x{self.N}x1.gpw' ), mode = 'all' )

            # Save Vibrational data for visualization
            if self.bandpath == None:
                path    = [[ 0,0,0 ],]
            else:
                path    = self.mol.bandpath( self.bandpath, self.mol.cell, npoints = 20 ) #len(self.bandpath) )
                path    = path.kpts
            
            ph.read( acoustic = True )

            #ph.write_modes( path, center = True )
            
            self.phonon_energies, self.phonon_DOS = ph.dos( kpts = (40, 40, 40), npts = 3000, delta=5e-4)

            # Calculate the Helmholtz free energy
            self.therm   = CrystalThermo(   phonon_energies = self.phonon_energies,
                                            phonon_DOS      = self.phonon_DOS,
                                            potentialenergy = self.poten )


        '''
        # Cycle through T, P to build Gibbs/Helmholtz Values
        if not self.is_crystal:
            print( f'Calculating Gibbs(T,P) for {self.mol_name}...' )
        else:
            print( f'Calculating Helmholtz(T) for {self.mol_name}' )

        Gibbs = np.zeros( (self.steps, self.steps), dtype = np.float64 )

        for j, t in enumerate( self.T ):
            if not self.is_crystal:
                for k, p in enumerate( self.P ):
                    Gibbs[j][k] = therm.get_gibbs_energy( t, p, verbose = False )
            else:
                Gibbs[j][:] = therm.get_helmholtz_energy( t, verbose = False )


        
        # Save the values
        T_P     = np.array( [ self.T, self.P ] )
        np.save( os.path.join(self.dir_base, 'Temp_Press.npy' ), T_P )
        if not self.is_crystal:
            print( f"Saving Gibbs data..." )
            np.save( os.path.join(self.dir_base, 'Gibbs.npy' ), np.array(Gibbs) )
        else:
            print( f"Saving Helmholtz data..." )
            if self.settings == "crystals":
                np.save( os.path.join(self.dir_base, f'Helmholtz_SUPERCELL_{self.N}x{self.N}x1.npy' ), np.array(Gibbs) )
            else:
                np.save( os.path.join(self.dir_base, f'Helmholtz_SUPERCELL_{self.N}x{self.N}x{self.N}.npy' ), np.array(Gibbs) ) '''
