from ase import Atoms
from ase.build import bulk, diamond100, mx2, fcc111, graphene, molecule
from ase.spacegroup import crystal

naph_positions   = [
    [ 0.0000, 0.7076, 0.0000 ],
    [ 0.0000, -0.7076, 0.0000 ],
    [ 1.2250, 1.3944, 0.0000 ],
    [ 1.2250, -1.3944, 0.0000 ],
    [ -1.2250, 1.3943, 0.0000 ],
    [ -1.2250, -1.3943, 0.0000 ],
    [ 2.4327, 0.6959, 0.0000 ],
    [ 2.4327, -0.6958, 0.0000 ],
    [ -2.4327, 0.6958, 0.0000 ],
    [ -2.4327, -0.6958, 0.0000 ],
    [ 1.2489, 2.4821, 0.0000 ],
    [ 1.2489, -2.4822, 0.0000 ],
    [ -1.2489, 2.4821, 0.0000 ],
    [ -1.2489, -2.4822, 0.0000 ],
    [ 3.3732, 1.2391, 0.0000 ],
    [ 3.3732, -1.2390, 0.0000 ],
    [ -3.3732, 1.2390, 0.0000 ],
    [ -3.3732, -1.2390, 0.0000 ]
]
naph_symbols = [ 'C', 'C', 'C', 'C', 'C', 'C', 'C', 'C', 'C', 'C', 'H', 'H', 'H', 'H', 'H', 'H', 'H', 'H' ]


# Se8 Settings
# Values from Pubchem: https://pubchem.ncbi.nlm.nih.gov/compound/Cyclooctaselenium#section=2D-Structure 
Se8_xs      = [ 3.7071,
                2.7071,
                4.4142,
                2,
                4.4142,
                2,
                3.7071,
                2.7071 ]
Se8_ys      = [ -1.2071,
                -1.2071,
                -0.5,
                -0.5,
                0.5,
                0.5,
                1.2071,
                1.2071 ]

Se8_zs      = [ 0 for _ in range( len( Se8_xs ) ) ]
Se8_pos     = list( zip( Se8_xs, Se8_ys, Se8_zs ) )
Se8_symb    = [ 'Se' for _ in range( len( Se8_xs ) ) ]

# Values already stored in ASE  
Si_cryst    = bulk("Si")

Si_100      = Si_cryst.copy()
Si_010      = Si_cryst.copy()
Si_001      = Si_cryst.copy()

Si_100.rotate(90, 'x')
Si_010.rotate(90, 'y')
Si_001.rotate(90, 'z')


# Build out the alpha, beta quartz. Values from MatProj
# Different chiralities exist for alpha-quartz. But, give the structure (just oxygen up or down), the total energy shouldn't
# be too different. So, just going to use one.
# The chosen one is because of how closely it matches the orientation of the beta state

# https://next-gen.materialsproject.org/materials/mp-7000
alpha_quartz    = crystal(['Si', 'O'], [ (0.531089, 0.531089, 0.), ( 0.269223, 0.413394, 0.784891) ], spacegroup = 152, cellpar = [ 4.91, 4.91, 5.43, 90., 90., 120.], size = (1,1,1))

Si_alpha_100    = alpha_quartz.copy()
Si_alpha_010    = alpha_quartz.copy()
Si_alpha_001    = alpha_quartz.copy()

Si_alpha_100.rotate(90, 'x')
Si_alpha_010.rotate(90, 'y')
Si_alpha_001.rotate(90, 'z')


# https://next-gen.materialsproject.org/materials/mp-10851
beta_quartz     = crystal(['Si', 'O'], [ (0, .5, 0.333333), (0.582789, 0.791395, 5./6.) ], spacegroup = 181, cellpar = [ 5.06, 5.06, 5.54, 90., 90., 120.], size = (1,1,1) )

Si_beta_100     = beta_quartz.copy()
Si_beta_010     = beta_quartz.copy()
Si_beta_001     = beta_quartz.copy()


Si_beta_100.rotate(90, 'x')
Si_beta_010.rotate(90, 'y')
Si_beta_001.rotate(90, 'z')

molecules = {
    'MoO2':{
        'symbols': [ 'Mo', 'O', 'O' ],
        'positions': [  [ 2.866, 0., 0. ],              # Values taken from Pubchem - https://pubchem.ncbi.nlm.nih.gov/compound/29320
                        [ 3.732, 0.5, 0. ],
                        [ 2.00, -0.5, 0. ] ],
        'type': 'molecules',
        'object': Atoms( symbols = [ 'Mo', 'O', 'O' ], positions = [ [ 2.866, 0., 0. ], [ 3.732, 0.5, 0. ], [ 2.00, -0.5, 0. ] ] )
    },
    'MoO3':{
        'symbols': [ 'Mo', 'O', 'O', 'O' ],
        'positions': [  [ 2.866, 0.25, 0. ],            # Values taken from Pubchem - https://pubchem.ncbi.nlm.nih.gov/compound/14802
                        [ 3.732, 0.75, 0. ],
                        [ 2.00, 0.75, 0. ],
                        [ 2.866, -0.75, 0. ] ],
        'type': 'molecules',
        'object': Atoms( symbols = [ 'Mo', 'O', 'O', 'O' ], positions = [ [ 2.866, 0.25, 0. ], [ 3.732, 0.75, 0. ], [ 2.00, 0.75, 0. ], [ 2.866, -0.75, 0. ] ] )
    },
    'Se':{
        'symbols': [ 'Se' ],
        'positions': [  [ 0., 0., 0. ] ],
        'type': 'molecules',
        'object': Atoms( symbols = 'Se' )
    },
    'Se8':{
        'symbols': Se8_symb,
        'positions': Se8_pos,
        'type': 'molecules',
        'object': Atoms( symbols = Se8_symb, positions = Se8_pos )
    },
    'O2':{
        'symbols': [ 'O', 'O' ],
        'positions': [  [ 2., 0., 0. ],                 # Values taken from Pubchem - https://pubchem.ncbi.nlm.nih.gov/compound/977
                        [ 3., 0., 0. ] ],
        'type': 'molecules',
        'object': Atoms( symbols = [ 'O', 'O' ], positions = [ [ 2., 0., 0. ], [ 3., 0., 0. ] ] )
    },
    'O3':{
        'symbols': [ 'O', 'O', 'O' ],
        'positions': [  [ 2.866, -0.25, 0. ],
                        [ 2., 0.25, 0. ],                 # Values taken from Pubchem - https://pubchem.ncbi.nlm.nih.gov/compound/Ozone
                        [ 3.732, 0.25, 0. ] ],
        'type': 'molecules',
        'object': Atoms( symbols = [ 'O', 'O', 'O' ], positions = [ [ 2.866, -0.25, 0. ], [ 2., 0.25, 0. ], [ 3.732, 0.25, 0. ] ] )
    },
    'H2':{
        'symbols': [ 'H', 'H' ],
        'positions': [  [ 2., 0., 0. ],                 # Values taken from Pubchem - https://pubchem.ncbi.nlm.nih.gov/compound/783
                        [ 3., 0., 0. ] ],
        'type': 'molecules',
        'object': Atoms( symbols = [ 'H', 'H' ], positions = [ [ 2., 0., 0. ], [ 3., 0., 0. ] ] )
    },
    'H2O':{
        'symbols': [ 'H', 'H', 'O' ],
        'positions': [  [ 3.0739, 0.115, 0. ],              # Values taken from Pubchem - https://pubchem.ncbi.nlm.nih.gov/compound/962
                        [ 2., 0.115, 0. ],
                        [ 2.5369, -0.115, 0. ] ],
        'type': 'molecules',
        'object': Atoms( symbols = [ 'H', 'H', 'O' ], positions = [ [ 3.0739, 0.115, 0. ], [ 2., 0.115, 0. ], [ 2.5369, -0.115, 0. ] ] )

    },
    'C':{
        'symbols': [ 'C' ],
        'positions': [  [ 0., 0., 0. ] ],
        'type': 'molecules',
        'object': Atoms( symbols = 'C' )
    },
    'CH4':{
        'symbols': [ 'C', 'H', 'H', 'H', 'H' ],
        'positions': [],
        'type': 'molecules',
        'object': molecule( name = 'CH4' )
    },
    'MoSe2':{
        'symbols': [ 'Mo', 'Se', 'Se' ],
        'positions': [],
        'type': 'crystals',
        'object': mx2( formula = 'MoSe2', a = 3.29, thickness = 3.17, kind = '2H' )      # 'a' value taken from Materials Project = https://next-gen.materialsproject.org/materials/mp-1023934
    },
    'MoS2':{
        'symbols': [ 'Mo', 'S', 'S' ],
        'positions': [],
        'type': 'crystals',
        'object': mx2( formula = 'MoS2' )      # Default values are for MoS2
    },
    'Graphene':{
        'symbols': [ 'C', 'C' ],
        'positions': [],
        'type': 'crystals',
        'object': graphene()
    },
    'Copper_111':{
        'symbols': ['Cu'],
        'positions': [],
        'type': 'substrates',
        'object': fcc111( 'Cu', size = (1,1,1), periodic = True )
    },
    'Nickel_111':{
        'symbols': ['Ni'],
        'positions': [],
        'type': 'substrates',
        'object': fcc111( 'Ni', size = (1,1,1), periodic = True )
    },
    'Diamond_100':{
        'symbols': [ 'C', 'C' ],
        'positions': [],
        'type': 'substrates',
        'object': diamond100( symbol='C', size = (1,1,1), periodic = True )
    },
    'SiO2_Beta':{
        'symbols': [ 'Si', 'O',  'O' ],
        'positions': [],                                    
        'type': 'substrates',
        'object': beta_quartz
    },
    'SiO2_Beta_100':{
        'symbols': [ 'Si', 'O',  'O' ],
        'positions': [],                                    
        'type': 'substrates',
        'object': Si_beta_100
    },
    'SiO2_Beta_010':{
        'symbols': [ 'Si', 'O',  'O' ],
        'positions': [],                                    
        'type': 'substrates',
        'object': Si_beta_010
    },
    'SiO2_Beta_001':{
        'symbols': [ 'Si', 'O',  'O' ],
        'positions': [],                                    
        'type': 'substrates',
        'object': Si_beta_001
    },
    'SiO2_Alpha':{
        'symbols': [ 'Si', 'O',  'O' ],
        'positions': [],                                    
        'type': 'substrates',
        'object': alpha_quartz
    },
    'SiO2_Alpha_100':{
        'symbols': [ 'Si', 'O',  'O' ],
        'positions': [],                                    
        'type': 'substrates',
        'object': Si_alpha_100
    },
    'SiO2_Alpha_010':{
        'symbols': [ 'Si', 'O',  'O' ],
        'positions': [],                                    
        'type': 'substrates',
        'object': Si_alpha_010
    },
    'SiO2_Alpha_001':{
        'symbols': [ 'Si', 'O',  'O' ],
        'positions': [],                                    
        'type': 'substrates',
        'object': Si_alpha_001
    },
    'Si':{
        'symbols': [ 'Si' ],
        'positions': [],                                  
        'type': 'substrates',
        'object': Si_cryst
    },
    'Si_100':{
        'symbols': [ 'Si' ],
        'positions': [],                                  
        'type': 'substrates',
        'object': Si_100
    },
    'Si_010':{
        'symbols': [ 'Si' ],
        'positions': [],                                  
        'type': 'substrates',
        'object': Si_010
    },
    'Si_001':{
        'symbols': [ 'Si' ],
        'positions': [],                                  
        'type': 'substrates',
        'object': Si_001
    },
    'Naphthalene':{
        'symbols': naph_symbols,
        'positions': naph_positions,
        'type': 'molecules',
        'object': Atoms( 'C10H8', positions = naph_positions )
    },
    'Naphthalene_ring':{
        'symbols': naph_symbols[:-8],
        'positions': naph_positions[:-8],
        'type': 'molecules',
        'object': Atoms( 'C10', positions = naph_positions[:-8] )
    }
}
