
import numpy as np

# assuming 'arr' is your 3D array
arr = np.random.rand(10,10,10)  # replace this with your array

# Get the indices of elements > 0
indices = np.where(arr > 0)

print(indices)


'''import json 

missing = 0.2
atomic_values   = {}
covalent_radii = [
    missing,  # X
    0.31,  # H
    0.28,  # He
    1.28,  # Li
    0.96,  # Be
    0.84,  # B
    0.76,  # C
    0.71,  # N
    0.66,  # O
    0.57,  # F
    0.58,  # Ne
    1.66,  # Na
    1.41,  # Mg
    1.21,  # Al
    1.11,  # Si
    1.07,  # P
    1.05,  # S
    1.02,  # Cl
    1.06,  # Ar
    2.03,  # K
    1.76,  # Ca
    1.70,  # Sc
    1.60,  # Ti
    1.53,  # V
    1.39,  # Cr
    1.39,  # Mn
    1.32,  # Fe
    1.26,  # Co
    1.24,  # Ni
    1.32,  # Cu
    1.22,  # Zn
    1.22,  # Ga
    1.20,  # Ge
    1.19,  # As
    1.20,  # Se
    1.20,  # Br
    1.16,  # Kr
    2.20,  # Rb
    1.95,  # Sr
    1.90,  # Y
    1.75,  # Zr
    1.64,  # Nb
    1.54,  # Mo
    1.47,  # Tc
    1.46,  # Ru
    1.42,  # Rh
    1.39,  # Pd
    1.45,  # Ag
    1.44,  # Cd
    1.42,  # In
    1.39,  # Sn
    1.39,  # Sb
    1.38,  # Te
    1.39,  # I
    1.40,  # Xe
    2.44,  # Cs
    2.15,  # Ba
    2.07,  # La
    2.04,  # Ce
    2.03,  # Pr
    2.01,  # Nd
    1.99,  # Pm
    1.98,  # Sm
    1.98,  # Eu
    1.96,  # Gd
    1.94,  # Tb
    1.92,  # Dy
    1.92,  # Ho
    1.89,  # Er
    1.90,  # Tm
    1.87,  # Yb
    1.87,  # Lu
    1.75,  # Hf
    1.70,  # Ta
    1.62,  # W
    1.51,  # Re
    1.44,  # Os
    1.41,  # Ir
    1.36,  # Pt
    1.36,  # Au
    1.32,  # Hg
    1.45,  # Tl
    1.46,  # Pb
    1.48,  # Bi
    1.40,  # Po
    1.50,  # At
    1.50,  # Rn
    2.60,  # Fr
    2.21,  # Ra
    2.15,  # Ac
    2.06,  # Th
    2.00,  # Pa
    1.96,  # U
    1.90,  # Np
    1.87,  # Pu
    1.80,  # Am
    1.69,  # Cm
    missing,  # Bk
    missing,  # Cf
    missing,  # Es
    missing,  # Fm
    missing,  # Md
    missing,  # No
    missing,  # Lr
    missing,  # Rf
    missing,  # Db
    missing,  # Sg
    missing,  # Bh
    missing,  # Hs
    missing,  # Mt
]

# Jmol colors.  See: http://jmol.sourceforge.net/jscolors/#color_U
jmol_colors = [
    (1.000, 0.000, 0.000),  # None
    (1.000, 1.000, 1.000),  # H
    (0.851, 1.000, 1.000),  # He
    (0.800, 0.502, 1.000),  # Li
    (0.761, 1.000, 0.000),  # Be
    (1.000, 0.710, 0.710),  # B
    (0.565, 0.565, 0.565),  # C
    (0.188, 0.314, 0.973),  # N
    (1.000, 0.051, 0.051),  # O
    (0.565, 0.878, 0.314),  # F
    (0.702, 0.890, 0.961),  # Ne
    (0.671, 0.361, 0.949),  # Na
    (0.541, 1.000, 0.000),  # Mg
    (0.749, 0.651, 0.651),  # Al
    (0.941, 0.784, 0.627),  # Si
    (1.000, 0.502, 0.000),  # P
    (1.000, 1.000, 0.188),  # S
    (0.122, 0.941, 0.122),  # Cl
    (0.502, 0.820, 0.890),  # Ar
    (0.561, 0.251, 0.831),  # K
    (0.239, 1.000, 0.000),  # Ca
    (0.902, 0.902, 0.902),  # Sc
    (0.749, 0.761, 0.780),  # Ti
    (0.651, 0.651, 0.671),  # V
    (0.541, 0.600, 0.780),  # Cr
    (0.612, 0.478, 0.780),  # Mn
    (0.878, 0.400, 0.200),  # Fe
    (0.941, 0.565, 0.627),  # Co
    (0.314, 0.816, 0.314),  # Ni
    (0.784, 0.502, 0.200),  # Cu
    (0.490, 0.502, 0.690),  # Zn
    (0.761, 0.561, 0.561),  # Ga
    (0.400, 0.561, 0.561),  # Ge
    (0.741, 0.502, 0.890),  # As
    (1.000, 0.631, 0.000),  # Se
    (0.651, 0.161, 0.161),  # Br
    (0.361, 0.722, 0.820),  # Kr
    (0.439, 0.180, 0.690),  # Rb
    (0.000, 1.000, 0.000),  # Sr
    (0.580, 1.000, 1.000),  # Y
    (0.580, 0.878, 0.878),  # Zr
    (0.451, 0.761, 0.788),  # Nb
    (0.329, 0.710, 0.710),  # Mo
    (0.231, 0.620, 0.620),  # Tc
    (0.141, 0.561, 0.561),  # Ru
    (0.039, 0.490, 0.549),  # Rh
    (0.000, 0.412, 0.522),  # Pd
    (0.753, 0.753, 0.753),  # Ag
    (1.000, 0.851, 0.561),  # Cd
    (0.651, 0.459, 0.451),  # In
    (0.400, 0.502, 0.502),  # Sn
    (0.620, 0.388, 0.710),  # Sb
    (0.831, 0.478, 0.000),  # Te
    (0.580, 0.000, 0.580),  # I
    (0.259, 0.620, 0.690),  # Xe
    (0.341, 0.090, 0.561),  # Cs
    (0.000, 0.788, 0.000),  # Ba
    (0.439, 0.831, 1.000),  # La
    (1.000, 1.000, 0.780),  # Ce
    (0.851, 1.000, 0.780),  # Pr
    (0.780, 1.000, 0.780),  # Nd
    (0.639, 1.000, 0.780),  # Pm
    (0.561, 1.000, 0.780),  # Sm
    (0.380, 1.000, 0.780),  # Eu
    (0.271, 1.000, 0.780),  # Gd
    (0.188, 1.000, 0.780),  # Tb
    (0.122, 1.000, 0.780),  # Dy
    (0.000, 1.000, 0.612),  # Ho
    (0.000, 0.902, 0.459),  # Er
    (0.000, 0.831, 0.322),  # Tm
    (0.000, 0.749, 0.220),  # Yb
    (0.000, 0.671, 0.141),  # Lu
    (0.302, 0.761, 1.000),  # Hf
    (0.302, 0.651, 1.000),  # Ta
    (0.129, 0.580, 0.839),  # W
    (0.149, 0.490, 0.671),  # Re
    (0.149, 0.400, 0.588),  # Os
    (0.090, 0.329, 0.529),  # Ir
    (0.816, 0.816, 0.878),  # Pt
    (1.000, 0.820, 0.137),  # Au
    (0.722, 0.722, 0.816),  # Hg
    (0.651, 0.329, 0.302),  # Tl
    (0.341, 0.349, 0.380),  # Pb
    (0.620, 0.310, 0.710),  # Bi
    (0.671, 0.361, 0.000),  # Po
    (0.459, 0.310, 0.271),  # At
    (0.259, 0.510, 0.588),  # Rn
    (0.259, 0.000, 0.400),  # Fr
    (0.000, 0.490, 0.000),  # Ra
    (0.439, 0.671, 0.980),  # Ac
    (0.000, 0.729, 1.000),  # Th
    (0.000, 0.631, 1.000),  # Pa
    (0.000, 0.561, 1.000),  # U
    (0.000, 0.502, 1.000),  # Np
    (0.000, 0.420, 1.000),  # Pu
    (0.329, 0.361, 0.949),  # Am
    (0.471, 0.361, 0.890),  # Cm
    (0.541, 0.310, 0.890),  # Bk
    (0.631, 0.212, 0.831),  # Cf
    (0.702, 0.122, 0.831),  # Es
    (0.702, 0.122, 0.729),  # Fm
    (0.702, 0.051, 0.651),  # Md
    (0.741, 0.051, 0.529),  # No
    (0.780, 0.000, 0.400),  # Lr
    (0.800, 0.000, 0.349),  # Rf
    (0.820, 0.000, 0.310),  # Db
    (0.851, 0.000, 0.271),  # Sg
    (0.878, 0.000, 0.220),  # Bh
    (0.902, 0.000, 0.180),  # Hs
    (0.922, 0.000, 0.149),  # Mt
]

atomic_names = [
    '', 'Hydrogen', 'Helium', 'Lithium', 'Beryllium', 'Boron',
    'Carbon', 'Nitrogen', 'Oxygen', 'Fluorine', 'Neon', 'Sodium',
    'Magnesium', 'Aluminium', 'Silicon', 'Phosphorus', 'Sulfur',
    'Chlorine', 'Argon', 'Potassium', 'Calcium', 'Scandium',
    'Titanium', 'Vanadium', 'Chromium', 'Manganese', 'Iron',
    'Cobalt', 'Nickel', 'Copper', 'Zinc', 'Gallium', 'Germanium',
    'Arsenic', 'Selenium', 'Bromine', 'Krypton', 'Rubidium',
    'Strontium', 'Yttrium', 'Zirconium', 'Niobium', 'Molybdenum',
    'Technetium', 'Ruthenium', 'Rhodium', 'Palladium', 'Silver',
    'Cadmium', 'Indium', 'Tin', 'Antimony', 'Tellurium',
    'Iodine', 'Xenon', 'Caesium', 'Barium', 'Lanthanum',
    'Cerium', 'Praseodymium', 'Neodymium', 'Promethium',
    'Samarium', 'Europium', 'Gadolinium', 'Terbium',
    'Dysprosium', 'Holmium', 'Erbium', 'Thulium', 'Ytterbium',
    'Lutetium', 'Hafnium', 'Tantalum', 'Tungsten', 'Rhenium',
    'Osmium', 'Iridium', 'Platinum', 'Gold', 'Mercury',
    'Thallium', 'Lead', 'Bismuth', 'Polonium', 'Astatine',
    'Radon', 'Francium', 'Radium', 'Actinium', 'Thorium',
    'Protactinium', 'Uranium', 'Neptunium', 'Plutonium',
    'Americium', 'Curium', 'Berkelium', 'Californium',
    'Einsteinium', 'Fermium', 'Mendelevium', 'Nobelium',
    'Lawrencium', 'Rutherfordium', 'Dubnium', 'Seaborgium',
    'Bohrium', 'Hassium', 'Meitnerium' ]

# atomic_values

for i, rad in enumerate( covalent_radii ):
    atomic_values[i] = { 'name': atomic_names[i], 'radius': rad, 'color': jmol_colors[i] }

with open('convert.txt', 'w') as convert_file: 
     convert_file.write( json.dumps(atomic_values) )'''