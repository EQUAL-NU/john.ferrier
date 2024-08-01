'''

    Name:           Blender Builder 

    Description:    Quite simply, this script generates files necessary for visualizing simulations in
                    Blender. While this script only creates the files, future work would include
                    generating the actual Blender files and rendering them in a headless mode (for cluster use)

    Date:           22 February 2024

    Author:         John Ferrier, NEU Physics

    Conda_env:      VDB (Python 3.11)

'''

import subprocess


class BuildBlender:

    def __init__(self) -> None:
        pass

    # Runs the python script
    def run_script_in_blender( blender_path:str = "blender", script_path:str = "", iter_dir:str = "", repeat_int:int = 3, anim_dur:int = 120, anim_fps:int = 60, sliced:int = 0 ):
        # Run the script in Blender
        # iter_dir"     str     # The directory with the iterated wave function files. i.e. /path/to/files
        # repeat_int:   int     # How many times to repeat the domain in xy. i.e. 3 -> 3x3x1 repeat
        # anim_dur:     int     # How long in seconds that the animation should exist for. i.e. 60 -> 1 minute
        # anim_fps:     int     # The Frames Per Second of the animation. i.e. 24 -> 24 fps
        # sliced:       int     # Whether or not to slice the wave functions along a calculated axis. i.e. 0->False, 1->True
        subprocess.run( [ blender_path, '--background', '--python', script_path, iter_dir, repeat_int, anim_dur, anim_fps, sliced ] )



if __name__ == "__main__":

    BB  = BuildBlender()

    # Blender is usually at this place for Windows computers
    blender_path    = "C:\\Program Files\\Blender Foundation\\Blender 4.2\\blender.exe"

    # Script is in my Northeastern OneDrive
    script_path     = "C:\\Users\\jpfer\\OneDrive - Northeastern University\\General\\Data Storage\\John Ferrier\\Code\\Python\\Blender\\DFT_Renderer.py"

    # Maybe as a test, just sent the CH4 path?
    iter_dir        = "C:\\Users\\jpfer\\OneDrive - Northeastern University\\General\\Data Storage\\John Ferrier\\Code\\DFT\\molecules\\CH4\\pw\\dzp\\PBE\\680\\h_0.18\\vac_6.0\\Wavefunctions"

    # For testing, let's not repeat
    repeat_int      = 1

    # Animation can be short
    anim_dur        = 60

    # FPS gotta look smooth
    anim_fps        = 60

    # And don't slice the values for now (creating electron shells in VDB files)
    sliced          = 0

    BB.run_script_in_blender( blender_path, script_path, iter_dir, repeat_int, anim_dur, anim_fps, sliced )


