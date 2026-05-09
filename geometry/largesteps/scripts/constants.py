import os

ROOT_DIR = os.path.dirname(os.path.dirname(__file__))
GEOMETRY_DIR = os.path.dirname(ROOT_DIR)
VNC_DIR = os.path.dirname(GEOMETRY_DIR)
OUTPUT_DIR = os.path.realpath(os.path.join(GEOMETRY_DIR, "output")) # Change this if you want to output the data and figures elsewhere
SCENES_DIR = os.path.realpath(os.path.join(VNC_DIR, "scenes", "geometry"))
REMESH_DIR = os.path.join(ROOT_DIR, "ext/botsch-kobbelt-remesher-libigl/build")
BLEND_SCENE = os.path.join(SCENES_DIR, "render.blend")
BLENDER_EXEC = "blender" # Change this if you have a different blender installation
