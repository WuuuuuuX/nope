import argparse
import numpy as np
import os
import sys

root_repo = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(os.path.dirname(__file__))
sys.path.append(root_repo)
sys.path.append(os.path.dirname(root_repo))
from src.lib3d.blender27 import utils 
from src.lib3d.blender27 import blender_interface

p = argparse.ArgumentParser(
    description="Renders given obj file by rotation a camera around it."
)
p.add_argument(
    "--mesh_fpath",
    type=str,
    required=True,
    help="The path the output will be dumped to.",
)
p.add_argument(
    "--obj_dir",
    type=str,
    required=True,
    help="The path the output will be dumped to.",
)
p.add_argument(
    "--gpu_id",
    type=str,
    default="0",
    help="The path the output will be dumped to.",
)

p.add_argument(
    "--disable_output", action="store_true", help="Whether show output of blender"
)

argv = sys.argv
argv = sys.argv[sys.argv.index("--") + 1 :]

opt = p.parse_args(argv)
os.environ["CUDA_VISIBLE_DEVICES"] = opt.gpu_id
instance_name = opt.mesh_fpath.split("/")[-3]

renderer = blender_interface.BlenderInterface(resolution=512)

poses = np.load(os.path.join(opt.obj_dir, "poses.npz"))
intrinsic = np.array([[525, 0.0, 256], [0.0, 525, 256], [0.0, 0.0, 1.0]])
blender_poses = {}
for name_pose in ["query", "ref"]:
    blender_poses[name_pose] = [
        utils.cv_cam2world_to_bcam2world(np.linalg.inv(m)) for m in poses[name_pose]
    ]

obj_location = np.zeros((1, 3))
rot_mat = np.eye(3)
hom_coords = np.array([[0.0, 0.0, 0.0, 1.0]]).reshape(1, 4)
obj_pose = np.concatenate((rot_mat, obj_location.reshape(3, 1)), axis=-1)
obj_pose = np.concatenate((obj_pose, hom_coords), axis=0)

if opt.disable_output:
    # redirect output to log file
    logfile = os.path.join(opt.obj_dir, "render.log")
    open(logfile, "a").close()
    old = os.dup(1)
    sys.stdout.flush()
    os.close(1)
    os.open(logfile, os.O_WRONLY)

renderer.import_mesh(opt.mesh_fpath, scale=1.0, object_world_matrix=obj_pose)
renderer.render(
    opt.obj_dir,
    blender_poses,
    write_cam_params=True,
)

if opt.disable_output:
    # disable output redirection
    os.close(1)
    os.dup(old)
    os.close(old)
    os.system("rm {}".format(logfile))
