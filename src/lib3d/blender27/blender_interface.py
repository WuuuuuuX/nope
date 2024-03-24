import os
from src.lib3d.blender27 import utils
import bpy
import math
import numpy as np


class BlenderInterface:
    def __init__(self, resolution=128, background_color=(0, 0, 0)):
        self.resolution = resolution

        # Delete the default cube (default selected)
        bpy.ops.object.delete()

        # Deselect all. All new object added to the scene will automatically selected.
        self.blender_renderer = bpy.context.scene.render
        self.blender_renderer.use_antialiasing = False
        self.blender_renderer.resolution_x = resolution
        self.blender_renderer.resolution_y = resolution
        self.blender_renderer.resolution_percentage = 100
        self.blender_renderer.image_settings.file_format = (
            "PNG"  # set output format to .png
        )

        self.blender_renderer.alpha_mode = "SKY"
        world = bpy.context.scene.world
        # world.use_nodes = True
        # bg = world.node_tree.nodes['Background']
        # bg.inputs[0].default_value[:3] = background_color
        # bg.inputs[1].default_value = 0.0

        world.horizon_color = background_color
        world.light_settings.use_environment_light = True
        world.light_settings.environment_color = "SKY_COLOR"
        world.light_settings.environment_energy = 1.0

        # bpy.ops.object.light_add(type="AREA")
        lamp1 = bpy.data.lamps["Lamp"]
        lamp1.type = "SUN"
        lamp1.shadow_method = "NOSHADOW"
        lamp1.use_specular = False
        lamp1.energy = 1.0

        bpy.ops.object.lamp_add(type="SUN")
        lamp2 = bpy.data.lamps["Sun"]
        lamp2.shadow_method = "NOSHADOW"
        lamp2.use_specular = False
        lamp2.energy = 1.0
        bpy.data.objects["Sun"].rotation_euler = bpy.data.objects["Lamp"].rotation_euler
        bpy.data.objects["Sun"].rotation_euler[0] += 180

        bpy.ops.object.lamp_add(type="SUN")
        lamp2 = bpy.data.lamps["Sun.001"]
        lamp2.shadow_method = "NOSHADOW"
        lamp2.use_specular = False
        lamp2.energy = 0.3
        bpy.data.objects["Sun.001"].rotation_euler = bpy.data.objects[
            "Lamp"
        ].rotation_euler
        bpy.data.objects["Sun.001"].rotation_euler[0] += 90

        # Set up the camera
        self.camera = bpy.context.scene.camera
        self.camera.data.sensor_height = self.camera.data.sensor_width  # Square sensor
        utils.set_camera_focal_length_in_world_units(
            self.camera.data, 525.0 / 512 * resolution
        )  # Set focal length to a common value (kinect)

        bpy.ops.object.select_all(action="DESELECT")

    def import_mesh(self, fpath, scale=1.0, object_world_matrix=None):
        ext = os.path.splitext(fpath)[-1]
        if ext == ".obj":
            bpy.ops.import_scene.obj(filepath=str(fpath), split_mode="OFF")
        elif ext == ".ply":
            bpy.ops.import_mesh.ply(filepath=str(fpath))

        obj = bpy.context.selected_objects[0]
        utils.dump(bpy.context.selected_objects)

        if object_world_matrix is not None:
            obj.matrix_world = object_world_matrix

        bpy.ops.object.origin_set(type="ORIGIN_GEOMETRY", center="BOUNDS")
        bounding_box = obj.bound_box
        center_box = np.mean(bounding_box, axis=0)
        obj.location = -center_box
        # obj.location = (0.0, 0.0, 0.0)  # center the bounding box!
        obj.rotation_euler[0] = math.pi / 2
        if scale != 1.0:
            bpy.ops.transform.resize(value=(scale, scale, scale))

        # Disable transparency & specularities
        M = bpy.data.materials
        for i in range(len(M)):
            M[i].use_transparency = False
            M[i].specular_intensity = 0.0

        # Disable texture interpolation
        T = bpy.data.textures
        for i in range(len(T)):
            try:
                T[i].use_interpolation = False
                T[i].use_mipmap = False
                T[i].use_filter_size_min = True
                T[i].filter_type = "BOX"
            except:
                continue

    def render(
        self,
        output_dir,
        blender_cam2world_matrices,
        write_cam_params=False,
    ):
        # if write_cam_params:
        #     K = utils.get_calibration_matrix_K_from_blender(self.camera.data)
        for name_pose in ["query", "ref"]:
            for i in range(len(blender_cam2world_matrices[name_pose])):
                self.camera.matrix_world = blender_cam2world_matrices[name_pose][i]
                self.blender_renderer.filepath = os.path.join(
                    output_dir, "%06d_%s.png" % (i, name_pose)
                )
                bpy.ops.render.render(write_still=True)

        # Remember which meshes were just imported
        meshes_to_remove = []
        for ob in bpy.context.selected_objects:
            meshes_to_remove.append(ob.data)

        bpy.ops.object.delete()

        # Remove the meshes from memory too
        for mesh in meshes_to_remove:
            bpy.data.meshes.remove(mesh)
