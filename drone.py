import trimesh, os
import pytest
import trimesh

import pybullet as p
import pybullet_data

from matplotlib.patches import Circle, Rectangle
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import animation
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import matplotlib.colors


def create_simple_drone_mesh(scale=0.2):
    """Create a simple drone mesh if URDF loading fails"""
    try:
        # Create a body sphere
        body = trimesh.primitives.Cylinder(radius=0.2 * scale, height=0.05 * scale, sections=8)
        body.visual.face_colors = [100, 100, 100, 255]

        # Create 4 rotors
        rotors = []
        for dx, dy in [(1, 1), (1, -1), (-1, 1), (-1, -1)]:
            rotor = trimesh.primitives.Cylinder(radius=0.1 * scale, height=0.06 * scale, sections=8)
            rotor.apply_translation([dx * 0.3 * scale, dy * 0.3 * scale, 0.1 * scale])
            rotor.visual.face_colors = [50, 50, 50, 255]
            rotors.append(rotor)

        parts = [body] + rotors
        combined = trimesh.util.concatenate(parts)
        return combined

    except Exception as e:
        print(f"Error creating simple drone mesh: {e}")
        return None


def load_drone_model(urdf_path="assets/drone.urdf", scale=0.2):
    """Load a drone URDF model file"""
    try:
        # Check if file exists
        if not os.path.exists(urdf_path):
            print(f"URDF file not found at {urdf_path}")
            return None

        # Initialize PyBullet in DIRECT mode (no visualization)
        physics_client = p.connect(p.DIRECT)

        # Load URDF
        model_id = p.loadURDF(urdf_path, useFixedBase=1)

        # Get all visual shapes
        visual_shapes = []
        for link_id in range(p.getNumJoints(model_id) + 1):  # +1 for base link
            # Get visual shape data
            for shape in p.getVisualShapeData(model_id, link_id):
                shape_type = shape[2]  # 0=GEOM_SPHERE, 1=GEOM_BOX, 2=GEOM_CAPSULE, 3=GEOM_CYLINDER, 4=GEOM_PLANE, 5=GEOM_MESH
                if shape_type == 5:  # MESH
                    mesh_scale = shape[3]
                    mesh_file = shape[4].decode("utf-8")  # Convert bytes to string
                    visual_shapes.append((mesh_file, mesh_scale))

        # Create trimesh objects from the shapes
        meshes = []
        for mesh_file, mesh_scale in visual_shapes:
            try:
                # Load the mesh file
                # Check if it's a local file or from pybullet_data
                if os.path.exists(mesh_file):
                    mesh = trimesh.load(mesh_file)
                else:
                    # It might be relative to URDF path
                    mesh_path = os.path.join(os.path.dirname(urdf_path), mesh_file)
                    if os.path.exists(mesh_path):
                        mesh = trimesh.load(mesh_path)
                    else:
                        # Try pybullet_data
                        pybullet_mesh_path = os.path.join(pybullet_data.getDataPath(), mesh_file)
                        if os.path.exists(pybullet_mesh_path):
                            mesh = trimesh.load(pybullet_mesh_path)
                        else:
                            print(f"Mesh file not found: {mesh_file}")
                            continue

                # Apply scaling
                mesh.apply_scale([mesh_scale[0], mesh_scale[1], mesh_scale[2]])
                meshes.append(mesh)

            except Exception as e:
                print(f"Error loading mesh {mesh_file}: {e}")

        # Disconnect PyBullet
        p.disconnect()

        # Combine all meshes into one
        if meshes:
            try:
                combined_mesh = trimesh.util.concatenate(meshes)
                # Center and scale the final mesh
                combined_mesh.apply_scale(scale)
                combined_mesh.apply_translation(-combined_mesh.centroid)
                return combined_mesh

            except Exception as e:
                print(f"Error combining meshes: {e}")
                return None
        else:
            # If no meshes were loaded, create a simple drone-like mesh
            print("No meshes found in URDF, creating a simple drone mesh")
            print()
            return create_simple_drone_mesh(scale / 2)

    except Exception as e:
        print(f"Error loading URDF model: {e}")
        return None
