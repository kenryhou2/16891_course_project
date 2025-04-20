#!/usr/bin/env python3
from matplotlib.patches import Circle, Rectangle
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import animation
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import matplotlib.colors
import trimesh, os
import pybullet as p
import pybullet_data
from drone import load_drone_model, create_simple_drone_mesh
import copy

Colors = ["green", "blue", "orange", "purple", "cyan", "magenta", "yellow", "red"]


class Animation3D:
    def __init__(self, my_map, starts, goals, paths):
        """
        Initialize the 3D animation

        Args:
            my_map: 3D grid map where True represents an obstacle and False represents free space
            starts: List of start locations (x, y, z)
            goals: List of goal locations (x, y, z)
            paths: List of paths, where each path is a list of (x, y, z) locations
        """

        self.my_map = my_map
        self.starts = starts
        self.goals = goals
        self.paths = paths if paths else []

        self.fig = plt.figure(figsize=(12, 10))
        self.ax = self.fig.add_subplot(111, projection="3d")

        # create boundary patch
        x_min, x_max = -0.5, len(my_map) - 0.5
        y_min, y_max = -0.5, len(my_map[0]) - 0.5
        z_min, z_max = -0.5, len(my_map[0][0]) - 0.5

        self.ax.set_xlim(x_min, x_max)
        self.ax.set_ylim(y_min, y_max)
        self.ax.set_zlim(z_min, z_max)

        self.ax.set_xlabel("X")
        self.ax.set_ylabel("Y")
        self.ax.set_zlabel("Z")

        # self.ax.set_box_aspect([1, 1, 1])

        # Similar to original, keep track of patches and artists
        self.artists = []
        self.agents = dict()
        self.agents_models = dict()
        self.agent_annotations = dict()
        self.agent_trajectories = dict()
        self.start_cubes = []
        self.goal_cubes = []

        # Plot obstacles as cubes
        self.plot_obstacles()

        # Create start and goal markers (cubes)
        self.plot_starts_and_goals()

        # create agents:
        self.T = 0

        self.init_agents()

        # Animate
        self.animation = animation.FuncAnimation(
            self.fig, self.animate_func, init_func=self.init_func, frames=int(self.T + 1) * 10, interval=250, blit=True
        )

    def create_cube(self, position, size=0.4, color="gray", alpha=0.5):
        """Create a 3D cube at the specified position"""
        x, y, z = position

        # Define the 8 vertices of the cube
        vertices = [
            [x - size / 2, y - size / 2, z - size / 2],
            [x + size / 2, y - size / 2, z - size / 2],
            [x + size / 2, y + size / 2, z - size / 2],
            [x - size / 2, y + size / 2, z - size / 2],
            [x - size / 2, y - size / 2, z + size / 2],
            [x + size / 2, y - size / 2, z + size / 2],
            [x + size / 2, y + size / 2, z + size / 2],
            [x - size / 2, y + size / 2, z + size / 2],
        ]

        # Define the 6 faces using indices of vertices
        faces = [
            [vertices[0], vertices[1], vertices[2], vertices[3]],  # bottom
            [vertices[4], vertices[5], vertices[6], vertices[7]],  # top
            [vertices[0], vertices[1], vertices[5], vertices[4]],  # front
            [vertices[2], vertices[3], vertices[7], vertices[6]],  # back
            [vertices[0], vertices[3], vertices[7], vertices[4]],  # left
            [vertices[1], vertices[2], vertices[6], vertices[5]],  # right
        ]

        # Create a Poly3DCollection
        collection = Poly3DCollection(faces, alpha=alpha, linewidths=1, edgecolors="black")
        collection.set_facecolor(color)

        # Add the cube to the plot
        return self.ax.add_collection3d(collection)

    def plot_obstacles(self):
        """Plot obstacles as cubes"""
        for i in range(len(self.my_map)):
            for j in range(len(self.my_map[0])):
                for k in range(len(self.my_map[0][0])):
                    if self.my_map[i][j][k]:
                        self.create_cube((i, j, k), size=0.8, color="gray", alpha=0.25)

    def plot_starts_and_goals(self):
        """Plot start and goal locations as cubes"""
        for i, (start, goal) in enumerate(zip(self.starts, self.goals)):
            color = Colors[i % len(Colors)]

            # Start location (cube)
            start_cube = self.create_cube(start, size=0.6, color=color, alpha=0.2)
            self.start_cubes.append(start_cube)

            # Goal location (cube)
            goal_cube = self.create_cube(goal, size=0.6, color=color, alpha=0.2)
            self.goal_cubes.append(goal_cube)

            # Add labels
            self.ax.text(start[0], start[1], start[2] + 0.5, f"S{i}", color=color, fontweight="bold", ha="center")
            self.ax.text(goal[0], goal[1], goal[2] + 0.5, f"G{i}", color=color, fontweight="bold", ha="center")

    def init_agents(self):
        """Initialize agent representations"""

        drone_mesh = load_drone_model(urdf_path="assets/quadrotor.urdf", scale=2.0)
        self.drone_mesh = drone_mesh

        for i, path in enumerate(self.paths):
            if not path:
                continue

            self.T = max(self.T, len(path) - 1)  # Maximum timestep across all agents
            color = Colors[i % len(Colors)]  # Get color for this agent

            # Create a sphere marker and store reference to the sphere
            sphere = self.ax.scatter(path[0][0], path[0][1], path[0][2], color=color, s=800, alpha=0.08, edgecolors="black")
            self.agents[i] = sphere

            if drone_mesh is not None:
                mesh_copy = copy.deepcopy(drone_mesh)
                mesh_copy.apply_translation([path[0][0], path[0][1], path[0][2]])

                # Create a Poly3DCollection from the mesh
                vertices = mesh_copy.vertices
                faces = mesh_copy.faces
                drone_poly = Poly3DCollection(vertices[faces], alpha=0.8, linewidths=0.5)
                drone_poly.set_facecolor(color)
                drone_poly.set_edgecolor("black")

                self.agents_models[i] = self.ax.add_collection3d(drone_poly)

            # Store original color
            self.agents[i].original_face_color = color

            # Add annotation
            annotation = self.ax.text3D(
                path[0][0], path[0][1], path[0][2] + 0.3, str(i), color="white", fontweight="bold", ha="center", va="center", zorder=100, fontsize=12
            )
            self.agent_annotations[i] = annotation

            # Path trajectory
            self.agent_trajectories[i] = self.ax.plot([], [], [], "-", color=color, alpha=0.5, linewidth=2)[0]

            # Add to artists collection
            self.artists.append(annotation)
            self.artists.append(self.agent_trajectories[i])

    def save(self, file_name, speed):
        """Save the animation to a file"""
        self.animation.save(file_name, writer='ffmpeg', fps=10 * speed, dpi=200, savefig_kwargs={"pad_inches": 0.5, "bbox_inches": "tight"})
        print(f"Animation saved to {file_name}")

    @staticmethod
    def show():
        plt.show()

    def init_func(self):
        """Initialize the animation"""
        self.ax.view_init(elev=25, azim=-38)

        return self.artists

    def animate_func(self, t):
        """Update function for animation at timestep t"""
        t = t / 10
        for k, path in enumerate(self.paths):
            if k in self.agents and path:
                # Get interpolated position
                pos = self.get_state(t, path)

                # Update agent position (sphere)
                self.agents[k]._offsets3d = ([pos[0]], [pos[1]], [pos[2]])

                if k in self.agents_models:
                    self.agents_models[k].remove()
                    # Create a new model at the updated position
                    if self.drone_mesh is not None:
                        mesh_copy = copy.deepcopy(self.drone_mesh)
                        mesh_copy.apply_translation([pos[0], pos[1], pos[2]])

                        vertices = mesh_copy.vertices
                        faces = mesh_copy.faces
                        drone_poly = Poly3DCollection(vertices[faces], alpha=0.8, linewidths=0.5)
                        drone_poly.set_facecolor(Colors[k % len(Colors)])
                        drone_poly.set_edgecolor("black")

                        self.agents_models[k] = self.ax.add_collection3d(drone_poly)

                # Update annotation position to match the sphere
                self.agent_annotations[k].set_position((pos[0], pos[1]))
                self.agent_annotations[k].set_3d_properties(pos[2] + 0.3)

                # Update agent trajectory
                if int(t) > 0:
                    # Show path until current point
                    path_slice = path[: min(int(t) + 2, len(path))]
                    xs = [p[0] for p in path_slice]
                    ys = [p[1] for p in path_slice]
                    zs = [p[2] for p in path_slice]
                    self.agent_trajectories[k].set_data(xs, ys)
                    self.agent_trajectories[k].set_3d_properties(zs)

        # Check for collisions
        self.check_collisions(t)

        # Update title with current time
        self.ax.set_title(f"Time: {t:.1f}")

        return self.artists

    def check_collisions(self, t):
        """Check for collisions between agents at time t"""
        # Reset all colors
        for i, agent in self.agents.items():
            color = Colors[i % len(Colors)]
            agent._facecolors[0] = np.array(matplotlib.colors.to_rgba(color))

        # Check agent-agent collisions
        for i in range(len(self.paths)):
            for j in range(i + 1, len(self.paths)):
                if i in self.agents and j in self.agents and self.paths[i] and self.paths[j]:
                    pos_i = self.get_state(t, self.paths[i])
                    pos_j = self.get_state(t, self.paths[j])

                    # Check if agents are too close in 3D space
                    distance = np.linalg.norm(np.array(pos_i) - np.array(pos_j))
                    if distance < 0.7:
                        # Mark collision by changing colors to red
                        self.agents[i]._facecolors[0] = np.array(matplotlib.colors.to_rgba("red"))
                        self.agents[j]._facecolors[0] = np.array(matplotlib.colors.to_rgba("red"))
                        print(f"COLLISION! (agent-agent) ({i}, {j}) at time {t:.1f}")

    @staticmethod
    def get_state(t, path):
        """
        Get interpolated state at time t
        This is directly based on the original code
        """
        if int(t) <= 0:
            return np.array(path[0])
        elif int(t) >= len(path):
            return np.array(path[-1])
        else:
            pos_last = np.array(path[int(t) - 1])
            pos_next = np.array(path[int(t)])
            pos = (pos_next - pos_last) * (t - int(t)) + pos_last
            return pos
