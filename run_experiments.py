#!/usr/bin/python
import args_parser
import glob
from pathlib import Path
from cbs import CBSSolver

# from pbs import PBSSolver
# from independent import IndependentSolver
# from joint_state import JointStateSolver
# from prioritized import PrioritizedPlanningSolver

from visualize import Animation3D
from utils import get_sum_of_cost

SOLVER = "CBS"


def print_mapf_instance(my_map, starts, goals):
    print("Start locations")
    print_locations(my_map, starts)
    print("Goal locations")
    print_locations(my_map, goals)


def print_3d_map(my_map):
    """
    Print a 3D map layer by layer
    """
    depth = len(my_map[0][0])
    height = len(my_map)
    width = len(my_map[0])

    for z in range(depth):
        print(f"Layer {z}:")
        for x in range(height):
            for y in range(width):
                if my_map[x][y][z]:
                    print("@", end=" ")  # Obstacle
                else:
                    print(".", end=" ")  # Free space
            print()
        print()


def print_locations(my_map, locations):
    """
    Print agent locations on the map
    """
    depth = len(my_map[0][0])
    height = len(my_map)
    width = len(my_map[0])

    # Create a 3D grid to mark agent positions
    locations_map = [[[-1 for _ in range(depth)] for _ in range(width)] for _ in range(height)]

    for i in range(len(locations)):
        x, y, z = locations[i]
        locations_map[x][y][z] = i

    for z in range(depth):
        print(f"Layer {z}:")
        for x in range(height):
            for y in range(width):
                if locations_map[x][y][z] >= 0:
                    print(str(locations_map[x][y][z]), end=" ")
                elif my_map[x][y][z]:
                    print("@", end=" ")
                else:
                    print(".", end=" ")
            print()
        print()


# def import_mapf_instance(filename):
#     f = Path(filename)
#     if not f.is_file():
#         raise BaseException(filename + " does not exist.")
#     f = open(filename, "r")
#     # first line: #rows #columns
#     line = f.readline()
#     rows, columns = [int(x) for x in line.split(" ")]
#     rows = int(rows)
#     columns = int(columns)
#     # #rows lines with the map
#     my_map = []
#     for r in range(rows):
#         line = f.readline()
#         my_map.append([])
#         for cell in line:
#             if cell == "@":
#                 my_map[-1].append(True)
#             elif cell == ".":
#                 my_map[-1].append(False)
#     # #agents
#     line = f.readline()
#     num_agents = int(line)
#     # #agents lines with the start/goal positions
#     starts = []
#     goals = []
#     for a in range(num_agents):
#         line = f.readline()
#         sx, sy, gx, gy = [int(x) for x in line.split(" ")]
#         starts.append((sx, sy))
#         goals.append((gx, gy))
#     f.close()
#     return my_map, starts, goals


def import_mapf_instance(filename):
    """
    Import a 3D MAPF instance from a file

    File format:
    num_rows num_cols num_layers
    Layer 0 represented as a grid of . and @ characters
    ...
    Layer n-1 represented as a grid of . and @ characters
    num_agents
    sx sy sz gx gy gz (for each agent)
    """
    f = Path(filename)
    if not f.is_file():
        raise BaseException(filename + " does not exist.")

    with open(filename, "r") as f:
        # First line: #rows #columns #layers
        line = f.readline().strip()
        rows, columns, layers = [int(x) for x in line.split()]

        # Initialize 3D map
        my_map = [[[False for _ in range(layers)] for _ in range(columns)] for _ in range(rows)]

        # Read each layer
        for z in range(layers):
            for x in range(rows):
                line = f.readline().strip()
                for y, cell in enumerate(line):
                    if cell == "@":
                        my_map[x][y][z] = True  # Obstacle
                    elif cell == ".":
                        my_map[x][y][z] = False  # Free space

            # Skip any blank lines or comments between layers
            next_line = f.readline().strip()
            while next_line.startswith("#") or not next_line:
                next_line = f.readline().strip()

            # Push back the last read line if it's not a comment or blank
            if not next_line.startswith("#") and next_line:
                f.seek(f.tell() - len(next_line) - 1)

        # Number of agents
        line = f.readline().strip()
        num_agents = int(line)

        # Agent start/goal positions
        starts = []
        goals = []
        for _ in range(num_agents):
            line = f.readline().strip()
            sx, sy, sz, gx, gy, gz = [int(x) for x in line.split()]
            starts.append((sx, sy, sz))
            goals.append((gx, gy, gz))

    return my_map, starts, goals


if __name__ == "__main__":
    parser = args_parser.create_parser()
    args = parser.parse_args()

    for file in sorted(glob.glob(args.instance)):

        print("***Import an instance***")
        my_map, starts, goals = import_mapf_instance(file)
        print_mapf_instance(my_map, starts, goals)

        if args.solver == "CBS":
            print("***Run CBS***")
            cbs = CBSSolver(my_map, starts, goals)
            paths = cbs.find_solution()
        else:
            raise RuntimeError("Unknown solver!")

        if not args.batch:
            print("***Test paths on a simulation***")
            animation = Animation3D(my_map, starts, goals, paths)
            animation.save("output.mp4", 1.0)
            animation.show()
