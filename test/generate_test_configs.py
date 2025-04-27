#!/usr/bin/env python3
import numpy as np
import PyKDL
import kdl_parser_py.urdf as kdl_urdf
import pybullet as p
import pybullet_data
import time
import threading
import queue
from urdf_parser_py.urdf import URDF

# ----------------------------------------------------------
# A) Build a KDL chain for FK + joint limits
# ----------------------------------------------------------
def load_robot_arm_chain(urdf_path, base_link='base_link', ee_link='ee_link'):
    robot = URDF.from_xml_file(urdf_path)
    ok, tree = kdl_urdf.treeFromUrdfModel(robot)
    if not ok:
        raise RuntimeError("Failed to parse URDF")
    chain     = tree.getChain(base_link, ee_link)
    fk_solver = PyKDL.ChainFkSolverPos_recursive(chain)

    limits = []
    for j in robot.joints:
        if j.type != 'fixed':
            limits.append((j.limit.lower, j.limit.upper))
    return fk_solver, chain, np.array(limits)


def fk_end_effector_z(q, fk_solver):
    q_kdl = PyKDL.JntArray(len(q))
    for i, qi in enumerate(q):
        q_kdl[i] = qi
    frame = PyKDL.Frame()
    fk_solver.JntToCart(q_kdl, frame)
    return frame.p[2]


# # ----------------------------------------------------------
# # B) Sample start/end configs above ground + within limits
# # ----------------------------------------------------------
# def sample_feasible_configs(n, joint_limits, fk_solver, chain, min_z=0.0):
#     """
#     Rejection‑sample n starts + n ends so that:
#       1) each q lies in joint_limits
#       2) every link (segment origin) in 'chain' has z > min_z
#     """
#     dof   = joint_limits.shape[0]
#     lows  = joint_limits[:,0]
#     highs = joint_limits[:,1]

#     # cache the segments once
#     segments = [chain.getSegment(i) for i in range(chain.getNrOfSegments())]

#     def sample_q():
#         return np.random.uniform(lows, highs)

#     def all_links_above(q):
#         # pack into KDL JointArray
#         q_kdl = PyKDL.JntArray(dof)
#         for i, qi in enumerate(q):
#             q_kdl[i] = qi

#         T = PyKDL.Frame()
#         joint_idx = 0
#         for seg in segments:
#             joint = seg.getJoint()
#             if joint.getTypeName() != "None":
#                 T = T * joint.pose(q_kdl[joint_idx])
#                 joint_idx += 1
#             T = T * seg.getFrameToTip()
#             if T.p[2] <= min_z:
#                 return False
#         return True

#     def is_feasible(q):
#         # joint limits
#         if np.any(q < lows) or np.any(q > highs):
#             return False
#         # full‑chain above ground
#         return all_links_above(q)

#     starts, ends = [], []
#     while len(starts) < n:
#         q0 = sample_q()
#         if is_feasible(q0):
#             starts.append(q0)
#     while len(ends) < n:
#         q1 = sample_q()
#         if is_feasible(q1):
#             ends.append(q1)
#     return starts, ends
def sample_feasible_configs(n, joint_limits, fk_solver, chain, min_z=0.0):
    """
    Rejection‑sample n starts + n ends so that:
      1) each q lies in joint_limits (with overrides for J1 and J2)
      2) every link (segment origin) in 'chain' has z > min_z
    """
    dof   = joint_limits.shape[0]
    # Original limits
    lows  = joint_limits[:,0].copy()
    highs = joint_limits[:,1].copy()
    # Override for shoulder_lift (joint index 1) and elbow (joint index 2)
    # Joint 1 (shoulder_lift) between -2.14 and -1.0 radians
    # Joint 2 (elbow) between -2.19 and  2.14 radians
    lows[1], highs[1] = -2.14, -1.0
    lows[2], highs[2] = -2.19,  2.14

    # cache the segments once
    segments = [chain.getSegment(i) for i in range(chain.getNrOfSegments())]

    def sample_q():
        return np.random.uniform(lows, highs)

    def all_links_above(q):
        # pack into KDL JointArray
        q_kdl = PyKDL.JntArray(dof)
        for i, qi in enumerate(q):
            q_kdl[i] = qi

        T = PyKDL.Frame()
        joint_idx = 0
        for seg in segments:
            joint = seg.getJoint()
            if joint.getTypeName() != "None":
                T = T * joint.pose(q_kdl[joint_idx])
                joint_idx += 1
            T = T * seg.getFrameToTip()
            if T.p[2] <= min_z:
                return False
        return True

    def is_feasible(q):
        # joint limits
        if np.any(q < lows) or np.any(q > highs):
            return False
        # full‑chain above ground
        return all_links_above(q)

    starts, ends = [], []
    while len(starts) < n:
        q0 = sample_q()
        if is_feasible(q0):
            starts.append(q0)
    while len(ends) < n:
        q1 = sample_q()
        if is_feasible(q1):
            ends.append(q1)
    return starts, ends



# ----------------------------------------------------------
# C) Overlap & density utilities
# ----------------------------------------------------------
def hemisphere_overlap_fraction(R, d):
    if d <= 0:    return 1.0
    if d >= 2*R:  return 0.0
    Vs    = np.pi*(4*R + d)*(2*R - d)**2/(12*d)
    Vh    = 0.5 * Vs
    Vhemi = 2/3 * np.pi * R**3
    return Vh / Vhemi

def average_workspace_overlap(bases, R):
    """
    Pairwise‑average overlap fraction for n fixed bases.
    """
    n = len(bases)
    if n < 2:
        return 0.0
    tot, cnt = 0.0, 0
    for i in range(n):
        for j in range(i+1, n):
            d = np.linalg.norm(bases[i] - bases[j])
            tot += hemisphere_overlap_fraction(R, d)
            cnt += 1
    return tot / cnt

def volumetric_density(n, avg_ov):
    """
    Total normalized overlap per arm = avg_ov * (n-1)
    """
    return avg_ov * (n - 1)

# ----------------------------------------------------------
# D) Equidistant base positions
# ----------------------------------------------------------
def generate_base_positions_equidistant(n, radius):
    """
    Place n bases equally spaced on a circle of given radius.
    """
    return np.array([
        [radius*np.cos(2*np.pi*i/n),
         radius*np.sin(2*np.pi*i/n),
         0.0]
        for i in range(n)
    ])

def find_circle_radius_for_overlap(n: int,
                                   R: float,
                                   target_ov: float,
                                   tol: float = 1e-3,
                                   max_radius: float = 5.0,
                                   max_iter: int = 40) -> float:
    """
    Find the circle radius at which n equidistant arms (reach R) have
    average pairwise overlap ≈ target_ov (in [0,1]), via bisection.
    
    Args:
      n:          number of arms
      R:          reach radius of each arm
      target_ov:  desired avg pairwise overlap fraction (0→1)
      tol:        acceptable error in overlap
      max_radius: upper bound on circle radius to search
      max_iter:   max bisection iterations
    
    Returns:
      circle_radius: the radius (in meters)
    """
    # helper: overlap at given radius
    def ov_at_radius(r):
        bases = generate_base_positions_equidistant(n, r)
        return average_workspace_overlap(bases, R)
    
    # bracket: at r=0, overlap=1; at r=2R, overlap=0
    lo, hi = 0.0, 2*R
    ov_lo, ov_hi = ov_at_radius(lo), ov_at_radius(hi)
    if not (ov_lo >= target_ov >= ov_hi):
        raise ValueError(f"Target {target_ov} not in [{ov_hi:.3f},{ov_lo:.3f}]")
    
    for _ in range(max_iter):
        mid = 0.5*(lo+hi)
        ov_mid = ov_at_radius(mid)
        if abs(ov_mid - target_ov) < tol:
            return mid
        if ov_mid > target_ov:
            lo = mid
        else:
            hi = mid
    return 0.5*(lo+hi)


# ----------------------------------------------------------
# E) Test‐case generator (equidistant)
# ----------------------------------------------------------
def generate_test_case_equidistant(n_arms, urdf_path,
                                   reach_radius=1.3,
                                   circle_radius=1.0):
    # 1) Load URDF, FK, limits
    fk_solver, chain, joint_limits = load_robot_arm_chain(urdf_path)

    # 2) Equidistant bases on circle
    bases = generate_base_positions_equidistant(n_arms, circle_radius)

    # 3) Compute average overlap
    avg_ov = average_workspace_overlap(bases, reach_radius)

    # 4) Sample feasible start/end
    starts, ends = sample_feasible_configs(
        n_arms,
        joint_limits,
        fk_solver,
        chain,
        min_z=0.0)

    # 5) Compute densities
    vol_den = volumetric_density(n_arms, avg_ov)

    return {
        'bases':       bases,
        'avg_overlap': avg_ov,
        'starts':      starts,
        'ends':        ends,
        'vol_density': vol_den,
        'fk_solver':   fk_solver,
        'joint_limits': joint_limits,
        'chain':       chain
    }

# def visualize(urdf_path, bases, starts, ends, R):
#     """
#     bases:    list of [x,y,0] base positions, shape (n,3)
#     starts:   list of n joint arrays (len = dof)
#     ends:     list of n joint arrays
#     R:        hemisphere radius (m)
#     """
#     # 1) Start the GUI
#     p.connect(p.GUI)
#     p.setAdditionalSearchPath(pybullet_data.getDataPath())
#     p.resetSimulation()
#     p.setGravity(0, 0, 0)
#     p.loadURDF("plane.urdf")

#     n = len(bases)
#     dof = len(starts[0])

#     # Draw 1 m axes at the origin of the very first base
#     base0 = bases[0]
#     p.addUserDebugLine(base0, [base0[0]+1, base0[1], base0[2]], [1,0,0], 2)  # X-axis
#     p.addUserDebugLine(base0, [base0[0], base0[1]+1, base0[2]], [0,1,0], 2)  # Y-axis
#     p.addUserDebugLine(base0, [base0[0], base0[1], base0[2]+1], [0,0,1], 2)  # Z-axis

#     # 2) Load start & end instances for each arm
#     start_ids = []
#     end_ids   = []
#     for i in range(n):
#         base_pos = bases[i]
#         sid = p.loadURDF(urdf_path, base_pos, [0,0,0,1], useFixedBase=True)
#         eid = p.loadURDF(urdf_path, base_pos, [0,0,0,1], useFixedBase=True)
#         start_ids.append(sid)
#         end_ids.append(eid)

#         # set joint states
#         for j in range(dof):
#             p.resetJointState(sid, j, starts[i][j])
#             p.resetJointState(eid, j, ends[i][j])

#         # zero out motors so poses stay fixed
#         for uid in (sid, eid):
#             for j in range(p.getNumJoints(uid)):
#                 qi = p.getJointState(uid, j)[0]
#                 p.setJointMotorControl2(
#                     bodyIndex=uid,
#                     jointIndex=j,
#                     controlMode=p.POSITION_CONTROL,
#                     targetPosition=qi,
#                     positionGain=0.1,
#                     velocityGain=1.0
#                 )

#         # apply transparency/color
#         for link in range(-1, p.getNumJoints(sid)):
#             p.changeVisualShape(sid, link, rgbaColor=[0, 0, 1, 0.3])
#         for link in range(-1, p.getNumJoints(eid)):
#             p.changeVisualShape(eid, link, rgbaColor=[1, 0, 0, 0.3])

#     # 3) Draw a wire-frame hemisphere shell
#     N_theta, N_phi = 20, 40
#     # build a grid of local hemisphere vertices
#     hemi = [[
#         (
#             R * np.sin(theta) * np.cos(phi),
#             R * np.sin(theta) * np.sin(phi),
#             R * np.cos(theta)
#         )
#         for phi in np.linspace(0, 2*np.pi, N_phi)
#     ] for theta in np.linspace(0, np.pi/2, N_theta)]

#     color = [0.8, 0.8, 0.0]  # yellow
#     lw    = 1               # thin lines

#     for base in bases:
#         # latitude circles
#         for ring in hemi:
#             for idx in range(len(ring)):
#                 a = (np.array(ring[idx]) + base).tolist()
#                 b = (np.array(ring[(idx+1) % len(ring)]) + base).tolist()
#                 p.addUserDebugLine(a, b, color, lineWidth=lw)
#         # meridian lines
#         for phi_idx in range(N_phi):
#             for theta_idx in range(N_theta-1):
#                 a = (np.array(hemi[theta_idx][phi_idx]) + base).tolist()
#                 b = (np.array(hemi[theta_idx+1][phi_idx]) + base).tolist()
#                 p.addUserDebugLine(a, b, color, lineWidth=lw)

#     # 4) Keep GUI alive
#     while p.isConnected():
#         time.sleep(1.0)
def visualize(
    urdf_path,
    bases,
    starts,
    ends,
    R,
    show_end=True,
    show_hemisphere=False
):
    """
    urdf_path      : path to the robot URDF
    bases          : list of [x,y,0] base positions, shape (n,3)
    starts, ends   : list of n joint–angle lists (len = dof)
    R              : hemisphere radius (m)
    show_end       : if False, the 'end' arm poses are not loaded or drawn
    show_hemisphere: if False, the overlap hemispheres are not drawn
    """
    # 1) Start PyBullet
    p.connect(p.GUI)
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    p.resetSimulation()
    p.setGravity(0, 0, 0)
    p.loadURDF("plane.urdf")

    n = len(bases)
    dof = len(starts[0])

   

    # 2) Load start (and optionally end) arms
    start_ids = []
    end_ids   = []
    for i in range(n):
        base_pos = bases[i]
        sid = p.loadURDF(urdf_path, base_pos, [0,0,0,1], useFixedBase=True)
        start_ids.append(sid)

        # set start joints
        for j in range(dof):
            p.resetJointState(sid, j, starts[i][j])
        # fix start pose
        for j in range(dof):
            qi = p.getJointState(sid, j)[0]
            p.setJointMotorControl2(sid, j, p.POSITION_CONTROL,
                                     targetPosition=qi,
                                     positionGain=0.1,
                                     velocityGain=1.0)
        # color start
        for link in range(-1, p.getNumJoints(sid)):
            p.changeVisualShape(sid, link, rgbaColor=[0, 0, 1, 0.3])

        if show_end:
            eid = p.loadURDF(urdf_path, base_pos, [0,0,0,1], useFixedBase=True)
            end_ids.append(eid)
            for j in range(dof):
                p.resetJointState(eid, j, ends[i][j])
            for j in range(dof):
                qi = p.getJointState(eid, j)[0]
                p.setJointMotorControl2(eid, j, p.POSITION_CONTROL,
                                         targetPosition=qi,
                                         positionGain=0.1,
                                         velocityGain=1.0)
            for link in range(-1, p.getNumJoints(eid)):
                p.changeVisualShape(eid, link, rgbaColor=[1, 0, 0, 0.3])

    # 3) Optionally draw the wireframe hemisphere
    if show_hemisphere:
        N_theta, N_phi = 20, 40
        hemi = [[
            (
                R * np.sin(theta) * np.cos(phi),
                R * np.sin(theta) * np.sin(phi),
                R * np.cos(theta)
            )
            for phi in np.linspace(0, 2*np.pi, N_phi)
        ] for theta in np.linspace(0, np.pi/2, N_theta)]

        color = [0.8, 0.8, 0.0]
        lw    = 1
        for base in bases:
            # latitude circles
            for ring in hemi:
                for idx in range(len(ring)):
                    a = (np.array(ring[idx]) + base).tolist()
                    b = (np.array(ring[(idx+1)%len(ring)]) + base).tolist()
                    p.addUserDebugLine(a, b, color, lineWidth=lw)
            # meridians
            for phi_idx in range(N_phi):
                for theta_idx in range(N_theta-1):
                    a = (np.array(hemi[theta_idx][phi_idx]) + base).tolist()
                    b = (np.array(hemi[theta_idx+1][phi_idx]) + base).tolist()
                    p.addUserDebugLine(a, b, color, lineWidth=lw)

    # 4) Keep the GUI alive until the user closes it
    while p.isConnected():
        time.sleep(1.0)

def print_joint_values(robot_ids, dof):
    """Query and print joint angles for each arm."""
    for i, uid in enumerate(robot_ids):
        angles = [p.getJointState(uid, j)[0] for j in range(dof)]
        print(f"Arm {i} joints: {angles}")

# def visualize_interactive(urdf_path, bases, start_qs, R):
#     """
#     urdf_path:  path to your URDF
#     bases:      list of [x,y,0] base positions, shape (n_arms,3)
#     start_qs:   list of n_arms joint vectors (len=dof) → initial pose
#     R:          reach‐radius (m) to draw hemisphere boundary
#     """
#     # --- 1) connect and reset ---
#     p.connect(p.GUI)
#     p.setAdditionalSearchPath(pybullet_data.getDataPath())
#     p.resetSimulation()
#     p.setGravity(0, 0, 0)
#     p.loadURDF("plane.urdf")

#     n_arms = len(bases)
#     dof    = len(start_qs[0])

#     # --- 2) draw reach hemisphere boundaries (circle on ground) ---
#     circle_pts = 64
#     for bx, by, _ in bases:
#         pts = []
#         angles = np.linspace(0, 2*np.pi, circle_pts, endpoint=True)
#         for θ in angles:
#             x = bx + R * np.cos(θ)
#             y = by + R * np.sin(θ)
#             pts.append((x, y, 0.001))  # slight lift so it shows over the plane
#         # connect point i to i+1
#         for i in range(circle_pts):
#             p.addUserDebugLine(pts[i], pts[(i+1) % circle_pts], [1,1,1])

#     # --- 3) load all arms at their start poses ---
#     robot_ids = []
#     for i, base in enumerate(bases):
#         uid = p.loadURDF(urdf_path, base, [0,0,0,1], useFixedBase=True)
#         robot_ids.append(uid)
#         # reset each joint to its start configuration
#         for j in range(dof):
#             p.resetJointState(uid, j, start_qs[i][j])

#     # --- 4) create sliders initialized at start_qs ---
#     slider_ids = []
#     # (you could replace [-π,π] with your robot’s actual limits)
#     low_limits  = [-np.pi]*dof
#     high_limits = [ np.pi]*dof
#     for i in range(n_arms):
#         for j in range(dof):
#             name = f"arm{i}_joint{j}"
#             init = start_qs[i][j]
#             sid  = p.addUserDebugParameter(name, low_limits[j], high_limits[j], init)
#             slider_ids.append((i, j, sid))

#     # --- 5) main loop: read sliders → update joint states → step sim ---
#     try:
#         while p.isConnected():
#             for (i, j, sid) in slider_ids:
#                 θ = p.readUserDebugParameter(sid)
#                 p.resetJointState(robot_ids[i], j, θ)
#             p.stepSimulation()
#             time.sleep(1/240.)
#     except KeyboardInterrupt:
#         p.disconnect()

# def visualize_interactive(urdf_path, bases, start_qs, R):
#     """
#     urdf_path: path to URDF
#     bases:     list of [x,y,0] base positions
#     start_qs:  list of initial joint arrays
#     R:         reach radius (m)
#     """
#     # --- Connect and reset sim ---
#     p.connect(p.GUI)
#     p.setAdditionalSearchPath(pybullet_data.getDataPath())
#     p.resetSimulation()
#     p.setGravity(0, 0, 0)
#     p.loadURDF("plane.urdf")

#     n_arms = len(bases)
#     dof    = len(start_qs[0])

#     # --- Draw reach circles on the ground ---
#     circle_pts = 64
#     for bx, by, _ in bases:
#         pts = []
#         angles = np.linspace(0, 2*np.pi, circle_pts, endpoint=True)
#         for θ in angles:
#             x = bx + R * np.cos(θ)
#             y = by + R * np.sin(θ)
#             pts.append((x, y, 0.001))
#         for i in range(circle_pts):
#             p.addUserDebugLine(pts[i], pts[(i+1) % circle_pts], [1,1,1])

#     # --- Load arms at start poses ---
#     robot_ids = []
#     for i, base in enumerate(bases):
#         uid = p.loadURDF(urdf_path, base, [0,0,0,1], useFixedBase=True)
#         robot_ids.append(uid)
#         for j in range(dof):
#             p.resetJointState(uid, j, start_qs[i][j])

#     # --- Create sliders initialized to start_qs ---
#     slider_ids = []
#     low_limits  = [-np.pi]*dof
#     high_limits = [ np.pi]*dof
#     for i in range(n_arms):
#         for j in range(dof):
#             name = f"arm{i}_joint{j}"
#             init = start_qs[i][j]
#             sid  = p.addUserDebugParameter(name, low_limits[j], high_limits[j], init)
#             slider_ids.append((i, j, sid))

#     # --- Main loop: update joints + print every 5s + step sim ---
#     last_print = time.time()
#     try:
#         while p.isConnected():
#             # 1) read sliders → update joints
#             for (i, j, sid) in slider_ids:
#                 θ = p.readUserDebugParameter(sid)
#                 p.resetJointState(robot_ids[i], j, θ)

#             # 2) check timer and print
#             now = time.time()
#             if now - last_print >= 5.0:
#                 print_joint_values(robot_ids, dof)
#                 last_print = now

#             # 3) step
#             p.stepSimulation()
#             time.sleep(1/240.)
#     except KeyboardInterrupt:
#         p.disconnect()

def start_input_listener():
    """
    Background thread that reads stdin.
    Whenever the user types 's' or 'e' (and hits Enter),
    that character is pushed into a Queue for the main loop to consume.
    """
    q = queue.Queue()

    def listen():
        while True:
            cmd = input().strip().lower()
            if cmd in ('s', 'e'):
                q.put(cmd)
    t = threading.Thread(target=listen, daemon=True)
    t.start()
    return q

def visualize_interactive(urdf_path, bases, reach_radius):
    # --- set up PyBullet ---
    p.connect(p.GUI)
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    p.resetSimulation()
    p.setGravity(0, 0, 0)
    NUM_JOINTS = 6

    # load arms
    uids = []
    for base in bases:
        uid = p.loadURDF(urdf_path, basePosition=base, useFixedBase=True)
        uids.append(uid)

    # sliders for exactly six joints per arm
    slider_ids = []
    for i, uid in enumerate(uids):
        sliders = []
        for j in range(NUM_JOINTS):
            info = p.getJointInfo(uid, j)
            lower, upper = info[8], info[9]
            if lower >= upper:
                lower, upper = -3.1416, 3.1416
            sliders.append(
                p.addUserDebugParameter(f"arm{i}_joint{j}", lower, upper, 0.0)
            )
        slider_ids.append(sliders)

    cmd_queue = start_input_listener()
    config_idx = 1
    saved_start = None

    print("Use sliders to move each of the 6 joints.")
    print("Type 's' + Enter → save START; 'e' + Enter → save END.")

    while True:
        # apply slider values to first 6 joints
        for i, uid in enumerate(uids):
            for j in range(NUM_JOINTS):
                q_target = p.readUserDebugParameter(slider_ids[i][j])
                p.resetJointState(uid, j, q_target)

        p.stepSimulation()

        # check for 's'/'e'
        try:
            cmd = cmd_queue.get_nowait()
        except queue.Empty:
            cmd = None

        if cmd == 's':
            saved_start = []
            for i, uid in enumerate(uids):
                qs = [p.getJointState(uid, j)[0] for j in range(NUM_JOINTS)]
                saved_start.append(qs)
                print(f"Arm {i} Start Config {config_idx}: {qs}")

        elif cmd == 'e':
            if saved_start is None:
                print("Warning: No START saved. Press 's' first.")
            else:
                for i, uid in enumerate(uids):
                    qs = [p.getJointState(uid, j)[0] for j in range(NUM_JOINTS)]
                    print(f"Arm {i} End   Config {config_idx}: {qs}")
                config_idx += 1
                saved_start = None

        time.sleep(0.01)

def visualize_interactive_dual(urdf_path, bases, starts, ends, R):
    """
    urdf_path: path to robot URDF
    bases:    list of [x,y,0] base positions, shape (n,3)
    starts:   list of n joint arrays (len = dof) for start poses
    ends:     list of n joint arrays (len = dof) for end poses
    R:        hemisphere radius (m)
    """
    # --- PyBullet init ---
    p.connect(p.GUI)
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    p.resetSimulation()
    p.setGravity(0,0,0)
    p.loadURDF("plane.urdf")

    n_arms = len(bases)
    dof    = len(starts[0])

    # --- Load and color start/end arms ---
    start_ids = []
    end_ids   = []
    for i in range(n_arms):
        base = bases[i].tolist()
        # Start arm (semi-transparent blue)
        sid = p.loadURDF(urdf_path, base, [0,0,0,1], useFixedBase=True)
        for j in range(dof):
            p.resetJointState(sid, j, starts[i][j])
        start_ids.append(sid)
        # End arm (semi-transparent red)
        eid = p.loadURDF(urdf_path, base, [0,0,0,1], useFixedBase=True)
        for j in range(dof):
            p.resetJointState(eid, j, ends[i][j])
        end_ids.append(eid)

    # Apply color/transparency
    for sid in start_ids:
        num_links = p.getNumJoints(sid)
        for link in range(-1, num_links):
            p.changeVisualShape(sid, link, rgbaColor=[0, 0, 1, 0.3])
    for eid in end_ids:
        num_links = p.getNumJoints(eid)
        for link in range(-1, num_links):
            p.changeVisualShape(eid, link, rgbaColor=[1, 0, 0, 0.3])

    # --- Create interactive sliders for start AND end angles ---
    slider_ids = []  # (arm_index, 'start'/'end', joint_index, slider_id)
    for i in range(n_arms):
        for j in range(dof):
            # start sliders
            sid = p.addUserDebugParameter(f"arm{i}_j{j}_start", -np.pi, np.pi, starts[i][j])
            slider_ids.append((i, 'start', j, sid))
            # end sliders
            eid = p.addUserDebugParameter(f"arm{i}_j{j}_end",   -np.pi, np.pi, ends[i][j])
            slider_ids.append((i, 'end', j, eid))

    # --- Precompute hemisphere points once ---
    pts = []
    N_theta, N_phi = 30, 30
    for ti in range(N_theta):
        theta = (ti/(N_theta-1)) * (np.pi/2)
        for pj in range(N_phi):
            phi = (pj/(N_phi-1)) * (2*np.pi)
            x = R * np.sin(theta) * np.cos(phi)
            y = R * np.sin(theta) * np.sin(phi)
            z = R * np.cos(theta)
            pts.append([x, y, z])

    # --- Main interactive loop ---
    while p.isConnected():
        # Update joint angles from sliders
        for (i, which, j, pid) in slider_ids:
            angle = p.readUserDebugParameter(pid)
            if which == 'start':
                p.resetJointState(start_ids[i], j, angle)
            else:
                p.resetJointState(end_ids[i], j, angle)

        # Draw/update hemisphere regions
        # (clear old, then add new) – PyBullet debug points cannot be removed, so draw once
        for base in bases:
            trans = (np.array(pts) + base).tolist()
            colors = [[1,1,0] for _ in trans]
            p.addUserDebugPoints(trans, colors, pointSize=2)

        p.stepSimulation()
        time.sleep(1/240.)


def compute_all_joint_positions(q, fk_solver, chain):
    dof = len(q)
    q_kdl = PyKDL.JntArray(dof)
    for i, qi in enumerate(q):
        q_kdl[i] = qi

    positions = []
    frame = PyKDL.Frame()
    for idx in range(chain.getNrOfSegments()):
        fk_solver.JntToCart(q_kdl, frame, idx)
        positions.append((float(frame.p[0]),
                          float(frame.p[1]),
                          float(frame.p[2])))
    return positions


if __name__ == "__main__":
    urdf = "../assets/ur5e/ur5e.urdf"
    n_arms       = 2
    reach_radius = 1.1      # UR10e reach [m]
    target_pct = 0.45      # target overlap fraction

    # find the radius
    circle_rad = find_circle_radius_for_overlap(
        n_arms,
        reach_radius,
        target_pct,
        tol=1e-3
    )
    print(f"Circle radius for {target_pct*100:.1f}% overlap: {circle_rad:.3f} m")

    tc = generate_test_case_equidistant(
        n_arms, urdf,
        reach_radius=reach_radius,
        circle_radius=circle_rad
    )

    fk = tc['fk_solver']
    chain = tc['chain']

    print(f"Equidistant circle radius: {circle_rad} m")
    print(f"Avg pairwise overlap:      {tc['avg_overlap']*100:5.1f}%")
    print(f"Volumetric density:        {tc['vol_density']:.2f}")
    print("Base positions:")
    # enumerate over bases, separating x, y, z with commas
    for i, base in enumerate(tc['bases']):
        print(f"  Arm {i} Base Position: [{base[0]:.3f}, {base[1]:.3f}, {base[2]:.3f}]")
        #print orientation of the base
        
    print("Start / End joint configs:")
    for i,(q0,q1) in enumerate(zip(tc['starts'], tc['ends'])):
        z0 = fk_end_effector_z(q0, tc['fk_solver'])
        z1 = fk_end_effector_z(q1, tc['fk_solver'])
        start_joint_pos = compute_all_joint_positions(q0, tc['fk_solver'], tc['chain'])
        end_joint_pos = compute_all_joint_positions(q1, tc['fk_solver'], tc['chain'])
        
    
    starts = [
        [0.0, -1.1243596076965332, -1.4550533294677734, 0.5621795654296875, 2.7116904258728027, -0.859804630279541], 
        [0.0, 1.78574800491333, -1.5873312950134277, 1.2897064685821533, 1.5873308181762695, -0.7275266647338867], 
        #  [0.0, -0.9259433746337891, -1.6534700393676758, 1.058220624923706, -0.5291104316711426, 1.7196087837219238] ,
        #   [0.0, 4.695854663848877, -1.4550533294677734, -0.3968327045440674, -1.7196087837219238, 1.3889145851135254] 
    ]
    ends = [
         [0.0, -3.836050033569336, -1.918025016784668, 1.2566368579864502, 3.3730788230895996, -4.166744232177734],
          [0.0, 3.3069396018981934, 0.2645554542541504, -0.7936656475067139, 4.629716396331787, 0.19841623306274414] ,
        #   [0.0, -2.976245641708374, -3.3730785846710205, 1.5542614459991455, 0.5291104316711426, -1.7196087837219238] ,
        #   [0.0, 2.843967914581299, -0.3968329429626465, 0.6944572925567627, -2.843968152999878, 0.33069419860839844]  
    ]
    visualize(urdf, tc['bases'], starts, ends, R=reach_radius)
    # visualize_interactive(urdf, tc['bases'], reach_radius)
    # visualize_interactive_dual(urdf, tc['bases'], starts, ends, reach_radius)