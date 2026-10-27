"""Shell-model helpers for the torsion examples: a box spar, Saint-Venant shear-flow loads,
and the twist, distortion, warping and bimoment measured from a shell solution.
"""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np

from laminate_modelling_package import FEAMesh


def box_spar(width, height, length, laminates, n_width=6, n_height=4, n_span=30, kinematics=None):
    """Quad8 shell box spar along z: caps (top, bottom) of ``width``, webs of ``height``.

    ``laminates`` maps "cap_top", "cap_bottom", "web_le", "web_te" to laminates (walls on
    their mid-surfaces). Returns an object with ``mesh``, ``node_sets`` (root, tip),
    ``element_sets`` and ``span_positions``, usable with ``ThinWalledSection.from_blade_mesh``.
    """
    from laminate_modelling_package import FSDTModel

    kin = kinematics or FSDTModel()
    b, h = width, height
    corners = [(-b / 2, -h / 2), (b / 2, -h / 2), (b / 2, h / 2), (-b / 2, h / 2)]
    names = ["cap_bottom", "web_te", "cap_top", "web_le"]
    counts = [n_width, n_height, n_width, n_height]
    ring = []  # counter-clockwise points round the section, quadratic spacing (2 per element)
    ring_wall = []
    for k in range(4):
        p, q = np.array(corners[k]), np.array(corners[(k + 1) % 4])
        m = 2 * counts[k]
        for i in range(m):
            ring.append(p + (q - p) * i / m)
            ring_wall.append(k)
    nr = len(ring)
    zs = np.linspace(0.0, length, 2 * n_span + 1)
    mesh = FEAMesh()
    nid = {}
    n = 1
    for j, z in enumerate(zs):
        for i in range(nr):
            if i % 2 and j % 2:
                continue
            mesh.add_node(n, (ring[i][0], ring[i][1], z))
            nid[(i, j)] = n
            n += 1
    elements = {name: [] for name in names}
    eid = 1
    for jj in range(n_span):
        j = 2 * jj
        for ii in range(0, nr, 2):
            i2 = (ii + 2) % nr
            conn = [nid[(ii, j)], nid[(i2, j)], nid[(i2, j + 2)], nid[(ii, j + 2)],
                    nid[(ii + 1, j)], nid[(i2, j + 1)], nid[(ii + 1, j + 2)], nid[(ii, j + 1)]]
            name = names[ring_wall[ii]]
            mesh.add_element(eid, conn, laminates[name], kin, element_type="quad8", material_axis=(0.0, 0.0, 1.0))
            elements[name].append(eid)
            eid += 1
    all_nodes = mesh.node_ids
    sets = {"root": [m for m in all_nodes if abs(mesh.node_coordinates(m)[2]) < 1e-12],
            "tip": [m for m in all_nodes if abs(mesh.node_coordinates(m)[2] - length) < 1e-12 * max(1, length)],
            "all": all_nodes}
    elements["all"] = mesh.element_ids
    return SimpleNamespace(mesh=mesh, node_sets=sets, element_sets=elements,
                           span_positions=np.linspace(0.0, length, n_span + 1),
                           nodes=lambda name: sets[name], elements=lambda name: elements[name])


def section_nodes(mesh, z, tol=1e-9):
    return [n for n in mesh.node_ids if abs(mesh.node_coordinates(n)[2] - z) <= tol * max(1.0, abs(z))]


def add_shear_flow_torque(case, mesh, section, z, torque):
    """Nodal forces at section ``z`` that carry ``torque`` as the Saint-Venant shear flows.

    ``section`` is the ``ThinWalledSection`` cut from the mesh at ``z``. Each wall's
    flow q b along its tangent is shared between its two end nodes, so the loads are
    self-equilibrated and add up to the torque exactly.
    """
    scale = torque / section.J_membrane if section.J_membrane > 0 else 0.0
    for k in range(len(section.walls)):
        f = 0.5 * section.shear_flow[k] * scale * section.length[k] * section.tangent[k]
        for p in (section.start[k], section.end[k]):
            m = section.mesh_nodes[p]
            case.add_nodal_load(m, 0, f[0])
            case.add_nodal_load(m, 1, f[1])
    return case


def shell_section_motion(solution, mesh, section, z):
    """In-plane displacements, warping and mean twist of the section at z from a shell solution.

    The mean twist is the least-squares rotation of the section nodes (uniform weights),
    the same measure as ``GBTSolution.mean_twist`` and an RBE3 reference node.
    """
    nodes = section_nodes(mesh, z)
    lookup = {m: i for i, m in enumerate(section.mesh_nodes)}
    P = section.points
    d = np.zeros((len(P), 3))
    for m in nodes:
        if m in lookup:
            d[lookup[m]] = solution.node_displacement(m)[:3]
    c = P.mean(axis=0)
    r = P - c
    ut = d[:, :2] - d[:, :2].mean(axis=0)
    twist = float(np.sum(r[:, 0] * ut[:, 1] - r[:, 1] * ut[:, 0]) / np.sum(r * r))
    return SimpleNamespace(displacement=d, twist=twist)


def _omega_at(section, xy):
    """Saint-Venant warping function at points (projected on the nearest wall)."""
    P = section.points
    out = np.zeros(len(xy))
    for n, p in enumerate(xy):
        best, val = np.inf, 0.0
        for k in range(len(section.walls)):
            a, b = P[section.start[k]], P[section.end[k]]
            t = np.clip(np.dot(p - a, b - a) / np.dot(b - a, b - a), 0.0, 1.0)
            dist = np.linalg.norm(a + t * (b - a) - p)
            if dist < best:
                best = dist
                val = (1 - t) * section.omega[section.start[k]] + t * section.omega[section.end[k]]
        out[n] = val
    return out


def shell_bimoment(results, mesh, section, z_rows):
    """Bimoment B = sum N_zz omega dA / dz over each element row (z_a, z_b) of the shell results.

    Returns the row mid-points and the bimoments.
    """
    g = results.gauss_points()
    omega = _omega_at(section, g.xyz[:, :2])
    zc, B = [], []
    for za, zb in z_rows:
        sel = (g.xyz[:, 2] >= za) & (g.xyz[:, 2] <= zb)
        zc.append(0.5 * (za + zb))
        B.append(float(np.sum(g.N[sel, 0] * omega[sel] * g.weight[sel]) / (zb - za)))
    return np.array(zc), np.array(B)


def shell_wall_axial(results, z_rows):
    """N_zz (local x = span) at the Gauss points of an element row, with their (x, y)."""
    g = results.gauss_points()
    za, zb = z_rows
    sel = (g.xyz[:, 2] >= za) & (g.xyz[:, 2] <= zb)
    return g.xyz[sel], g.N[sel, 0]


def add_rib(mesh, bcs, z, centre=(0.0, 0.0)):
    """A rib at section z: a reference node tied to the section's in-plane translations (RBE2).

    The section stays rigid in its plane and free to warp. Returns the reference node,
    whose rotation about z is the rib's twist and which takes a torque as a moment.
    """
    from laminate_modelling_package import add_reference_node, rigid_link

    ref = add_reference_node(mesh, (centre[0], centre[1], z))
    rigid_link(mesh, bcs, ref, section_nodes(mesh, z), dofs=(0, 1))
    bcs.fix([ref], [2, 3, 4])
    return ref


def loop_order(section, walls):
    """Walls of a closed loop in walking order, as (wall, forward) pairs, and the arc length at each start."""
    walls = list(walls)
    adj = {}
    for k in walls:
        for p in (section.start[k], section.end[k]):
            adj.setdefault(int(p), []).append(k)
    k0 = walls[0]
    order, p = [], int(section.start[k0])
    used = set()
    k = k0
    while True:
        fwd = int(section.start[k]) == p
        order.append((k, fwd))
        used.add(k)
        p = int(section.end[k]) if fwd else int(section.start[k])
        nxt = [w for w in adj[p] if w not in used]
        if not nxt:
            break
        k = nxt[0]
    s = np.concatenate([[0.0], np.cumsum([section.length[k] for k, _ in order])])
    return order, s


def contour_coordinate(section, order, s0, xy):
    """Arc length along a wall loop (from ``loop_order``) of the points ``xy`` (nearest wall)."""
    P = section.points
    out = np.zeros(len(xy))
    for n, p in enumerate(xy):
        best = (np.inf, 0.0)
        for (k, fwd), s_start in zip(order, s0[:-1]):
            a, b = (P[section.start[k]], P[section.end[k]]) if fwd else (P[section.end[k]], P[section.start[k]])
            t = np.clip(np.dot(p - a, b - a) / np.dot(b - a, b - a), 0.0, 1.0)
            dist = np.linalg.norm(a + t * (b - a) - p)
            if dist < best[0]:
                best = (dist, s_start + t * section.length[k])
        out[n] = best[1]
    return out
