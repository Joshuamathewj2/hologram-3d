"""
src/renderer/objects.py
========================
3D object definitions using VBOs (Vertex Buffer Objects) via PyOpenGL.

WHY VBOs?
---------
Immediate mode (glBegin/glVertex3fv/glEnd) sends vertex data from CPU to GPU
per-frame with a function call per vertex.  This is deprecated in OpenGL 3+ and
has measurable overhead even in older contexts.

VBO approach:
 1. Upload vertex data to GPU memory *once* on construction.
 2. Each frame: bind VAO/VBO, draw.  No per-vertex CPU→GPU copy.
 3. Cleaner abstraction: each object owns its own GPU buffers.

For GLU quadrics (sphere, cylinder) the geometry is generated at runtime by
GLU — we keep those as-is since GLU doesn't expose raw vertices and the objects
are secondary in complexity.

API
---
class HologramObject:
    render() → None

Each concrete class implements render().
"""

import ctypes
import math
from typing import List, Tuple

import numpy as np
from OpenGL.GL import *


# ---------------------------------------------------------------------------
# Base class
# ---------------------------------------------------------------------------

class HologramObject:
    name: str = "Object"

    def render(self) -> None:
        raise NotImplementedError

    def cleanup(self) -> None:
        """Release GPU resources."""
        pass


# ---------------------------------------------------------------------------
# Generic wireframe VBO object
# ---------------------------------------------------------------------------

class WireframeVBO(HologramObject):
    """
    Generic wireframe object backed by a VBO.

    vertices : list of (x, y, z) tuples
    edges    : list of (i, j) index pairs
    color    : (r, g, b) floats [0, 1]
    """

    def __init__(
        self,
        name: str,
        vertices: List[Tuple[float, float, float]],
        edges: List[Tuple[int, int]],
        color: Tuple[float, float, float],
    ):
        self.name = name
        self._color = color

        # Build a flat vertex array in edge-list order (each edge = 2 vertices)
        raw: List[float] = []
        for (i, j) in edges:
            v0 = vertices[i]
            v1 = vertices[j]
            raw.extend(v0)
            raw.extend(v1)

        self._vertex_count = len(edges) * 2
        arr = np.array(raw, dtype=np.float32)

        # Upload to GPU
        self._vbo = glGenBuffers(1)
        glBindBuffer(GL_ARRAY_BUFFER, self._vbo)
        glBufferData(GL_ARRAY_BUFFER, arr.nbytes, arr, GL_STATIC_DRAW)
        glBindBuffer(GL_ARRAY_BUFFER, 0)

    def render(self) -> None:
        glColor3f(*self._color)
        glEnableClientState(GL_VERTEX_ARRAY)
        glBindBuffer(GL_ARRAY_BUFFER, self._vbo)
        glVertexPointer(3, GL_FLOAT, 0, None)
        glDrawArrays(GL_LINES, 0, self._vertex_count)
        glBindBuffer(GL_ARRAY_BUFFER, 0)
        glDisableClientState(GL_VERTEX_ARRAY)

    def cleanup(self) -> None:
        glDeleteBuffers(1, [self._vbo])


# ---------------------------------------------------------------------------
# Object catalogue
# ---------------------------------------------------------------------------

def build_object_catalogue() -> List[HologramObject]:
    """
    Construct all objects.  Must be called AFTER OpenGL context is ready
    (VBO upload requires an active GL context).
    """
    objects: List[HologramObject] = []

    # 1. Cube
    _cube_verts = [
        ( 1,-1,-1),( 1, 1,-1),(-1, 1,-1),(-1,-1,-1),
        ( 1,-1, 1),( 1, 1, 1),(-1,-1, 1),(-1, 1, 1),
    ]
    _cube_edges = [
        (0,1),(0,3),(0,4),(2,1),(2,3),(2,7),
        (6,3),(6,4),(6,7),(5,1),(5,4),(5,7),
    ]
    objects.append(WireframeVBO("CUBE", _cube_verts, _cube_edges, (0, 1, 1)))

    # 2. Sphere
    rings = 16
    sectors = 16
    _sph_verts = []
    for r in range(rings + 1):
        lat = math.pi * r / rings - math.pi / 2
        y = math.sin(lat)
        r_xz = math.cos(lat)
        for s in range(sectors):
            lon = 2 * math.pi * s / sectors
            x = r_xz * math.cos(lon)
            z = r_xz * math.sin(lon)
            _sph_verts.append((x, y, z))
    
    _sph_edges = []
    for r in range(rings):
        for s in range(sectors):
            cur = r * sectors + s
            next_s = r * sectors + ((s + 1) % sectors)
            next_r = (r + 1) * sectors + s
            # Horizontal ring
            _sph_edges.append((cur, next_s))
            # Vertical line
            if r < rings:
                _sph_edges.append((cur, next_r))
    objects.append(WireframeVBO("SPHERE", _sph_verts, _sph_edges, (1, 0.5, 0)))

    # 3. Square Pyramid (current pyramid)
    _sqpyr_verts = [(0,1,0),(-1,-1,1),(1,-1,1),(1,-1,-1),(-1,-1,-1)]
    _sqpyr_edges = [(0,1),(0,2),(0,3),(0,4),(1,2),(2,3),(3,4),(4,1)]
    objects.append(WireframeVBO("SQUARE PYRAMID", _sqpyr_verts, _sqpyr_edges, (1, 0, 1)))

    # 4. Square Prism
    _sqpri_verts = [
        ( 0.6,-1,-0.6),( 0.6, 1,-0.6),(-0.6, 1,-0.6),(-0.6,-1,-0.6),
        ( 0.6,-1, 0.6),( 0.6, 1, 0.6),(-0.6,-1, 0.6),(-0.6, 1, 0.6),
    ]
    objects.append(WireframeVBO("SQUARE PRISM", _sqpri_verts, _cube_edges, (0, 1, 0.5)))

    # 5. Pentagonal Pyramid
    _pent_pyr_verts = [(0, 1, 0)]
    for i in range(5):
        a = 2 * math.pi * i / 5.0
        _pent_pyr_verts.append((math.cos(a), -1, math.sin(a)))
    _pent_pyr_edges = []
    for i in range(5):
        _pent_pyr_edges.append((0, i+1))
        _pent_pyr_edges.append((i+1, (i+1)%5 + 1))
    objects.append(WireframeVBO("PENTAGONAL PYRAMID", _pent_pyr_verts, _pent_pyr_edges, (1, 1, 0)))

    # 6. Pentagonal Prism
    _pent_pri_verts = []
    for y in [1.0, -1.0]:
        for i in range(5):
            a = 2 * math.pi * i / 5.0
            _pent_pri_verts.append((math.cos(a), y, math.sin(a)))
    _pent_pri_edges = []
    for i in range(5):
        _pent_pri_edges.append((i, (i+1)%5))
        _pent_pri_edges.append((i+5, (i+1)%5 + 5))
        _pent_pri_edges.append((i, i+5))
    objects.append(WireframeVBO("PENTAGONAL PRISM", _pent_pri_verts, _pent_pri_edges, (0.5, 0, 1)))

    # 7. Cone
    _cone_verts = [(0, 1, 0)]
    for i in range(16):
        a = 2 * math.pi * i / 16.0
        _cone_verts.append((math.cos(a), -1, math.sin(a)))
    _cone_edges = []
    for i in range(16):
        _cone_edges.append((0, i+1))
        _cone_edges.append((i+1, (i+1)%16 + 1))
    objects.append(WireframeVBO("CONE", _cone_verts, _cone_edges, (1, 0, 0.5)))

    # 8. Hexagonal Pyramid
    _hex_pyr_verts = [(0, 1, 0)]
    for i in range(6):
        a = 2 * math.pi * i / 6.0
        _hex_pyr_verts.append((math.cos(a), -1, math.sin(a)))
    _hex_pyr_edges = []
    for i in range(6):
        _hex_pyr_edges.append((0, i+1))
        _hex_pyr_edges.append((i+1, (i+1)%6 + 1))
    objects.append(WireframeVBO("HEXAGONAL PYRAMID", _hex_pyr_verts, _hex_pyr_edges, (0, 1, 0)))

    # 9. Hexagonal Prism
    _hex_pri_verts = []
    for y in [1.0, -1.0]:
        for i in range(6):
            a = 2 * math.pi * i / 6.0
            _hex_pri_verts.append((math.cos(a), y, math.sin(a)))
    _hex_pri_edges = []
    for i in range(6):
        _hex_pri_edges.append((i, (i+1)%6))             # top hex
        _hex_pri_edges.append((i+6, (i+1)%6 + 6))       # bottom hex
        _hex_pri_edges.append((i, i+6))                 # vertical sides
    objects.append(WireframeVBO("HEXAGONAL PRISM", _hex_pri_verts, _hex_pri_edges, (0, 0.5, 1)))

    # 10. Triangular Pyramid
    _tri_pyr_verts = [
        (0, 1, 0),                       # Apex
        (0, -1, 1),                      # Front
        (0.866, -1, -0.5),               # Bottom right
        (-0.866, -1, -0.5)               # Bottom left
    ]
    _tri_pyr_edges = [
        (0,1), (0,2), (0,3),   # Edges to apex
        (1,2), (2,3), (3,1)    # Base triangle
    ]
    objects.append(WireframeVBO("TRIANGULAR PYRAMID", _tri_pyr_verts, _tri_pyr_edges, (1, 0.5, 0.5)))

    # 11. Triangular Prism
    _tri_pri_verts = [
        (0, 1, 1), (-0.866, 1, -0.5), (0.866, 1, -0.5),
        (0, -1, 1), (-0.866, -1, -0.5), (0.866, -1, -0.5)
    ]
    _tri_pri_edges = [
        (0,1),(1,2),(2,0),
        (3,4),(4,5),(5,3),
        (0,3),(1,4),(2,5)
    ]
    objects.append(WireframeVBO("TRIANGULAR PRISM", _tri_pri_verts, _tri_pri_edges, (0, 1, 0)))

    return objects
