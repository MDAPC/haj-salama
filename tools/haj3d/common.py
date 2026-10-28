"""Shared helpers for building the 3D Haj Salama in Blender: meshes, toon materials, outlines."""
import bmesh
import bpy
import numpy as np


def srgb(*c):
    """sRGB 0-255 -> linear RGBA for material colours."""
    out = []
    for v in c[:3]:
        v = v / 255.0
        out.append(v / 12.92 if v <= 0.04045 else ((v + 0.055) / 1.055) ** 2.4)
    return (*out, 1.0)


def link(ob, parent=None):
    bpy.context.scene.collection.objects.link(ob)
    if parent is not None:
        ob.parent = parent
    return ob


def obj_from_bm(name, bm, parent=None, smooth=True):
    me = bpy.data.meshes.new(name)
    bm.to_mesh(me)
    bm.free()
    if smooth:
        me.shade_smooth()
    return link(bpy.data.objects.new(name, me), parent)


def get_co(me):
    a = np.empty(len(me.vertices) * 3)
    me.vertices.foreach_get("co", a)
    return a.reshape(-1, 3)


def set_co(me, co):
    me.vertices.foreach_set("co", np.asarray(co, np.float64).ravel())
    me.update()


def uv_sphere(name, radius, segs=32, rings=16, parent=None, loc=(0, 0, 0), scale=(1, 1, 1)):
    bm = bmesh.new()
    bmesh.ops.create_uvsphere(bm, u_segments=segs, v_segments=rings, radius=radius)
    ob = obj_from_bm(name, bm, parent)
    ob.location = loc
    ob.scale = scale
    return ob


def skin_object(name, verts, edges, radii, subsurf=2, parent=None, root=0):
    """Organic tube shapes (nose, limbs, fingers, brows) from a vertex skeleton.
    Returns a plain mesh object (modifiers applied), so it can be weighted and keyed."""
    me = bpy.data.meshes.new(name + "_skel")
    me.from_pydata([tuple(v) for v in verts], [tuple(e) for e in edges], [])
    tmp = bpy.data.objects.new(name + "_skel", me)
    bpy.context.scene.collection.objects.link(tmp)
    sk = tmp.modifiers.new("skin", "SKIN")
    sk.use_smooth_shade = True
    data = me.skin_vertices[0].data
    for i, r in enumerate(radii):
        data[i].radius = (r, r) if np.isscalar(r) else tuple(r)
        data[i].use_root = (i == root)
    if subsurf:
        ss = tmp.modifiers.new("sub", "SUBSURF")
        ss.levels = ss.render_levels = subsurf
    dg = bpy.context.evaluated_depsgraph_get()
    out = bpy.data.meshes.new_from_object(tmp.evaluated_get(dg))
    out.name = name
    bpy.data.objects.remove(tmp)
    bpy.data.meshes.remove(me)
    out.shade_smooth()
    return link(bpy.data.objects.new(name, out), parent)


META_K = 0.6726  # a lone metaball's surface sits at ~0.67 of its radius (threshold 0.6, stiffness 2)


def meta_mesh(name, elements, resolution=0.003, parent=None):
    """Blend soft shapes into one smooth surface.
    elements: (center, surface_radii(x,y,z), negative?) tuples."""
    mb = bpy.data.metaballs.new(name + "_mb")
    mb.resolution = mb.render_resolution = resolution
    mb.threshold = 0.6
    for c, r, *neg in elements:
        r = (r, r, r) if np.isscalar(r) else r
        rm = max(r)
        el = mb.elements.new(type="ELLIPSOID")
        el.co = c
        el.radius = rm / META_K
        el.size_x, el.size_y, el.size_z = (v / rm for v in r)
        el.use_negative = bool(neg and neg[0])
    tmp = bpy.data.objects.new(name + "_mb", mb)
    bpy.context.scene.collection.objects.link(tmp)
    dg = bpy.context.evaluated_depsgraph_get()
    me = bpy.data.meshes.new_from_object(tmp.evaluated_get(dg))
    me.name = name
    bpy.data.objects.remove(tmp)
    bpy.data.metaballs.remove(mb)
    me.shade_smooth()
    return link(bpy.data.objects.new(name, me), parent)


# ------------------------------------------------------------------ materials

def _toon_tree(m, color_socket_value=None, shade=0.55, soft=0.22, pos=0.36):
    """Diffuse light -> stepped ramp -> multiplied by base colour -> emission.
    Returns (nodes, links, multiply node) so callers can feed a computed colour."""
    nt = m.node_tree
    nt.nodes.clear()
    N, L = nt.nodes, nt.links
    diff = N.new("ShaderNodeBsdfDiffuse")
    s2r = N.new("ShaderNodeShaderToRGB")
    ramp = N.new("ShaderNodeValToRGB")
    ramp.color_ramp.interpolation = "EASE"
    el = ramp.color_ramp.elements
    el[0].position, el[0].color = pos, (shade, shade, shade, 1)
    el[1].position, el[1].color = pos + soft, (1, 1, 1, 1)
    mul = N.new("ShaderNodeVectorMath")
    mul.operation = "MULTIPLY"
    if color_socket_value is not None:
        mul.inputs[1].default_value = color_socket_value[:3]
    emit = N.new("ShaderNodeEmission")
    out = N.new("ShaderNodeOutputMaterial")
    L.new(diff.outputs[0], s2r.inputs[0])
    L.new(s2r.outputs["Color"], ramp.inputs["Fac"])
    L.new(ramp.outputs["Color"], mul.inputs[0])
    L.new(mul.outputs["Vector"], emit.inputs["Color"])
    L.new(emit.outputs[0], out.inputs["Surface"])
    return N, L, mul


def toon(name, rgb, shade=0.55, soft=0.22):
    m = bpy.data.materials.new(name)
    _toon_tree(m, srgb(*rgb), shade, soft)
    return m


def toon_attr(name, attr="paint", shade=0.6, soft=0.22):
    """Toon material whose base colour comes from a painted colour attribute."""
    m = bpy.data.materials.new(name)
    N, L, mul = _toon_tree(m, None, shade, soft)
    a = N.new("ShaderNodeAttribute")
    a.attribute_name = attr
    L.new(a.outputs["Color"], mul.inputs[1])
    return m


def paint(ob, rgb_array, attr="paint"):
    """Per-vertex sRGB 0-255 colours -> linear colour attribute."""
    c = np.asarray(rgb_array, float) / 255.0
    lin = np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)
    rgba = np.concatenate([lin, np.ones((len(lin), 1))], 1).astype(np.float32)
    ca = ob.data.color_attributes.new(attr, "FLOAT_COLOR", "POINT")
    ca.data.foreach_set("color", rgba.ravel())


def flat(name, rgb):
    m = bpy.data.materials.new(name)
    nt = m.node_tree
    nt.nodes.clear()
    e = nt.nodes.new("ShaderNodeEmission")
    e.inputs["Color"].default_value = srgb(*rgb)
    o = nt.nodes.new("ShaderNodeOutputMaterial")
    nt.links.new(e.outputs[0], o.inputs["Surface"])
    return m


_OUTLINE = {}


def outline(ob, thickness=0.003):
    """Inverted-hull ink line: a flipped, slightly fatter shell whose camera-facing side is culled."""
    if "m" not in _OUTLINE:
        m = flat("ink", (14, 10, 10))
        m.use_backface_culling = True
        _OUTLINE["m"] = m
    me = ob.data
    if len(me.materials) == 0:
        me.materials.append(None)
    me.materials.append(_OUTLINE["m"])
    mod = ob.modifiers.new("outline", "SOLIDIFY")
    mod.thickness = thickness
    mod.offset = 1
    mod.use_flip_normals = True
    mod.use_rim = False
    mod.material_offset = len(me.materials) - 1
    return mod


def assign(ob, mat):
    me = ob.data
    if len(me.materials) and me.materials[0] is None:
        me.materials[0] = mat
    elif len(me.materials) == 0:
        me.materials.append(mat)
    else:
        me.materials[0] = mat
