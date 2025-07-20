import os
import trimesh
import numpy as np
import xml.dom.minidom
import os
import trimesh
import xml.etree.ElementTree as ET
from lpanlib.poselib.skeleton.skeleton3d import SkeletonTree, SkeletonState, SkeletonMotion


def create_sphere(pos, size, MESH_SIMPLIFY=True):
    if pos == '':
        pos = [0, 0, 0]
    else:
        pos = [float(x) for x in pos.split(' ')]
    R = np.identity(4)
    R[:3, 3] = np.array(pos).T
    mesh = trimesh.creation.icosphere(subdivisions=3, radius=float(size))
    mesh.apply_transform(R)

    if MESH_SIMPLIFY:
        face_count = 50
    else:
        face_count = 5000

    return mesh.simplify_quadric_decimation(face_count)

def create_capsule(from_to, size, MESH_SIMPLIFY=True):
    from_to = [float(x) for x in from_to.split(' ')]
    start_point = np.array(from_to[:3])
    end_point = np.array(from_to[3:])

    # 计算pos
    pos = (start_point + end_point) / 2.0

    # 计算rot
    # 用罗德里格公式, 由向量vec2求旋转矩阵
    vec1 = np.array([0, 0, 1.0])
    vec2 = (start_point - end_point)
    height = np.linalg.norm(vec2)
    vec2 = vec2 / np.linalg.norm(vec2)
    if vec2[2] != 1.0: # (如果方向相同时, 公式不适用, 所以需要判断一下)
        i = np.identity(3)
        v = np.cross(vec1, vec2)
        v_mat = [[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]]
        s = np.linalg.norm(v)
        c = np.dot(vec1, vec2)
        R_mat = i + v_mat + np.matmul(v_mat, v_mat) * (1 - c) / (s * s)
    else:
        R_mat = np.identity(3)

    # 做transform
    T = np.identity(4)
    T[0:3, 0:3] = R_mat
    T[0:3, 3] = pos.T
    mesh = trimesh.creation.capsule(height, float(size))
    mesh.apply_transform(T)

    if MESH_SIMPLIFY:
        face_count = 50
    else:
        face_count = 1000

    return mesh.simplify_quadric_decimation(face_count)

def create_box(pos, size, MESH_SIMPLIFY=True):
    if pos == '':
        pos = [0, 0, 0]
    else:
        pos = [float(x) for x in pos.split(' ')]
    
    size = [float(x) * 2 for x in size.split(' ')]
    
    R = np.identity(4)
    R[:3, 3] = np.array(pos).T
    mesh = trimesh.creation.box(size)
    mesh.apply_transform(R)

    if MESH_SIMPLIFY:
        face_count = 50
    else:
        face_count = 1000

    return mesh.simplify_quadric_decimation(face_count)

def parse_geom_elements_from_xml(xml_path, MESH_SIMPLIFY=True): # only support box, sphere, mesh, and capsule (fromto format)
    dom = xml.dom.minidom.parse(xml_path)
    root = dom.documentElement

    # support mesh type rigid body
    geoms = {}
    for info in root.getElementsByTagName('mesh'):
        name = info.getAttribute("name")
        file_path = os.path.join(os.path.dirname(xml_path), info.getAttribute("file"))
        geoms[name] = trimesh.load(file_path, process=False)

    body = root.getElementsByTagName('body')
    body_names = []
    body_meshes = []
    for b in body:
        name = b.getAttribute('name')
        child = b.childNodes

        mesh = []
        for c in child:
            if c.nodeType == 1:
                if c.nodeName == 'geom':
                    if c.getAttribute('type') == 'sphere':
                        size = c.getAttribute('size')
                        pos = c.getAttribute('pos')
                        mesh.append(create_sphere(pos, size, MESH_SIMPLIFY))
                    elif c.getAttribute('type') == 'box':
                        pos = c.getAttribute('pos')
                        size = c.getAttribute('size')
                        mesh.append(create_box(pos, size, MESH_SIMPLIFY))
                    elif c.getAttribute('type') == 'mesh':
                        key = c.getAttribute('mesh')
                        mesh.append(geoms[key])
                    else:
                        from_to = c.getAttribute('fromto')
                        size = c.getAttribute('size')
                        mesh.append(create_capsule(from_to, size, MESH_SIMPLIFY))
        mesh = trimesh.util.concatenate(mesh)

        body_names.append(name)
        body_meshes.append(mesh)
    
    return body_names, body_meshes


def parse_mesh_elements_from_xml(xml_path):
    """
    根据G1骨架的实际节点顺序创建网格，确保与骨架完全一致
    """
    # 先创建G1骨架获取正确的节点顺序
    g1_skeleton = SkeletonTree.from_mjcf_g1(xml_path)
    
    rigidbody_names = []
    rigidbody_meshes = []

    # 按照骨架的实际节点顺序创建网格
    for name in g1_skeleton.node_names:
        mesh = create_dummy_capsule(name)
        rigidbody_names.append(name)
        rigidbody_meshes.append(mesh)

    print(f"[Info] Created {len(rigidbody_meshes)} dummy capsule meshes for joints: {rigidbody_names}")
    return rigidbody_names, rigidbody_meshes


def create_dummy_capsule(name, radius=0.04, height=0.2):
    """为每个具体部位设置明确的颜色和尺寸，并调整anchor point"""
    if name == 'pelvis':
        height, radius = 0.25, 0.06  # 骨盆部位的高度和半径
        color = [160, 160, 160]  # 颜色：灰色
        anchor_offset = 0.0  # pelvis保持中心
    elif name == 'torso':
        height, radius = 0.30, 0.08  # 躯干部位的高度和半径
        color = [100, 100, 100]  # 颜色：暗灰色
        anchor_offset = -height/4  # 向下偏移，顶部连接pelvis
    elif name == 'head':
        height, radius = 0.15, 0.08  # 头部的高度和半径
        color = [255, 150, 200]  # 颜色：浅粉色
        anchor_offset = height/4   # 向上偏移，底部连接torso
    elif 'thigh' in name:
        height, radius = 0.25, 0.055  # 大腿的高度和半径
        color = [0, 100, 255]  # 颜色：蓝色
        anchor_offset = height/4   # 向上偏移，顶部连接pelvis
    elif 'shin' in name:
        height, radius = 0.25, 0.04  # 小腿的高度和半径
        color = [255, 255, 0]  # 颜色：黄色
        anchor_offset = height/4   # 向上偏移，顶部连接thigh
    elif 'foot' in name:
        height, radius = 0.10, 0.04  # 脚部的高度和半径
        color = [255, 128, 0]  # 颜色：橙色
        anchor_offset = height/4   # 向上偏移，顶部连接shin
    elif 'upper_arm' in name:
        height, radius = 0.20, 0.045  # 上臂的高度和半径
        color = [0, 200, 255]  # 颜色：亮蓝色
        anchor_offset = height/4   # 向上偏移，顶部连接torso
    elif 'lower_arm' in name:
        height, radius = 0.20, 0.035  # 前臂的高度和半径
        color = [255, 200, 0]  # 颜色：橙黄色
        anchor_offset = height/4   # 向上偏移，顶部连接upper_arm
    elif 'hand' in name:
        height, radius = 0.08, 0.03  # 手部的高度和半径
        color = [255, 0, 0]  # 颜色：红色
        anchor_offset = height/4   # 向上偏移，顶部连接lower_arm
    else:
        height, radius = 0.2, 0.04  # 默认的高度和半径
        color = [180, 180, 180]  # 颜色：浅灰色
        anchor_offset = 0.0

    mesh = trimesh.creation.capsule(radius=radius, height=height, count=[8, 8])
    
    # 应用anchor offset - 移动胶囊使关节位置在合适的连接点
    if anchor_offset != 0.0:
        offset_transform = np.eye(4)
        offset_transform[2, 3] = anchor_offset  # Z轴偏移
        mesh.apply_transform(offset_transform)
    
    mesh.visual.vertex_colors = np.tile(color + [255], (mesh.vertices.shape[0], 1))  # 设置每个顶点的颜色
    return mesh

