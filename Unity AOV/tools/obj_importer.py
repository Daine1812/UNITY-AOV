import os
import sys
import json
from dataclasses import dataclass
from typing import List, Tuple, Dict, Optional

# Minimal OBJ importer (no external deps)

@dataclass
class ObjMesh:
	positions: List[Tuple[float, float, float]]
	normals: List[Tuple[float, float, float]]
	texcoords: List[Tuple[float, float]]
	# Unified vertex stream after indexing by (v, vt, vn)
	vertices: List[Tuple[float, float, float]]
	normals_u: List[Tuple[float, float, float]]
	uvs_u: List[Tuple[float, float]]
	indices: List[int]


def parse_obj(obj_path: str) -> ObjMesh:
	"""
	Parse a Wavefront OBJ file. Supports v/vt/vn and faces with arbitrary polygon size.
	Faces are triangulated via fan method. Builds a unified vertex stream indexed by (v,vt,vn).
	"""
	positions: List[Tuple[float, float, float]] = []
	normals: List[Tuple[float, float, float]] = []
	texcoords: List[Tuple[float, float]] = []

	# After unifying v/vt/vn tuples
	vertex_map: Dict[Tuple[int, int, int], int] = {}
	vertices_u: List[Tuple[float, float, float]] = []
	normals_u: List[Tuple[float, float, float]] = []
	uvs_u: List[Tuple[float, float]] = []
	indices: List[int] = []

	def add_vertex(vi: int, ti: int, ni: int) -> int:
		key = (vi, ti, ni)
		idx = vertex_map.get(key)
		if idx is not None:
			return idx
		# OBJ is 1-based; allow negative indices
		v = positions[vi - 1 if vi > 0 else len(positions) + vi]
		n = (0.0, 0.0, 0.0)
		t = (0.0, 0.0)
		if ni != 0 and normals:
			n = normals[ni - 1 if ni > 0 else len(normals) + ni]
		if ti != 0 and texcoords:
			t = texcoords[ti - 1 if ti > 0 else len(texcoords) + ti]
		idx = len(vertices_u)
		vertex_map[key] = idx
		vertices_u.append(v)
		normals_u.append(n)
		uvs_u.append(t)
		return idx

	with open(obj_path, "rt", encoding="utf8", errors="ignore") as f:
		for line in f:
			line = line.strip()
			if not line or line.startswith("#"):
				continue
			parts = line.split()
			tok = parts[0]
			if tok == "v" and len(parts) >= 4:
				x, y, z = float(parts[1]), float(parts[2]), float(parts[3])
				positions.append((x, y, z))
			elif tok == "vn" and len(parts) >= 4:
				x, y, z = float(parts[1]), float(parts[2]), float(parts[3])
				normals.append((x, y, z))
			elif tok == "vt" and len(parts) >= 3:
				u, v = float(parts[1]), float(parts[2])
				texcoords.append((u, v))
			elif tok == "f" and len(parts) >= 4:
				# Support formats: v, v/vt, v//vn, v/vt/vn
				# Build face as list of unified indices then triangulate via fan
				face: List[int] = []
				for p in parts[1:]:
					if "/" in p:
						sp = p.split("/")
						vi = int(sp[0]) if sp[0] else 0
						ti = int(sp[1]) if len(sp) > 1 and sp[1] else 0
						ni = int(sp[2]) if len(sp) > 2 and sp[2] else 0
					else:
						vi = int(p)
						ti = 0
						ni = 0
					face.append(add_vertex(vi, ti, ni))
				# triangulate fan: (0,i,i+1)
				for i in range(1, len(face) - 1):
					# Reverse winding to match MeshExporter which outputs flipped order
					indices.extend([face[i + 1], face[i], face[0]])

	return ObjMesh(
		positions=positions,
		normals=normals,
		texcoords=texcoords,
		vertices=vertices_u,
		normals_u=normals_u,
		uvs_u=uvs_u,
		indices=indices,
	)


def to_unity_convention(mesh: ObjMesh, flip_x: bool = True) -> ObjMesh:
	"""
	Convert coordinates to match the exporter convention (which negates X when writing OBJ).
	"""
	if not flip_x:
		return mesh

	def fx3(v):
		return (-v[0], v[1], v[2])

	mesh.vertices = [fx3(v) for v in mesh.vertices]
	mesh.normals_u = [fx3(n) for n in mesh.normals_u]
	return mesh


# Optional: Integrate with Unity AOV environment to replace an existing Mesh

def try_replace_mesh_in_assets(assets_path: str, obj_mesh: ObjMesh, target_name: Optional[str] = None, target_path_id: Optional[int] = None, out_dir: Optional[str] = None) -> bool:
	"""
	Attempt to load a Unity assets/bundle, find a Mesh by name or path_id, and replace its geometry.
	This currently supports meshes that use legacy direct arrays (no modern VertexData streams).
	Returns True if replacement succeeded and was saved.
	"""
	try:
		from Unity\ AOV.environment import Environment  # type: ignore
		from Unity\ AOV.enums.ClassIDType import ClassIDType  # type: ignore
		from Unity\ AOV.classes.Mesh import Mesh as UnityMesh  # type: ignore
	except Exception:
		# Fallback import path if the package name differs (e.g., UnityPy_AOV)
		try:
			from Unity_AOV.environment import Environment  # type: ignore
			from Unity_AOV.enums.ClassIDType import ClassIDType  # type: ignore
			from Unity_AOV.classes.Mesh import Mesh as UnityMesh  # type: ignore
		except Exception:
			return False

	env = Environment(assets_path)
	candidate = None
	for obj in env.objects:
		if obj.type.name != "Mesh":
			continue
		if target_path_id is not None and obj.path_id == target_path_id:
			candidate = obj
			break
		if target_name and obj.container and os.path.basename(obj.container).startswith(target_name):
			candidate = obj
			break

	if not candidate:
		return False

	m = candidate.read()
	# Heuristic: if modern vertex streams are present, abort (not implemented here)
	if hasattr(m, "m_VertexData") and getattr(m.m_VertexData, "m_VertexCount", 0) > 0:
		return False

	# Populate legacy arrays
	m.m_VertexCount = len(obj_mesh.vertices)
	m.m_Vertices = [c for v in obj_mesh.vertices for c in (v[0], v[1], v[2])]
	m.m_Normals = [c for n in obj_mesh.normals_u for c in (n[0], n[1], n[2])]
	m.m_UV0 = [c for t in obj_mesh.uvs_u for c in (t[0], t[1])]
	# Index data
	m.m_Use16BitIndices = max(obj_mesh.indices) < 65535 if obj_mesh.indices else True
	# Many engines store index buffer as m_IndexBuffer; also retain m_Indices to keep exporter compatibility
	m.m_Indices = list(obj_mesh.indices)
	m.m_IndexBuffer = list(obj_mesh.indices)

	# Single SubMesh covering all indices
	# If the SubMesh type exists, create a lightweight instance with required fields
	try:
		from Unity\ AOV.classes.Mesh import SubMesh  # type: ignore
		sm = object.__new__(SubMesh)
		sm.firstByte = 0
		sm.indexCount = len(obj_mesh.indices)
		sm.topology = 0  # assume triangles
		sm.triangleCount = len(obj_mesh.indices) // 3
		sm.baseVertex = 0
		sm.firstVertex = 0
		sm.vertexCount = len(obj_mesh.vertices)
		sm.localAABB = getattr(m, "m_LocalAABB", None)
		m.m_SubMeshes = [sm]
	except Exception:
		# Fallback: try to assign a typetree-like dict
		m.m_SubMeshes = [{
			"firstByte": 0,
			"indexCount": len(obj_mesh.indices),
			"topology": 0,
			"triangleCount": len(obj_mesh.indices) // 3,
			"baseVertex": 0,
			"firstVertex": 0,
			"vertexCount": len(obj_mesh.vertices),
		}]

	# Save back into the asset and write files
	m.save()
	if out_dir:
		env.out_path = out_dir
	env.save(pack="none")
	return True


def main(argv: List[str]) -> int:
	import argparse
	p = argparse.ArgumentParser(description="OBJ Importer for Unity AOV tools")
	p.add_argument("obj", help="Path to .obj file to import")
	p.add_argument("--to-json", dest="json_out", help="Output parsed mesh as JSON to this path")
	p.add_argument("--assets", dest="assets", help="Path to a Unity assets/bundle file to modify")
	p.add_argument("--mesh-name", dest="mesh_name", help="Target Mesh name prefix (matches container basename)")
	p.add_argument("--mesh-path-id", dest="mesh_path_id", type=int, help="Target Mesh path_id")
	p.add_argument("--out", dest="out_dir", help="Output directory for modified assets (default ./output)")
	p.add_argument("--no-flip-x", dest="no_flip_x", action="store_true", help="Do not flip X axis (by default X is negated)")
	args = p.parse_args(argv)

	if not os.path.exists(args.obj):
		print(f"OBJ not found: {args.obj}")
		return 2

	mesh = parse_obj(args.obj)
	mesh = to_unity_convention(mesh, flip_x=(not args.no_flip_x))

	if args.json_out:
		payload = {
			"vertices": [c for v in mesh.vertices for c in (v[0], v[1], v[2])],
			"normals": [c for n in mesh.normals_u for c in (n[0], n[1], n[2])],
			"uv0": [c for t in mesh.uvs_u for c in (t[0], t[1])],
			"indices": mesh.indices,
		}
		os.makedirs(os.path.dirname(os.path.abspath(args.json_out)) or ".", exist_ok=True)
		with open(args.json_out, "wt", encoding="utf8") as f:
			json.dump(payload, f)
		print(f"Wrote JSON: {args.json_out}")

	if args.assets:
		ok = try_replace_mesh_in_assets(
			assets_path=args.assets,
			obj_mesh=mesh,
			target_name=args.mesh_name,
			target_path_id=args.mesh_path_id,
			out_dir=args.out_dir or os.path.join(os.getcwd(), "output"),
		)
		if not ok:
			print("Mesh replacement not supported for this asset or target not found.")
			return 3
		print("Mesh replaced and assets saved.")

	return 0


if __name__ == "__main__":
	sys.exit(main(sys.argv[1:]))