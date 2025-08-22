import os
import sys
import json
import importlib.util
from dataclasses import dataclass
from typing import List, Tuple, Dict, Optional, Any

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

def _load_env_class() -> Optional[object]:
	# Load Environment as part of its package to satisfy relative imports inside environment.py
	base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
	package_name = os.path.basename(base_dir)
	parent_dir = os.path.dirname(base_dir)
	if parent_dir not in sys.path:
		sys.path.insert(0, parent_dir)
	# Try normal package import first, e.g., Unity_AOV.environment
	try:
		mod = importlib.import_module(f"{package_name}.environment")
		return getattr(mod, "Environment", None)
	except Exception:
		pass
	# Create a synthetic package if needed and load via spec with a qualified name
	try:
		import types
		from importlib.machinery import ModuleSpec
		# Synthesize the package module
		if package_name not in sys.modules:
			pkg = types.ModuleType(package_name)
			pkg.__path__ = [base_dir]
			pkg.__package__ = package_name
			pkg.__spec__ = ModuleSpec(name=package_name, loader=None, is_package=True)
			sys.modules[package_name] = pkg
		# Load environment as submodule of that package
		env_path = os.path.join(base_dir, "environment.py")
		spec = importlib.util.spec_from_file_location(f"{package_name}.environment", env_path)
		if spec and spec.loader:
			mod = importlib.util.module_from_spec(spec)
			mod.__package__ = package_name
			sys.modules[f"{package_name}.environment"] = mod
			spec.loader.exec_module(mod)
			return getattr(mod, "Environment", None)
	except Exception:
		return None
	return None


def list_meshes(assets_path: str) -> List[Dict[str, Any]]:
	Environment = _load_env_class()
	items: List[Dict[str, Any]] = []
	if Environment is not None:
		try:
			env = Environment(assets_path)
			for obj in env.objects:
				if obj.type.name != "Mesh":
					continue
				m = obj.read(return_typetree_on_error=True)
				modern = bool(getattr(m, "m_VertexData", None) and getattr(m.m_VertexData, "m_VertexCount", 0) > 0)
				items.append({
					"path_id": obj.path_id,
					"container": obj.container,
					"modern_vertex": modern,
				})
		except Exception:
			pass
	if items:
		return items
	# Fallback: UnityPy direct load
	try:
		import UnityPy
		up_env = UnityPy.load(assets_path)
		for obj in up_env.objects:
			if obj.type.name != "Mesh":
				continue
			try:
				m = obj.read()
			except Exception:
				m = None
			modern = bool(getattr(m, "m_VertexData", None) and getattr(m.m_VertexData, "m_VertexCount", 0) > 0) if m else False
			items.append({
				"path_id": obj.path_id,
				"container": getattr(obj, "container", None),
				"modern_vertex": modern,
			})
		return items
	except Exception:
		return []


def try_replace_mesh_in_assets(assets_path: str, obj_mesh: ObjMesh, target_name: Optional[str] = None, target_path_id: Optional[int] = None, out_dir: Optional[str] = None, debug: bool = False) -> Tuple[bool, str]:
	"""
	Attempt to load a Unity assets/bundle, find a Mesh by name or path_id, and replace its geometry.
	This currently supports meshes that use legacy direct arrays (no modern VertexData streams).
	Returns (ok, reason/message).
	"""
	Environment = _load_env_class()
	# First attempt with native Environment
	if Environment is not None:
		try:
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
			if candidate:
				m = candidate.read(return_typetree_on_error=True)
				if hasattr(m, "m_VertexData") and getattr(m.m_VertexData, "m_VertexCount", 0) > 0:
					return False, "Target mesh uses modern VertexData (not supported)"
				m.m_VertexCount = len(obj_mesh.vertices)
				m.m_Vertices = [c for v in obj_mesh.vertices for c in (v[0], v[1], v[2])]
				m.m_Normals = [c for n in obj_mesh.normals_u for c in (n[0], n[1], n[2])]
				m.m_UV0 = [c for t in obj_mesh.uvs_u for c in (t[0], t[1])]
				m.m_Use16BitIndices = max(obj_mesh.indices) < 65535 if obj_mesh.indices else True
				m.m_Indices = list(obj_mesh.indices)
				m.m_IndexBuffer = list(obj_mesh.indices)
				m.m_SubMeshes = [{
					"firstByte": 0,
					"indexCount": len(obj_mesh.indices),
					"topology": 0,
					"triangleCount": len(obj_mesh.indices) // 3,
					"baseVertex": 0,
					"firstVertex": 0,
					"vertexCount": len(obj_mesh.vertices),
					"localAABB": getattr(m, "m_LocalAABB", None),
				}]
				m.save_typetree()
				try:
					m.assets_file.mark_changed()
				except Exception:
					pass
				if out_dir:
					env.out_path = out_dir
				os.makedirs(env.out_path, exist_ok=True)
				env.save(pack="none")
				return True, os.path.join(env.out_path, os.path.basename(assets_path))
		except Exception as e:
			if debug:
				print(f"[DEBUG] Native Environment failed: {e}")
	# Fallback with UnityPy direct editing
	try:
		import UnityPy
		up_env = UnityPy.load(assets_path)
		candidate = None
		for obj in up_env.objects:
			if obj.type.name != "Mesh":
				continue
			if target_path_id is not None and obj.path_id == target_path_id:
				candidate = obj
				break
			if target_name and getattr(obj, "container", None) and os.path.basename(obj.container).startswith(target_name):
				candidate = obj
				break
		if not candidate:
			return False, "Target mesh not found"
		m = candidate.read()
		if hasattr(m, "m_VertexData") and getattr(m.m_VertexData, "m_VertexCount", 0) > 0:
			return False, "Target mesh uses modern VertexData (not supported)"
		m.m_VertexCount = len(obj_mesh.vertices)
		m.m_Vertices = [c for v in obj_mesh.vertices for c in (v[0], v[1], v[2])]
		m.m_Normals = [c for n in obj_mesh.normals_u for c in (n[0], n[1], n[2])]
		m.m_UV0 = [c for t in obj_mesh.uvs_u for c in (t[0], t[1])]
		m.m_Use16BitIndices = max(obj_mesh.indices) < 65535 if obj_mesh.indices else True
		m.m_Indices = list(obj_mesh.indices)
		m.m_IndexBuffer = list(obj_mesh.indices)
		m.m_SubMeshes = [{
			"firstByte": 0,
			"indexCount": len(obj_mesh.indices),
			"topology": 0,
			"triangleCount": len(obj_mesh.indices) // 3,
			"baseVertex": 0,
			"firstVertex": 0,
			"vertexCount": len(obj_mesh.vertices),
			"localAABB": getattr(m, "m_LocalAABB", None),
		}]
		# Try save; UnityPy objects also support save_typetree in most cases
		try:
			m.save_typetree()
		except Exception:
			try:
				m.save()
			except Exception as e2:
				return False, f"Failed to save mesh data: {e2}"
		if out_dir:
			up_env.out_path = out_dir
		os.makedirs(getattr(up_env, "out_path", os.path.join(os.getcwd(), "output")), exist_ok=True)
		try:
			up_env.save()
		except Exception as e:
			return False, f"Failed to save env: {e}"
		return True, os.path.join(getattr(up_env, "out_path", "."), os.path.basename(assets_path))
	except Exception as e:
		return False, f"UnityPy fallback failed: {e}"


def main(argv: List[str]) -> int:
	import argparse
	p = argparse.ArgumentParser(description="OBJ Importer for Unity AOV tools")
	p.add_argument("obj", nargs="?", help="Path to .obj file to import")
	p.add_argument("--to-json", dest="json_out", help="Output parsed mesh as JSON to this path")
	p.add_argument("--assets", dest="assets", help="Path to a Unity assets/bundle file to modify")
	p.add_argument("--mesh-name", dest="mesh_name", help="Target Mesh name prefix (matches container basename)")
	p.add_argument("--mesh-path-id", dest="mesh_path_id", type=int, help="Target Mesh path_id")
	p.add_argument("--out", dest="out_dir", help="Output directory for modified assets (default ./output)")
	p.add_argument("--no-flip-x", dest="no_flip_x", action="store_true", help="Do not flip X axis (by default X is negated)")
	p.add_argument("--list-meshes", dest="list_meshes", action="store_true", help="List meshes (path_id, container, modern/legacy) in the assets")
	p.add_argument("--debug", dest="debug", action="store_true", help="Print debug info")
	args = p.parse_args(argv)

	if args.list_meshes:
		if not args.assets:
			print("--assets is required for --list-meshes")
			return 2
		items = list_meshes(args.assets)
		if not items:
			print("No meshes found or Environment not available")
			return 3
		for it in items:
			print(f"path_id={it['path_id']} modern_vertex={it['modern_vertex']} container={it['container']}")
		return 0

	if args.json_out and not args.obj:
		print("OBJ path is required when using --to-json")
		return 2

	if args.obj and not os.path.exists(args.obj):
		print(f"OBJ not found: {args.obj}")
		return 2

	if args.obj:
		mesh = parse_obj(args.obj)
		mesh = to_unity_convention(mesh, flip_x=(not args.no_flip_x))
	else:
		mesh = None

	if args.json_out and mesh:
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

	if args.assets and mesh:
		ok, msg = try_replace_mesh_in_assets(
			assets_path=args.assets,
			obj_mesh=mesh,
			target_name=args.mesh_name,
			target_path_id=args.mesh_path_id,
			out_dir=args.out_dir or os.path.join(os.getcwd(), "output"),
			debug=args.debug,
		)
		if not ok:
			print(f"Mesh replacement failed: {msg}")
			return 3
		print(f"Mesh replaced and assets saved to: {msg}")

	return 0


if __name__ == "__main__":
	sys.exit(main(sys.argv[1:]))