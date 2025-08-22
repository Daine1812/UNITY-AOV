import os
import sys
import json
import importlib.util
from dataclasses import dataclass
from typing import List, Tuple, Dict, Optional, Any
import struct

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


def scan_dir_for_meshes(scan_path: str, find_name: Optional[str] = None) -> List[Dict[str, Any]]:
	"""Recursively scan a directory to find files that contain Mesh objects. Uses UnityPy per file.
	If find_name is provided, only return matches whose container basename startswith(find_name).
	"""
	results: List[Dict[str, Any]] = []
	try:
		import UnityPy
	except Exception:
		return results

	if os.path.isfile(scan_path):
		paths = [scan_path]
	else:
		paths = []
		for root, _, files in os.walk(scan_path):
			for fn in files:
				paths.append(os.path.join(root, fn))

	for fp in paths:
		try:
			env = UnityPy.load(fp)
		except Exception:
			continue
		try:
			for obj in env.objects:
				if obj.type.name != "Mesh":
					continue
				container = getattr(obj, "container", None)
				base = os.path.basename(container) if container else None
				if find_name and (not base or not base.startswith(find_name)):
					continue
				results.append({
					"file": fp,
					"path_id": obj.path_id,
					"container": container,
				})
		except Exception:
			continue
	return results


def _update_aabb(m, verts: List[Tuple[float, float, float]]):
	if not verts:
		return
	try:
		xs = [v[0] for v in verts]
		ys = [v[1] for v in verts]
		zs = [v[2] for v in verts]
		minx, maxx = min(xs), max(xs)
		miny, maxy = min(ys), max(ys)
		minz, maxz = min(zs), max(zs)
		cx = (minx + maxx) * 0.5
		cy = (miny + maxy) * 0.5
		cz = (minz + maxz) * 0.5
		ex = (maxx - cx)
		ey = (maxy - cy)
		ez = (maxz - cz)
		if hasattr(m, "m_LocalAABB") and m.m_LocalAABB:
			la = m.m_LocalAABB
			# Common AABB layout: m_Center (Vector3), m_Extent (Vector3)
			if hasattr(la, "m_Center") and hasattr(la, "m_Extent"):
				if hasattr(la.m_Center, "x"):
					la.m_Center.x, la.m_Center.y, la.m_Center.z = cx, cy, cz
					la.m_Extent.x, la.m_Extent.y, la.m_Extent.z = ex, ey, ez
				else:
					# Fallback if stored as tuples/dicts
					la.m_Center = getattr(la, "m_Center", (0, 0, 0))
					la.m_Extent = getattr(la, "m_Extent", (0, 0, 0))
					try:
						la.m_Center = (cx, cy, cz)
						la.m_Extent = (ex, ey, ez)
					except Exception:
						pass
			# Also update first submesh localAABB when possible
			if hasattr(m, "m_SubMeshes") and m.m_SubMeshes:
				sm0 = m.m_SubMeshes[0]
				if hasattr(sm0, "localAABB") and sm0.localAABB and hasattr(sm0.localAABB, "m_Center"):
					if hasattr(sm0.localAABB.m_Center, "x"):
						sm0.localAABB.m_Center.x, sm0.localAABB.m_Center.y, sm0.localAABB.m_Center.z = cx, cy, cz
						sm0.localAABB.m_Extent.x, sm0.localAABB.m_Extent.y, sm0.localAABB.m_Extent.z = ex, ey, ez
	except Exception:
		pass


def _update_submesh_and_flags(m, num_vertices: int, num_indices: int):
	try:
		if hasattr(m, "m_SubMeshes") and m.m_SubMeshes:
			sm0 = m.m_SubMeshes[0]
			if hasattr(sm0, "indexCount"):
				sm0.indexCount = num_indices
			if hasattr(sm0, "vertexCount"):
				sm0.vertexCount = num_vertices
			if hasattr(sm0, "firstVertex"):
				sm0.firstVertex = 0
			if hasattr(sm0, "firstByte"):
				sm0.firstByte = 0
			if hasattr(sm0, "baseVertex"):
				sm0.baseVertex = 0
			if hasattr(sm0, "topology"):
				try:
					# 0 usually equals triangles
					sm0.topology = 0 if isinstance(sm0.topology, int) else getattr(sm0.topology, "Triangles", 0)
				except Exception:
					pass
		# Helpful flags
		if hasattr(m, "m_IsReadable"):
			m.m_IsReadable = True
		if hasattr(m, "m_KeepVertices"):
			m.m_KeepVertices = True
		if hasattr(m, "m_KeepIndices"):
			m.m_KeepIndices = True
		# Clear baked collision meshes if present
		if hasattr(m, "m_BakedConvexCollisionMesh"):
			m.m_BakedConvexCollisionMesh = b""
		if hasattr(m, "m_BakedTriangleCollisionMesh"):
			m.m_BakedTriangleCollisionMesh = b""
		# Index format for >=2017.3
		if hasattr(m, "m_IndexFormat"):
			m.m_IndexFormat = 0 if num_indices == 0 or (num_vertices <= 65535) else 1
		# Compression flags as safe defaults
		if hasattr(m, "m_MeshCompression"):
			m.m_MeshCompression = 0
		if hasattr(m, "m_StreamCompression"):
			m.m_StreamCompression = 0
		if hasattr(m, "m_MeshUsageFlags"):
			try:
				m.m_MeshUsageFlags = int(max(0, min(255, int(m.m_MeshUsageFlags))))
			except Exception:
				m.m_MeshUsageFlags = 0
	except Exception:
		pass


def _rebuild_vertex_data_modern(m, verts: List[Tuple[float, float, float]], norms: List[Tuple[float, float, float]], uvs: List[Tuple[float, float]], debug: bool=False):
	"""Builds minimal m_VertexData (pos3, normal3, uv0-2) for Unity 2019+.
	Packs floats as big-endian to match MeshHelper.BytesToFloatArray.
	"""
	try:
		N = len(verts)
		# Ensure normals/uvs lengths match
		if len(norms) != N:
			norms = [(0.0, 0.0, 1.0)] * N
		if len(uvs) != N:
			uvs = [(0.0, 0.0)] * N
		stride = 4 * (3 + 3 + 2)  # pos3 + normal3 + uv2
		# Build interleaved data
		buf = bytearray(N * stride)
		off = 0
		for i in range(N):
			x,y,z = verts[i]
			nx,ny,nz = norms[i]
			u,v = uvs[i]
			for val in (x,y,z,nx,ny,nz,u,v):
				buf[off:off+4] = struct.pack('<f', float(val))
				off += 4
		# Channel indices: 0=pos,1=normal,4=uv0
		from importlib import import_module
		MeshMod = import_module(f"{m.__class__.__module__}")
		ChannelInfo = getattr(MeshMod, 'ChannelInfo')
		VertexDataCls = getattr(MeshMod, 'VertexData')
		# If m_VertexData missing, create a blank-like object
		vd = getattr(m, 'm_VertexData', None)
		if vd is None:
			vd = object.__new__(VertexDataCls)
		m.m_VertexData = vd
		vd.m_VertexCount = N
		# Optionally set m_CurrentChannels bitmask if field exists (mainly <2018, safe otherwise)
		if not hasattr(vd, 'm_CurrentChannels'):
			try:
				vd.m_CurrentChannels = 0
			except Exception:
				pass
		# Build channels list with correct indices (0=pos,1=normal,2=tangent,3=color,4=uv0)
		channels = []
		for i in range(12):
			ci = object.__new__(ChannelInfo)
			ci.stream = 0
			ci.offset = 0
			ci.format = 0
			ci.dimension = 0
			channels.append(ci)
		# pos
		channels[0].offset = 0
		channels[0].dimension = 3
		# normal
		channels[1].offset = 12
		channels[1].dimension = 3
		# uv0 at index 4
		channels[4].offset = 24
		channels[4].dimension = 2
		vd.m_Channels = channels
		# Set stream 0 metadata
		try:
			StreamInfo = getattr(MeshMod, 'StreamInfo')
			si0 = object.__new__(StreamInfo)
			si0.channelMask = (1 << 0) | (1 << 1) | (1 << 4)
			si0.offset = 0
			si0.stride = stride
			si0.dividerOp = 0
			si0.frequency = 0
			vd.m_Streams = {0: si0}
		except Exception:
			vd.m_Streams = {0: object()}
		vd.m_DataSize = bytes(buf)
		# Clear compressed mesh to avoid overrides
		cm = getattr(m, 'm_CompressedMesh', None)
		if cm is not None:
			for fld in ('m_Vertices','m_UV','m_Normals','m_Tangents','m_Weights','m_BoneIndices','m_Triangles','m_FloatColors'):
				try:
					pv = getattr(cm, fld, None)
					if pv is not None and hasattr(pv, 'm_NumItems'):
						pv.m_NumItems = 0
				except Exception:
					pass
		# Disable external stream to force inline vertex data
		try:
			sd = getattr(m, 'm_StreamData', None)
			if sd is not None:
				sd.path = ""
				sd.offset = 0
				sd.size = 0
		except Exception:
			pass
		# Also set index format
		if hasattr(m, 'm_IndexFormat'):
			m.m_IndexFormat = 0 if N <= 65535 else 1
		if debug:
			print(f"[DEBUG] Rebuilt VertexData: N={N}, stride={stride}")
	except Exception as e:
		if debug:
			print(f"[DEBUG] Rebuild VertexData failed: {e}")
		raise


def _clamp_vertexdata_bytes(vd):
	try:
		chs = getattr(vd, 'm_Channels', None)
		if chs:
			for c in chs:
				if hasattr(c, 'stream'):
					c.stream = int(max(0, min(255, int(c.stream))))
				if hasattr(c, 'offset'):
					c.offset = int(max(0, min(255, int(c.offset))))
				if hasattr(c, 'format'):
					c.format = int(max(0, min(255, int(c.format))))
				if hasattr(c, 'dimension'):
					c.dimension = int(max(0, min(255, int(c.dimension))))
		s = getattr(vd, 'm_Streams', None)
		if isinstance(s, dict) and 0 in s:
			si = s[0]
			if hasattr(si, 'stride'):
				si.stride = int(max(0, min(255, int(getattr(si, 'stride', 0)))))
			if hasattr(si, 'dividerOp'):
				si.dividerOp = int(max(0, min(255, int(getattr(si, 'dividerOp', 0)))))
			if hasattr(si, 'frequency'):
				si.frequency = int(max(0, min(65535, int(getattr(si, 'frequency', 0)))))
	except Exception:
		pass


def try_replace_mesh_in_assets(assets_path: str, obj_mesh: ObjMesh, target_name: Optional[str] = None, target_path_id: Optional[int] = None, out_dir: Optional[str] = None, debug: bool = False, sanitize: bool = False, force_legacy: bool = False, export_only: Optional[str] = None) -> Tuple[bool, str]:
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
				# Populate geometry
				m.m_VertexCount = len(obj_mesh.vertices)
				m.m_Vertices = [c for v in obj_mesh.vertices for c in (v[0], v[1], v[2])]
				m.m_Normals = [c for n in obj_mesh.normals_u for c in (n[0], n[1], n[2])]
				m.m_UV0 = [c for t in obj_mesh.uvs_u for c in (t[0], t[1])]
				m.m_Use16BitIndices = max(obj_mesh.indices) < 65535 if obj_mesh.indices else True
				m.m_Indices = list(obj_mesh.indices)
				# Avoid assigning m_IndexBuffer directly for legacy formats
				# writer will serialize from m_Indices/m_Use16BitIndices
				_update_submesh_and_flags(m, len(obj_mesh.vertices), len(obj_mesh.indices))
				_update_aabb(m, obj_mesh.vertices)
				# Vertex data handling
				if force_legacy:
					vd = getattr(m, 'm_VertexData', None)
					if vd is not None:
						try:
							vd.m_VertexCount = 0
							vd.m_Channels = []
							vd.m_Streams = {}
							vd.m_DataSize = b""
						except Exception:
							pass
					# disable external stream
					sd = getattr(m, 'm_StreamData', None)
					if sd is not None:
						try:
							sd.path = ""; sd.offset = 0; sd.size = 0
						except Exception:
							pass
					# clear compressed
					cm = getattr(m, 'm_CompressedMesh', None)
					if cm is not None:
						for fld in ('m_Vertices','m_UV','m_Normals','m_Tangents','m_Weights','m_BoneIndices','m_Triangles','m_FloatColors'):
							pv = getattr(cm, fld, None)
							if pv is not None and hasattr(pv, 'm_NumItems'):
								pv.m_NumItems = 0
					# ensure index buffer for legacy triangle build
					m.m_IndexBuffer = list(obj_mesh.indices)
					# ensure at least one SubMesh object for exporters
					try:
						from importlib import import_module
						MeshMod = import_module(f"{m.__class__.__module__}")
						SubMesh = getattr(MeshMod, 'SubMesh')
						if not getattr(m, 'm_SubMeshes', None):
							sm = object.__new__(SubMesh)
							sm.firstByte = 0
							sm.indexCount = len(obj_mesh.indices)
							sm.topology = 0
							sm.baseVertex = 0
							sm.firstVertex = 0
							sm.vertexCount = len(obj_mesh.vertices)
							sm.localAABB = getattr(m, 'm_LocalAABB', None)
							m.m_SubMeshes = [sm]
					except Exception:
						if not getattr(m, 'm_SubMeshes', None):
							class _SM: pass
							sm = _SM(); sm.firstByte=0; sm.indexCount=len(obj_mesh.indices); sm.topology=0; sm.baseVertex=0; sm.firstVertex=0; sm.vertexCount=len(obj_mesh.vertices); sm.localAABB=getattr(m,'m_LocalAABB',None)
							m.m_SubMeshes = [sm]
				else:
					try:
						ver = getattr(m, 'version', (2022,3,5))
						if ver >= (2019,):
							_rebuild_vertex_data_modern(m, obj_mesh.vertices, obj_mesh.normals_u, obj_mesh.uvs_u, debug)
							# Provide index buffer for triangle build
							m.m_IndexBuffer = list(obj_mesh.indices)
							# Clamp byte fields in channels/streams
							_clamp_vertexdata_bytes(m.m_VertexData)
					except Exception as e:
						if debug:
							print(f"[DEBUG] VertexData rebuild skipped: {e}")
				# Sanitize optional channels to avoid viewer strictness
				if sanitize:
					for field in ("m_Tangents", "m_Colors", "m_UV1", "m_UV2", "m_UV3", "m_UV4", "m_UV5", "m_UV6", "m_UV7"):
						if hasattr(m, field):
							setattr(m, field, [])
					if hasattr(m, "m_Skin"):
						m.m_Skin = []
					if hasattr(m, "m_Shapes"):
						try:
							m.m_Shapes.shapes = []
							m.m_Shapes.channels = []
							m.m_Shapes.fullWeights = []
						except Exception:
							pass
				# Update existing submeshes if any
				try:
					submeshes = getattr(m, "m_SubMeshes", None)
					if submeshes and len(submeshes) > 0:
						sm0 = submeshes[0]
						# Attribute-style if it's an object; dict-style otherwise
						if hasattr(sm0, "indexCount"):
							sm0.indexCount = len(obj_mesh.indices)
							sm0.vertexCount = len(obj_mesh.vertices)
							# keep topology and others as-is
						elif isinstance(sm0, dict):
							sm0["indexCount"] = len(obj_mesh.indices)
							sm0["vertexCount"] = len(obj_mesh.vertices)
				except Exception:
					pass
				# Export-only handling
				if export_only:
					try:
						os.makedirs(os.path.dirname(os.path.abspath(export_only)) or ".", exist_ok=True)
						with open(export_only, "w", encoding="utf8") as f:
							f.write(m.export())
						return True, export_only
					except Exception as e:
						return False, f"Export failed: {e}"
				# Save using class save and mark changed
				try:
					m.save()
				except Exception as e:
					if debug:
						print(f"[DEBUG] m.save() failed: {e}")
					# fallback to typetree only if available
					if hasattr(m, "save_typetree"):
						m.save_typetree()
					else:
						return False, f"Mesh class doesn't support save_typetree and save() failed: {e}"
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
		# Avoid assigning m_IndexBuffer directly for legacy formats
		_update_submesh_and_flags(m, len(obj_mesh.vertices), len(obj_mesh.indices))
		_update_aabb(m, obj_mesh.vertices)
		# Vertex data handling (UnityPy fallback)
		if force_legacy:
			vd = getattr(m, 'm_VertexData', None)
			if vd is not None:
				try:
					vd.m_VertexCount = 0
					vd.m_Channels = []
					vd.m_Streams = {}
					vd.m_DataSize = b""
				except Exception:
					pass
			sd = getattr(m, 'm_StreamData', None)
			if sd is not None:
				try:
					sd.path = ""; sd.offset = 0; sd.size = 0
				except Exception:
					pass
			cm = getattr(m, 'm_CompressedMesh', None)
			if cm is not None:
				for fld in ('m_Vertices','m_UV','m_Normals','m_Tangents','m_Weights','m_BoneIndices','m_Triangles','m_FloatColors'):
					pv = getattr(cm, fld, None)
					if pv is not None and hasattr(pv, 'm_NumItems'):
						pv.m_NumItems = 0
			# ensure index buffer for legacy triangle build
			m.m_IndexBuffer = list(obj_mesh.indices)
			# ensure at least one SubMesh object for exporters
			try:
				from importlib import import_module
				MeshMod = import_module(f"{m.__class__.__module__}")
				SubMesh = getattr(MeshMod, 'SubMesh')
				if not getattr(m, 'm_SubMeshes', None):
					sm = object.__new__(SubMesh)
					sm.firstByte = 0
					sm.indexCount = len(obj_mesh.indices)
					sm.topology = 0
					sm.baseVertex = 0
					sm.firstVertex = 0
					sm.vertexCount = len(obj_mesh.vertices)
					sm.localAABB = getattr(m, 'm_LocalAABB', None)
					m.m_SubMeshes = [sm]
			except Exception:
				if not getattr(m, 'm_SubMeshes', None):
					class _SM: pass
					sm = _SM(); sm.firstByte=0; sm.indexCount=len(obj_mesh.indices); sm.topology=0; sm.baseVertex=0; sm.firstVertex=0; sm.vertexCount=len(obj_mesh.vertices); sm.localAABB=getattr(m,'m_LocalAABB',None)
					m.m_SubMeshes = [sm]
		else:
			try:
				ver = getattr(m, 'version', (2022,3,5))
				if ver >= (2019,):
					_rebuild_vertex_data_modern(m, obj_mesh.vertices, obj_mesh.normals_u, obj_mesh.uvs_u, debug)
					m.m_IndexBuffer = list(obj_mesh.indices)
					_clamp_vertexdata_bytes(m.m_VertexData)
			except Exception as e:
				if debug:
					print(f"[DEBUG] VertexData rebuild skipped: {e}")
		# Sanitize optional channels to avoid viewer strictness
		if sanitize:
			for field in ("m_Tangents", "m_Colors", "m_UV1", "m_UV2", "m_UV3", "m_UV4", "m_UV5", "m_UV6", "m_UV7"):
				if hasattr(m, field):
					setattr(m, field, [])
			if hasattr(m, "m_Skin"):
				m.m_Skin = []
			if hasattr(m, "m_Shapes"):
				try:
					m.m_Shapes.shapes = []
					m.m_Shapes.channels = []
					m.m_Shapes.fullWeights = []
				except Exception:
					pass
		# Update existing submeshes if any
		try:
			submeshes = getattr(m, "m_SubMeshes", None)
			if submeshes and len(submeshes) > 0:
				sm0 = submeshes[0]
				if hasattr(sm0, "indexCount"):
					sm0.indexCount = len(obj_mesh.indices)
					sm0.vertexCount = len(obj_mesh.vertices)
				elif isinstance(sm0, dict):
					sm0["indexCount"] = len(obj_mesh.indices)
					sm0["vertexCount"] = len(obj_mesh.vertices)
		except Exception:
			pass
		# Export-only handling
		if export_only:
			try:
				os.makedirs(os.path.dirname(os.path.abspath(export_only)) or ".", exist_ok=True)
				with open(export_only, "w", encoding="utf8") as f:
					f.write(m.export())
				return True, export_only
			except Exception as e:
				return False, f"Export failed: {e}"
		# Try save
		try:
			m.save()
		except Exception as e:
			return False, f"UnityPy Mesh can't be saved: {e}"
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
	p.add_argument("--scan-dir", dest="scan_dir", help="Scan a directory recursively to locate files containing meshes")
	p.add_argument("--find-name", dest="find_name", help="When used with --scan-dir, filter meshes whose container basename startswith this name")
	p.add_argument("--sanitize", dest="sanitize", action="store_true", help="Clear optional channels (UV1..7, tangents, colors, skin, shapes) to improve compatibility")
	p.add_argument("--force-legacy", dest="force_legacy", action="store_true", help="Strip VertexData/CompressedMesh to save using legacy arrays")
	p.add_argument("--export-only", dest="export_only", help="Export replaced mesh to this OBJ path without saving the asset")
	args = p.parse_args(argv)

	if args.scan_dir:
		matches = scan_dir_for_meshes(args.scan_dir, args.find_name)
		if not matches:
			print("No mesh matches found in directory")
			return 3
		for m in matches:
			print(f"file={m['file']} path_id={m['path_id']} container={m['container']}")
		return 0

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
			sanitize=getattr(args, "sanitize", False),
			force_legacy=getattr(args, "force_legacy", False),
			export_only=args.export_only,
		)
		if not ok:
			print(f"Mesh replacement failed: {msg}")
			return 3
		print(f"Mesh replaced and assets saved to: {msg}")

	return 0


if __name__ == "__main__":
	sys.exit(main(sys.argv[1:]))