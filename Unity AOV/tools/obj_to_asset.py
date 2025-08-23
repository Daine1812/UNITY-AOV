import os
import sys
import argparse
from pathlib import Path
from typing import List, Tuple, Optional

# Use the local Unity AOV package
try:
	import UnityPy_AOV as UA
	UA_load = UA.load
	from UnityPy_AOV.enums import ClassIDType, GfxPrimitiveType
except Exception as e:
	print("[ERROR] Không thể import Unity AOV package trong workspace. Lỗi:", e)
	print("Hãy đảm bảo thư mục 'Unity AOV' nằm trong workspace và có thể import.")
	sys.exit(1)


def parse_obj(fp: Path) -> Tuple[List[Tuple[float, float, float]], List[Tuple[float, float]], List[Tuple[float, float, float]], List[Tuple[int, int, int]]]:
	"""
	Đọc file OBJ đơn giản: v, vt, vn, f (tam giác). Trả về (vertices, uvs, normals, faces_indices)
	- faces_indices: danh sách bộ ba chỉ số (i0, i1, i2) 0-based cho vertex
	"""
	vertices: List[Tuple[float, float, float]] = []
	uvs: List[Tuple[float, float]] = []
	normals: List[Tuple[float, float, float]] = []
	faces: List[Tuple[int, int, int]] = []

	with fp.open("r", encoding="utf-8", errors="ignore") as f:
		for line in f:
			line = line.strip()
			if not line or line.startswith("#"):
				continue
			parts = line.split()
			if not parts:
				continue
			t = parts[0]
			if t == "v" and len(parts) >= 4:
				x, y, z = float(parts[1]), float(parts[2]), float(parts[3])
				# Unity hệ trục: đảo trục X cho khớp export truyền thống
				vertices.append((-x, y, z))
			elif t == "vt" and len(parts) >= 3:
				u, v = float(parts[1]), float(parts[2])
				# Lật V theo quy ước phổ biến
				uvs.append((u, 1.0 - v))
			elif t == "vn" and len(parts) >= 4:
				nx, ny, nz = float(parts[1]), float(parts[2]), float(parts[3])
				normals.append((-nx, ny, nz))
			elif t == "f" and len(parts) >= 4:
				# Chỉ hỗ trợ tam giác; nếu là polygon, triangulate fan
				idx_tokens = parts[1:]
				# Triangulate (v0,v1,v2), (v0,v2,v3), ...
				base = idx_tokens[0]
				for k in range(1, len(idx_tokens) - 1):
					a = base
					b = idx_tokens[k]
					c = idx_tokens[k + 1]
					def parse_idx(tok: str) -> int:
						# format: v, v/vt, v//vn, v/vt/vn (1-based)
						v_str = tok.split("/")[0]
						return int(v_str) - 1
					faces.append((parse_idx(a), parse_idx(b), parse_idx(c)))

	return vertices, uvs, normals, faces


def find_template_with_mesh(hints: List[Path]) -> Tuple[object, object]:
	"""
	Tìm một SerializedFile trong các file gợi ý có chứa Mesh.
	Return (env, mesh_obj_reader)
	"""
	searched: List[Path] = []
	for hint in hints:
		if hint.is_dir():
			for p in hint.rglob("*.asset"):
				searched.append(p)
			for p in hint.rglob("CAB-*"):
				searched.append(p)
		elif hint.is_file():
			searched.append(hint)

	# Duyệt và tìm Mesh đầu tiên
	for p in searched:
		try:
			env = UA_load(str(p))
			for obj in env.objects:
				if obj.type == ClassIDType.Mesh:
					return env, obj
		except Exception:
			continue

	raise RuntimeError("Không tìm thấy template .asset có Mesh. Hãy cung cấp --template trỏ tới file .asset chứa Mesh.")


def apply_obj_to_mesh(mesh, vertices, uvs, normals, faces, mesh_name: str = "ImportedMesh"):
	"""
	Ghi dữ liệu OBJ vào instance Mesh đã đọc từ template.
	"""
	mesh.name = mesh_name

	# Vertices
	mesh.m_VertexCount = len(vertices)
	mesh.m_Vertices = [c for v in vertices for c in (v[0], v[1], v[2])]

	# UV0 (nếu có)
	if uvs and len(uvs) == mesh.m_VertexCount:
		mesh.m_UV0 = [c for uv in uvs for c in (uv[0], uv[1])]
	else:
		mesh.m_UV0 = []

	# Normals (nếu có)
	if normals and len(normals) == mesh.m_VertexCount:
		mesh.m_Normals = [c for n in normals for c in (n[0], n[1], n[2])]
	else:
		mesh.m_Normals = []

	# Indices (triangles)
	indices: List[int] = []
	for (i0, i1, i2) in faces:
		indices.extend([i0, i1, i2])
	mesh.m_Indices = indices

	# SubMesh đơn (toàn bộ mặt)
	class SubMeshLite:
		def __init__(self, count: int):
			self.firstByte = 0
			self.indexCount = count
			self.topology = GfxPrimitiveType.Triangles
			self.firstVertex = 0
			self.vertexCount = mesh.m_VertexCount
			class AABB:
				def __init__(self):
					self.m_Center = (0.0, 0.0, 0.0)
					self.m_Extent = (0.5, 0.5, 0.5)
			self.localAABB = AABB()

	mesh.m_SubMeshes = [SubMeshLite(len(indices))]

	# Local AABB
	if vertices:
		xs = [v[0] for v in vertices]
		ys = [v[1] for v in vertices]
		zs = [v[2] for v in vertices]
		min_x, max_x = min(xs), max(xs)
		min_y, max_y = min(ys), max(ys)
		min_z, max_z = min(zs), max(zs)
		center = ((min_x + max_x) / 2.0, (min_y + max_y) / 2.0, (min_z + max_z) / 2.0)
		extent = ((max_x - min_x) / 2.0, (max_y - min_y) / 2.0, (max_z - min_z) / 2.0)
		try:
			mesh.m_LocalAABB.m_Center = center
			mesh.m_LocalAABB.m_Extent = extent
		except Exception:
			pass



def save_serialized_asset_copy(env, out_path: Path) -> None:
	"""
	Lưu lại SerializedFile chứa mesh đã chỉnh sửa ra file .asset mới.
	"""
	assets = env.assets
	if not assets:
		raise RuntimeError("Environment không có SerializedFile để lưu.")
	sf = assets[0]
	data = sf.save()
	out_path.parent.mkdir(parents=True, exist_ok=True)
	out_path.write_bytes(data)


def main():
	parser = argparse.ArgumentParser(description="Import OBJ vào .asset (Mesh) dựa trên Unity AOV")
	parser.add_argument("--obj", required=True, help="Đường dẫn file .obj nguồn")
	parser.add_argument("--out", required=True, help="Đường dẫn file .asset đầu ra")
	parser.add_argument("--template", help="File .asset mẫu (chứa sẵn Mesh). Nếu không có, tool sẽ tự tìm trong workspace")
	parser.add_argument("--name", default="ImportedMesh", help="Tên Mesh trong asset")
	args = parser.parse_args()

	obj_path = Path(args.obj)
	out_path = Path(args.out)
	template_path = Path(args.template) if args.template else None

	if not obj_path.exists():
		print(f"[ERROR] Không tìm thấy OBJ: {obj_path}")
		sys.exit(2)

	# Parse OBJ
	vertices, uvs, normals, faces = parse_obj(obj_path)
	if not vertices or not faces:
		print("[ERROR] OBJ không hợp lệ hoặc thiếu vertices/faces")
		sys.exit(3)

	# Load template + lấy Mesh
	env = None
	mesh_obj_reader = None
	if template_path:
		if not template_path.exists():
			print(f"[ERROR] Không tìm thấy template: {template_path}")
			sys.exit(4)
		env = UA_load(str(template_path))
		for obj in env.objects:
			if obj.type == ClassIDType.Mesh:
				mesh_obj_reader = obj
				break
		if mesh_obj_reader is None:
			print("[ERROR] Template không chứa Mesh. Hãy chọn template khác hoặc bỏ --template để tool tự tìm.")
			sys.exit(5)
	else:
		# Tự tìm trong workspace
		workspace = Path("/").joinpath("workspace") if Path("/workspace").exists() else Path.cwd()
		try:
			env, mesh_obj_reader = find_template_with_mesh([
				workspace,
				workspace / "asetoutput",
				workspace / "Unity AOV",
				workspace / "Assetbundle Tool",
			])
		except Exception as e:
			print("[ERROR]", e)
			sys.exit(6)

	# Đọc Mesh class, áp OBJ, ghi lại
	mesh = mesh_obj_reader.read()
	apply_obj_to_mesh(mesh, vertices, uvs, normals, faces, args.name)
	# Ghi raw data vào object (Object.save -> write typetree -> set into reader)
	mesh.save()

	# Lưu thành .asset mới
	save_serialized_asset_copy(env, out_path)
	print(f"[OK] Đã tạo asset: {out_path}")


if __name__ == "__main__":
	main()