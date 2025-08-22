import os
from py3daov import config
import py3daov

I_O = input("Nhập thư mục: ").strip()

dirs = I_O.split()
if len(dirs) == 1:
    input_dir = dirs[0]
    output_dir = dirs[0]
elif len(dirs) == 2:
    input_dir, output_dir = dirs
else:
    print("Lỗi: nhập đúng hộ cái")
    exit()

os.makedirs(output_dir, exist_ok=True)

print("1. to UABE")
print("2. to AOV")
choice = input("    ==> ").strip()

if choice == "1":
    mF = config.bAOV
    rF = config.notAOV
elif choice == "2":
    mF = config.notAOV
    rF = config.bAOV
else:
    print("Lựa chọn không hợp lệ.")
    exit()

for fileN in os.listdir(input_dir):
    inp = os.path.join(input_dir, fileN)
    outp = os.path.join(output_dir, fileN)

    if not os.path.isfile(inp):
        continue

    print(f"Đang xẻ mổ: {fileN}")
    mF()
    try:
        env = py3daov.load(inp)
        rF()
        with open(outp, "wb") as f:
            f.write(env.file.save("none"))
    except Exception as e:
        print(f"Lỗi khi xử lý file {fileN}: {e}")
        continue

print("Xong")
