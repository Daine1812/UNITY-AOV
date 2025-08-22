import os
import UnityPy_AOV
from PIL import Image

def extract_assets(input_dir, output_dir):
    """
    Trích xuất tài nguyên (hình ảnh Texture2D) từ các tệp AssetBundle.

    Args:
        input_dir (str): Đường dẫn tới thư mục chứa các tệp AssetBundle.
        output_dir (str): Đường dẫn tới thư mục lưu các tài nguyên đã trích xuất.
    """
    for file_name in os.listdir(input_dir):
        if file_name.endswith(".assetbundle"):
            file_path = os.path.join(input_dir, file_name)
            print(f"Loading AssetBundle: {file_path}")
            
            # Tạo thư mục đầu ra cho từng file
            asset_output_dir = os.path.join(output_dir, os.path.splitext(file_name)[0])
            os.makedirs(asset_output_dir, exist_ok=True)

            env = UnityPy_AOV.load(file_path)

            for obj in env.objects:
                if obj.type.name == "Texture2D":
                    data = obj.read()
                    dest = os.path.join(asset_output_dir, f'{data.m_Name}.png')
                    print(f"Exporting Texture2D: {dest}")
                    img = data.image
                    img.save(dest)

def import_assets(input_dir, output_dir):
    """
    Đóng gói lại tài nguyên đã chỉnh sửa vào một tệp AssetBundle mới.

    Args:
        input_dir (str): Đường dẫn tới thư mục chứa các tệp tài nguyên đã chỉnh sửa (hình ảnh PNG).
        output_dir (str): Đường dẫn tới thư mục lưu AssetBundle đầu ra.
    """
    for folder_name in os.listdir(input_dir):
        folder_path = os.path.join(input_dir, folder_name)

        if os.path.isdir(folder_path):
            print(f"Loading input directory for re-import: {folder_path}")
            
            
            original_assetbundle = os.path.join("C:/Users/acer/Documents/texture2d", f"{folder_name}.assetbundle")
            env = UnityPy_AOV.load(original_assetbundle)

            for obj in env.objects:
                if obj.type.name == "Texture2D":
                    data = obj.read()
                    image_path = os.path.join(folder_path, f"{data.m_Name}.png")
                    if os.path.exists(image_path):
                        print(f"Re-importing image: {image_path}")
                        pil_img = Image.open(image_path)
                        data.image = pil_img
                        data.save()

            
            output_file = os.path.join(output_dir, f"{folder_name}.assetbundle")
            with open(output_file, "wb") as f:
                if hasattr(env, "file") and env.file:
                    bundle_data = env.file.save("lz4")
                    f.write(bundle_data)
                    print(f"AssetBundle saved: {output_file}")
                else:
                    print(f"Error: Unable to save AssetBundle for {folder_name}")

def main_menu():
    input_directory = "C:/Users/acer/Documents/texture2d"
    output_directory = "C:/Users/acer/Documents/texture2d/asetoutput/"

    while True:
        print("\nMenu:")
        print("1. Trích xuất tài nguyên từ AssetBundle")
        print("2. Đóng gói tài nguyên đã chỉnh sửa vào AssetBundle")
        print("3. Thoát")
        choice = input("Vui lòng chọn (1/2/3): ")

        if choice == "1":
            print("Bắt đầu trích xuất tài nguyên...")
            extract_assets(input_directory, output_directory)
            print("Hoàn thành trích xuất tài nguyên.")
        elif choice == "2":
            print("Bắt đầu đóng gói tài nguyên...")
            import_assets(output_directory, output_directory)
            print("Hoàn thành đóng gói tài nguyên.")
        elif choice == "3":
            print("Thoát chương trình.")
            break
        else:
            print("Lựa chọn không hợp lệ. Vui lòng thử lại.")

if __name__ == "__main__":
    main_menu()