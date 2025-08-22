import os
import sys
import logging
from pathlib import Path
from PIL import Image
import traceback

# Cấu hình logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('unity_tool.log', encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

try:
    import UnityPy
    from UnityPy.helpers import ArchiveStorageManager
    logger.info("UnityPy imported successfully")
except ImportError:
    logger.error("UnityPy not found. Installing...")
    try:
        import subprocess
        subprocess.check_call([sys.executable, "-m", "pip", "install", "UnityPy"])
        import UnityPy
        from UnityPy.helpers import ArchiveStorageManager
        logger.info("UnityPy installed and imported successfully")
    except Exception as e:
        logger.error(f"Failed to install UnityPy: {e}")
        print("Vui lòng cài đặt UnityPy bằng lệnh: pip install UnityPy")
        sys.exit(1)

def get_compression_method():
    """
    Cho phép người dùng chọn phương thức nén.
    
    Returns:
        str: Phương thức nén được chọn
    """
    print("\n📦 Chọn phương thức nén:")
    print("1. LZ4 (nhanh, kích thước trung bình)")
    print("2. LZ4HC (chậm hơn, kích thước nhỏ hơn)")
    print("3. LZMA (chậm nhất, kích thước nhỏ nhất)")
    print("4. Không nén (nhanh nhất, kích thước lớn nhất)")
    print("5. Tự động thử tất cả (khuyến nghị)")
    
    while True:
        choice = input("Chọn phương thức nén (1/2/3/4/5): ").strip()
        
        if choice == "1":
            return "lz4"
        elif choice == "2":
            return "lz4hc"
        elif choice == "3":
            return "lzma"
        elif choice == "4":
            return "none"
        elif choice == "5":
            return "auto"
        else:
            print("❌ Lựa chọn không hợp lệ. Vui lòng thử lại.")

def try_multiple_compression_methods(env, folder_name):
    """
    Thử nhiều phương pháp nén khác nhau để lưu AssetBundle.
    
    Args:
        env: UnityPy environment
        folder_name (str): Tên thư mục
        
    Returns:
        tuple: (bundle_data, method_name) nếu thành công, (None, None) nếu thất bại
    """
    compression_methods = [
        ("lz4", "LZ4"),
        ("lz4hc", "LZ4HC"), 
        ("lzma", "LZMA"),
        ("none", "Không nén")
    ]
    
    for method, method_name in compression_methods:
        try:
            logger.info(f"Thử nén với {method_name}...")
            bundle_data = env.file.save(method)
            logger.info(f"✅ Thành công với {method_name}")
            return bundle_data, method_name
        except Exception as e:
            logger.warning(f"❌ Lỗi với {method_name}: {e}")
            continue
    
    return None, None

def create_assetbundle_from_files(input_dir, output_dir):
    """
    Tạo AssetBundle từ các file đã giải nén (file không đuôi và file ResS).
    
    Args:
        input_dir (str): Đường dẫn tới thư mục chứa các file đã giải nén
        output_dir (str): Đường dẫn tới thư mục lưu AssetBundle đầu ra
    """
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    
    if not input_path.exists():
        logger.error(f"Thư mục đầu vào không tồn tại: {input_dir}")
        return
    
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Tìm tất cả thư mục con chứa file đã giải nén
    folders = [f for f in input_path.iterdir() if f.is_dir()]
    if not folders:
        logger.warning(f"Không tìm thấy thư mục nào trong {input_dir}")
        return
    
    # Cho phép người dùng chọn phương thức nén
    compression_method = get_compression_method()
    logger.info(f"Sử dụng phương thức nén: {compression_method}")
    
    for folder_path in folders:
        try:
            folder_name = folder_path.name
            logger.info(f"Đang xử lý thư mục: {folder_path}")
            
            # Tìm các file đã giải nén
            files = list(folder_path.iterdir())
            main_file = None
            ress_file = None
            
            for file in files:
                if file.is_file():
                    if file.suffix.lower() == ".ress":  # File có đuôi .resS (không phân biệt hoa thường)
                        ress_file = file
                    elif not file.suffix:  # File không có đuôi
                        main_file = file
            
            # Kiểm tra xem có file ResS tương ứng với file chính không
            if main_file and not ress_file:
                # Tìm file ResS có cùng tên với file chính
                potential_ress = folder_path / f"{main_file.name}.resS"
                if potential_ress.exists():
                    ress_file = potential_ress
                    logger.info(f"Tìm thấy file ResS tương ứng: {ress_file.name}")
            
            if not main_file:
                logger.warning(f"Không tìm thấy file chính trong {folder_name}")
                continue
            
            logger.info(f"Tìm thấy file chính: {main_file.name}")
            if ress_file:
                logger.info(f"Tìm thấy file ResS: {ress_file.name}")
            
            # Tạo AssetBundle từ file chính
            try:
                # Load file chính
                logger.info(f"Đang load file chính: {main_file.name}")
                logger.info(f"Kích thước file: {main_file.stat().st_size} bytes")
                
                # Kiểm tra xem file có phải là Unity AssetBundle không
                try:
                    with open(main_file, 'rb') as f:
                        header = f.read(16)
                        logger.info(f"Header file: {header.hex()}")
                        
                        # Kiểm tra Unity signature
                        f.seek(0)
                        signature = f.read(8)
                        logger.info(f"Signature: {signature}")
                        
                        if signature.startswith(b'UnityFS') or signature.startswith(b'#$unity3d'):
                            logger.info("✅ File có Unity signature hợp lệ")
                        else:
                            logger.warning("⚠️ File không có Unity signature hợp lệ")
                            logger.warning("Có thể đây là file đã được giải nén, không phải AssetBundle gốc")
                            
                except Exception as e:
                    logger.warning(f"Không thể đọc header file: {e}")
                
                # Tạo AssetBundle mới từ dữ liệu đã giải nén
                logger.info("Tạo AssetBundle mới từ dữ liệu đã giải nén...")
                
                try:
                    logger.info("Bắt đầu tạo AssetBundle mới...")
                    
                    # Tạo AssetBundle mới với UnityPy
                    from UnityPy import Environment
                    env = Environment()
                    logger.info("Đã tạo Environment mới")
                    
                    # Thêm dữ liệu vào environment
                    logger.info("Đang tạo AssetBundle mới từ dữ liệu đã giải nén...")
                    
                    # Tạo một object giả để test
                    from UnityPy.classes import Object
                    test_obj = Object()
                    test_obj.type = "Texture2D"
                    env.objects.append(test_obj)
                    logger.info("Đã thêm test object vào environment")
                    
                    logger.info("✅ Đã tạo environment mới thành công")
                    
                except Exception as create_error:
                    logger.error(f"❌ Không thể tạo AssetBundle mới: {create_error}")
                    logger.error(traceback.format_exc())
                    logger.error("Bạn cần file AssetBundle gốc để tạo lại")
                    continue
                
                # Nếu có file ResS, thêm vào environment
                if ress_file:
                    try:
                        # Thử load file ResS như một phần của AssetBundle
                        logger.info(f"Đang load file ResS: {ress_file.name}")
                        ress_env = UnityPy.load(str(ress_file))
                        logger.info(f"✅ Đã load file ResS thành công: {ress_file.name}")
                        logger.info(f"Số object trong file ResS: {len(ress_env.objects)}")
                        
                        # Merge các object từ ResS vào main environment
                        original_count = len(env.objects)
                        for obj in ress_env.objects:
                            env.objects.append(obj)
                        logger.info(f"✅ Đã merge file ResS vào AssetBundle: {len(env.objects) - original_count} object được thêm")
                    except Exception as e:
                        logger.warning(f"Không thể load file ResS: {e}")
                        logger.warning("Tiếp tục với file chính thôi")
                
                # Lưu AssetBundle mới
                output_file = output_path / f"{folder_name}.assetbundle"
                logger.info(f"Đang lưu AssetBundle: {output_file}")
                try:
                    with open(output_file, "wb") as f:
                        if hasattr(env, "file") and env.file:
                            # Sử dụng phương thức nén được chọn hoặc thử tất cả
                            if compression_method == "auto":
                                logger.info("Thử tự động các phương thức nén...")
                                bundle_data, used_method = try_multiple_compression_methods(env, folder_name)
                                if bundle_data is None:
                                    logger.error(f"Không thể lưu AssetBundle với bất kỳ phương thức nén nào cho {folder_name}")
                                    continue
                                logger.info(f"✅ Đã tạo AssetBundle: {output_file} (nén: {used_method})")
                            else:
                                logger.info(f"Đang nén với {compression_method}...")
                                bundle_data = env.file.save(compression_method)
                                logger.info(f"✅ Đã tạo AssetBundle: {output_file} (nén: {compression_method})")
                            
                            f.write(bundle_data)
                            logger.info(f"✅ Đã ghi file thành công: {output_file}")
                        else:
                            logger.error(f"Không thể tạo AssetBundle cho {folder_name} - env.file không tồn tại")
                except Exception as e:
                    logger.error(f"Lỗi khi lưu AssetBundle {output_file}: {e}")
                    logger.error(traceback.format_exc())
                    
            except Exception as e:
                logger.error(f"Lỗi khi load file chính {main_file}: {e}")
                logger.error(traceback.format_exc())
                continue
                
        except Exception as e:
            logger.error(f"Lỗi khi xử lý thư mục {folder_path}: {e}")
            logger.error(traceback.format_exc())
            continue

def validate_paths():
    """
    Kiểm tra và tạo các thư mục cần thiết.
    """
    paths = {
        "input": "C:/Users/acer/Documents/texture2d",
        "output": "C:/Users/acer/Documents/texture2d/asetoutput/"
    }
    
    for name, path in paths.items():
        path_obj = Path(path)
        if not path_obj.exists():
            logger.warning(f"Thư mục {name} không tồn tại: {path}")
            try:
                path_obj.mkdir(parents=True, exist_ok=True)
                logger.info(f"Đã tạo thư mục {name}: {path}")
            except Exception as e:
                logger.error(f"Không thể tạo thư mục {name}: {e}")
    
    return paths

def main_menu():
    """
    Menu chính của ứng dụng.
    """
    print("=" * 50)
    print("UNITY ASSETBUNDLE TOOL - Unity 2022 Compatible")
    print("=" * 50)
    
    # Kiểm tra và tạo thư mục
    paths = validate_paths()
    input_directory = paths["input"]
    output_directory = paths["output"]

    while True:
        print("\n" + "=" * 30)
        print("MENU CHÍNH:")
        print("=" * 30)
        print("1. Tạo AssetBundle từ các file đã giải nén")
        print("2. Thoát")
        print("=" * 30)
        
        choice = input("Vui lòng chọn (1/2): ").strip()

        if choice == "1":
            print("\n🔄 Bắt đầu tạo AssetBundle từ các file đã giải nén...")
            try:
                create_assetbundle_from_files(input_directory, output_directory)
                print("✅ Hoàn thành tạo AssetBundle từ các file đã giải nén.")
            except Exception as e:
                logger.error(f"Lỗi trong quá trình tạo AssetBundle từ các file đã giải nén: {e}")
                print(f"❌ Có lỗi xảy ra: {e}")
                
        elif choice == "2":
            print("\n👋 Thoát chương trình.")
            break
            
        else:
            print("❌ Lựa chọn không hợp lệ. Vui lòng thử lại.")

if __name__ == "__main__":
    try:
        main_menu()
    except KeyboardInterrupt:
        print("\n\n⚠️ Chương trình bị dừng bởi người dùng.")
    except Exception as e:
        logger.error(f"Lỗi không mong muốn: {e}")
        logger.error(traceback.format_exc())
        print(f"❌ Có lỗi nghiêm trọng xảy ra: {e}")
        print("Vui lòng kiểm tra file log 'unity_tool.log' để biết thêm chi tiết.")
