from pathlib import Path


class PackageInitializer:
    def __init__(self, target_root: str):
        self.target_root = Path(target_root)
        self.order_log = self.target_root / "order.txt"

    def ensure_target_root(self):
        self.target_root.mkdir(parents=True, exist_ok=True)

    def ensure_order_file(self):
        self.ensure_target_root()
        if not self.order_log.exists():
            self.order_log.touch()

    def init_package(self):
        self.ensure_target_root()
        self.ensure_order_file()

    def create_map_folder(self, folder_name: str) -> Path:
        """
        创建谱面目录。
        若目录已存在则抛异常，由上层决定是否跳过或报错。
        """
        self.init_package()
        dest_dir = self.target_root / folder_name

        if dest_dir.exists():
            raise FileExistsError(f"目录已存在: {dest_dir}")

        dest_dir.mkdir(parents=True, exist_ok=False)
        return dest_dir

    def get_order_log_path(self) -> Path:
        self.init_package()
        return self.order_log