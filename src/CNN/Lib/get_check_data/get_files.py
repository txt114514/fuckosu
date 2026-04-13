from pathlib import Path
import zipfile
import tempfile
import shutil

EXPORT_DIR = Path("/home/dev/workspace/osu-lazer/exports")
TARGET_ROOT = Path("/home/dev/workspace/training_package/match-completed_package")
ORDER_LOG = TARGET_ROOT / "order.txt"

KEYWORD = "normal"


def is_target_osu(path: Path, keyword: str) -> bool:
    return (
        path.is_file()
        and path.suffix.lower() == ".osu"
        and keyword.lower() in path.name.lower()
    )


def load_registered_names(order_file: Path) -> list[str]:
    if not order_file.exists():
        return []

    lines = order_file.read_text(encoding="utf-8").splitlines()
    return [line.strip() for line in lines if line.strip()]


def main(keyword: str = KEYWORD):
    TARGET_ROOT.mkdir(parents=True, exist_ok=True)

    osz_files = sorted(
        EXPORT_DIR.glob("*.osz"),
        key=lambda p: p.stat().st_mtime_ns
    )

    if not osz_files:
        print(f"没有在 {EXPORT_DIR} 中找到 .osz 文件")
        return

    # 已经登记过的 .osu 文件名（不带后缀）
    registered_names = load_registered_names(ORDER_LOG)
    registered_set = set(registered_names)

    success_count = 0
    skip_count = 0

    for osz_path in osz_files:
        try:
            with tempfile.TemporaryDirectory() as tmpdir:
                tmp_path = Path(tmpdir)

                with zipfile.ZipFile(osz_path, "r") as zf:
                    zf.extractall(tmp_path)

                matched_osu_files = sorted(
                    [p for p in tmp_path.rglob("*") if is_target_osu(p, keyword)],
                    key=lambda p: p.name.lower()
                )

                if not matched_osu_files:
                    print(f"[跳过] {osz_path.name}：未找到文件名包含 {keyword!r} 的 .osu 文件")
                    skip_count += 1
                    continue

                chosen_osu = matched_osu_files[0]
                osu_base_name = chosen_osu.stem

                # 如果 order.txt 已经记录过，则跳过整个流程
                if osu_base_name in registered_set:
                    print(f"[跳过] {osz_path.name}：{osu_base_name} 已在 order.txt 中登记过")
                    skip_count += 1
                    continue

                # 文件夹名使用 .osu 文件名（不带后缀）
                dest_dir = TARGET_ROOT / osu_base_name

                # 未登记过但目录已存在，说明发生重名冲突或目录状态异常，直接报错
                if dest_dir.exists():
                    raise FileExistsError(
                        f"目标文件夹已存在但 order.txt 未登记，发生重名冲突或状态不一致：{dest_dir}"
                    )

                dest_dir.mkdir(parents=True, exist_ok=False)

                dest_file = dest_dir / chosen_osu.name
                shutil.copy2(chosen_osu, dest_file)

                # 追加登记到 order.txt
                with ORDER_LOG.open("a", encoding="utf-8") as f:
                    if ORDER_LOG.stat().st_size > 0:
                        f.write("\n")
                    f.write(osu_base_name)

                registered_set.add(osu_base_name)

                print(f"[完成] {osz_path.name} -> {dest_file}")
                success_count += 1

                if len(matched_osu_files) > 1:
                    print(f"       注意：发现多个匹配 .osu，已复制第一个：{chosen_osu.name}")

        except zipfile.BadZipFile:
            print(f"[错误] {osz_path.name} 不是有效的 ZIP/.osz 文件")
            skip_count += 1
        except Exception as e:
            print(f"[错误] 处理 {osz_path.name} 时出错：{e}")
            raise

    print()
    print(f"处理完成：成功 {success_count} 个，跳过/失败 {skip_count} 个")
    print(f"排序记录已写入：{ORDER_LOG}")


if __name__ == "__main__":
    main()