from pathlib import Path
from PIL import Image

ROOT = Path("dataset/ai_image_detector")

SPLITS = ["train", "validation", "test"]
CLASSES = ["ai_generated", "authentic"]

VALID_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".webp",
    ".bmp"
}


def check_folder(folder: Path):
    total = 0
    valid = 0
    corrupt = 0
    unsupported = 0

    for file in folder.rglob("*"):
        if not file.is_file():
            continue

        total += 1

        if file.suffix.lower() not in VALID_EXTENSIONS:
            unsupported += 1
            continue

        try:
            with Image.open(file) as img:
                img.verify()

            valid += 1

        except Exception:
            corrupt += 1

    return total, valid, corrupt, unsupported


print("=" * 70)
print("SALVUS AI IMAGE DATASET CHECK")
print("=" * 70)

grand_total = 0

for split in SPLITS:

    print(f"\n[{split.upper()}]")

    split_total = 0

    for cls in CLASSES:

        folder = ROOT / split / cls

        if not folder.exists():
            print(f"{cls:15} : FOLDER NOT FOUND")
            continue

        total, valid, corrupt, unsupported = check_folder(folder)

        print(
            f"{cls:15} : "
            f"total={total}, "
            f"valid={valid}, "
            f"corrupt={corrupt}, "
            f"unsupported={unsupported}"
        )

        split_total += valid

    grand_total += split_total

    print(f"Valid images in {split}: {split_total}")

print("\n" + "=" * 70)
print(f"TOTAL VALID IMAGES: {grand_total}")
print("=" * 70)