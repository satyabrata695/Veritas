from pathlib import Path
import csv
import shutil
import sys


# ============================================================
# CONFIGURATION
# ============================================================

# CHANGE ONLY THIS PATH
# Point this to the extracted READFake dataset.
READFAKE_ROOT = Path(
    r"C:\Users\padhi\Downloads\READFake_Dataset_CVPR2023"
)

PROJECT_ROOT = Path(__file__).resolve().parent

RGB_DIR = READFAKE_ROOT / "READFake-RGB-images"
ANNOTATION_DIR = READFAKE_ROOT / "READFake-Image-Annotation"

DATASET_DIR = PROJECT_ROOT / "dataset"
DEEPFAKE_DIR = DATASET_DIR / "deepfake_detector"


# ============================================================
# READFAKE LABEL MAPPING
# ============================================================

# Based on the annotation data you showed:
#
# img_label = 2 -> REAL
# img_label = 0 -> FAKE

LABEL_MAP = {
    "2": "real",
    "0": "fake",
}


# ============================================================
# OUTPUT DIRECTORIES
# ============================================================

SPLITS = {
    "train": ANNOTATION_DIR / "READFake-Image-Annotation-Train.csv",
    "validation": ANNOTATION_DIR / "READFake-Image-Annotation-Validation.csv",
    "test": ANNOTATION_DIR / "READFake-Image-Annotation-Test.csv",
}


# ============================================================
# CHECK PATHS
# ============================================================

def check_paths():
    print("\nChecking paths...\n")

    if not READFAKE_ROOT.exists():
        print(f"ERROR: READFake root not found:")
        print(READFAKE_ROOT)
        sys.exit(1)

    if not RGB_DIR.exists():
        print(f"ERROR: RGB image folder not found:")
        print(RGB_DIR)
        sys.exit(1)

    if not ANNOTATION_DIR.exists():
        print(f"ERROR: Annotation folder not found:")
        print(ANNOTATION_DIR)
        sys.exit(1)

    if not DEEPFAKE_DIR.exists():
        print(f"ERROR: Your project deepfake_detector folder was not found:")
        print(DEEPFAKE_DIR)
        sys.exit(1)

    print("All required paths found.\n")


# ============================================================
# BUILD IMAGE INDEX
# ============================================================

def build_image_index():
    """
    Creates a lookup table from filename/stem to actual image path.

    This makes the script more tolerant of extensions such as:
        .jpg
        .jpeg
        .png
    """

    print("Indexing READFake RGB images...")

    extensions = {
        ".jpg",
        ".jpeg",
        ".png",
        ".bmp",
        ".webp",
    }

    index = {}

    for image_path in RGB_DIR.rglob("*"):
        if not image_path.is_file():
            continue

        if image_path.suffix.lower() not in extensions:
            continue

        # Full filename
        index[image_path.name.lower()] = image_path

        # Filename without extension
        index[image_path.stem.lower()] = image_path

    print(f"Indexed {len(index)} filename entries.\n")

    return index


# ============================================================
# FIND IMAGE
# ============================================================

def find_image(imgid, image_index):
    """
    Finds an image using:
      1. Exact filename
      2. Filename without extension

    If the CSV includes 'abc.jpg' and actual file is abc.png,
    the stem match can still find it.
    """

    imgid = str(imgid).strip()

    if not imgid:
        return None

    key = imgid.lower()

    # Exact filename
    if key in image_index:
        return image_index[key]

    # Remove extension if CSV contains one
    stem = Path(imgid).stem.lower()

    if stem in image_index:
        return image_index[stem]

    return None


# ============================================================
# PROCESS ONE SPLIT
# ============================================================

def process_split(split_name, csv_path, image_index):

    print("=" * 70)
    print(f"Processing: {split_name}")
    print(f"CSV: {csv_path}")
    print("=" * 70)

    if not csv_path.exists():
        # Fallback to alternate file names if present
        alt_names = {
            "train": ANNOTATION_DIR / "READFake-Image-Annotation-Training-Data.csv",
            "validation": ANNOTATION_DIR / "READFake-Image-Annotation-Validation-Data.csv",
            "test": ANNOTATION_DIR / "READFake-Image-Annotation-Testing-Data.csv",
        }
        alt_path = alt_names.get(split_name)
        if alt_path and alt_path.exists():
            csv_path = alt_path
        else:
            print(f"ERROR: CSV not found: {csv_path}")
            return

    real_dir = DEEPFAKE_DIR / split_name / "real"
    fake_dir = DEEPFAKE_DIR / split_name / "fake"

    real_dir.mkdir(parents=True, exist_ok=True)
    fake_dir.mkdir(parents=True, exist_ok=True)

    real_count = 0
    fake_count = 0
    missing_count = 0
    invalid_label_count = 0

    with csv_path.open(
        "r",
        encoding="utf-8-sig",
        newline=""
    ) as f:

        reader = csv.DictReader(f)

        # ----------------------------------------------------
        # Check required columns
        # ----------------------------------------------------

        if reader.fieldnames is None:
            print("ERROR: CSV has no header.")
            return

        print("Columns found:")
        print(reader.fieldnames)

        if "imgid" not in reader.fieldnames:
            print("ERROR: 'imgid' column not found.")
            return

        if "img_label" not in reader.fieldnames:
            print("ERROR: 'img_label' column not found.")
            return

        print()

        # ----------------------------------------------------
        # Process rows
        # ----------------------------------------------------

        for row_number, row in enumerate(reader, start=2):

            imgid = str(row.get("imgid", "")).strip()
            label = str(row.get("img_label", "")).strip()

            if not imgid:
                print(f"Row {row_number}: empty imgid -> skipped")
                continue

            if label not in LABEL_MAP:
                print(
                    f"Row {row_number}: "
                    f"unknown img_label={label} "
                    f"for image={imgid}"
                )
                invalid_label_count += 1
                continue

            image_path = find_image(imgid, image_index)

            if image_path is None:
                print(
                    f"Missing image: {imgid}"
                )
                missing_count += 1
                continue

            destination_class = LABEL_MAP[label]

            if destination_class == "real":
                destination_dir = real_dir
            else:
                destination_dir = fake_dir

            destination = destination_dir / image_path.name

            # Avoid accidentally overwriting an existing different file.
            if destination.exists():
                print(
                    f"Already exists, skipping: "
                    f"{destination.name}"
                )
                continue

            shutil.copy2(image_path, destination)

            if destination_class == "real":
                real_count += 1
            else:
                fake_count += 1

    print()
    print(f"{split_name.upper()} SUMMARY")
    print(f"Real images copied : {real_count}")
    print(f"Fake images copied : {fake_count}")
    print(f"Missing images     : {missing_count}")
    print(f"Invalid labels     : {invalid_label_count}")
    print()


# ============================================================
# MAIN
# ============================================================

def main():

    print("\n")
    print("=" * 70)
    print("READFake -> Salvus Deepfake Dataset Preparation")
    print("=" * 70)
    print()

    check_paths()

    image_index = build_image_index()

    for split_name, csv_path in SPLITS.items():
        process_split(
            split_name,
            csv_path,
            image_index
        )

    print("=" * 70)
    print("DONE")
    print("=" * 70)
    print()
    print("Your existing folder structure was NOT changed.")
    print()
    print("Images were placed into:")
    print(DEEPFAKE_DIR)
    print()


if __name__ == "__main__":
    main()
