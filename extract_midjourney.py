from pathlib import Path
from zipfile import ZipFile, BadZipFile
import shutil
import sys


# ============================================================
# SETTINGS
# ============================================================

# Your downloaded GenImage archive
ZIP_PATH = Path(r"C:\imagenet_midjourney.zip")

# Automatically use the current Veritas project folder
PROJECT_ROOT = Path(__file__).resolve().parent

# Your existing Veritas destination
DESTINATION = (
    PROJECT_ROOT
    / "dataset"
    / "ai_image_detector"
    / "train"
    / "ai_generated"
    / "midjourney"
)

# Number of usable images we want
TARGET_IMAGES = 1000

# Image formats we accept
VALID_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".webp",
    ".bmp",
}


# ============================================================
# FIND MIDJOURNEY AI FILES
# ============================================================

def get_candidate_files(zip_file):
    """
    Find files inside the ZIP belonging to:

        train/ai/

    We ignore:
        train/nature/
        val/
        other folders
    """

    candidates = []

    for info in zip_file.infolist():

        if info.is_dir():
            continue

        name = info.filename.replace("\\", "/")
        lower_name = name.lower()

        # We only want train/ai/
        if "/train/ai/" not in f"/{lower_name}":
            continue

        extension = Path(name).suffix.lower()

        if extension not in VALID_EXTENSIONS:
            continue

        candidates.append(info)

    return candidates


# ============================================================
# EXTRACT ONE FILE SAFELY
# ============================================================

def extract_one(zip_file, info, destination):
    """
    Extract one image from the ZIP.
    Returns True if successful.
    """

    filename = Path(info.filename).name

    # Avoid accidental duplicate names
    output_path = destination / filename

    try:
        with zip_file.open(info, "r") as source:
            with output_path.open("wb") as target:
                shutil.copyfileobj(source, target)

        # Basic sanity check:
        if output_path.stat().st_size == 0:
            output_path.unlink(missing_ok=True)
            return False

        return True

    except Exception as exc:
        # Remove partial output if one was created
        output_path.unlink(missing_ok=True)

        print(
            f"SKIPPED: {filename}"
        )
        print(
            f"Reason: {type(exc).__name__}: {exc}"
        )

        return False


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 70)
    print("VERITAS - MIDJOURNEY SUBSET EXTRACTION")
    print("=" * 70)
    print()

    # --------------------------------------------------------
    # Check ZIP exists
    # --------------------------------------------------------

    if not ZIP_PATH.exists():
        print("ERROR:")
        print(f"ZIP file not found:")
        print(ZIP_PATH)
        print()
        print("Change ZIP_PATH in the script to the correct location.")
        sys.exit(1)

    # --------------------------------------------------------
    # Create destination
    # --------------------------------------------------------

    DESTINATION.mkdir(
        parents=True,
        exist_ok=True
    )

    print(f"ZIP:")
    print(ZIP_PATH)
    print()

    print("Destination:")
    print(DESTINATION)
    print()

    print(f"Target images: {TARGET_IMAGES}")
    print()

    # --------------------------------------------------------
    # Open ZIP
    # --------------------------------------------------------

    try:
        zip_file = ZipFile(ZIP_PATH, "r")
    except BadZipFile:
        print("ERROR: The ZIP archive appears to be corrupted.")
        print("Do not continue until the ZIP has been downloaded correctly.")
        sys.exit(1)
    except Exception as exc:
        print("ERROR opening ZIP:")
        print(type(exc).__name__, exc)
        sys.exit(1)

    with zip_file:

        print("ZIP opened successfully.")
        print()

        # ----------------------------------------------------
        # Find candidates
        # ----------------------------------------------------

        candidates = get_candidate_files(zip_file)

        print(
            f"Candidate train/ai images found in archive: "
            f"{len(candidates)}"
        )
        print()

        if not candidates:
            print(
                "ERROR: No train/ai image files were found."
            )
            print()
            print(
                "The internal ZIP folder structure may be different."
            )
            sys.exit(1)

        # ----------------------------------------------------
        # Extract successfully readable images
        # ----------------------------------------------------

        extracted = 0
        attempted = 0

        for info in candidates:

            if extracted >= TARGET_IMAGES:
                break

            attempted += 1

            filename = Path(info.filename).name

            print(
                f"[{extracted + 1}/{TARGET_IMAGES}] "
                f"Trying: {filename}"
            )

            success = extract_one(
                zip_file,
                info,
                DESTINATION
            )

            if success:
                extracted += 1

        # ----------------------------------------------------
        # Result
        # ----------------------------------------------------

        print()
        print("=" * 70)
        print("EXTRACTION COMPLETE")
        print("=" * 70)
        print()

        print(f"Requested images : {TARGET_IMAGES}")
        print(f"Successfully extracted : {extracted}")
        print(f"Archive entries attempted : {attempted}")
        print()

        print("Destination:")
        print(DESTINATION)
        print()

        if extracted >= TARGET_IMAGES:
            print("SUCCESS: 1,000 Midjourney AI images are ready.")
        else:
            print(
                "WARNING: Could not extract the requested number "
                "of images."
            )

            print()
            print(
                "This may mean the ZIP contains corrupted/unavailable "
                "entries."
            )


if __name__ == "__main__":
    main()