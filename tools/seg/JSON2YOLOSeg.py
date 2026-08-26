import os
import json


# =========================================
# JSON(LabelMe Polygon) -> YOLO Seg
# =========================================
class JSON2YOLOSeg:

    def __init__(self, classes, output_dir, image_folder=None):

        self.classes = classes

        self.class_map = {
            name: idx
            for idx, name in enumerate(classes)
        }

        self.output_dir = output_dir
        self.image_folder = image_folder

        os.makedirs(self.output_dir, exist_ok=True)

    # =========================================
    # Convert Single JSON
    # =========================================
    def convert(self, json_path):

        try:

            with open(json_path, 'r', encoding='utf-8') as f:

                data = json.load(f)

        except Exception as e:

            print(f"[ERR] Failed to read: {json_path}")
            print(e)

            return

        image_width = data.get("imageWidth")
        image_height = data.get("imageHeight")

        if not image_width or not image_height:
            # Fallback: search image in image_folder or json directory
            base_name = os.path.splitext(os.path.basename(json_path))[0]
            search_folders = []
            if self.image_folder and os.path.exists(self.image_folder):
                search_folders.append(self.image_folder)
            search_folders.append(os.path.dirname(json_path))

            for fld in search_folders:
                for ext in ['.jpg', '.jpeg', '.png', '.bmp', '.webp', '.tif', '.tiff', '.JPG', '.PNG', '.JPEG', '.BMP']:
                    c_path = os.path.join(fld, base_name + ext)
                    if os.path.exists(c_path):
                        try:
                            from PIL import Image
                            with Image.open(c_path) as im:
                                image_width, image_height = im.size
                            break
                        except Exception:
                            try:
                                import cv2
                                img = cv2.imread(c_path)
                                if img is not None:
                                    image_height, image_width = img.shape[:2]
                                    break
                            except Exception:
                                pass
                if image_width and image_height:
                    break

        if not image_width or not image_height:

            print(f"[WARN] Missing image size: {json_path}")

            return

        lines = []

        # =========================================
        # Parse Shapes
        # =========================================
        for shape in data.get("shapes", []):

            try:

                label = str(shape["label"]).strip()

                if label not in self.class_map:

                    print(f"[WARN] Unknown class: {label}")

                    continue

                class_id = self.class_map[label]

                shape_type = shape.get("shape_type", "polygon")
                points = shape.get("points", [])

                # Support rectangle shape in LabelMe by converting to 4 polygon vertices
                if shape_type == "rectangle" and len(points) == 2:
                    p1, p2 = points[0], points[1]
                    points = [[p1[0], p1[1]], [p2[0], p1[1]], [p2[0], p2[1]], [p1[0], p2[1]]]
                elif shape_type != "polygon" or len(points) < 3:

                    print(f"[WARN] Skip non-polygon/rectangle shape ({shape_type}): {json_path}")

                    continue

                seg_points = []

                for pt in points:

                    x = float(pt[0])
                    y = float(pt[1])

                    # normalize
                    nx = x / image_width
                    ny = y / image_height

                    # clamp
                    nx = max(0.0, min(1.0, nx))
                    ny = max(0.0, min(1.0, ny))

                    seg_points.append(f"{nx:.6f}")
                    seg_points.append(f"{ny:.6f}")

                line = f"{class_id} " + " ".join(seg_points)

                lines.append(line)

            except Exception as e:

                print(f"[ERR] Shape parse failed: {json_path}")
                print(e)

        # =========================================
        # Save TXT
        # =========================================
        base_name = os.path.splitext(
            os.path.basename(json_path)
        )[0]

        output_path = os.path.join(
            self.output_dir,
            base_name + ".txt"
        )

        try:

            with open(output_path, "w", encoding="utf-8") as f:

                f.write("\n".join(lines))

            print(f"[OK] {json_path} -> {output_path}")

        except Exception as e:

            print(f"[ERR] Save failed: {output_path}")
            print(e)

    # =========================================
    # Batch Convert
    # =========================================
    def batch_convert(self, folder_path):

        if not os.path.exists(folder_path):

            print(f"[WARN] Folder not found: {folder_path}")

            return

        files = sorted(os.listdir(folder_path))

        json_files = [
            f for f in files
            if f.lower().endswith(".json")
        ]

        print(f"[INFO] Found {len(json_files)} json files")

        for fn in json_files:

            json_path = os.path.join(folder_path, fn)

            self.convert(json_path)


# =========================================
# MAIN
# =========================================
if __name__ == "__main__":

    classes = ["1"]

    seg_converter = JSON2YOLOSeg(

        classes=classes,

        output_dir=r"Z:\VisionTek\Ultralytics\Ultralytics_YOLO\Datasets\train\labels"
    )

    seg_converter.batch_convert(

        r"Z:\VisionTek\Ultralytics\Ultralytics_YOLO\Datasets\JSON"
    )

    print("=== DONE ===")