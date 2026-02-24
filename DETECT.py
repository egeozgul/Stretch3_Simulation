import cv2
import cv2.aruco as aruco
import argparse

# -----------------------------
# All OpenCV ArUco dictionaries
# -----------------------------
ARUCO_DICTS = {
    "DICT_4X4_50": aruco.DICT_4X4_50,
    "DICT_4X4_100": aruco.DICT_4X4_100,
    "DICT_4X4_250": aruco.DICT_4X4_250,
    "DICT_4X4_1000": aruco.DICT_4X4_1000,

    "DICT_5X5_50": aruco.DICT_5X5_50,
    "DICT_5X5_100": aruco.DICT_5X5_100,
    "DICT_5X5_250": aruco.DICT_5X5_250,
    "DICT_5X5_1000": aruco.DICT_5X5_1000,

    "DICT_6X6_50": aruco.DICT_6X6_50,
    "DICT_6X6_100": aruco.DICT_6X6_100,
    "DICT_6X6_250": aruco.DICT_6X6_250,
    "DICT_6X6_1000": aruco.DICT_6X6_1000,

    "DICT_7X7_50": aruco.DICT_7X7_50,
    "DICT_7X7_100": aruco.DICT_7X7_100,
    "DICT_7X7_250": aruco.DICT_7X7_250,
    "DICT_7X7_1000": aruco.DICT_7X7_1000,

    "DICT_ARUCO_ORIGINAL": aruco.DICT_ARUCO_ORIGINAL
}


# -----------------------------
# Detect dictionary + IDs
# -----------------------------
def detect_marker(frame):

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    for name, dict_id in ARUCO_DICTS.items():

        dictionary = aruco.getPredefinedDictionary(dict_id)
        detector = aruco.ArucoDetector(dictionary)

        corners, ids, rejected = detector.detectMarkers(gray)

        if ids is not None and len(ids) > 0:
            print("\n✅ MARKER FOUND")
            print("Dictionary:", name)
            print("IDs:", ids.flatten())

            # Draw detection
            aruco.drawDetectedMarkers(frame, corners, ids)

            cv2.putText(
                frame,
                f"{name} | ID: {ids.flatten()}",
                (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 255, 0),
                2,
            )

            return frame, True

    return frame, False


# -----------------------------
# Main
# -----------------------------
def main():

    parser = argparse.ArgumentParser()
    parser.add_argument("--image", type=str, help="Path to image")
    args = parser.parse_args()

    # ---- IMAGE MODE ----
    if args.image:
        frame = cv2.imread(args.image)

        result, found = detect_marker(frame)

        if not found:
            print("❌ No ArUco marker detected")

        cv2.imshow("Result", result)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    # ---- WEBCAM MODE ----
    else:
        cap = cv2.VideoCapture(0)

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            result, _ = detect_marker(frame)

            cv2.imshow("Unknown ArUco Detector", result)

            if cv2.waitKey(1) & 0xFF == 27:  # ESC to quit
                break

        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
