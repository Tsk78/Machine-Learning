from ultralytics import YOLO

def train_hand_model():
    # Initialize with official nano pose model as base weights
    model = YOLO("yolo11n-pose.pt")

    # Train on official Ultralytics hand-keypoints dataset
    # 'device="mps"' leverages Apple Silicon Metal GPU acceleration
    model.train(
        data="hand-keypoints.yaml",
        epochs=35,
        imgsz=640,
        batch=16,
        device="mps",
        workers=4,
        project="runs/hand_pose",
        name="yolo11n_hand"
    )

if __name__ == "__main__":
    train_hand_model()