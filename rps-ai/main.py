import os
import cv2
from rps_ai.vision.gesture import GestureDetector
from rps_ai.vision.classifier import GestureClassifier
from rps_ai.game.engine import GameEngine, GameState
from rps_ai.utils.overlay import overlay_transparent


class AssetManager:
    """Preloads and resizes move icon assets."""

    def __init__(self, target_size=(100, 100)):
        self.icons = {}
        moves = ["ROCK", "PAPER", "SCISSORS"]
        
        for move in moves:
            path = f"assets/{move.lower()}.png"
            if os.path.exists(path):
                img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
                if img is not None:
                    self.icons[move] = cv2.resize(img, target_size, interpolation=cv2.INTER_AREA)

    def get(self, move: str):
        return self.icons.get(move, None)


def draw_hud(frame, engine, assets: AssetManager):
    h, w, _ = frame.shape

    # 1. Scoreboard Header Banner
    score_text = f"PLAYER: {engine.user_score}  |  AI: {engine.ai_score}"
    cv2.rectangle(frame, (0, 0), (w, 60), (30, 30, 30), -1)
    cv2.putText(frame, score_text, (w // 2 - 180, 42), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)

    # 2. State-Driven Overlays
    if engine.state == GameState.WAITING:
        cv2.putText(frame, "Press SPACE to Start Round", (w // 2 - 220, h - 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 255), 2)

    elif engine.state == GameState.COUNTDOWN:
        # Status indicator
        cv2.putText(frame, "AI: Analyzing habits & locking choice...", (20, h - 70),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 1)
        
        # Central countdown timer display
        cv2.putText(frame, str(engine.timer_val), (w // 2 - 30, h // 2 + 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 4.0, (0, 255, 255), 8)

    elif engine.state == GameState.SHOW_RESULT:
        # Result Banner (Green for Player Win, Red for AI Win, Yellow for Draw)
        color = (0, 255, 0) if "WIN" in engine.round_result and "YOU" in engine.round_result else (
            (0, 0, 255) if "AI" in engine.round_result else (0, 255, 255)
        )
        cv2.putText(frame, engine.round_result, (w // 2 - 120, h // 2 - 80),
                    cv2.FONT_HERSHEY_SIMPLEX, 2.0, color, 5)

        # Draw Player Move Icon & Text
        player_icon = assets.get(engine.player_move)
        if player_icon is not None:
            overlay_transparent(frame, player_icon, w // 2 - 180, h // 2 - 20)
        cv2.putText(frame, f"You: {engine.player_move}", (w // 2 - 200, h // 2 + 110),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

        # Center VS Label
        cv2.putText(frame, "VS", (w // 2 - 20, h // 2 + 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 255), 3)

        # Draw AI Move Icon & Text
        ai_icon = assets.get(engine.ai_move)
        if ai_icon is not None:
            overlay_transparent(frame, ai_icon, w // 2 + 80, h // 2 - 20)
        cv2.putText(frame, f"AI: {engine.ai_move}", (w // 2 + 70, h // 2 + 110),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)


def main():
    cap = cv2.VideoCapture(0)
    detector = GestureDetector()
    classifier = GestureClassifier()
    engine = GameEngine()
    assets = AssetManager(target_size=(100, 100))

    print("RPS Game Running! Controls: [SPACE] Start Round | [R] Reset Score | [Q] Quit")

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)
        h, w, _ = frame.shape

        # Detect landmarks and predict current frame gesture
        frame, landmarks = detector.detect(frame)
        current_gesture = classifier.classify(landmarks)

        # Advance state machine logic
        engine.update(current_gesture)

        # Real-time gesture diagnostic overlay in lower-left corner
        cv2.putText(frame, f"Live Gesture: {current_gesture}", (20, h - 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        # Render complete HUD
        draw_hud(frame, engine, assets)

        cv2.imshow("Rock Paper Scissors AI", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord(' '):
            engine.start_round()
        elif key == ord('r'):
            engine.reset_scores()
            classifier.reset_buffer()
        elif key == ord('q'):
            break

    # Resource cleanup
    cap.release()
    detector.close()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()