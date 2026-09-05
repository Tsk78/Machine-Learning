# main.py

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


def draw_stats_overlay(frame, engine: GameEngine):
    """Renders a stats panel showing past AI predictions and success rate."""
    h, w, _ = frame.shape
    panel_w, panel_h = 420, 360
    x1, y1 = (w - panel_w) // 2, (h - panel_h) // 2
    x2, y2 = x1 + panel_w, y1 + panel_h

    # Semi-transparent background box
    overlay = frame.copy()
    cv2.rectangle(overlay, (x1, y1), (x2, y2), (20, 20, 20), -1)
    cv2.addWeighted(overlay, 0.85, frame, 0.15, 0, frame)
    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 255), 2)

    # Panel Header
    cv2.putText(frame, "AI PREDICTION ACCURACY", (x1 + 60, y1 + 35),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

    # Overall Metrics
    accuracy_str = f"Success Rate: {engine.accuracy_rate:.1f}%"
    total_str = f"Total Predictions: {engine.total_predictions} ({engine.correct_predictions} Correct)"
    
    cv2.putText(frame, accuracy_str, (x1 + 30, y1 + 75),
                cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 0), 2)
    cv2.putText(frame, total_str, (x1 + 30, y1 + 105),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 200, 200), 1)

    # Divider Line
    cv2.line(frame, (x1 + 20, y1 + 120), (x2 - 20, y1 + 120), (100, 100, 100), 1)

    # Recent History Title
    cv2.putText(frame, "Recent Predictions (Last 5 Rounds):", (x1 + 30, y1 + 145),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)

    # List last 5 predictions
    history_subset = engine.prediction_history[-5:]
    if not history_subset:
        cv2.putText(frame, "No completed rounds yet.", (x1 + 30, y1 + 185),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (150, 150, 150), 1)
    else:
        y_offset = y1 + 175
        for item in reversed(history_subset):
            status = "CORRECT" if item["correct"] else "WRONG"
            color = (0, 255, 0) if item["correct"] else (0, 0, 255)
            row_text = f"R{item['round']}: Pred {item['predicted']} vs Played {item['actual']} [{status}]"
            
            cv2.putText(frame, row_text, (x1 + 25, y_offset),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.48, color, 1)
            y_offset += 28

    # Footer note
    cv2.putText(frame, "Press [P] to close", (x1 + 140, y2 - 15),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180, 180, 180), 1)


def draw_hud(frame, engine: GameEngine, assets: AssetManager):
    h, w, _ = frame.shape

    # 1. Scoreboard Header Banner
    score_text = f"PLAYER: {engine.user_score}  |  AI: {engine.ai_score}"
    cv2.rectangle(frame, (0, 0), (w, 60), (30, 30, 30), -1)
    cv2.putText(frame, score_text, (w // 2 - 180, 42), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)

    # 2. State-Driven Overlays
    if engine.state == GameState.WAITING:
        cv2.putText(frame, "Press SPACE to Start | Press P for AI Stats", (w // 2 - 280, h - 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

    elif engine.state == GameState.COUNTDOWN:
        cv2.rectangle(frame, (10, h - 90), (380, h - 10), (0, 0, 0), -1)
        cv2.rectangle(frame, (10, h - 90), (380, h - 10), (0, 255, 255), 1)
        
        cv2.putText(frame, "AI BRAIN:", (20, h - 65),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
        cv2.putText(frame, f"Predicted You: {engine.predicted_player_move}", (20, h - 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        cv2.putText(frame, f"AI Locked Move: {engine.ai_move}", (20, h - 18),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        cv2.putText(frame, str(engine.timer_val), (w // 2 - 30, h // 2 + 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 4.0, (0, 255, 255), 8)

    elif engine.state == GameState.SHOW_RESULT:
        color = (0, 255, 0) if "WIN" in engine.round_result and "YOU" in engine.round_result else (
            (0, 0, 255) if "AI" in engine.round_result else (0, 255, 255)
        )
        cv2.putText(frame, engine.round_result, (w // 2 - 120, h // 2 - 80),
                    cv2.FONT_HERSHEY_SIMPLEX, 2.0, color, 5)

        player_icon = assets.get(engine.player_move)
        if player_icon is not None:
            overlay_transparent(frame, player_icon, w // 2 - 180, h // 2 - 20)
        cv2.putText(frame, f"You played: {engine.player_move}", (w // 2 - 200, h // 2 + 110),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

        cv2.putText(frame, "VS", (w // 2 - 20, h // 2 + 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 255), 3)

        ai_icon = assets.get(engine.ai_move)
        if ai_icon is not None:
            overlay_transparent(frame, ai_icon, w // 2 + 80, h // 2 - 20)
        cv2.putText(frame, f"AI played: {engine.ai_move}", (w // 2 + 70, h // 2 + 110),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

    # 3. Draw Prediction Stats Overlay if Toggled
    if engine.show_stats:
        draw_stats_overlay(frame, engine)


def main():
    cap = cv2.VideoCapture(0)
    detector = GestureDetector()
    classifier = GestureClassifier()
    engine = GameEngine()
    assets = AssetManager(target_size=(100, 100))

    print("RPS Game Running! Controls: [SPACE] Start Round | [P] Toggle Stats | [R] Reset Score | [Q] Quit")

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)
        h, w, _ = frame.shape

        frame, landmarks = detector.detect(frame)
        current_gesture = classifier.classify(landmarks)

        engine.update(current_gesture)

        cv2.putText(frame, f"Live Gesture: {current_gesture}", (20, h - 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        draw_hud(frame, engine, assets)

        cv2.imshow("Rock Paper Scissors AI", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord(' '):
            engine.start_round()
        elif key == ord('p') or key == ord('P'):
            engine.toggle_stats()
        elif key == ord('r') or key == ord('R'):
            engine.reset_scores()
            classifier.reset_buffer()
        elif key == ord('q') or key == ord('Q'):
            break

    cap.release()
    detector.close()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()