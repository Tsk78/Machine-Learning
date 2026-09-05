# rps_ai/game/engine.py

import time
from enum import Enum
from rps_ai.game.predictor import HumanBehaviorPredictor


class GameState(Enum):
    WAITING = 0
    COUNTDOWN = 1
    SHOW_RESULT = 2


class GameEngine:
    def __init__(self, countdown_duration: float = 3.0, result_duration: float = 2.5):
        self.state = GameState.WAITING
        self.countdown_duration = countdown_duration
        self.result_duration = result_duration

        self.user_score = 0
        self.ai_score = 0
        self.start_time = 0.0
        self.timer_val = 3

        self.player_move = "NO_HAND"
        self.ai_move = "NO_HAND"
        self.predicted_player_move = "NONE"
        self.round_result = ""

        # Prediction Tracking Stats
        self.show_stats = False  # Toggle state for 'P' key
        self.total_predictions = 0
        self.correct_predictions = 0
        self.prediction_history = []  # Stores dicts of past round predictions

        self.predictor = HumanBehaviorPredictor()

    def toggle_stats(self):
        """Toggles the visibility of the past predictions HUD."""
        self.show_stats = not self.show_stats

    @property
    def accuracy_rate(self) -> float:
        """Returns AI prediction success rate percentage."""
        if self.total_predictions == 0:
            return 0.0
        return (self.correct_predictions / self.total_predictions) * 100.0

    def start_round(self):
        """Starts round, predicts player gesture, and locks AI move."""
        if self.state in [GameState.WAITING, GameState.SHOW_RESULT]:
            self.state = GameState.COUNTDOWN
            self.start_time = time.time()
            self.timer_val = int(self.countdown_duration)

            # Record predicted human move & AI counter-move
            if hasattr(self.predictor, "_predict_player_move"):
                self.predicted_player_move = self.predictor._predict_player_move()
            else:
                self.predicted_player_move = "ROCK"

            self.ai_move = self.predictor.COUNTER_MOVES[self.predicted_player_move]

    def update(self, current_gesture: str):
        """Updates round timer state machine."""
        if self.state == GameState.COUNTDOWN:
            elapsed = time.time() - self.start_time
            remaining = self.countdown_duration - elapsed
            self.timer_val = max(1, int(remaining) + 1)

            if elapsed >= self.countdown_duration:
                self._evaluate_round(current_gesture)

        elif self.state == GameState.SHOW_RESULT:
            elapsed = time.time() - self.start_time
            if elapsed >= self.result_duration:
                self.state = GameState.WAITING

    def _evaluate_round(self, player_gesture: str):
        """Calculates round winner and records stats."""
        self.player_move = player_gesture
        self.state = GameState.SHOW_RESULT
        self.start_time = time.time()

        if self.player_move not in ["ROCK", "PAPER", "SCISSORS"]:
            self.round_result = "NO GESTURE DETECTED!"
            return

        # Record accuracy metrics
        was_correct = (self.predicted_player_move == self.player_move)
        self.total_predictions += 1
        if was_correct:
            self.correct_predictions += 1

        self.prediction_history.append({
            "round": self.total_predictions,
            "predicted": self.predicted_player_move,
            "actual": self.player_move,
            "correct": was_correct
        })

        # Win condition check
        if self.player_move == self.ai_move:
            self.round_result = "DRAW!"
            result_for_history = "DRAW"
        elif (
            (self.player_move == "ROCK" and self.ai_move == "SCISSORS") or
            (self.player_move == "PAPER" and self.ai_move == "ROCK") or
            (self.player_move == "SCISSORS" and self.ai_move == "PAPER")
        ):
            self.round_result = "YOU WIN!"
            self.user_score += 1
            result_for_history = "YOU WIN!"
        else:
            self.round_result = "AI WINS!"
            self.ai_score += 1
            result_for_history = "AI WINS!"

        # Train Markov predictor on round outcome
        self.predictor.record_round(self.player_move, self.ai_move, result_for_history)

    def reset_scores(self) -> None:
        """Resets scores, stats, and state back to WAITING."""
        self.user_score = 0
        self.ai_score = 0
        self.player_move = "NO_HAND"
        self.ai_move = "NO_HAND"
        self.predicted_player_move = "NONE"
        self.round_result = ""
        self.total_predictions = 0
        self.correct_predictions = 0
        self.prediction_history.clear()
        self.state = GameState.WAITING