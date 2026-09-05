import math
import random
import time
from enum import Enum, auto
from rps_ai.game.predictor import HumanBehaviorPredictor


class GameState(Enum):
    WAITING = auto()
    COUNTDOWN = auto()
    EVALUATE = auto()
    SHOW_RESULT = auto()


class GameEngine:
    """Game engine where AI predictions are locked BEFORE player shows hand."""
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
            self.round_result = ""

            self.predictor = HumanBehaviorPredictor()

    def reset_scores(self) -> None:
        """Resets the game state and scores back to zero."""
        self.user_score = 0
        self.ai_score = 0
        self.player_move = "NO_HAND"
        self.ai_move = "NO_HAND"
        self.round_result = ""
        self.state = GameState.WAITING
    def start_round(self):
        """Starts round and LOCKS AI prediction immediately."""
        if self.state in [GameState.WAITING, GameState.SHOW_RESULT]:
            self.state = GameState.COUNTDOWN
            self.start_time = time.time()
            self.timer_val = int(self.countdown_duration)

            # --- KEY FIX: AI predicts move BEFORE player shows their hand ---
            self.ai_move = self.predictor.predict_ai_move()

    def update(self, current_player_move: str):
        now = time.time()

        if self.state == GameState.COUNTDOWN:
            elapsed = now - self.start_time
            remaining = self.countdown_duration - elapsed
            self.timer_val = max(1, int(math.ceil(remaining)))

            if remaining <= 0:
                self.state = GameState.EVALUATE

        elif self.state == GameState.EVALUATE:
            # Capture actual player move at countdown zero
            self.player_move = current_player_move if current_player_move in ["ROCK", "PAPER", "SCISSORS"] else "ROCK"
            
            # Evaluate winner against pre-selected AI move
            self._evaluate_winner()

            # Train predictor with round outcome
            self.predictor.record_round(self.player_move, self.ai_move, self.round_result)

            self.start_time = time.time()
            self.state = GameState.SHOW_RESULT

        elif self.state == GameState.SHOW_RESULT:
            if now - self.start_time >= self.result_duration:
                self.state = GameState.WAITING

    def _evaluate_winner(self):
        winning_rules = {"ROCK": "SCISSORS", "PAPER": "ROCK", "SCISSORS": "PAPER"}

        if self.player_move == self.ai_move:
            self.round_result = "DRAW!"
        elif winning_rules.get(self.player_move) == self.ai_move:
            self.user_score += 1
            self.round_result = "YOU WIN!"
        else:
            self.ai_score += 1
            self.round_result = "AI WINS!"