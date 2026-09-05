import random
from collections import defaultdict


class HumanBehaviorPredictor:
    """Predicts human choices using transition matrices and psychological heuristics."""

    COUNTER_MOVES = {
        "ROCK": "PAPER",
        "PAPER": "SCISSORS",
        "SCISSORS": "ROCK"
    }

    BEATEN_BY = {
        "PAPER": "ROCK",
        "SCISSORS": "PAPER",
        "ROCK": "SCISSORS"
    }

    def __init__(self):
        # 2nd-order Markov Chain: tracks pairs of past moves (e.g., ("ROCK", "PAPER") -> "SCISSORS")
        self.history_matrix = defaultdict(lambda: {"ROCK": 0, "PAPER": 0, "SCISSORS": 0})
        self.move_history = []
        self.last_result = None  # "WIN", "LOSS", "DRAW" relative to player

    def record_round(self, player_move: str, ai_move: str, player_result: str):
        """Updates move sequence history and transition probabilities."""
        if player_move not in self.COUNTER_MOVES:
            return

        self.move_history.append(player_move)
        self.last_result = player_result

        # Update 2nd-order transition matrix if at least 3 moves recorded
        if len(self.move_history) >= 3:
            prev_pair = (self.move_history[-3], self.move_history[-2])
            self.history_matrix[prev_pair][player_move] += 1

    def _predict_player_move(self) -> str:
        """Determines what move the AI expects the human to throw next."""
        # Fallback for initial rounds: apply Win-Stay, Lose-Shift heuristic
        if len(self.move_history) < 3:
            return self._predict_by_psychology()

        # Markov Chain sequence prediction
        prev_pair = (self.move_history[-2], self.move_history[-1])
        counts = self.history_matrix[prev_pair]

        total_observations = sum(counts.values())
        if total_observations > 0:
            # Use lambda to avoid Pylance type issues with dict.get
            return max(counts, key=lambda k: counts[k])
        else:
            return self._predict_by_psychology()

    def predict_ai_move(self) -> str:
        """Returns the winning counter-move to the predicted player gesture."""
        predicted_player_move = self._predict_player_move()
        return self.COUNTER_MOVES[predicted_player_move]

    def _predict_by_psychology(self) -> str:
        """Applies 'Win-Stay, Lose-Shift' human behavioral heuristics."""
        if not self.move_history:
            return random.choice(["ROCK", "PAPER", "SCISSORS"])

        last_player_move = self.move_history[-1]

        if self.last_result == "YOU WIN!":
            # Players who win tend to repeat their winning move
            return last_player_move
        elif self.last_result == "AI WINS!":
            # Players who lose tend to shift to the move that counter-punches what the AI just played
            return self.COUNTER_MOVES[last_player_move]
        else:
            # On draws, players tend to cycle clockwise (Rock -> Paper -> Scissors)
            return self.COUNTER_MOVES[last_player_move]