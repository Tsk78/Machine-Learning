import json
import os
import random
from collections import Counter, defaultdict
from typing import Dict, List, Optional, Tuple


class HumanBehaviorPredictor:
    """Predicts human choices using multi-order Markov chains and cross-state tracking."""

    COUNTER_MOVES: Dict[str, str] = {
        "ROCK": "PAPER",
        "PAPER": "SCISSORS",
        "SCISSORS": "ROCK"
    }

    BEATEN_BY: Dict[str, str] = {
        "PAPER": "ROCK",
        "SCISSORS": "PAPER",
        "ROCK": "SCISSORS"
    }

    def __init__(self, save_path: str = "data/ai_memory.json", decay_factor: float = 0.995):
        self.save_path: str = save_path
        self.decay_factor: float = decay_factor

        self.history_matrix: defaultdict[Tuple[str, str], Dict[str, float]] = defaultdict(
            lambda: {"ROCK": 0.0, "PAPER": 0.0, "SCISSORS": 0.0}
        )
        self.order1_matrix: defaultdict[str, Dict[str, float]] = defaultdict(
            lambda: {"ROCK": 0.0, "PAPER": 0.0, "SCISSORS": 0.0}
        )
        self.cross_state_matrix: defaultdict[Tuple[str, str, str], Dict[str, float]] = defaultdict(
            lambda: {"ROCK": 0.0, "PAPER": 0.0, "SCISSORS": 0.0}
        )
        self.move_history: List[str] = []
        self.last_ai_move: Optional[str] = None
        self.last_result: Optional[str] = None

        self.reset_memory()

    def reset_memory(self) -> None:
        """Resets memory state completely between simulation runs."""
        self.history_matrix.clear()
        self.order1_matrix.clear()
        self.cross_state_matrix.clear()
        self.move_history.clear()
        self.last_ai_move = None
        self.last_result = None

    def record_round(self, player_move: str, ai_move: str, player_result: str) -> None:
        if player_move not in self.COUNTER_MOVES:
            return

        # Smooth historical weight decay
        for state_key in list(self.history_matrix.keys()):
            for m in self.history_matrix[state_key]:
                self.history_matrix[state_key][m] *= self.decay_factor

        for state_key_1 in list(self.order1_matrix.keys()):
            for m in self.order1_matrix[state_key_1]:
                self.order1_matrix[state_key_1][m] *= self.decay_factor

        for state_key_cs in list(self.cross_state_matrix.keys()):
            for m in self.cross_state_matrix[state_key_cs]:
                self.cross_state_matrix[state_key_cs][m] *= self.decay_factor

        # Record context updates
        if self.move_history and self.last_ai_move and self.last_result:
            cs_key = (self.move_history[-1], self.last_ai_move, str(self.last_result))
            self.cross_state_matrix[cs_key][player_move] += 1.0

        if len(self.move_history) >= 1:
            self.order1_matrix[self.move_history[-1]][player_move] += 1.0

        if len(self.move_history) >= 2:
            prev_pair = (self.move_history[-2], self.move_history[-1])
            self.history_matrix[prev_pair][player_move] += 1.0

        self.move_history.append(player_move)
        self.last_ai_move = ai_move
        self.last_result = player_result
        self.save_memory()

    def _get_markov_distributions(self, weights: Dict[str, float], min_samples: float = 0.5) -> Dict[str, float]:
        total = sum(weights.values())
        if total < min_samples:
            return {"ROCK": 0.0, "PAPER": 0.0, "SCISSORS": 0.0}

        sorted_w = sorted(weights.values(), reverse=True)
        top_w = sorted_w[0]
        second_w = sorted_w[1]

        # Calculate dominance margin to suppress noise
        margin = (top_w - second_w) / total if total > 0 else 0.0
        sample_weight = min(total / 1.5, 1.0)
        
        # Suppress near-uniform noisy distributions
        if margin < 0.15:
            return {"ROCK": 0.0, "PAPER": 0.0, "SCISSORS": 0.0}

        return {m: (w / total) * margin * sample_weight for m, w in weights.items()}

    def _predict_player_move(self) -> str:
        move_scores: Dict[str, float] = {"ROCK": 0.0, "PAPER": 0.0, "SCISSORS": 0.0}

        # Model A: Unconditional Static Frequency
        top_freq_move = None
        if len(self.move_history) >= 5:
            counts = Counter(self.move_history)
            total_moves = len(self.move_history)
            top_freq_move, top_count = counts.most_common(1)[0]
            top_ratio = top_count / total_moves

            for m, count in counts.items():
                ratio = count / total_moves
                if ratio >= 0.35:
                    # Smooth quadratic boost for higher frequency dominance
                    bias_strength = (ratio - 0.33) * 3.5
                    move_scores[m] += max(bias_strength, 0.1)

        # Model B: Cross-State Outcomes (WSLS detection)
        if self.move_history and self.last_ai_move and self.last_result:
            cs_key = (self.move_history[-1], self.last_ai_move, str(self.last_result))
            if cs_key in self.cross_state_matrix:
                dist = self._get_markov_distributions(self.cross_state_matrix[cs_key], min_samples=0.5)
                for m, score in dist.items():
                    move_scores[m] += score * 2.0

        # Model C: Order-2 Markov Chains
        if len(self.move_history) >= 2:
            prev_pair = (self.move_history[-2], self.move_history[-1])
            if prev_pair in self.history_matrix:
                dist = self._get_markov_distributions(self.history_matrix[prev_pair], min_samples=0.8)
                for m, score in dist.items():
                    move_scores[m] += score * 1.5

        # Model D: Order-1 Markov Chains (Sequential & Cycler pattern)
        if len(self.move_history) >= 1:
            last_move = self.move_history[-1]
            if last_move in self.order1_matrix:
                dist = self._get_markov_distributions(self.order1_matrix[last_move], min_samples=0.5)
                for m, score in dist.items():
                    move_scores[m] += score * 1.6

        # Select move with the highest overall score
        best_move = max(move_scores, key=lambda k: move_scores[k])
        if move_scores[best_move] >= 0.12:
            return best_move

        # Fallback to dominant unconditional frequency before psychology rules
        if top_freq_move:
            return top_freq_move

        return self._predict_by_psychology()

    def predict_ai_move(self) -> str:
        predicted_player_move = self._predict_player_move()
        return self.COUNTER_MOVES[predicted_player_move]

    def _predict_by_psychology(self) -> str:
        if not self.move_history:
            return random.choice(["ROCK", "PAPER", "SCISSORS"])

        last_player_move = self.move_history[-1]
        res = str(self.last_result).upper()
        if "WIN" in res and "AI" not in res:
            return last_player_move
        return self.COUNTER_MOVES[last_player_move]

    def save_memory(self) -> None:
        try:
            os.makedirs(os.path.dirname(self.save_path), exist_ok=True)
            serialized_o2 = {f"{k[0]},{k[1]}": v for k, v in self.history_matrix.items()}
            serialized_cs = {f"{k[0]},{k[1]},{k[2]}": v for k, v in self.cross_state_matrix.items()}
            data = {
                "move_history": self.move_history[-100:],
                "history_matrix": serialized_o2,
                "order1_matrix": dict(self.order1_matrix),
                "cross_state_matrix": serialized_cs
            }
            with open(self.save_path, "w") as f:
                json.dump(data, f, indent=2)
        except Exception:
            pass

    def load_memory(self) -> None:
        if not os.path.exists(self.save_path):
            return
        try:
            with open(self.save_path, "r") as f:
                data = json.load(f)
                self.move_history = data.get("move_history", [])

                raw_o2 = data.get("history_matrix", {})
                for key_str, weights in raw_o2.items():
                    k1, k2 = key_str.split(",")
                    for move, w in weights.items():
                        self.history_matrix[(k1, k2)][move] = float(w)

                raw_o1 = data.get("order1_matrix", {})
                for prev_move, weights in raw_o1.items():
                    for move, w in weights.items():
                        self.order1_matrix[prev_move][move] = float(w)

                raw_cs = data.get("cross_state_matrix", {})
                for key_str, weights in raw_cs.items():
                    parts = key_str.split(",")
                    if len(parts) == 3:
                        p_move, a_move, res = parts
                        for move, w in weights.items():
                            self.cross_state_matrix[(p_move, a_move, res)][move] = float(w)
        except Exception:
            pass