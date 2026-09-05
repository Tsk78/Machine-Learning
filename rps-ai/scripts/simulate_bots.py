import os
import random
import sys

# Ensure module path includes the project root
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from rps_ai.game.predictor import HumanBehaviorPredictor


class CounterMoves:
    MAP = {"ROCK": "PAPER", "PAPER": "SCISSORS", "SCISSORS": "ROCK"}


# --- Automated Human Strategy Bots ---

class BotSequential:
    """Plays in a fixed, predictable cycle: Rock -> Paper -> Scissors -> Rock..."""
    def __init__(self):
        self.sequence = ["ROCK", "PAPER", "SCISSORS"]
        self.index = 0

    def play(self, last_ai_move, last_result):
        move = self.sequence[self.index % 3]
        self.index += 1
        return move


class BotWinStayLoseShift:
    """If won: repeats same move. If lost/tied: shifts to the move that beats AI's last move."""
    def __init__(self):
        self.last_move = "ROCK"

    def play(self, last_ai_move, last_result):
        if last_result == "YOU WIN!":
            return self.last_move
        elif last_ai_move in CounterMoves.MAP:
            self.last_move = CounterMoves.MAP[last_ai_move]
            return self.last_move
        self.last_move = random.choice(["ROCK", "PAPER", "SCISSORS"])
        return self.last_move


class BotBiasedRock:
    """Favors Rock 60% of the time, split evenly between Paper and Scissors for the rest."""
    def play(self, last_ai_move, last_result):
        return random.choices(
            ["ROCK", "PAPER", "SCISSORS"],
            weights=[0.60, 0.20, 0.20]
        )[0]


class BotMarkovCycler:
    """Follows a probabilistic pattern: 70% chance to cycle to the next move, 30% chance to repeat."""
    def __init__(self):
        self.last_move = "ROCK"

    def play(self, last_ai_move, last_result):
        if random.random() < 0.70:
            self.last_move = CounterMoves.MAP[self.last_move]
        return self.last_move


# --- Simulation Runner ---

def run_simulation(bot_instance, bot_name: str, rounds: int = 100):
    test_save_path = f"data/test_{bot_name.lower().replace(' ', '_')}.json"
    
    # Remove old test memory file if present
    if os.path.exists(test_save_path):
        os.remove(test_save_path)

    # Initialize predictor ONCE before starting round loop
    predictor = HumanBehaviorPredictor(save_path=test_save_path, decay_factor=0.95)

    correct_predictions = 0
    ai_wins = 0
    player_wins = 0
    ties = 0

    last_ai_move = None
    last_result = None

    for r in range(1, rounds + 1):
        # 1. AI predicts and locks move
        predicted_player_move = predictor._predict_player_move()
        ai_move = predictor.predict_ai_move()

        # 2. Bot chooses its move
        player_move = bot_instance.play(last_ai_move, last_result)

        # 3. Evaluate prediction accuracy
        if predicted_player_move == player_move:
            correct_predictions += 1

        # 4. Evaluate game outcome
        if ai_move == player_move:
            last_result = "TIE"
            ties += 1
        elif CounterMoves.MAP[player_move] == ai_move:
            last_result = "AI WINS!"
            ai_wins += 1
        else:
            last_result = "YOU WIN!"
            player_wins += 1

        # 5. Record round in predictor memory
        predictor.record_round(player_move, ai_move, last_result)
        last_ai_move = ai_move

    # Clean up test memory file
    if os.path.exists(test_save_path):
        os.remove(test_save_path)

    accuracy = (correct_predictions / rounds) * 100
    win_rate = (ai_wins / rounds) * 100

    print(f"| {bot_name:<20} | {accuracy:>8.1f}% | {ai_wins:>6} | {player_wins:>8} | {ties:>5} | {win_rate:>8.1f}% |")


def main():
    print("=" * 73)
    print(f"| {'Bot Strategy':<20} | {'Predict %':>9} | {'AI Win':>6} | {'User Win':>8} | {'Ties':>5} | {'AI Win %':>8} |")
    print("=" * 73)

    bots = [
        (BotSequential(), "Sequential (R->P->S)"),
        (BotWinStayLoseShift(), "Win-Stay Lose-Shift"),
        (BotBiasedRock(), "60% Rock Biased"),
        (BotMarkovCycler(), "70% Cycler Pattern")
    ]

    for bot, name in bots:
        run_simulation(bot, name, rounds=100)

    print("=" * 73)


if __name__ == "__main__":
    main()