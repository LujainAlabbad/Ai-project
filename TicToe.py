import tkinter as tk
from tkinter import font as tkfont
from tkinter import messagebox
import math
import random
import time

# Project layout:
# tic tac toe game on a 3x3 board where it is ai vs ai
# first alphabeta pruning vs alphabeta pruning
# second mct vs mct
# lastly ab vs mct
# allowed only three pieces on the board for each player
# when they place a fourth the third (oldest) is removed
# 20 minute time limit for each game
# winning condition is 3 of the same piece x or o in a row

# game logic variables
win_combinations = [[0,1,2],[3,4,5],[6,7,8],[0,3,6],[1,4,7],[2,5,8],[0,4,8],[2,4,6]]
max_pieces = 3
timer_limit = 1200
mct_iterations = 800  # Increased for the full tree search

# game state class to manage the board and player turns

class gameState:
    def __init__(self, board=None, player_o=None, opponent_x=None, turn='X'):
        self.board = board if board else ['' for _ in range(9)]
        self.player_o = player_o if player_o else []
        self.opponent_x = opponent_x if opponent_x else []
        self.player_turn = turn

    def clone(self):
        return gameState(self.board[:], self.player_o[:], self.opponent_x[:], self.player_turn)

    def get_legal_moves(self):
        return [i for i in range(9) if self.board[i] == '']

def check_win(board):
    for combo in win_combinations:
        if board[combo[0]] != '' and board[combo[0]] == board[combo[1]] == board[combo[2]]:
            return board[combo[0]]
    if '' not in board: return 'Draw'
    return None

def apply_move(state, index, player):
    s = state.clone()
    pieces = s.player_o if player == 'O' else s.opponent_x
    if len(pieces) >= max_pieces:
        oldest_index = pieces.pop(0)
        s.board[oldest_index] = ''
    s.board[index] = player
    pieces.append(index)
    s.player_turn = 'O' if player == 'X' else 'X'
    return s

#  ALPHA-BETA PRUNING and move selection

def ab_minimax(state, depth, max_depth, alpha, beta, maximizing):
    winner = check_win(state.board)
    if winner == 'X': return 10 - depth
    if winner == 'O': return depth - 10
    
    moves = state.get_legal_moves()
    if depth >= max_depth or not moves or winner == 'Draw': return 0
    
    if maximizing:
        best = -math.inf
        for move in moves:
            new_state = apply_move(state, move, 'X')
            score = ab_minimax(new_state, depth + 1, max_depth, alpha, beta, False)
            best = max(score, best)
            alpha = max(alpha, best)
            if beta <= alpha: break
        return best
    else:
        best = math.inf
        for move in moves:
            new_state = apply_move(state, move, 'O')
            score = ab_minimax(new_state, depth + 1, max_depth, alpha, beta, True)
            best = min(score, best)
            beta = min(beta, best)
            if beta <= alpha: break
        return best

def get_ab_move(state, player, depth_limit):
    moves = state.get_legal_moves()
    if not moves: return None
    best_move = None
    best_score = -math.inf if player == 'X' else math.inf
    
    for move in moves:
        new_state = apply_move(state, move, player)
        score = ab_minimax(new_state, 0, depth_limit, -math.inf, math.inf, player == 'O')
        if player == 'X' and score > best_score:
            best_score, best_move = score, move
        elif player == 'O' and score < best_score:
            best_score, best_move = score, move
            
    return best_move if best_move is not None else random.choice(moves)

# MONTE CARLO TREE SEARCH Selection → Expansion → Simulation → Backpropagation
class MCTSNode:
    def __init__(self, state, parent=None, move=None):
        self.state = state
        self.parent = parent
        self.move = move
        self.children = []
        self.wins = 0
        self.visits = 0
        self.untried_moves = state.get_legal_moves()

    def uct_select_child(self):
        # Selection logic using UCT formula
        return max(self.children, key=lambda c: (c.wins / c.visits) + math.sqrt(2 * math.log(self.visits) / c.visits))

    def expand(self):
        move = self.untried_moves.pop()
        next_state = apply_move(self.state, move, self.state.player_turn)
        child_node = MCTSNode(next_state, parent=self, move=move)
        self.children.append(child_node)
        return child_node

    def update(self, result):
        self.visits += 1
        self.wins += result

def mcts_best_move(root_state, iterations):
    root = MCTSNode(root_state)

    for _ in range(iterations):
        node = root
        temp_state = root_state.clone()

        # 1. Selection
        while not node.untried_moves and node.children:
            node = node.uct_select_child()
            temp_state = apply_move(temp_state, node.move, temp_state.player_turn)

        # 2. Expansion
        if node.untried_moves:
            node = node.expand()
            temp_state = apply_move(temp_state, node.move, temp_state.player_turn)

        # 3. Simulation (Playout)
        while True:
            winner = check_win(temp_state.board)
            if winner: break
            moves = temp_state.get_legal_moves()
            if not moves: break
            temp_state = apply_move(temp_state, random.choice(moves), temp_state.player_turn)

        # 4. Backpropagation
        result_winner = check_win(temp_state.board)
        while node is not None:
            # Reward based on whose turn it was to move into this state
            reward = 0
            if result_winner == 'Draw': reward = 0.5
            elif result_winner == (node.parent.state.player_turn if node.parent else 'N/A'): reward = 1
            
            node.update(reward)
            node = node.parent

    return max(root.children, key=lambda c: c.visits).move

# GUI Application using Tkinter

class TicTacToeApp:
    def __init__(self, root):
        self.root = root
        self.root.title("AI Tic Tac Toe")
        self.state = gameState()
        self.time_left = timer_limit
        self.game_running = False
        self.scores = {'X': 0, 'O': 0, 'Draw': 0}

        self.label_font = tkfont.Font(family="Arial", size=11, weight="bold")
        tk.Label(root, text="Agent Configuration:", font=self.label_font).pack(pady=5)

        # Agent Variables
        self.x_agent_var = tk.StringVar(value='AB')
        self.o_agent_var = tk.StringVar(value='MCT')
        self.x_ab_depth = tk.IntVar(value=5)
        self.o_ab_depth = tk.IntVar(value=5)
        self.x_mct_iters = tk.IntVar(value=1000)
        self.o_mct_iters = tk.IntVar(value=1000)

        # Trace variables to toggle UI depth visibility
        self.x_agent_var.trace_add("write", self.update_visibility)
        self.o_agent_var.trace_add("write", self.update_visibility)

        # Controls Frame
        self.ctrl_frame = tk.Frame(root)
        self.ctrl_frame.pack(pady=10)

        # X Controls
        tk.Label(self.ctrl_frame, text="X Agent:").grid(row=0, column=0, sticky="e")
        tk.OptionMenu(self.ctrl_frame, self.x_agent_var, 'Human', 'AB', 'MCT').grid(row=0, column=1, padx=5)
        
        self.x_depth_label = tk.Label(self.ctrl_frame, text="X Depth:")
        self.x_depth_menu = tk.OptionMenu(self.ctrl_frame, self.x_ab_depth, 2, 5, 10)
        
        self.x_mct_label = tk.Label(self.ctrl_frame, text="X Iters:")
        self.x_mct_menu = tk.OptionMenu(self.ctrl_frame, self.x_mct_iters, 200, 500, 1000)

        # O Controls
        tk.Label(self.ctrl_frame, text="O Agent:").grid(row=1, column=0, sticky="e", pady=5)
        tk.OptionMenu(self.ctrl_frame, self.o_agent_var, 'Human', 'AB', 'MCT').grid(row=1, column=1, padx=5)
        
        self.o_depth_label = tk.Label(self.ctrl_frame, text="O Depth:")
        self.o_depth_menu = tk.OptionMenu(self.ctrl_frame, self.o_ab_depth, 2, 5, 10)
        
        self.o_mct_label = tk.Label(self.ctrl_frame, text="O Iters:")
        self.o_mct_menu = tk.OptionMenu(self.ctrl_frame, self.o_mct_iters, 200, 500, 1000)

        self.score_label = tk.Label(root, text="X Wins: 0 | O Wins: 0 | Draws: 0", font=self.label_font, fg="darkblue")
        self.score_label.pack(pady=5)

        self.timer_label = tk.Label(root, text="Time: 20:00", font=("Courier", 15), fg="red")
        self.timer_label.pack()

        # Initialize Visibility
        self.update_visibility()

        self.board_frame = tk.Frame(root)
        self.board_frame.pack(pady=10)
        self.buttons = []
        for i in range(9):
            btn = tk.Button(self.board_frame, text="", font=("Arial", 20), width=4, height=2,
                            command=lambda i=i: self.human_click(i))
            btn.grid(row=i//3, column=i%3)
            self.buttons.append(btn)

        self.start_btn = tk.Button(root, text="Start Match", command=self.start_match, bg="green", fg="white", width=15)
        self.start_btn.pack(pady=10)

    def update_visibility(self, *args):
        # Handle X settings visibility
        if self.x_agent_var.get() == 'AB':
            self.x_depth_label.grid(row=0, column=2, sticky="e")
            self.x_depth_menu.grid(row=0, column=3, padx=5)
            self.x_mct_label.grid_remove()
            self.x_mct_menu.grid_remove()
        elif self.x_agent_var.get() == 'MCT':
            self.x_mct_label.grid(row=0, column=2, sticky="e")
            self.x_mct_menu.grid(row=0, column=3, padx=5)
            self.x_depth_label.grid_remove()
            self.x_depth_menu.grid_remove()
        else:
            self.x_depth_label.grid_remove()
            self.x_depth_menu.grid_remove()
            self.x_mct_label.grid_remove()
            self.x_mct_menu.grid_remove()

        # Handle O settings visibility
        if self.o_agent_var.get() == 'AB':
            self.o_depth_label.grid(row=1, column=2, sticky="e")
            self.o_depth_menu.grid(row=1, column=3, padx=5)
            self.o_mct_label.grid_remove()
            self.o_mct_menu.grid_remove()
        elif self.o_agent_var.get() == 'MCT':
            self.o_mct_label.grid(row=1, column=2, sticky="e")
            self.o_mct_menu.grid(row=1, column=3, padx=5)
            self.o_depth_label.grid_remove()
            self.o_depth_menu.grid_remove()
        else:
            self.o_depth_label.grid_remove()
            self.o_depth_menu.grid_remove()
            self.o_mct_label.grid_remove()
            self.o_mct_menu.grid_remove()

    def start_match(self):
        self.state = gameState()
        self.time_left = timer_limit
        self.game_running = True
        self.start_btn.config(state="disabled")
        self.refresh_board()
        self.run_timer()
        self.ai_loop()

    def run_timer(self):
        if self.time_left > 0 and self.game_running:
            self.time_left -= 1
            mins, secs = divmod(self.time_left, 60)
            self.timer_label.config(text=f"Time: {mins:02d}:{secs:02d}")
            self.root.after(1000, self.run_timer)
        elif self.time_left <= 0:
            self.end_match("Draw")

    def human_click(self, index):
        if not self.game_running: return
        agent = self.x_agent_var.get() if self.state.player_turn == 'X' else self.o_agent_var.get()
        if agent == 'Human' and self.state.board[index] == '':
            self.state = apply_move(self.state, index, self.state.player_turn)
            self.refresh_board()
            self.ai_loop()

    def ai_loop(self):
        if not self.game_running: return
        winner = check_win(self.state.board)
        if winner:
            self.end_match(winner)
            return

        player = self.state.player_turn
        agent = self.x_agent_var.get() if player == 'X' else self.o_agent_var.get()

        if agent == 'Human': return

        move = None
        if agent == 'AB':
            depth = self.x_ab_depth.get() if player == 'X' else self.o_ab_depth.get()
            move = get_ab_move(self.state, player, depth)
        elif agent == 'MCT':
            iters = self.x_mct_iters.get() if player == 'X' else self.o_mct_iters.get()
            move = mcts_best_move(self.state, iters)

        if move is not None:
            self.state = apply_move(self.state, move, player)
            self.refresh_board()
            self.root.after(600, self.ai_loop)

    def refresh_board(self):
        for i in range(9):
            char = self.state.board[i]
            self.buttons[i].config(text=char, fg="blue" if char == 'X' else "orange")

    def end_match(self, result):
        self.game_running = False
        self.start_btn.config(state="normal")
        if result in self.scores:
            self.scores[result] += 1
        self.score_label.config(
            text=f"X Wins: {self.scores['X']} | O Wins: {self.scores['O']} | Draws: {self.scores['Draw']}"
        )
        messagebox.showinfo("Game Result", f"Winner: {result}")

def run_gui_mode():
    root = tk.Tk()
    app = TicTacToeApp(root)
    root.mainloop()

if __name__ == "__main__":
    run_gui_mode()