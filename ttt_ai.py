import tkinter as tk
from tkinter import font as tkfont
from tkinter import messagebox
import math #to use infinity for the minimax algorithm
import random #to use random for the random move generator for mct
import time 
import copy 

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
max_pieces = 3 #the maximum number of pieces for each player at any time
timer_limit = 1200 #the time limit for 20 minutes in seconds
mct_iterations = 400 
ab_depth = 5 

class gameState:
    def __init__(self, board=None, player_o=None, opponent_x=None, turn='X'):
        self.board = board if board else ['' for i in range(9)]
        self.player_o = player_o if player_o else []
        self.opponent_x = opponent_x if opponent_x else [] 
        self.player_turn = turn

    def clone(self):
        s = gameState()
        s.board = self.board[:]
        s.player_o = self.player_o[:]
        s.opponent_x = self.opponent_x[:]
        s.player_turn = self.player_turn
        return s

def check_win(board):
    # win check function: defining the winning conditions for a 3x3 grid 
    for combo in win_combinations:
        if board[combo[0]] != '' and board[combo[0]] == board[combo[1]] == board[combo[2]]:
            return board[combo[0]]
    if '' not in board: return 'Draw'
    return None

def apply_move(state, index, player):
    # a function to remove oldest piece when a fourth is placed 
    s = state.clone()
    pieces = s.player_o if player == 'O' else s.opponent_x
    if len(pieces) >= max_pieces:
        oldest_index = pieces.pop(0) 
        s.board[oldest_index] = '' 
    s.board[index] = player
    pieces.append(index)
    s.player_turn = 'O' if player == 'X' else 'X'
    return s

def legal_moves(state):
    # return empty cells for next moves placement 
    return [i for i in range(9) if state.board[i] == '']

# algorithms

def ab_minimax(state, depth, alpha, beta, maximizing):
    winner = check_win(state.board)
    if winner == 'X': return 10 - depth
    if winner == 'O': return depth - 10
    moves = legal_moves(state)
    if depth >= ab_depth or not moves or winner == 'Draw': return 0
    if maximizing:
        best = -math.inf
        for move in moves:
            new_state = apply_move(state, move, 'X')
            score = ab_minimax(new_state, depth+1, alpha, beta, False)
            best = max(score, best)
            alpha = max(alpha, best)
            if beta <= alpha: break
        return best
    else:
        best = math.inf
        for move in moves:
            new_state = apply_move(state, move, 'O')
            score = ab_minimax(new_state, depth+1, alpha, beta, True)
            best = min(score, best)
            beta = min(beta, best)
            if beta <= alpha: break
        return best

def get_ab_move(state, player):
    moves = legal_moves(state)
    if not moves: return None
    best_move = None
    best_score = -math.inf if player == 'X' else math.inf
    for move in moves:
        new_state = apply_move(state, move, player)
        score = ab_minimax(new_state, 0, -math.inf, math.inf, player == 'O')
        if player == 'X' and score > best_score:
            best_score, best_move = score, move
        elif player == 'O' and score < best_score:
            best_score, best_move = score, move
    return best_move if best_move is not None else random.choice(moves)

def random_play(state, next_player):
    current = state.clone()
    turn = next_player
    for _ in range(15):
        winner = check_win(current.board)
        if winner: return winner
        moves = legal_moves(current)
        if not moves: return 'Draw'
        move = random.choice(moves)
        current = apply_move(current, move, turn)
        turn = 'O' if turn == 'X' else 'X'
    return 'Draw'

def mct_best_move(state, player):
    moves = legal_moves(state)
    if not moves: return None
    wins = [0] * len(moves)
    opponent = 'O' if player == 'X' else 'X'
    for i in range(mct_iterations):
        mi = i % len(moves)
        new_state = apply_move(state, moves[mi], player)
        winner = random_play(new_state, opponent)
        if winner == player: wins[mi] += 1
        elif winner == 'Draw': wins[mi] += 0.5
    return moves[wins.index(max(wins))]

def choose_ai_move(state, player, mode):
    if mode == 'ab vs ab':
        return get_ab_move(state, player)
    elif mode == 'mct vs mct':
        return mct_best_move(state, player)
    elif mode == 'ab vs mct':
        return get_ab_move(state, player) if player == 'X' else mct_best_move(state, player)
    elif mode == 'human vs ai':
        return mct_best_move(state, player) if player == 'O' else None

# gui timer and score tracking

class TicTacToeApp:
    def __init__(self, root):
        self.root = root
        self.root.title("AI Comparison: 3-Piece Game")
        self.state = gameState()
        self.time_left = timer_limit
        self.game_running = False
        
        # History Tracking for each mode
        self.scores = {
            'ab vs ab': {'X': 0, 'O': 0, 'Draw': 0},
            'mct vs mct': {'X': 0, 'O': 0, 'Draw': 0},
            'ab vs mct': {'X': 0, 'O': 0, 'Draw': 0},
            'human vs ai': {'Human': 0, 'AI': 0, 'Draw': 0}
        }
        
        # UI Elements
        self.label_font = tkfont.Font(family="Arial", size=11, weight="bold")
        
        tk.Label(root, text="Select Game Mode:", font=self.label_font).pack(pady=5)
        self.mode_var = tk.StringVar(value='ab vs mct')
        modes = ['ab vs ab', 'mct vs mct', 'ab vs mct', 'human vs ai']
        self.mode_menu = tk.OptionMenu(root, self.mode_var, *modes, command=self.update_score_display)
        self.mode_menu.pack()

        self.score_label = tk.Label(root, text="Score for this mode - X: 0 | O: 0 | Draws: 0", font=("Arial", 10))
        self.score_label.pack(pady=5)

        self.timer_label = tk.Label(root, text="Time: 20:00", font=("Courier", 15), fg="red")
        self.timer_label.pack()

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

    def update_score_display(self, *args):
        mode = self.mode_var.get()
        s = self.scores[mode]
        if mode == 'human vs ai':
            self.score_label.config(text=f"Human: {s['Human']} | AI: {s['AI']} | Draws: {s['Draw']}")
        else:
            self.score_label.config(text=f"X Wins: {s['X']} | O Wins: {s['O']} | Draws: {s['Draw']}")

    def start_match(self):
        self.state = gameState()
        self.time_left = timer_limit
        self.game_running = True
        self.start_btn.config(state="disabled")
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
        if self.mode_var.get() == 'human vs ai' and self.game_running:
            if self.state.board[index] == '' and self.state.player_turn == 'X':
                self.state = apply_move(self.state, index, 'X')
                self.refresh_board()
                self.ai_loop()

    def ai_loop(self):
        if not self.game_running: return
        
        winner = check_win(self.state.board)
        if winner:
            self.end_match(winner)
            return

        mode = self.mode_var.get()
        # If it's Human vs AI and it's X's turn, wait for human
        if mode == 'human vs ai' and self.state.player_turn == 'X':
            return

        move = choose_ai_move(self.state, self.state.player_turn, mode)
        if move is not None:
            self.state = apply_move(self.state, move, self.state.player_turn)
            self.refresh_board()
            self.root.after(600, self.ai_loop)

    def refresh_board(self):
        for i in range(9):
            char = self.state.board[i]
            self.buttons[i].config(text=char, fg="blue" if char == 'X' else "orange")

    def end_match(self, result):
        self.game_running = False
        self.start_btn.config(state="normal")
        mode = self.mode_var.get()
        
        # update score tracking
        if mode == 'human vs ai':
            if result == 'X': self.scores[mode]['Human'] += 1
            elif result == 'O': self.scores[mode]['AI'] += 1
            else: self.scores[mode]['Draw'] += 1
        else:
            if result in ['X', 'O', 'Draw']:
                self.scores[mode][result] += 1
        
        self.update_score_display()
        messagebox.showinfo("Game Result", f"Winner: {result}")

if __name__ == "__main__":
    root = tk.Tk()
    app = TicTacToeApp(root)
    root.mainloop()