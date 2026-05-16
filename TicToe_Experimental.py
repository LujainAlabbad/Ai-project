import tkinter as tk
from tkinter import font as tkfont
from tkinter import messagebox, ttk
import math
import random
import time
import threading

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

    if not root.children:
        moves = root_state.get_legal_moves()
        return random.choice(moves) if moves else None
    return max(root.children, key=lambda c: c.visits).move

def run_single_game(x_agent, o_agent, x_param, o_param, move_limit=300):
    state = gameState()
    x_time = 0.0
    o_time = 0.0
    moves = 0

    for _ in range(move_limit):
        winner = check_win(state.board)
        if winner:
            return winner, x_time, o_time, moves

        player = state.player_turn
        agent = x_agent if player == 'X' else o_agent
        param  = x_param  if player == 'X' else o_param

        t0 = time.perf_counter()
        if agent == 'AB':
            move = get_ab_move(state, player, param)
        else:
            move = mcts_best_move(state, param)
        elapsed = (time.perf_counter() - t0) * 1000

        if player == 'X':
            x_time += elapsed
        else:
            o_time += elapsed

        if move is None:
            break
        state = apply_move(state, move, player)
        moves += 1

    winner = check_win(state.board) or 'Draw'
    return winner, x_time, o_time, moves

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

        # ── Notebook (tabs) ──────────────────────────
        self.notebook = ttk.Notebook(root)
        self.notebook.pack(fill='both', expand=True, padx=8, pady=8)

        self.game_tab = tk.Frame(self.notebook)
        self.exp_tab  = tk.Frame(self.notebook)
        self.notebook.add(self.game_tab, text="  Game  ")
        self.notebook.add(self.exp_tab,  text="  Run Experiments  ")

        self._build_game_tab()
        self._build_experiment_tab()

    def _build_game_tab(self):
        tab = self.game_tab

        tk.Label(tab, text="Agent Configuration:", font=self.label_font).pack(pady=5)

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
        self.ctrl_frame = tk.Frame(tab)
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

        self.score_label = tk.Label(tab, text="X Wins: 0 | O Wins: 0 | Draws: 0", font=self.label_font, fg="darkblue")
        self.score_label.pack(pady=5)

        self.timer_label = tk.Label(tab, text="Time: 20:00", font=("Courier", 15), fg="red")
        self.timer_label.pack()

        # Initialize Visibility
        self.update_visibility()

        self.board_frame = tk.Frame(tab)
        self.board_frame.pack(pady=10)
        self.buttons = []
        for i in range(9):
            btn = tk.Button(self.board_frame, text="", font=("Arial", 20), width=4, height=2,
                            command=lambda i=i: self.human_click(i))
            btn.grid(row=i//3, column=i%3)
            self.buttons.append(btn)

        self.start_btn = tk.Button(tab, text="Start Match", command=self.start_match, bg="green", fg="white", width=15)
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

    def _build_experiment_tab(self):
        tab = self.exp_tab
        bold = tkfont.Font(family="Arial", size=11, weight="bold")
        head = tkfont.Font(family="Arial", size=13, weight="bold")

        tk.Label(tab, text="Experiment Settings", font=head).pack(pady=(12, 4))

        cfg = tk.Frame(tab)
        cfg.pack(pady=6)

        tk.Label(cfg, text="Matchup:", font=bold).grid(row=0, column=0, sticky="e", padx=6, pady=4)
        self.exp_matchup = tk.StringVar(value="AB vs AB")
        mu_menu = tk.OptionMenu(cfg, self.exp_matchup, "AB vs AB", "MCT vs MCT", "AB vs MCT",
                                command=self._on_matchup_change)
        mu_menu.config(width=12)
        mu_menu.grid(row=0, column=1, sticky="w", padx=6)

        self.x_param_label = tk.Label(cfg, text="X Depth (AB):", font=bold)
        self.x_param_label.grid(row=1, column=0, sticky="e", padx=6, pady=4)
        self.exp_x_param = tk.IntVar(value=5)
        self.exp_x_menu = tk.OptionMenu(cfg, self.exp_x_param, 2, 5, 10)
        self.exp_x_menu.config(width=8)
        self.exp_x_menu.grid(row=1, column=1, sticky="w", padx=6)

        self.o_param_label = tk.Label(cfg, text="O Depth (AB):", font=bold)
        self.o_param_label.grid(row=2, column=0, sticky="e", padx=6, pady=4)
        self.exp_o_param = tk.IntVar(value=5)
        self.exp_o_menu = tk.OptionMenu(cfg, self.exp_o_param, 2, 5, 10)
        self.exp_o_menu.config(width=8)
        self.exp_o_menu.grid(row=2, column=1, sticky="w", padx=6)

        tk.Label(cfg, text="Number of games:", font=bold).grid(row=3, column=0, sticky="e", padx=6, pady=4)
        self.exp_n_games = tk.IntVar(value=5)
        tk.OptionMenu(cfg, self.exp_n_games, 1, 3, 5, 10, 20).grid(row=3, column=1, sticky="w", padx=6)

        self.run_exp_btn = tk.Button(tab, text="▶  Run Experiment", font=bold,
                                     bg="#1a6b3a", fg="white", width=20, pady=4,
                                     command=self._start_experiment)
        self.run_exp_btn.pack(pady=10)

        self.exp_progress = ttk.Progressbar(tab, length=460, mode='determinate')
        self.exp_progress.pack(pady=(0, 6))

        self.exp_status = tk.Label(tab, text="", font=("Arial", 10), fg="gray")
        self.exp_status.pack()

        cols = ("Game", "Winner", "X Time (ms)", "O Time (ms)", "Moves")
        self.results_tree = ttk.Treeview(tab, columns=cols, show='headings', height=10)
        for c in cols:
            w = 80 if c != "Game" else 50
            self.results_tree.heading(c, text=c)
            self.results_tree.column(c, width=w, anchor='center')

        sb = ttk.Scrollbar(tab, orient="vertical", command=self.results_tree.yview)
        self.results_tree.configure(yscrollcommand=sb.set)
        self.results_tree.pack(side="left", fill="both", expand=True, padx=(10, 0), pady=6)
        sb.pack(side="left", fill="y", pady=6)

        sum_frame = tk.Frame(tab)
        sum_frame.pack(side="left", fill="y", padx=10, pady=6, anchor="n")

        tk.Label(sum_frame, text="Summary", font=head).pack(pady=(0, 8))
        self.sum_labels = {}
        for key in ("X Wins", "O Wins", "Draws",
                    "Avg X Time", "Avg O Time", "Avg Moves", "Avg Time/Move"):
            row = tk.Frame(sum_frame)
            row.pack(fill="x", pady=2)
            tk.Label(row, text=f"{key}:", font=bold, width=13, anchor="e").pack(side="left")
            lbl = tk.Label(row, text="—", font=("Arial", 11), anchor="w")
            lbl.pack(side="left", padx=4)
            self.sum_labels[key] = lbl

        self._on_matchup_change("AB vs AB")

    def _on_matchup_change(self, value):
        self.exp_x_menu.destroy()
        self.exp_o_menu.destroy()

        cfg = self.x_param_label.master

        if value == "AB vs AB":
            x_lbl, o_lbl = "X Depth (AB):", "O Depth (AB):"
            self.exp_x_param.set(5)
            self.exp_o_param.set(5)
            self.exp_x_menu = tk.OptionMenu(cfg, self.exp_x_param, 2, 5, 10)
            self.exp_x_menu.config(width=8)
            self.exp_x_menu.grid(row=1, column=1, sticky="w", padx=6)
            self.exp_o_menu = tk.OptionMenu(cfg, self.exp_o_param, 2, 5, 10)
            self.exp_o_menu.config(width=8)
            self.exp_o_menu.grid(row=2, column=1, sticky="w", padx=6)

        elif value == "MCT vs MCT":
            x_lbl, o_lbl = "X Iterations (MCT):", "O Iterations (MCT):"
            self.exp_x_param.set(500)
            self.exp_o_param.set(500)
            self.exp_x_menu = tk.OptionMenu(cfg, self.exp_x_param, 200, 500, 1000)
            self.exp_x_menu.config(width=8)
            self.exp_x_menu.grid(row=1, column=1, sticky="w", padx=6)
            self.exp_o_menu = tk.OptionMenu(cfg, self.exp_o_param, 200, 500, 1000)
            self.exp_o_menu.config(width=8)
            self.exp_o_menu.grid(row=2, column=1, sticky="w", padx=6)

        else:
            x_lbl, o_lbl = "X Depth (AB):", "O Iterations (MCT):"
            self.exp_x_param.set(5)
            self.exp_o_param.set(500)
            self.exp_x_menu = tk.OptionMenu(cfg, self.exp_x_param, 2, 5, 10)
            self.exp_x_menu.config(width=8)
            self.exp_x_menu.grid(row=1, column=1, sticky="w", padx=6)
            self.exp_o_menu = tk.OptionMenu(cfg, self.exp_o_param, 200, 500, 1000)
            self.exp_o_menu.config(width=8)
            self.exp_o_menu.grid(row=2, column=1, sticky="w", padx=6)

        self.x_param_label.config(text=x_lbl)
        self.o_param_label.config(text=o_lbl)

    def _start_experiment(self):
        self.run_exp_btn.config(state="disabled")
        for row in self.results_tree.get_children():
            self.results_tree.delete(row)
        for lbl in self.sum_labels.values():
            lbl.config(text="—")

        matchup  = self.exp_matchup.get()
        x_param  = self.exp_x_param.get()
        o_param  = self.exp_o_param.get()
        n_games  = self.exp_n_games.get()

        if matchup == "AB vs AB":
            x_agent, o_agent = "AB", "AB"
        elif matchup == "MCT vs MCT":
            x_agent, o_agent = "MCT", "MCT"
        else:
            x_agent, o_agent = "AB", "MCT"

        self.exp_progress['maximum'] = n_games
        self.exp_progress['value']   = 0

        t = threading.Thread(
            target=self._run_experiment_thread,
            args=(x_agent, o_agent, x_param, o_param, n_games),
            daemon=True
        )
        t.start()

    def _run_experiment_thread(self, x_agent, o_agent, x_param, o_param, n_games):
        results = []
        for g in range(1, n_games + 1):
            self.root.after(0, lambda g=g: self.exp_status.config(
                text=f"Running game {g} / {n_games}…"))
            winner, xt, ot, mv = run_single_game(x_agent, o_agent, x_param, o_param)
            results.append((g, winner, round(xt, 1), round(ot, 1), mv))
            self.root.after(0, self._append_result_row, g, winner, xt, ot, mv)
            self.root.after(0, lambda g=g: self.exp_progress.config(value=g))

        self.root.after(0, self._finish_experiment, results)

    def _append_result_row(self, g, winner, xt, ot, mv):
        tag = 'x' if winner == 'X' else ('o' if winner == 'O' else 'draw')
        self.results_tree.insert('', 'end',
            values=(g, winner, f"{xt:.1f}", f"{ot:.1f}", mv),
            tags=(tag,))
        self.results_tree.tag_configure('x',    background='#d0e8ff')
        self.results_tree.tag_configure('o',    background='#ffd0a0')
        self.results_tree.tag_configure('draw', background='#e8e8e8')

    def _finish_experiment(self, results):
        wins_x = sum(1 for r in results if r[1] == 'X')
        wins_o = sum(1 for r in results if r[1] == 'O')
        draws  = sum(1 for r in results if r[1] == 'Draw')
        avg_xt = sum(r[2] for r in results) / len(results)
        avg_ot = sum(r[3] for r in results) / len(results)
        avg_mv = sum(r[4] for r in results) / len(results)
        avg_total_time   = sum(r[2] + r[3] for r in results) / len(results)
        avg_time_per_move = avg_total_time / avg_mv if avg_mv else 0

        self.sum_labels["X Wins"].config(text=str(wins_x))
        self.sum_labels["O Wins"].config(text=str(wins_o))
        self.sum_labels["Draws"].config(text=str(draws))
        self.sum_labels["Avg X Time"].config(text=f"{avg_xt:.1f} ms")
        self.sum_labels["Avg O Time"].config(text=f"{avg_ot:.1f} ms")
        self.sum_labels["Avg Moves"].config(text=f"{avg_mv:.1f}")
        self.sum_labels["Avg Time/Move"].config(text=f"{avg_time_per_move:.1f} ms")

        self.exp_status.config(text=f"✔ Done — {len(results)} games completed.", fg="green")
        self.run_exp_btn.config(state="normal")


def run_gui_mode():
    root = tk.Tk()
    app = TicTacToeApp(root)
    root.mainloop()

if __name__ == "__main__":
    run_gui_mode()
