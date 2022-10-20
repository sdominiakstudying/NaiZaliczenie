### Rules: https://en.wikipedia.org/wiki/Five_Field_Kono

### Authors: Mateusz Pioch (s21331), Stanisław Dominiak (s18864)

### To prepare the environmnent you first need to install EasyAI,
### for example through the use of the "sudo pip install easyAI" command.

from easyAI import TwoPlayerGame, Human_Player, AI_Player, Negamax
import numpy as np

DIRECTIONS = list(map(np.array, [[1,1],[-1,1],[1,-1],[-1,-1]]))

to_char = lambda a: "ABCDE"[a[0]]
to_array = lambda s: np.array(["ABCDE".index(s[0]), int(s[1]) - 1])

class FiveFieldKono( TwoPlayerGame ):
    """Since the game isn't too convoluted
    basically all of it is in the entire class. 
    First everything is initialised, then all the possible moves are explained,
    then the input and what happens is defined and in the end we describe
    the way that victory plays out.
    """
    

    def __init__(self, players):
        """ Most of the following are inherent to EasyAI. The first line of code
        is used to initialise the players, the second defines a starting board
        with the starting locations of all the pieces,
        and the last line elects who is the starting player.
        
        Returns
        -------
        None.        
        """
        
        self.players = players
        self.board = [
            [1, 1, 1, 1, 1],
            [1, 0, 0, 0, 1],
            [0, 0, 0, 0, 0],
            [2, 0, 0, 0, 2],
            [2, 2, 2, 2, 2]
        ]
        self.current_player = 1 # player 1 starts

    def possible_moves(self):
        """Lists all the possible moves that can be made.        

        Returns
        -------
        moves : str[]
            An array that contains all the possible moves that can be made.

        """
        moves = []
        for row_index, row in enumerate(self.board):
            for column_index, column in enumerate(row):
                if column == self.current_player:
                    for d in DIRECTIONS:
                        potential_move = [row_index+d[0], column_index+d[1]]
                        
                        if (0 <= potential_move[0] < 5) and (0 <= potential_move[1] < 5) and (self.board[potential_move[0]][potential_move[1]] == 0):
                            moves.append(to_char([row_index]) + str(column_index+1) + "=>" + to_char([row_index+d[0]]) + str(column_index+1+d[1]))
        #print(self.board)
        return moves
    def make_move(self,move): 
        """Used to determine what happens whenever a move is chosen.        

        Parameters
        ----------
        move : str
            In this context it's taken from the "moves" array, and
            presents the player with options.

        Returns
        -------
        None.

        """
        move_whole = move.split("=>")
        move_from = to_array(move_whole[0])
        move_to = to_array(move_whole[1])
        self.board[move_from[0]][move_from[1]] = 0
        self.board[move_to[0]][move_to[1]] = self.current_player

    def win(self): 
        """
        

        Returns
        -------
        bool
            An info if 

        """
        if self.current_player == 1:
            return self.board[4] == 1 and board[3][0] == 1 and board[3][4] == 1
        else:
            return self.board[0] == 2 and board[1][0] == 2 and board[1][4] == 2

    def is_over(self): 
        """
        

        Returns
        -------
        bool
            An info if the game is over.

        """        
        return self.win()
    def show(self):
        """ Prints both the board and all the moves for the player.
        

        Returns
        -------
        None.

        """
        print(
            "\n"
            + "\n".join(
                ["  1 2 3 4 5 "]
                + [
                    "ABCDE"[k]
                    + " "
                    + " ".join(
                        [[".", "1", "2"][self.board[k][i]] for i in range(5)]
                    )
                    for k in range(5)
                ]
                + [""]
            )
        )
        print("Possible moves for player " + str(self.current_player) + " are " + str(self.possible_moves()))
        #print("\n".join(str(d) for d in DIRECTIONS))
    def scoring(self): 
        """
        

        Returns
        -------
        int
            requirement for the AI - in the documentation: http://zulko.github.io/easyAI/get_started.html .

        """
        return 100 if game.win() else 0 # For the AI

# Start a match (and store the history of moves when it ends)
ai = Negamax(6) # The AI will think 6 moves in advance
game = FiveFieldKono( [ Human_Player(), AI_Player(ai) ] )
history = game.play()