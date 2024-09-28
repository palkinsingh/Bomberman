import numpy as np
import random
import heapq
import logging
from collections import deque

def setup(self):
    # Define the state space features
    state_features = {
        "tile_up": ['free_tile', 'coin', 'invalid', 'adjacent', 'explosion', 'ticking'],
        "tile_down": ['free_tile', 'coin', 'invalid', 'adjacent', 'explosion', 'ticking'],
        "tile_left": ['free_tile', 'coin', 'invalid', 'adjacent', 'explosion', 'ticking'],
        "tile_right": ['free_tile', 'coin', 'invalid', 'adjacent', 'explosion', 'ticking'],
        "compass": ["N", "S", "W", "E", "NP"],
        "compass_mode": ["coin", "crate", "escape", "attack"]
    }

    # Create the state space object
    self.state_space = GameState(state_features)

    # Initialize Q-table and parameters
    self.q_table = {}  # Initialize an empty Q-table
    self.epsilon = 1.0  # Initial epsilon value for exploration
    self.epsilon_decay = 0.999  # Epsilon decay rate
    self.epsilon_min = 0.01  # Minimum epsilon value
    self.bomb_dropped_recently = False  # Track whether a bomb was recently dropped
    self.visited_tiles = {}  # Initialize visited tiles map
    self.move_history = []  # Track recent moves

    self.visitation_map = None  # Track visited tiles

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(name)s] %(levelname)s: %(message)s')
logger = logging.getLogger('Ironman')

class GameState:
    def __init__(self, state_features):
        self.state_features = state_features
        self.state_mapping = self.create_state_mapping()

    def create_state_mapping(self):
        state_mapping = {}
        for feature_name, feature_values in self.state_features.items():
            state_mapping[feature_name] = {value: idx for idx, value in enumerate(feature_values)}
        return state_mapping

    def get_index(self, agent_state):
        index = 0
        for feature, value in agent_state.items():
            feature_index = self.state_mapping[feature].get(value, 0)  # Default to 0 if feature not found
            index = index * len(self.state_mapping[feature]) + feature_index
        return index

    def get_state(self, agent_state):
        return agent_state

def manhattan_distance(a, b):
    return abs(a[0] - b[0]) + abs(a[1] - b[1])

def simulate_move(position, direction):
    if direction == 'UP':
        return (position[0], position[1] - 1)
    elif direction == 'DOWN':
        return (position[0], position[1] + 1)
    elif direction == 'LEFT':
        return (position[0] - 1, position[1])
    elif direction == 'RIGHT':
        return (position[0] + 1, position[1])
    return position

def reconstruct_path(came_from, current):
    total_path = [current]
    while current in came_from:
        current = came_from[current]
        total_path.insert(0, current)
    return total_path

def get_neighbors(position, field):
    x, y = position
    neighbors = []
    directions = [(-1,0), (1,0), (0,-1), (0,1)]
    for dx, dy in directions:
        nx, ny = x + dx, y + dy
        if 0 <= nx < field.shape[0] and 0 <= ny < field.shape[1]:
            if field[nx, ny] == 0:  # Walkable tile
                neighbors.append((nx, ny))
    return neighbors

def find_nearest_coin(position, coins):
    if not coins:
        return None, float('inf')  # No coins left
    nearest_coin = min(coins, key=lambda coin: manhattan_distance(position, coin))
    return nearest_coin, manhattan_distance(position, nearest_coin)

def move_towards_coin(position, coin, field):
    if not coin:
        return 'WAIT'  # No coin to move to

    directions = ['UP', 'DOWN', 'LEFT', 'RIGHT']
    best_direction = 'WAIT'
    min_distance = float('inf')

    for direction in directions:
        new_position = simulate_move(position, direction)
        if field[new_position[0], new_position[1]] == 0:  # Walkable tile
            distance_to_coin = manhattan_distance(new_position, coin)
            if distance_to_coin < min_distance:
                min_distance = distance_to_coin
                best_direction = direction

    return best_direction

def compute_danger_map(game_state):
    field = game_state['field']
    bombs = game_state['bombs']
    explosions = game_state['explosion_map']
    width, height = field.shape

    danger_map = np.zeros((width, height), dtype=int)

    for bomb in bombs:
        bomb_x, bomb_y = bomb[0]
        bomb_timer = bomb[1]

        danger_map[bomb_x, bomb_y] = max(danger_map[bomb_x, bomb_y], 5 - bomb_timer)

        for direction in [(-1,0), (1,0), (0,-1), (0,1)]:
            dx, dy = direction
            for i in range(1, 4):  # Assuming blast length of 3
                x, y = bomb_x + dx * i, bomb_y + dy * i
                if x < 0 or x >= width or y < 0 or y >= height:
                    break
                if field[x, y] == -1:
                    break
                danger_map[x, y] = max(danger_map[x, y], 5 - bomb_timer)
                if field[x, y] == 1:
                    break

    # Mark areas as dangerous where explosions are currently hapening
    explosion_indices = np.where(explosions > 0)
    for x, y in zip(explosion_indices[0], explosion_indices[1]):
        danger_map[x, y] = 5  # Immediate danger

    return danger_map

def is_position_safe(position, game_state, agent):
    field = game_state['field']
    explosion_map = game_state['explosion_map']

    if (0 <= position[0] < field.shape[0]) and (0 <= position[1] < field.shape[1]):
        if field[position[0], position[1]] == 0 and explosion_map[position[0], position[1]] == 0:
            # Ensure bomb positions are considered
            bomb_positions = [bomb[0] for bomb in game_state['bombs']]
            if position not in bomb_positions:
                return True
    return False

def conservative_escape_check(game_state, agent, next_step):
    danger_map = compute_danger_map(game_state)
    if danger_map[next_step[0], next_step[1]] == 0:
        return True
    return False

def is_in_dead_end(position, field):
    x, y = position
    width, height = field.shape
    directions = [(-1,0), (1,0), (0,-1), (0,1)]
    open_paths = 0

    for dx, dy in directions:
        nx, ny = x + dx, y + dy
        if 0 <= nx < width and 0 <= ny < height and field[nx, ny] == 0:
            open_paths += 1

    return open_paths == 1  # Only one direction available, so it's a dead-end

def move_away_from_bomb(game_state, agent):
    position = game_state['self'][3]
    bomb_positions = [bomb[0] for bomb in game_state['bombs']]

    if not bomb_positions:
        return 'WAIT'  # No bomb to avoid, so we wait

    # Get the position of the first bomb (most relevant one for escape)
    bomb_x, bomb_y = bomb_positions[0]

    # Compare the agent's position to the bomb's position
    if bomb_x < position[0]:
        return 'RIGHT'  # Bomb is to the left, move right
    elif bomb_x > position[0]:
        return 'LEFT'  # Bomb is to the right, move left
    elif bomb_y < position[1]:
        return 'DOWN'  # Bomb is above, move down
    elif bomb_y > position[1]:
        return 'UP'  # Bomb is below, move up

    return 'WAIT'  

def move_towards_safest_direction(game_state, agent):
    position = game_state['self'][3]
    danger_map = compute_danger_map(game_state)
    bomb_positions = [bomb[0] for bomb in game_state['bombs']]

    # Check if the current position is safe; if so, stay put
    if danger_map[position[0], position[1]] == 0:
        return 'WAIT'

    directions = ['UP', 'DOWN', 'LEFT', 'RIGHT']
    best_direction = None
    min_danger = float('inf')
    max_distance_from_bombs = -float('inf')

    # Determine if the agent is in the top or bottom half of the map
    field = game_state['field']
    map_height = field.shape[1]  # Y-axis represents height

    if position[1] < map_height // 2:
        escape_priority = 'DOWN'
    else:
        escape_priority = 'UP'

    for direction in directions:
        new_position = simulate_move(position, direction)

        # Check if the new position is safe
        if is_position_safe(new_position, game_state, agent):
            danger_level = danger_map[new_position[0], new_position[1]]
            distance_from_bombs = min([manhattan_distance(new_position, bomb) for bomb in bomb_positions], default=0)

            if direction == escape_priority and danger_level == 0:
                return escape_priority  # Move backward when it's safe

            if danger_level < min_danger or (danger_level == min_danger and distance_from_bombs > max_distance_from_bombs):
                best_direction = direction
                min_danger = danger_level
                max_distance_from_bombs = distance_from_bombs

    if best_direction:
        return best_direction

    return 'WAIT'  # If no safe direction found

def find_safe_escape(game_state, position, danger_map, after_bomb=False):
    field = game_state['field']
    width, height = field.shape

    modified_danger_map = np.copy(danger_map)
    if after_bomb:
        modified_danger_map[position[0], position[1]] = max(modified_danger_map[position[0], position[1]], 1)
        for direction in [(-1,0), (1,0), (0,-1), (0,1)]:
            dx, dy = direction
            for i in range(1, 4):
                nx, ny = position[0] + dx*i, position[1] + dy*i
                if 0 <= nx < width and 0 <= ny < height:
                    if field[nx, ny] == -1:
                        break
                    modified_danger_map[nx, ny] = max(modified_danger_map[nx, ny], 1)
                    if field[nx, ny] == 1:
                        break

    safe_tiles = np.argwhere(modified_danger_map == 0)
    if not safe_tiles.size:
        return None  # No safe tiles

    min_distance = float('inf')
    best_path = None
    for safe_tile in safe_tiles:
        path = a_star_search(game_state, position, tuple(safe_tile), modified_danger_map)
        if path and len(path) < min_distance:
            min_distance = len(path)
            best_path = path
    return best_path

def include_agents_in_field(game_state):
    agent_locations = [item[3] for item in game_state["others"]]

    for x, y in agent_locations:
        game_state["field"][x][y] = 2

def a_star_search(game_state, start, goal, danger_map):
    field = game_state['field']
    width, height = field.shape
    
    # Example usage of width and height
    logger.info(f"Field dimensions: width={width}, height={height}")
    
    open_set = []
    heapq.heappush(open_set, (0, start))
    came_from = {}
    g_score = {start: 0}
    f_score = {start: manhattan_distance(start, goal)}

    while open_set:
        current = heapq.heappop(open_set)[1]

        if current == goal:
            return reconstruct_path(came_from, current)

        for neighbor in get_neighbors(current, field):
            tentative_g_score = g_score[current] + 1 + danger_map[neighbor[0], neighbor[1]] * 10

            if neighbor not in g_score or tentative_g_score < g_score[neighbor]:
                came_from[neighbor] = current
                g_score[neighbor] = tentative_g_score
                f_score[neighbor] = g_score[neighbor] + manhattan_distance(neighbor, goal)
                heapq.heappush(open_set, (f_score[neighbor], neighbor))

    return None  # No path found


def is_in_danger(game_state):
    position = game_state['self'][3]
    bombs = game_state['bombs']
    danger_map = compute_danger_map(game_state)

    if danger_map[position[0], position[1]] > 0:
        return True

    for bomb_position, bomb_timer in bombs:
        if bomb_timer <= 2 and np.sum(np.abs(np.array(bomb_position) - np.array(position))) <= 3:
            return True
    return False

def has_sufficient_escape_routes(position, game_state):
    field = game_state['field']
    neighbors = get_neighbors(position, field)
    safe_neighbors = [neighbor for neighbor in neighbors if is_position_safe(neighbor, game_state, None)]

    return len(safe_neighbors) >= 2

def should_attack_opponent(game_state, agent):
    position = game_state['self'][3]
    field = game_state['field']
    opponents = get_opponent_positions(game_state)

    nearest_opponent, distance = find_nearest_opponent(position, opponents)

    if nearest_opponent and distance <= 3:  # Attack only if the opponent is within 3 tiles
        if is_opponent_in_bomb_range(position, nearest_opponent, field):
            if can_place_bomb_safely(position, game_state):
                agent.target_opponent_direction = get_direction(position, nearest_opponent)
                logger.info(f"Considering dropping bomb near opponent at {nearest_opponent}. Distance: {distance}")
                return True
    return False



def get_opponent_positions(game_state):
    return [agent[3] for agent in game_state['others']]

def find_nearest_opponent(position, opponents):
    if not opponents:
        return None, float('inf')
    nearest_opponent = min(opponents, key=lambda opponent: manhattan_distance(position, opponent))
    distance = manhattan_distance(position, nearest_opponent)
    return nearest_opponent, distance

def is_opponent_in_bomb_range(position, opponent_position, field):
    x1, y1 = position
    x2, y2 = opponent_position

    if x1 == x2:
        step = 1 if y2 > y1 else -1
        for y in range(y1 + step, y2 + step, step):
            if field[x1, y] == -1:
                return False
        return abs(y2 - y1) <= 3
    elif y1 == y2:
        step = 1 if x2 > x1 else -1
        for x in range(x1 + step, x2 + step, step):
            if field[x, y1] == -1:
                return False
        return abs(x2 - x1) <= 3
    return False

def can_place_bomb_safely(position, game_state):
    simulated_game_state = game_state.copy()
    simulated_game_state['bombs'] = game_state['bombs'] + [(position, 3)]  # Assuming bomb timer starts at 3
    danger_map = compute_danger_map(simulated_game_state)

    escape_possible = False
    for direction in ['UP', 'DOWN', 'LEFT', 'RIGHT']:
        new_pos = simulate_move(position, direction)
        if is_position_safe(new_pos, simulated_game_state, None):
            escape_possible = True
            break
    return escape_possible

def get_direction(from_pos, to_pos):
    fx, fy = from_pos
    tx, ty = to_pos
    if tx < fx:
        return 'LEFT'
    elif tx > fx:
        return 'RIGHT'
    elif ty < fy:
        return 'UP'
    elif ty > fy:
        return 'DOWN'
    return 'WAIT'

def drop_bomb_if_in_range_of_agent(closest_agent, game_state, action, self):
    if closest_agent is None or action != 'BOMB' or not game_state['self'][2]:
        return 0

    agent_coord_x, agent_coord_y = game_state['self'][3]
    x, y = closest_agent

    if abs(y - agent_coord_y) <= 3 and x == agent_coord_x and agent_coord_x % 2 != 0:
        return 1
    if abs(x - agent_coord_x) <= 3 and y == agent_coord_y and agent_coord_y % 2 != 0:
        return 1

    return 0

def runs_towards_closest_agent_but_not_wall_or_crate(closest_agent, game_state, action, self):
    if closest_agent is None:
        return 0

    agent_coord_x, agent_coord_y = game_state['self'][3]
    x, y = closest_agent

    if action == 'UP' and abs(y - (agent_coord_y - 1)) < abs(y - agent_coord_y) and game_state['field'][agent_coord_x][agent_coord_y-1] == 0:
        return 1 if ((agent_coord_y - 1) % 2 != 0 or abs(y - (agent_coord_y - 1)) != 0) else 0
    if action == 'RIGHT' and abs(x - (agent_coord_x + 1)) < abs(x - agent_coord_x) and game_state['field'][agent_coord_x+1][agent_coord_y] == 0:
        return 1 if ((agent_coord_y + 1) % 2 != 0 or abs(x - (agent_coord_x + 1)) != 0) else 0
    if action == 'DOWN' and abs(y - (agent_coord_y + 1)) < abs(y - agent_coord_y) and game_state['field'][agent_coord_x][agent_coord_y+1] == 0:
        return 1 if ((agent_coord_y + 1) % 2 != 0 or abs(y - (agent_coord_y + 1)) != 0) else 0
    if action == 'LEFT' and abs(x - (agent_coord_x - 1)) < abs(x - agent_coord_x) and game_state['field'][agent_coord_x-1][agent_coord_y] == 0:
        return 1 if ((agent_coord_x - 1) % 2 != 0 or abs(x - (agent_coord_x - 1)) != 0) else 0

    return 0



def detect_oscillation(move_history, max_cycle_length):
    for cycle_length in range(2, max_cycle_length + 1):
        cycle = move_history[-cycle_length * 2:]
        if len(cycle) < cycle_length * 2:
            continue
        first_half = cycle[:cycle_length]
        second_half = cycle[cycle_length:]
        if first_half == second_half:
            return True
    return False

def state_to_features(game_state: dict):
    if game_state is None:
        return None

    position = np.array(game_state['self'][3])
    coins = np.array(game_state['coins'])
    bombs = game_state['bombs']
    explosion_map = game_state['explosion_map']  # Currently unused
    others = np.array([other[3] for other in game_state['others']])

    nearest_coin_dist = min([np.sum(np.abs(coin - position)) for coin in coins]) if coins.size > 0 else 0

    bomb_danger = 0
    for bomb_position, bomb_timer in bombs:
        if bomb_timer <= 2 and np.sum(np.abs(np.array(bomb_position) - position)) <= 3:
            bomb_danger = 1
            break

    nearest_opponent_dist = min([np.sum(np.abs(opponent - position)) for opponent in others]) if others.size > 0 else 0

    # Determine whether the agent's current position is in an explosion or nearsby danger
    explosion_danger = explosion_map[position[0], position[1]] > 0

    left = game_state['field'][position[0] - 1, position[1]] if position[0] > 0 else -1
    right = game_state['field'][position[0] + 1, position[1]] if position[0] < game_state['field'].shape[0] - 1 else -1
    up = game_state['field'][position[0], position[1] - 1] if position[1] > 0 else -1
    down = game_state['field'][position[0], position[1] + 1] if position[1] < game_state['field'].shape[1] - 1 else -1

    dead_end = int(left != 0 and right != 0 and up != 0 and down != 0)

    features = (position[0], position[1], nearest_coin_dist, bomb_danger, nearest_opponent_dist, left, right, up, down, dead_end, explosion_danger)
    return features


def find_closest_crates(game_state, action, self): 
    agent_coord_x = game_state['self'][3][0]
    agent_coord_y = game_state['self'][3][1]

    x_vals = np.where(game_state['field'] == 1)[1]
    y_vals = np.where(game_state['field'] == 1)[0]
    crate_locations = [(x, y) for (x, y) in zip(y_vals, x_vals) if y is not None]

    closest_crate = None
    closest_dist = 100

    for crate_x, crate_y in crate_locations:
        dist = np.linalg.norm([crate_x - agent_coord_x, crate_y - agent_coord_y])
        if dist < closest_dist:
            closest_dist = dist
            closest_crate = [crate_x, crate_y]

    return closest_crate

def move_away_from_walls_or_toward_crates(game_state, agent):
    position = game_state['self'][3]
    field = game_state['field']
    directions = ['UP', 'DOWN', 'LEFT', 'RIGHT']
    best_direction = None

    for direction in directions:
        new_position = simulate_move(position, direction)
        # Avoid moving towards walls
        if field[new_position[0], new_position[1]] != -1 and is_position_safe(new_position, game_state, agent):
            best_direction = direction
            break

    # If no safe direction, fallback to current exploration logic
    return best_direction if best_direction else explore_map(agent, game_state)

def include_agents_in_field(game_state):
    agent_locations = [item[3] for item in game_state["others"]]

    for x, y in agent_locations:
        game_state["field"][x][y] = 2

def find_escape_path_with_priority(game_state, agent):
    position = game_state['self'][3]
    danger_map = compute_danger_map(game_state)

    # Identify all possible escape routes
    escape_path = find_safe_escape_with_distance(game_state, position, danger_map, after_bomb=True)

    if escape_path:
        return escape_path
    else:
        return move_towards_safest_direction(game_state, agent)  # Fall back to moving to a safe spot

def find_safe_escape_with_distance(game_state, position, danger_map, after_bomb=False):
    field = game_state['field']
    width, height = field.shape

    modified_danger_map = np.copy(danger_map)
    if after_bomb:
        modified_danger_map[position[0], position[1]] = max(modified_danger_map[position[0], position[1]], 1)
        for direction in [(-1,0), (1,0), (0,-1), (0,1)]:
            dx, dy = direction
            for i in range(1, 4):  # Blast length of 3
                nx, ny = position[0] + dx*i, position[1] + dy*i
                if 0 <= nx < width and 0 <= ny < height:
                    if field[nx, ny] == -1:
                        break
                    modified_danger_map[nx, ny] = max(modified_danger_map[nx, ny], 1)
                    if field[nx, ny] == 1:
                        break

    # Identify safe tiles to escape to
    safe_tiles = np.argwhere(modified_danger_map == 0)
    if not safe_tiles.size:
        return None  # No safe tiles

    # Prioritize more distant safe tiles to avoid blast proximity
    best_path = None
    min_distance = float('inf')
    for safe_tile in safe_tiles:
        path = a_star_search(game_state, position, tuple(safe_tile), modified_danger_map)
        if path:
            distance_from_bomb = np.sum(np.abs(np.array(safe_tile) - np.array(position)))
            if distance_from_bomb < min_distance:
                min_distance = distance_from_bomb
                best_path = path

    return best_path

def find_opposite_direction(position, game_state):
    bombs = game_state['bombs']
    if bombs:
        bomb_x, bomb_y = bombs[0][0]  # Get the position of the first bomb
        x, y = position

        if bomb_x < x:
            return 'RIGHT'  # Bomb is to the left, move right
        elif bomb_x > x:
            return 'LEFT'  # Bomb is to the right, move left
        elif bomb_y < y:
            return 'DOWN'  # Bomb is above, move down
        elif bomb_y > y:
            return 'UP'  # Bomb is below, move up

    return None  # No bomb found, no specific direction to move

def explore_map(agent, game_state):
    position = game_state['self'][3]
    field = game_state['field']
    danger_map = compute_danger_map(game_state)

    # Initialize the visitation map if not already done
    if not hasattr(agent, 'visitation_map') or agent.visitation_map is None:
        agent.visitation_map = np.zeros_like(field)

    # Mark the current position as visited
    agent.visitation_map[position[0], position[1]] += 1

    directions = ['UP', 'DOWN', 'LEFT', 'RIGHT']
    safe_directions = []
    direction_scores = []

    recent_positions = set(agent.move_history[-10:])

    for direction in directions:
        new_position = simulate_move(position, direction)
        # Only move if the new position is safe and not in a danger zone
        if is_position_safe(new_position, game_state, None) and danger_map[new_position[0], new_position[1]] == 0:
            if new_position not in recent_positions and new_position not in agent.destroyed_crates:
                # Prefer directions that lead to less-visited areas
                score = -agent.visitation_map[new_position[0], new_position[1]]
                safe_directions.append(direction)
                direction_scores.append(score)

    if safe_directions:
        # Choose the direction that leads to the least visited tile
        best_direction = safe_directions[np.argmax(direction_scores)]
        return best_direction

    # If no safe unvisited direction, pick any safe direction not leading to recent positions or destroyed crates
    for direction in directions:
        new_position = simulate_move(position, direction)
        if is_position_safe(new_position, game_state, None) and new_position not in recent_positions and new_position not in agent.destroyed_crates:
            return direction

    # If no safe direction, wait or move cautiously
    return 'WAIT'

def is_in_corner(position, field):
    x, y = position
    width, height = field.shape

    # Check the four corners of the map
    if (x == 0 and y == 0) or (x == 0 and y == height - 1) or (x == width - 1 and y == 0) or (x == width - 1 and y == height - 1):
        return True

    # Check if surrounded by walls/crates on two sides (L-shape corner)
    if (field[x - 1, y] == -1 or field[x + 1, y] == -1) and (field[x, y - 1] == -1 or field[x, y + 1] == -1):
        return True

    return False

def should_drop_bomb(game_state, agent):
    position = game_state['self'][3]
    field = game_state['field']
    bombs_left = game_state['self'][2]

    # Avoid dropping bombs if none are left
    if bombs_left == 0:
        return False

    # Avoid dropping bombs in corners unless absolutely necessary
    if is_in_corner(position, field):
        logger.info("Agent is in a corner, avoiding bomb drop unless necessary.")
        return False

    # Prioritize dropping bombs near crates
    adjacent_crate_direction = get_adjacent_crates_prioritized(position, field)
    if adjacent_crate_direction:
        logger.info(f"Adjacent crate found at {adjacent_crate_direction}, considering bomb drop.")

        # Ensure there is a safe escape path after dropping the bomb
        escape_path = find_safe_escape_with_distance(game_state, position, compute_danger_map(game_state), after_bomb=True)

        if escape_path:
            agent.safe_position = position
            logger.info(f"Dropping bomb near crate in direction: {adjacent_crate_direction}. Escape path is {escape_path}")
            return True

    # If no escape path or no adjacent crates, do not drop the bomb
    logger.info("No safe escape path or no adjacent crates, avoiding bomb drop.")
    return False

def get_adjacent_crates_prioritized(position, field):
    x, y = position

    # Prioritize checking UP and DOWN first
    directions = [('UP', (0, -1)), ('DOWN', (0, 1)), ('LEFT', (-1, 0)), ('RIGHT', (1, 0))]

    for direction, (dx, dy) in directions:
        nx, ny = x + dx, y + dy
        if 0 <= nx < field.shape[0] and 0 <= ny < field.shape[1]:
            if field[nx, ny] == 1:  # Crate found
                return direction

    return None  # No adjacent crates

def act(self, game_state: dict) -> str:
    ACTIONS = ['UP', 'DOWN', 'LEFT', 'RIGHT', 'WAIT', 'BOMB']

    position = game_state['self'][3]
    coins = game_state['coins']  # Coins on the map
    bombs = game_state['bombs']  # Active bombs on the map
    step = game_state['step']

    logger.info(f"Step: {step}, Position: {position}, Coins on Map: {len(coins)}, Bombs on Map: {len(bombs)}")

    # Initialize necessary attributes if they are not present
    if not hasattr(self, 'bomb_dropped_recently'):
        self.bomb_dropped_recently = False
    if not hasattr(self, 'visited_tiles'):
        self.visited_tiles = {}
    if not hasattr(self, 'move_history'):
        self.move_history = []
    if not hasattr(self, 'destroyed_crates'):
        self.destroyed_crates = set()
    if not hasattr(self, 'target_opponent_direction'):
        self.target_opponent_direction = None
    if not hasattr(self, 'visitation_map'):
        self.visitation_map = None

    # **State Representation**: Compute agent state as features
    agent_state = state_to_features(game_state)

    # Epsilon-greedy strategy (only applies during training)
    if self.train:
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
        if random.random() < self.epsilon:
            chosen_action = random.choice(ACTIONS)
            logger.info("Exploring with epsilon-greedy strategy.")
        else:
            q_values = self.q_table.get(agent_state, np.zeros(len(ACTIONS)))
            chosen_action = ACTIONS[np.argmax(q_values)]
            logger.info("Exploiting with best action from Q-table.")
    else:
        q_values = self.q_table.get(agent_state, np.zeros(len(ACTIONS)))
        chosen_action = ACTIONS[np.argmax(q_values)]

    # **Track visited tiles**
    self.visited_tiles[position] = self.visited_tiles.get(position, 0) + 1

    # Track move history
    self.move_history.append(position)
    if len(self.move_history) > 20:  # Keep history of the last 20 steps
        self.move_history.pop(0)

    # **Escape after dropping a bomb**
    if self.bomb_dropped_recently:
        logger.info("Agent dropped a bomb recently, prioritizing escape.")
        if is_position_safe(position, game_state, self):
            logger.info("Escaped bomb blast radius, resuming normal behavior.")
            self.bomb_dropped_recently = False
        else:
            return move_towards_safest_direction(game_state, self)

    # **Danger handling**: Compute danger map and check if the agent is in immediate danger
    danger_map = compute_danger_map(game_state)
    if is_in_danger(game_state):
        return move_towards_safest_direction(game_state, self)

    # **Check if the current position is dangerous**
    if not is_position_safe(position, game_state, self):
        return move_towards_safest_direction(game_state, self)

    # **Bomb-dropping logic for opponents**
    if should_attack_opponent(game_state, self):
        logger.info("Considering dropping bomb near opponent.")
        if can_place_bomb_safely(position, game_state):
            self.bomb_dropped_recently = True
            logger.info("Dropping bomb near opponent.")
            return 'BOMB'

    # **Coin Collection**: Attempt to collect coins eagerly if it's safe
    if coins:
        nearest_coin, distance = find_nearest_coin(position, coins)
        logger.info(f"Nearest coin at {nearest_coin}, {distance} tiles away.")
        path = a_star_search(game_state, position, nearest_coin, danger_map)
        if path and len(path) > 1:
            next_step = path[1]
            if is_position_safe(next_step, game_state, self) and next_step not in self.move_history[-10:]:
                self.move_history.append(next_step)
                if next_step[0] < position[0]:
                    return 'LEFT'
                elif next_step[0] > position[0]:
                    return 'RIGHT'
                elif next_step[1] < position[1]:
                    return 'UP'
                elif next_step[1] > position[1]:
                    return 'DOWN'

    # **Bomb-dropping logic for crates**
    if should_drop_bomb(game_state, self):
        surrounding_tiles = get_neighbors(position, game_state['field'])
        visited_enough = all(self.visited_tiles.get(tile, 0) >= 1 for tile in surrounding_tiles)

        if visited_enough:
            logger.info("Dropping bomb and prioritizing escape.")
            self.bomb_dropped_recently = True
            return 'BOMB'
        else:
            logger.info("Surrounding tiles not visited enough, continuing exploration.")
            return explore_map(self, game_state)

    # **Fix for oscillation**: Detect and handle repeated movements
    if detect_oscillation(self.move_history, 4):
        logger.info("Oscillation detected: Avoiding recently visited positions.")

        # Prioritize moving to a less-visited tile not recently visited
        directions = ['UP', 'DOWN', 'LEFT', 'RIGHT']
        safe_directions = []
        direction_scores = []

        recent_positions = set(self.move_history[-10:])

        for direction in directions:
            new_position = simulate_move(position, direction)
            if is_position_safe(new_position, game_state, self):
                if new_position not in recent_positions and new_position not in self.destroyed_crates:
                    visits = self.visited_tiles.get(new_position, 0)
                    safe_directions.append(direction)
                    direction_scores.append(-visits)  # Less visited tiles have higher scores

        if safe_directions:
            best_direction = safe_directions[np.argmax(direction_scores)]
            return best_direction

        # If no unvisited safe positions, pick any safe direction not leading to recent positions or destroyed crates
        for direction in directions:
            new_position = simulate_move(position, direction)
            if is_position_safe(new_position, game_state, self) and new_position not in recent_positions and new_position not in self.destroyed_crates:
                return direction

        # As a last resort, pick any safe direction
        for direction in directions:
            new_position = simulate_move(position, direction)
            if is_position_safe(new_position, game_state, self):
                return direction

        # If no safe direction, wait
        return 'WAIT'

    # **Improve Exploration**: Explore the map, avoiding tiles where crates were destroyed
    logger.info("Exploring the map.")
    explore_action = explore_map(self, game_state)
    if explore_action != 'WAIT':
        return explore_action

    # **Fallback**: If no other actions, wait
    return 'WAIT'


