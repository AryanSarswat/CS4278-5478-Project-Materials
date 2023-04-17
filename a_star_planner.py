from queue import PriorityQueue

direction = ["forward", "left", "right"] #0, 1, 2

def plan(src, dest, map, heading):
    '''
    map is a map with 4 channels
    generate a sequence of coordianates as well as entering direction
    <go here>, <using this intention>
    convention: (y, x)
    heading: direction in the actual map (entering direction) bottom, right, top, left
    returns path as ((y, x), intention, (previous subpath))
    '''
    # probably can compute rough heading pointing to duckie
    dy = dest[0] - src[0] 
    dx = dest[1] - src[1]
    if abs(dy) >= abs(dx):
        heading = 2 if dy > 0 else 0
    else:
        heading = 3 if dx > 0 else 1
    print(f"estimated heading: {heading}")
    def isValidNode(node): # keep if true
        coordinate, _, _ = node
        return 0 <= coordinate[0] < len(map) and 0 <= coordinate[1] < len(map[0]) and map[coordinate[0]][coordinate[1]]
    pq = PriorityQueue()
    pq.put((heuristics(src, dest, 0), ((src, heading, 0),None)))
    print("Generating plan")
    while not pq.empty():
        h, path = pq.get()
        recentNode = path[0]
        y, x = recentNode[0]
        allNodes = []
        if recentNode[1] == 0: # enter from bottom
            allNodes = [((y-1, x), 0, 0), ((y, x-1), 1, 1), ((y, x+1), 3, 2)]
        elif recentNode[1] == 1: # enter from right
            allNodes = [((y, x-1), 1, 0), ((y+1, x), 2, 1), ((y-1, x), 0, 2)]
        elif recentNode[1] == 2: # enter from top
            allNodes = [((y+1, x), 2, 0), ((y, x+1), 3, 1), ((y, x-1), 1, 2)]
        elif recentNode[1] == 3: # enter from left
            allNodes = [((y, x+1), 3, 0), ((y-1, x), 0, 1), ((y+1, x), 2, 2)]
        else:
            print("Error")
        allNodes = list(filter(isValidNode, allNodes))
        for node in allNodes:
            coordinate, enterDir, intention = node
            if (coordinate, enterDir, intention) in path:
                continue # seen state before, no need to re-add to queue
            if coordinate == dest:
                return tuple([(coordinate, enterDir, intention)] + list(path))
            node_h = heuristics(coordinate, dest, intention)
            pq.put((node_h, tuple([(coordinate, enterDir, intention),] + list(path))  ) )
    print("Generating done")
    return None

def heuristics(current, dest, intention):
    return abs(current[0]-dest[0]) + abs(current[1]-dest[1]) + 1 if intention > 0 else 0

def parse(tiles):
    '''
    takes in tile configuration and returns a bool map
    '''
    map_data = [[False for _ in range(len(tiles[0]))] for _ in range(len(tiles))]
    for y, row in enumerate(tiles):
        for x, col in enumerate(row):
            map_data[y][x] = True if ("3way" in col or "straight" in col or "curve_" in col or "4way" in col) else False
    return map_data

def view_parsed_map(map_data, src, dest):
    '''
    takes in a parsed map to pretty view it
    '''
    total = ""
    for row in map_data:
        row_total = ""
        for col in row:
            row_total += "o " if col else "- "
        row_total += "\n"
        total += row_total
    print(total)
    return total

def generate_paths(path, filename):
    print("Saving at: " + filename)
    with open(filename, "w") as f:
        for i, node in enumerate(path[::-1][1:]):
            coordinate_yx, _, intention = node
            y, x = coordinate_yx
            intention = direction[intention]
            text = f"({x}, {y}), {intention}\n"
            f.write(text)


if __name__ == "__main__":
    '''
    xxxxox
    xxxxox
    xxxoox
    xxxoxx

    3,3 -> 2,3 -> 2,4 -> 1,4 -> 0,4
    '''
    map_1 = [[False, False, False, False, True,  False],
             [False, False, False, False, True,  False],
             [False, False, False, True,  True,  False], 
             [False, False, False, True,  False, False]]
    
    print(plan((3,3), (0, 4), map_1, 0)) #(y, x)

    '''
    xxoxox
    xxoxox
    xoooox
    xoxoxx

    3,3 -> 2,3 -> 2,4 -> 1,4 -> 0,4
    '''
    map_2 = [[False, False, True , False, True,  False],
             [False, False, True , False, True,  False],
             [False, True , True , True,  True,  False], 
             [False, True , False, True,  False, False]]    
    print(plan((3,3), (0, 4), map_2, 0)) #(y, x)

    '''
    xxoxox
    xxoxox
    xooxox
    xoxoxx

    None
    '''
    map_3 = [[False, False, True , False, True,  False],
             [False, False, True , False, True,  False],
             [False, True , True , False,  True,  False], 
             [False, True , False, True,  False, False]]    
    print(plan((3,3), (0, 4), map_3, 0)) #(y, x)