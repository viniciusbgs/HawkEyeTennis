def get_center_of_bb(bb):
    x1, y1, x2, y2 = bb
    center_x = int((x1 + x2) / 2)
    center_y = int((y1 + y2) / 2)
    return center_x, center_y

def get_distance(p1, p2):
    x1, y1 = p1
    x2, y2 = p2
    return ((x2 - x1)**2 + (y2 - y1)**2)**0.5

def get_foot_position(bb):
    x1, y1, x2, y2 = bb
    return (int((x1 + x2) / 2), y2)