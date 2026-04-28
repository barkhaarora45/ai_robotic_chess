import serial
import time


# ==============================
# SERIAL CONFIG
# ==============================
PORT = "COM8"
BAUD = 115200  # Changed from 115200 for stability

ser = serial.Serial(PORT, BAUD, timeout=1)
time.sleep(2)

# ==============================
# MOTION TUNING (OPTIMIZED!)
# ==============================
STEP_DELAY = 0.08   # Increased from 0.03
ANGLE_STEP = 3      # Increased from 1

# ==============================
# GRIPPER
# ==============================
GRIP_OPEN = 0
GRIP_CLOSE = 60

# ==============================
# HOME POSITION (5 DOF)
# ==============================
HOME = [90, 180, 90, 90, GRIP_OPEN]

# ==============================
# BOARD POSITIONS (4 DOF ONLY)
# ==============================
board = {
    "a8": [58,  58, 114,90],
    "b8": [76,  76,  92,90],
    "c8": [80,  70,  94,90],
    "d8": [105, 84, 76,82],
    "e8": [120, 80, 87,89],
    "f8": [153, 53, 123,90],
    "g8": [155, 50, 130,89],
    "h8": [161,	35,	150,89],
    "h7": [148,	27,	169,90],
    "g7": [142,	33,	143,90],
    "f7": [134,	54,	127,90],
    "e7": [113,	61,	110,90],
    "d7": [98,	62,	108,90],
    "c7": [85,	60,	114,90],
    "b7": [70,	53,	125,90],
    "a7": [60,  47, 134,90],
    "a6": [64,  30, 166,90],
    "b6": [75,  38, 151,90],
    "c6": [87, 50, 134,90],
    "d6": [97, 49, 136,90],
    "e6": [109, 44, 143,90],
    "f6": [118, 38, 153,90],
    "g6": [130, 30, 166,90],
    "h6": [137, 20, 180,82],
    "h5": [132, 21, 180,72],
    "g5": [123, 24, 180,85],
    "f5": [115, 24, 180,91],
    "e5": [107, 36, 165,89],
    "d5": [98, 33, 165,89],
    "c5": [89, 27, 176,89],
    "b5": [81, 23, 180,87],
    "a5": [72, 23, 180,80],
    "a4": [78, 21, 180,63],
    "b4": [85, 22, 180,72],
    "c4": [92, 25, 180,78],
}

# ==============================
# PIECE SIZE MAP
# ==============================
PIECE_SIZE = {
    "k": "big",
    "q": "big",
    "r": "small",
    "p": "small",
    "b": "medium",
    "n": "medium",
}

# ==============================
# SIZE OFFSETS
# ==============================
SIZE_OFFSETS = {
    "big":    {},
    "small":  {1:-10},
    "medium": {}
}

# ==============================
# LOW LEVEL SEND
# ==============================
def send(pose):
    ser.write((",".join(map(str, pose)) + "\n").encode())
    time.sleep(STEP_DELAY)

# ==============================
# MOVE SINGLE JOINT SMOOTHLY
# ==============================
def move_joint(current, joint_index, target):
    start = current[joint_index]
    step = ANGLE_STEP if target > start else -ANGLE_STEP

    for angle in range(start, target, step):
        current[joint_index] = angle
        send(current)

    current[joint_index] = target
    send(current)
    return current

# ==============================
# APPLY PIECE OFFSETS (5 DOF)
# ==============================
def adjusted_pose(square, piece_code):
    pose = board[square].copy()
    pose.append(GRIP_OPEN)

    size = PIECE_SIZE.get(piece_code.lower(), "medium")
    for idx, delta in SIZE_OFFSETS[size].items():
        pose[idx] += delta

    return pose

# ==============================
# MOVE HOME → PICK
# ==============================
def move_home_to_pick(current, pick):
    current = move_joint(current, 0, pick[0])  # base
    current = move_joint(current, 2, pick[2])  # elbow
    current = move_joint(current, 3, pick[3])  # wrist
    current = move_joint(current, 1, pick[1])  # shoulder
    return current

# ==============================
# RETURN HOME
# ==============================
def go_home(current):
    current = move_joint(current, 1, HOME[1])
    current = move_joint(current, 0, HOME[0])
    current = move_joint(current, 2, HOME[2])
    current = move_joint(current, 3, HOME[3])
    current[4] = GRIP_OPEN
    send(current)
    return current

# ==============================
# PICK & PLACE
# ==============================
def move_piece(src, dst, piece_code):
    print(f"Move {src} → {dst} | piece = {piece_code}")

    current = HOME.copy()
    send(current)
    time.sleep(0.5)  # Increased from 0.3

    # ---------- PICK ----------
    pick_pose = adjusted_pose(src, piece_code)
    current = move_home_to_pick(current, pick_pose)

    # CLOSE GRIPPER (pick)
    current[4] = GRIP_CLOSE
    send(current)
    time.sleep(0.5)  # Increased from 0.3

    # LIFT
    current = move_joint(current, 1, 138)

    # ---------- PLACE ----------
    place_pose = adjusted_pose(dst, piece_code)
    current = move_joint(current, 0, place_pose[0])
    current = move_joint(current, 2, place_pose[2])
    current = move_joint(current, 3, place_pose[3])
    current = move_joint(current, 1, place_pose[1])

    # OPEN GRIPPER (drop)
    current[4] = GRIP_OPEN
    send(current)
    time.sleep(0.5)  # Increased from 0.3

    # ---------- HOME ----------
    go_home(current)
    print("Move complete\n")

# ==============================
# MAIN
# ==============================
if __name__ == "__main__":
    try:
        move_piece("b7", "b6", "p")
    except KeyboardInterrupt:
        pass
    finally:
        ser.close()