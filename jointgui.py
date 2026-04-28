import tkinter as tk
import serial
import time

# =========================
# SERIAL CONFIG
# =========================
PORT = "COM8"        # change to your Arduino port
BAUD = 115200

ser = serial.Serial(PORT, BAUD, timeout=1)
time.sleep(2)  # wait for Arduino reset

# =========================
# SEND ANGLES FUNCTION
# =========================
def send_angles(_=None):
    angles = [
        base.get(),
        shoulder.get(),
        elbow.get(),
        wrist.get(),
        gripper.get()
    ]
    cmd = ",".join(map(str, angles)) + "\n"
    ser.write(cmd.encode())

# =========================
# GUI SETUP
# =========================
root = tk.Tk()
root.title("5-DOF Arm Joint Control (Arduino Nano)")
root.geometry("420x350")

# =========================
# SLIDERS
# =========================
base = tk.Scale(root, from_=0, to=180, orient="horizontal",
                label="Base", length=350, command=send_angles)
base.set(90)

shoulder = tk.Scale(root, from_=0, to=180, orient="horizontal",
                    label="Shoulder", length=350, command=send_angles)
shoulder.set(0)

elbow = tk.Scale(root, from_=0, to=180, orient="horizontal",
                 label="Elbow", length=350, command=send_angles)
elbow.set(90)

wrist = tk.Scale(root, from_=0, to=180, orient="horizontal",
                 label="Wrist", length=350, command=send_angles)
wrist.set(90)

gripper = tk.Scale(root, from_=0, to=180, orient="horizontal",
                   label="Gripper", length=350, command=send_angles)
gripper.set(30)

# =========================
# PACK WIDGETS
# =========================
for slider in [base, shoulder, elbow, wrist, gripper]:
    slider.pack(pady=5)

# =========================
# SAFE EXIT
# =========================
def on_close():
    ser.close()
    root.destroy()

root.protocol("WM_DELETE_WINDOW", on_close)

root.mainloop()