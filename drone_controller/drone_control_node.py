#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, QoSReliabilityPolicy, QoSHistoryPolicy
from px4_msgs.msg import VehicleCommand, TrajectorySetpoint, VehicleStatus, VehicleOdometry
import time
import threading
import tkinter as tk
from tkinter import ttk, messagebox
import numpy as np
import logging

# Constants
VEHICLE_CMD_ARM_DISARM = 400
VEHICLE_CMD_DO_SET_MODE = 176
PX4_CUSTOM_MODE_OFFBOARD = 6
MAIN_STATE_OFFBOARD = 7

# QoS Profile
px4_qos = QoSProfile(
    reliability=QoSReliabilityPolicy.BEST_EFFORT,
    history=QoSHistoryPolicy.KEEP_LAST,
    depth=30
)

# Flight States
class FlightState:
    GROUNDED = "Grounded"
    HOVERING = "Hovering"
    MOVING = "Moving"
    LANDING = "Landing"
    RETURNING = "Returning"

class PIDController:
    def __init__(self, Kp, Ki, Kd, max_output):
        self.Kp = Kp
        self.Ki = Ki
        self.Kd = Kd
        self.max_output = max_output
        self.reset()

    def reset(self):
        self.integral = 0.0
        self.previous_error = 0.0

    def compute(self, setpoint, measured_value, dt):
        error = setpoint - measured_value
        self.integral += error * dt
        self.integral = np.clip(self.integral, -self.max_output, self.max_output)
        derivative = (error - self.previous_error) / dt if dt > 0 else 0.0
        self.previous_error = error
        output = self.Kp * error + self.Ki * self.integral + self.Kd * derivative
        return np.clip(output, -self.max_output, self.max_output)

class DroneController(Node):
    def __init__(self, root):
        super().__init__('drone_controller')
        self.root = root
        self.current_position = [0.0, 0.0, 0.0]
        self.current_velocity = [0.0, 0.0, 0.0]
        self.armed = False
        self.main_state = 0
        self.offboard_mode = False
        self.home_position = None
        self.target_position = None
        self.flight_state = FlightState.GROUNDED
        self.control_thread = None
        self.running = True
        self.preflight_ok = True  # Force True for SITL testing

        # PID Controllers
        self.pid_x = PIDController(0.8, 0.05, 0.15, max_output=2.0)
        self.pid_y = PIDController(0.8, 0.05, 0.15, max_output=2.0)
        self.pid_z = PIDController(1.5, 0.1, 0.25, max_output=3.0)

        # Publishers
        self.vehicle_command_pub = self.create_publisher(VehicleCommand, '/fmu/in/vehicle_command', px4_qos)
        self.setpoint_pub = self.create_publisher(TrajectorySetpoint, '/fmu/in/trajectory_setpoint', px4_qos)

        # Subscribers
        self.status_sub = self.create_subscription(VehicleStatus, '/fmu/out/vehicle_status', self.status_callback, px4_qos)
        self.odom_sub = self.create_subscription(VehicleOdometry, '/fmu/out/vehicle_odometry', self.odom_callback, px4_qos)

        # Logging
        logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')
        self.logger = logging.getLogger(__name__)

        # Timers
        self.create_timer(0.2, self.offboard_heartbeat)
        self.create_timer(0.1, self.send_initial_setpoint)  # Continuous setpoint for offboard

        # GUI Setup
        self.init_gui()

    def init_gui(self):
        self.root.title("Drone Controller")
        self.root.geometry("600x400")
        self.root.protocol("WM_DELETE_WINDOW", self.cleanup_and_quit)

        status_frame = ttk.LabelFrame(self.root, text="Drone Status", padding=10)
        status_frame.pack(fill="x", padx=10, pady=5)

        self.armed_label = ttk.Label(status_frame, text="Armed: False")
        self.armed_label.grid(row=0, column=0, sticky="w")
        self.preflight_label = ttk.Label(status_frame, text="Preflight: True")
        self.preflight_label.grid(row=1, column=0, sticky="w")
        self.mode_label = ttk.Label(status_frame, text="Mode: MANUAL")
        self.mode_label.grid(row=2, column=0, sticky="w")
        self.state_label = ttk.Label(status_frame, text="State: Grounded")
        self.state_label.grid(row=3, column=0, sticky="w")
        self.pos_label = ttk.Label(status_frame, text="Pos: X=0.00 Y=0.00 Z=0.00")
        self.pos_label.grid(row=4, column=0, sticky="w")

        control_frame = ttk.LabelFrame(self.root, text="Controls", padding=10)
        control_frame.pack(fill="x", padx=10, pady=5)

        ttk.Button(control_frame, text="Takeoff & Hover", command=self.prompt_takeoff).grid(row=0, column=0, padx=5, pady=5)
        ttk.Button(control_frame, text="Move to Position", command=self.prompt_move).grid(row=0, column=1, padx=5, pady=5)
        ttk.Button(control_frame, text="Return Home", command=self.return_home).grid(row=1, column=0, padx=5, pady=5)
        ttk.Button(control_frame, text="Land", command=self.land).grid(row=1, column=1, padx=5, pady=5)
        ttk.Button(control_frame, text="Quit", command=self.cleanup_and_quit).grid(row=2, column=0, columnspan=2, pady=10)

        self.update_gui()

    def update_gui(self):
        if self.running:
            self.armed_label.config(text=f"Armed: {self.armed}")
            self.preflight_label.config(text=f"Preflight: {self.preflight_ok}")
            self.mode_label.config(text=f"Mode: {'OFFBOARD' if self.offboard_mode else 'MANUAL'}")
            self.state_label.config(text=f"State: {self.flight_state}")
            self.pos_label.config(text=f"Pos: X={self.current_position[0]:.2f} Y={self.current_position[1]:.2f} Z={-self.current_position[2]:.2f}")
            self.root.after(100, self.update_gui)

    def init_home_position(self):
        start_time = time.time()
        while rclpy.ok() and self.home_position is None and (time.time() - start_time) < 10:
            self.logger.debug("Waiting for initial odometry...")
            time.sleep(0.1)
        if self.home_position is None:
            self.logger.warning("No odometry received, using default home [0, 0, 0]")
            self.home_position = [0.0, 0.0, 0.0]

    def prompt_takeoff(self):
        height = self.get_input("Enter hover height (m):")
        if height is not None:
            self.takeoff_and_hover(height)

    def prompt_move(self):
        if self.armed:
            x = self.get_input("Enter X position:")
            if x is not None:
                y = self.get_input("Enter Y position:")
                if y is not None:
                    z = self.get_input("Enter Z position (height):")
                    if z is not None:
                        self.move_to_position(x, y, z)
        else:
            messagebox.showerror("Error", "Drone not armed!")

    def get_input(self, prompt):
        dialog = tk.Toplevel(self.root)
        dialog.title("Input")
        dialog.geometry("300x100")
        dialog.grab_set()

        ttk.Label(dialog, text=prompt).pack(pady=5)
        entry = ttk.Entry(dialog)
        entry.pack(pady=5)
        result = [None]

        def on_ok():
            try:
                result[0] = float(entry.get())
                dialog.destroy()
            except ValueError:
                messagebox.showerror("Error", "Invalid input! Please enter a number.")

        ttk.Button(dialog, text="OK", command=on_ok).pack(pady=5)
        dialog.wait_window()
        return result[0]

    def takeoff_and_hover(self, height):
        if height <= 0:
            messagebox.showerror("Error", "Height must be > 0")
            return
        self.flight_state = FlightState.HOVERING
        self.target_position = self.current_position.copy()
        self.start_control_loop()
        time.sleep(1)
        if self.prepare_offboard() and self._arm_vehicle():
            self.target_position[2] = -height
            self.logger.info(f"Takeoff initiated to height {height}m")

    def prepare_offboard(self):
        self.logger.info("Preparing for offboard mode")
        for _ in range(20):  # Send setpoints for 2 seconds
            setpoint = TrajectorySetpoint()
            setpoint.position = self.current_position
            setpoint.velocity = [0.0, 0.0, 0.0]
            setpoint.timestamp = int(self.get_clock().now().nanoseconds / 1000)
            self.setpoint_pub.publish(setpoint)
            self.logger.debug("Initial setpoint sent")
            time.sleep(0.1)

        for _ in range(5):
            self.set_flight_mode("OFFBOARD")
            time.sleep(1)
            if self.offboard_mode:
                self.logger.info("Offboard mode activated")
                return True
        self.logger.error("Failed to enter Offboard mode")
        messagebox.showerror("Error", "Failed to enter Offboard mode")
        return False

    def _arm_vehicle(self):
        if not self.offboard_mode:
            self.logger.error("Cannot arm: Not in Offboard mode")
            messagebox.showerror("Error", "Not in Offboard mode!")
            return False
        for _ in range(10):
            self.send_command(VEHICLE_CMD_ARM_DISARM, param1=1.0)
            time.sleep(0.2)
            if self.armed:
                self.logger.info("Vehicle armed successfully")
                return True
        self.logger.error("Arming failed after retries")
        messagebox.showerror("Error", "Arming failed after retries!")
        return False

    def set_flight_mode(self, mode):
        cmd = VehicleCommand()
        cmd.command = VEHICLE_CMD_DO_SET_MODE
        cmd.param1 = 1.0
        cmd.param2 = float(PX4_CUSTOM_MODE_OFFBOARD if mode == "OFFBOARD" else 0)
        cmd.target_system = 1
        cmd.target_component = 1
        cmd.source_system = 1
        cmd.source_component = 1
        cmd.from_external = True
        cmd.timestamp = int(self.get_clock().now().nanoseconds / 1000)
        self.vehicle_command_pub.publish(cmd)
        self.logger.debug(f"Set flight mode to {mode}")

    def send_command(self, command, param1=0.0, param7=0.0):
        cmd = VehicleCommand()
        cmd.command = command
        cmd.param1 = param1
        cmd.param7 = param7
        cmd.target_system = 1
        cmd.target_component = 1
        cmd.source_system = 1
        cmd.source_component = 1
        cmd.from_external = True
        cmd.timestamp = int(self.get_clock().now().nanoseconds / 1000)
        self.vehicle_command_pub.publish(cmd)
        self.logger.debug(f"Sent command: {command}, param1={param1}")

    def move_to_position(self, x, y, z):
        self.flight_state = FlightState.MOVING
        self.pid_x.reset()
        self.pid_y.reset()
        self.pid_z.reset()
        self.target_position = [x, y, -z]
        if not self.control_thread:
            self.start_control_loop()

    def return_home(self):
        if self.home_position:
            self.flight_state = FlightState.RETURNING
            self.move_to_position(*self.home_position)
            messagebox.showinfo("Info", "Returning to home")
        else:
            messagebox.showerror("Error", "Home not set!")

    def land(self):
        self.flight_state = FlightState.LANDING
        if self.control_thread:
            self.control_thread = None
        self.send_command(21)
        messagebox.showinfo("Info", "Landing initiated")
        time.sleep(3)
        self.send_command(VEHICLE_CMD_ARM_DISARM, param1=0.0)

    def start_control_loop(self):
        if not self.control_thread:
            self.control_thread = threading.Thread(target=self._control_loop, daemon=True)
            self.control_thread.start()

    def _control_loop(self):
        dt = 0.2
        while self.running and self.control_thread:
            if self.target_position and self.flight_state in [FlightState.HOVERING, FlightState.MOVING, FlightState.RETURNING]:
                vx = self.pid_x.compute(self.target_position[0], self.current_position[0], dt)
                vy = self.pid_y.compute(self.target_position[1], self.current_position[1], dt)
                vz = self.pid_z.compute(self.target_position[2], self.current_position[2], dt)
                setpoint = TrajectorySetpoint()
                setpoint.position = self.target_position
                setpoint.velocity = [vx, vy, vz]
                setpoint.timestamp = int(self.get_clock().now().nanoseconds / 1000)
                self.setpoint_pub.publish(setpoint)
                self.logger.debug(f"Setpoint sent: pos={self.target_position}, vel=[{vx:.2f}, {vy:.2f}, {vz:.2f}]")
            time.sleep(dt)

    def send_initial_setpoint(self):
        setpoint = TrajectorySetpoint()
        setpoint.position = self.current_position
        setpoint.velocity = [0.0, 0.0, 0.0]
        setpoint.timestamp = int(self.get_clock().now().nanoseconds / 1000)
        self.setpoint_pub.publish(setpoint)
        self.logger.debug("Continuous initial setpoint sent")

    def status_callback(self, msg):
        old_armed = self.armed
        old_offboard = self.offboard_mode
        self.armed = (msg.arming_state == 2)
        self.main_state = msg.main_state
        self.offboard_mode = (self.main_state == MAIN_STATE_OFFBOARD)
        # preflight_ok forced True, no update here
        if old_armed != self.armed or old_offboard != self.offboard_mode:
            self.logger.info(f"Status updated: armed={self.armed}, offboard={self.offboard_mode}, main_state={self.main_state}")

    def odom_callback(self, msg):
        self.current_position = list(msg.position)
        self.current_velocity = list(msg.velocity)
        if self.home_position is None:
            self.home_position = self.current_position.copy()
            self.logger.info(f"Home position set: {self.home_position}")

    def offboard_heartbeat(self):
        cmd = VehicleCommand()
        cmd.command = VEHICLE_CMD_DO_SET_MODE
        cmd.param1 = 1.0
        cmd.param2 = float(PX4_CUSTOM_MODE_OFFBOARD)
        cmd.target_system = 1
        cmd.target_component = 1
        cmd.source_system = 1
        cmd.source_component = 1
        cmd.from_external = True
        cmd.timestamp = int(self.get_clock().now().nanoseconds / 1000)
        self.vehicle_command_pub.publish(cmd)
        self.logger.debug("Offboard heartbeat sent")

    def cleanup_and_quit(self):
        self.running = False
        self.control_thread = None
        self.send_command(21)
        time.sleep(1)
        self.send_command(VEHICLE_CMD_ARM_DISARM, param1=0.0)
        self.destroy_node()
        rclpy.shutdown()
        self.root.quit()

def main():
    rclpy.init()
    root = tk.Tk()
    controller = DroneController(root)
    controller.init_home_position()
    spin_thread = threading.Thread(target=rclpy.spin, args=(controller,), daemon=True)
    spin_thread.start()
    root.mainloop()
    controller.cleanup_and_quit()

if __name__ == '__main__':
    main()
