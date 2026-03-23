"""
╔══════════════════════════════════════════════════════════════╗
║         HEAD MOVEMENT DETECTION - WebSocket Server           ║
║                                                              ║
║  Flow:                                                       ║
║  1. Python starts a WebSocket server on port 8000            ║
║  2. Frontend connects to ws://localhost:8000/ws              ║
║  3. Frontend sends webcam frames (JPEG bytes) every 200ms    ║
║  4. Python detects head pose from each frame                 ║
║  5. Python sends result back to frontend (JSON)              ║
║  6. Python also saves movements to your Node.js backend      ║
╚══════════════════════════════════════════════════════════════╝
"""

import asyncio
import websockets
import cv2
import mediapipe as mp
import numpy as np
import json
import requests
import time
from datetime import datetime

# ╔══════════════════════════════════════════════════════════════╗
# ║                      YOUR CONFIG                             ║
# ║           ← ONLY CHANGE THINGS IN THIS BLOCK →              ║
# ╚══════════════════════════════════════════════════════════════╝

WS_HOST         = "localhost"       # WebSocket server host
WS_PORT         = 8000              # ← must match NEXT_PUBLIC_WS_URL in frontend
                                    #   frontend uses: ws://localhost:8000/ws

BACKEND_URL     = "http://localhost:5000"  # ← Your Node.js backend

# How sensitive the detection is
YAW_THRESHOLD   = 25    # degrees left/right
PITCH_THRESHOLD = 20    # degrees up/down
ROLL_THRESHOLD  = 20    # degrees tilt

# Calibration — how many frames to collect before starting detection
CALIBRATION_FRAMES = 30

# How many frames to wait before sending "prolonged" warning
PROLONGED_FRAMES = 15   # ~3 seconds at 200ms per frame

# ══════════════════════════════════════════════════════════════════════════════

mp_face_mesh = mp.solutions.face_mesh

# 3D face model reference points
FACE_3D_MODEL = np.array([
    [0.0,    0.0,    0.0   ],   # Nose tip
    [0.0,  -330.0,  -65.0  ],   # Chin
    [-225.0, 170.0, -135.0 ],   # Left eye corner
    [225.0,  170.0, -135.0 ],   # Right eye corner
    [-150.0,-150.0, -125.0 ],   # Left mouth corner
    [150.0, -150.0, -125.0 ],   # Right mouth corner
], dtype=np.float64)

LANDMARK_IDS = [1, 152, 33, 263, 61, 291]


# ── Head pose calculation ─────────────────────────────────────────────────────

def get_head_pose(landmarks, w, h):
    image_points = np.array([
        [landmarks[i].x * w, landmarks[i].y * h]
        for i in LANDMARK_IDS
    ], dtype=np.float64)

    focal         = w
    camera_matrix = np.array([
        [focal, 0,     w / 2],
        [0,     focal, h / 2],
        [0,     0,     1    ]
    ], dtype=np.float64)

    dist = np.zeros((4, 1))
    _, rvec, _ = cv2.solvePnP(FACE_3D_MODEL, image_points, camera_matrix, dist)
    rmat, _    = cv2.Rodrigues(rvec)
    angles, *_ = cv2.RQDecomp3x3(rmat)

    yaw   = angles[1] * 360
    pitch = angles[0] * 360
    roll  = angles[2] * 360
    return yaw, pitch, roll


def get_reason(yaw, pitch, roll, baseline_yaw, baseline_pitch, baseline_roll):
    """Compare current angles against calibrated baseline."""
    dy = yaw   - baseline_yaw
    dp = pitch - baseline_pitch
    dr = roll  - baseline_roll

    if abs(dy) > YAW_THRESHOLD:
        return "yaw_left" if dy < 0 else "yaw_right"
    if dp > PITCH_THRESHOLD:
        return "pitch_up"
    if dp < -PITCH_THRESHOLD:
        return "pitch_down"
    if abs(dr) > ROLL_THRESHOLD:
        return "roll_left" if dr < 0 else "roll_right"
    return None


# ── Send movement to Node.js backend ─────────────────────────────────────────

def save_movement_to_backend(token, session_id, direction, counts):
    """Fire and forget — save to MongoDB via your existing REST API."""
    if not token or not session_id:
        return
    try:
        requests.post(
            f"{BACKEND_URL}/api/proctor/head/movement",
            json={
                "session_id":          session_id,
                "triggered_direction": direction,
                "counts":              counts,
            },
            headers={"Authorization": f"Bearer {token}"},
            timeout=3,
        )
        print(f"  [SAVED] {direction} → MongoDB")
    except Exception as e:
        print(f"  [WARN] Could not save to backend: {e}")


# ── Per-client state ──────────────────────────────────────────────────────────

class ClientState:
    def __init__(self):
        self.calibrated        = False
        self.calib_samples     = []
        self.baseline_yaw      = 0.0
        self.baseline_pitch    = 0.0
        self.baseline_roll     = 0.0
        self.prev_reason       = None   # last direction sent to frontend
        self.away_frame_count  = 0
        self.no_face_count     = 0
        self.multi_face_count  = 0

        # ── Debounce logic ────────────────────────────────────────────────────
        # A movement is counted ONLY when:
        #   1. Head moves to a new direction  (straight → left)
        #   2. Head returns to straight       (left → straight)  ← RESET
        #   3. Head moves again               (straight → left)  ← COUNT AGAIN
        #
        # This means holding your head to the left = counts ONCE only.
        # You must return to centre before it counts again.

        self.current_direction   = None   # direction currently being held
        self.counted_directions  = set()  # directions already counted THIS turn
        self.returned_to_center  = True   # True = head is straight, ready to count

        # No-face / multi-face debounce
        self.no_face_active      = False  # True while face is missing
        self.multi_face_active   = False  # True while multiple faces visible

        # MongoDB sync
        self.token      = None
        self.session_id = None
        self.counts     = {
            "left": 0, "right": 0, "up": 0, "down": 0,
            "tilt_left": 0, "tilt_right": 0,
            "no_face": 0, "multiple_faces": 0,
            "total": 0,
        }

    def should_count(self, reason):
        """
        Returns True only when this is a NEW movement after returning to centre.
        Prevents counting the same direction many times while holding position.
        """
        if reason is None:
            # Head returned to centre — reset so next movement can be counted
            self.returned_to_center  = True
            self.counted_directions  = set()
            self.current_direction   = None
            return False

        if not self.returned_to_center:
            # Still in a movement, head hasn't come back to centre yet
            return False

        if reason == self.current_direction:
            # Same direction still being held — don't count again
            return False

        # New direction after being at centre → COUNT IT
        self.current_direction  = reason
        self.returned_to_center = False   # must return to centre before counting again
        return True

    def update_count(self, direction):
        """Map reason string → counts key and increment."""
        key_map = {
            "yaw_left":   "left",
            "yaw_right":  "right",
            "pitch_up":   "up",
            "pitch_down": "down",
            "roll_left":  "tilt_left",
            "roll_right": "tilt_right",
            "no_face":    "no_face",
            "multi_face": "multiple_faces",
        }
        key = key_map.get(direction)
        if key:
            self.counts[key] += 1
            self.counts["total"] += 1
        return key


# ── Process one JPEG frame ────────────────────────────────────────────────────

def process_frame(frame_bytes, state: ClientState, face_mesh):
    """Decode frame, run MediaPipe, return result dict."""
    np_arr = np.frombuffer(frame_bytes, dtype=np.uint8)
    frame  = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

    if frame is None:
        return None

    h, w   = frame.shape[:2]
    rgb    = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = face_mesh.process(rgb)

    # ── No face ───────────────────────────────────────────────────────────────
    if not result.multi_face_landmarks:
        state.no_face_count += 1
        return {
            "type":           "pose",
            "no_face":        True,
            "multiple_faces": False,
            "looking_away":   True,
            "reason":         None,
            "calibrated":     state.calibrated,
            "yaw": 0, "pitch": 0, "roll": 0,
        }

    # ── Multiple faces ────────────────────────────────────────────────────────
    if len(result.multi_face_landmarks) > 1:
        state.multi_face_count += 1
        return {
            "type":           "pose",
            "no_face":        False,
            "multiple_faces": True,
            "looking_away":   True,
            "reason":         None,
            "calibrated":     state.calibrated,
            "yaw": 0, "pitch": 0, "roll": 0,
        }

    # ── One face — calculate pose ─────────────────────────────────────────────
    landmarks        = result.multi_face_landmarks[0].landmark
    yaw, pitch, roll = get_head_pose(landmarks, w, h)

    # Calibration phase
    if not state.calibrated:
        state.calib_samples.append((yaw, pitch, roll))
        if len(state.calib_samples) >= CALIBRATION_FRAMES:
            yaws   = [s[0] for s in state.calib_samples]
            pitches= [s[1] for s in state.calib_samples]
            rolls  = [s[2] for s in state.calib_samples]
            state.baseline_yaw   = float(np.median(yaws))
            state.baseline_pitch = float(np.median(pitches))
            state.baseline_roll  = float(np.median(rolls))
            state.calibrated     = True
            print(f"  [CALIBRATED] baseline yaw={state.baseline_yaw:.1f}  pitch={state.baseline_pitch:.1f}  roll={state.baseline_roll:.1f}")

        return {
            "type":           "pose",
            "no_face":        False,
            "multiple_faces": False,
            "looking_away":   False,
            "reason":         None,
            "calibrated":     False,
            "yaw":   round(yaw, 2),
            "pitch": round(pitch, 2),
            "roll":  round(roll, 2),
        }

    # Detection phase
    reason       = get_reason(yaw, pitch, roll, state.baseline_yaw, state.baseline_pitch, state.baseline_roll)
    looking_away = reason is not None

    # Track prolonged looking away
    if looking_away:
        state.away_frame_count += 1
    else:
        state.away_frame_count = 0

    return {
        "type":           "pose",
        "no_face":        False,
        "multiple_faces": False,
        "looking_away":   looking_away,
        "reason":         reason,
        "calibrated":     True,
        "yaw":            round(yaw, 2),
        "pitch":          round(pitch, 2),
        "roll":           round(roll, 2),
        "away_frames":    state.away_frame_count,
    }


# ── WebSocket handler (one per connected client) ──────────────────────────────

async def handle_client(websocket):
    client_ip = websocket.remote_address[0]
    print(f"\n[WS] Client connected: {client_ip}")

    state = ClientState()

    with mp_face_mesh.FaceMesh(
        max_num_faces            = 2,    # detect up to 2 to catch multi-face
        refine_landmarks         = True,
        min_detection_confidence = 0.6,
        min_tracking_confidence  = 0.6,
    ) as face_mesh:

        try:
            async for message in websocket:

                # ── JSON message (auth/session info from frontend) ────────────
                if isinstance(message, str):
                    try:
                        data = json.loads(message)

                        # Frontend sends token + session_id when exam starts
                        if data.get("type") == "auth":
                            state.token      = data.get("token")
                            state.session_id = data.get("session_id")
                            print(f"  [AUTH] token received, session: {state.session_id}")

                        # Frontend asks to recalibrate
                        elif data.get("type") == "recalibrate":
                            state.calibrated    = False
                            state.calib_samples = []
                            print("  [RECALIBRATE] Starting fresh calibration")
                            await websocket.send(json.dumps({"type": "recalibrating"}))

                    except json.JSONDecodeError:
                        pass
                    continue

                # ── Binary message = JPEG frame from frontend ─────────────────
                result = process_frame(message, state, face_mesh)
                if result is None:
                    continue

                reason = result.get("reason")

                # ── Head movement debounce logic ───────────────────────────────
                # should_count() returns True ONLY when:
                #   - Head moved to a NEW direction
                #   - AND head was at centre before this move
                # So holding your head left = counts ONCE only
                # Must return to centre before it counts again

                if state.calibrated and state.should_count(reason):
                    state.update_count(reason)
                    print(f"  [COUNT] {reason}  →  counts={state.counts}")

                    # Save to MongoDB (non-blocking)
                    if state.token and state.session_id:
                        asyncio.get_event_loop().run_in_executor(
                            None,
                            save_movement_to_backend,
                            state.token,
                            state.session_id,
                            reason,
                            dict(state.counts),
                        )

                # ── No face debounce — count only on NEW no-face event ────────
                no_face_now = result.get("no_face", False)
                if no_face_now and not state.no_face_active:
                    # Face just disappeared
                    state.no_face_active = True
                    state.update_count("no_face")
                    print(f"  [COUNT] no_face  →  counts={state.counts}")
                    if state.token and state.session_id:
                        asyncio.get_event_loop().run_in_executor(
                            None, save_movement_to_backend,
                            state.token, state.session_id, "no_face", dict(state.counts),
                        )
                elif not no_face_now:
                    state.no_face_active = False   # face came back — reset

                # ── Multiple faces debounce ───────────────────────────────────
                multi_now = result.get("multiple_faces", False)
                if multi_now and not state.multi_face_active:
                    state.multi_face_active = True
                    state.update_count("multi_face")
                    print(f"  [COUNT] multiple_faces  →  counts={state.counts}")
                    if state.token and state.session_id:
                        asyncio.get_event_loop().run_in_executor(
                            None, save_movement_to_backend,
                            state.token, state.session_id, "multiple_faces", dict(state.counts),
                        )
                elif not multi_now:
                    state.multi_face_active = False  # back to one face — reset

                # ── Prolonged warning → send event to frontend ────────────────
                if state.away_frame_count == PROLONGED_FRAMES:
                    prolonged_msg = {
                        "type":     "event",
                        "event":    reason or "looking_away",
                        "severity": "warning",
                        "duration": round(state.away_frame_count * 0.2, 1),
                    }
                    await websocket.send(json.dumps(prolonged_msg))

                state.prev_reason = reason

                # Send pose result back to frontend
                await websocket.send(json.dumps(result))

        except websockets.exceptions.ConnectionClosedOK:
            print(f"[WS] Client disconnected cleanly: {client_ip}")
        except websockets.exceptions.ConnectionClosedError as e:
            print(f"[WS] Client disconnected with error: {e}")
        except Exception as e:
            print(f"[WS] Unexpected error: {e}")

    print(f"[WS] Handler ended for {client_ip}")
    print(f"     Final counts: {state.counts}")


# ── Start the server ──────────────────────────────────────────────────────────

async def main():
    print("\n" + "═"*56)
    print("   HEAD MOVEMENT DETECTOR - WebSocket Server")
    print("═"*56)
    print(f"   Listening on : ws://{WS_HOST}:{WS_PORT}/ws")
    print(f"   Backend      : {BACKEND_URL}")
    print(f"   Yaw limit    : ±{YAW_THRESHOLD}°")
    print(f"   Pitch limit  : ±{PITCH_THRESHOLD}°")
    print(f"   Roll limit   : ±{ROLL_THRESHOLD}°")
    print(f"   Calibration  : {CALIBRATION_FRAMES} frames")
    print("═"*56)
    print("\n✅ Waiting for frontend to connect...\n")

    async with websockets.serve(
        handle_client,
        WS_HOST,
        WS_PORT,
        max_size      = 10 * 1024 * 1024,   # 10MB max frame size
        ping_interval = 20,
        ping_timeout  = 60,
    ):
        await asyncio.Future()   # run forever


if __name__ == "__main__":
    asyncio.run(main())
