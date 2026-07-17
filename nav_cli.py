#!/usr/bin/env python3
import os, sys, math, time, signal, threading, warnings
import numpy as np

# Suppress all warnings before any imports
warnings.filterwarnings("ignore")
os.environ["PYTHONWARNINGS"] = "ignore"

# Redirect stderr to kill FPS warnings from stretch_mujoco subprocess
import subprocess
sys.stderr = open(os.devnull, 'w')

from stretch_mujoco import StretchMujocoSimulator
from anchor_utils import load_anchors_from_xml

sys.stderr = sys.__stderr__  # restore stderr for our own errors

# ── Tuning ────────────────────────────────────────────────────────────────────
POS_TOL    = 0.15
ALIGN_THR  = math.radians(15)
MAX_ANG    = math.radians(45)
TURN_TOL   = math.radians(5)
MAX_LIN    = 0.25
MAX_ANG_V  = 0.8
KP_ANG     = 1.5

# ── Nav controller ────────────────────────────────────────────────────────────
class Nav:
    def __init__(self):
        self.tpos  = None
        self.tdir  = None
        self.active= False
        self.done  = False
        self.turn  = False
        self.ptol  = POS_TOL
        self.ttol  = TURN_TOL

    @staticmethod
    def wrap(a): return math.atan2(math.sin(a), math.cos(a))

    def cancel(self):
        self.tpos=self.tdir=None
        self.active=self.done=self.turn=False

    def goto(self, x, y, tol=0.15):
        self.tpos=np.array([x,y]); self.tdir=None
        self.ptol=tol; self.turn=False; self.active=True; self.done=False

    def turn_anchor(self, x, y, deg=5.0):
        self.tpos=np.array([x,y]); self.tdir=None
        self.ttol=math.radians(deg); self.turn=True; self.active=True; self.done=False

    def turn_abs(self, degrees, deg=5.0):
        self.tpos=None; self.tdir=math.radians(degrees%360)
        self.ttol=math.radians(deg); self.turn=True; self.active=True; self.done=False

    def control(self, x, y, yaw):
        if not self.active: return 0.0, 0.0
        p = np.array([x,y])

        if self.turn:
            des = self.tdir if self.tdir is not None else math.atan2(*(self.tpos-p)[::-1])
            err = self.wrap(des - yaw)
            if abs(err) <= self.ttol:
                self.done=True; self.active=False; return 0.0, 0.0
            return 0.0, float(np.clip(KP_ANG*err, -MAX_ANG_V, MAX_ANG_V))

        diff = self.tpos - p
        dist = np.linalg.norm(diff)
        if dist <= self.ptol:
            self.done=True; self.active=False; return 0.0, 0.0

        des = math.atan2(diff[1], diff[0])
        err = self.wrap(des - yaw)
        ang = float(np.clip(KP_ANG*err, -MAX_ANG_V, MAX_ANG_V))
        if   abs(err) > MAX_ANG:   lin = 0.0
        elif abs(err) <= ALIGN_THR: lin = MAX_LIN
        else:
            t = (abs(err)-ALIGN_THR)/(MAX_ANG-ALIGN_THR)
            lin = MAX_LIN*(1.0-t)
        return float(lin), ang

# ── Main ──────────────────────────────────────────────────────────────────────
class StretchNav:
    def __init__(self, xml):
        self.xml = xml
        # suppress warnings during sim creation
        sys.stderr = open(os.devnull, 'w')
        self.sim = StretchMujocoSimulator(scene_xml_path=xml)
        sys.stderr = sys.__stderr__

        self.nav = Nav()
        try:
            raw = load_anchors_from_xml(xml)
            self.anchors = {k: np.array(v['pos']) for k,v in raw.items()}
        except:
            self.anchors = {}

        self._x = 0.533; self._y = 2.317; self._yaw = 0.0; self._lt = None
        self.running = True

    def _pose(self):
        s = self.sim.pull_status()
        now = float(s.time)
        if self._lt is None or now <= self._lt:
            self._lt = now; return self._x, self._y, self._yaw
        dt = now - self._lt; self._lt = now
        xv = float(s.base.x_vel); tv = float(s.base.theta_vel)
        self._x   += xv * math.cos(self._yaw) * dt
        self._y   += xv * math.sin(self._yaw) * dt
        self._yaw  = math.atan2(math.sin(self._yaw + tv*dt), math.cos(self._yaw + tv*dt))
        return self._x, self._y, self._yaw

    def _nav_loop(self):
        i = 0
        while self.running:
            try:
                if not self.nav.active:
                    self.sim.set_base_velocity(0.0, 0.0)
                    time.sleep(0.05); continue
                x,y,yaw = self._pose()
                l,a = self.nav.control(x,y,yaw)
                self.sim.set_base_velocity(l, a)
                if not self.nav.active:
                    self.sim.set_base_velocity(0.0,0.0)
                    print("\n✓ Reached target\nnav> ", end='', flush=True)
                i+=1
                if i%20==0:
                    d = np.linalg.norm(self.nav.tpos-np.array([x,y])) if self.nav.tpos is not None else 0
                    print(f"\n  [{x:.2f},{y:.2f}] yaw={math.degrees(yaw):.0f}° dist={d:.2f}m\nnav> ", end='', flush=True)
            except: pass
            time.sleep(0.05)

    def start(self):
        # suppress all output during start
        old_out = sys.stdout
        sys.stdout = open(os.devnull, 'w')
        sys.stderr = open(os.devnull, 'w')
        try:
            self.sim.start(headless=False)
            time.sleep(3.0)
        finally:
            sys.stdout = old_out
            sys.stderr = sys.__stderr__

        t = threading.Thread(target=self._nav_loop, daemon=True)
        t.start()
        print(f"Ready. Anchors: {sorted(self.anchors.keys())}\n")

    def stop(self):
        self.running = False
        try:
            self.sim.set_base_velocity(0.0, 0.0)
            sys.stderr = open(os.devnull,'w')
            self.sim.stop()
            sys.stderr = sys.__stderr__
        except: pass

    def cli(self):
        print("Commands: anchors|pose|status|go A|pos X Y|turn A|turndeg D|vel L A|stop|stow|home|exit\n")
        while self.running:
            try:
                cmd = input("nav> ").strip()
            except (EOFError, KeyboardInterrupt):
                print("\nExiting..."); break
            if not cmd: continue
            p = cmd.split(); op = p[0].lower()
            try:
                if op=="exit": break
                elif op=="anchors":
                    for k,v in sorted(self.anchors.items()):
                        print(f"  {k}: ({v[0]:.2f},{v[1]:.2f})")
                elif op=="pose":
                    x,y,yaw=self._pose()
                    print(f"  pos=({x:.3f},{y:.3f}) yaw={math.degrees(yaw):.1f}°")
                elif op=="status":
                    s=self.sim.pull_status()
                    print(f"  base: x_vel={s.base.x_vel:.4f} theta_vel={s.base.theta_vel:.4f}")
                    print(f"  lift={s.lift.pos:.3f} arm={s.arm.pos:.3f}")
                elif op=="go":
                    a=p[1].upper(); tol=float(p[2]) if len(p)>2 else 0.15
                    if a not in self.anchors: print(f"Unknown: {a}"); continue
                    v=self.anchors[a]; self.nav.goto(float(v[0]),float(v[1]),tol)
                    print(f"→ {a} ({v[0]:.2f},{v[1]:.2f})")
                elif op=="pos":
                    self.nav.goto(float(p[1]),float(p[2]),float(p[3]) if len(p)>3 else 0.15)
                    print(f"→ ({p[1]},{p[2]})")
                elif op=="turn":
                    a=p[1].upper(); d=float(p[2]) if len(p)>2 else 5.0
                    if a not in self.anchors: print(f"Unknown: {a}"); continue
                    v=self.anchors[a]; self.nav.turn_anchor(float(v[0]),float(v[1]),d)
                    print(f"↻ toward {a}")
                elif op=="turndeg":
                    self.nav.turn_abs(float(p[1]),float(p[2]) if len(p)>2 else 5.0)
                    print(f"↻ to {p[1]}°")
                elif op=="vel":
                    self.nav.cancel()
                    self.sim.set_base_velocity(float(p[1]),float(p[2]))
                    print(f"vel lin={p[1]} ang={p[2]}")
                elif op=="stop":
                    self.nav.cancel(); self.sim.set_base_velocity(0.0,0.0); print("Stopped")
                elif op=="stow":
                    self.sim.stow(); print("Stowing...")
                elif op=="home":
                    self.sim.home(); print("Homing...")
                else:
                    print(f"Unknown: {op}")
            except Exception as e:
                print(f"Error: {e}")

def main():
    # Handle Ctrl+C cleanly
    node = None
    def handler(sig, frame):
        print("\nCtrl+C — stopping...")
        if node: node.stop()
        sys.exit(0)
    signal.signal(signal.SIGINT, handler)

    here = os.path.dirname(os.path.abspath(__file__))
    xml  = os.path.join(here, "table_world.xml")
    if not os.path.exists(xml):
        raise FileNotFoundError(f"Not found: {xml}")

    node = StretchNav(xml)
    node.start()
    try:
        node.cli()
    finally:
        node.stop()

if __name__ == "__main__":
    main()