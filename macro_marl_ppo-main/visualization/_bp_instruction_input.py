"""Overcooked-style instruction input for the Box-Pushing visualizations.

Mirrors the UX of test_overcooked_*.py but adapted to the BP env's pyglet
window:

    1. Press 't' on the game window to enter input mode (game pauses, the
       agents do nothing while you type).
    2. Type the instruction. An overlay shows the buffer in real time.
       Backspace deletes the last character.
    3. Press ENTER to confirm. An empty buffer clears the active
       instruction; a non-empty buffer sets it for the agents.
    4. Press ESC to cancel without changing the active instruction.

The shared `BPInstructionInput` controller registers pyglet `on_key_press`
and `on_text` handlers on the viewer's window, and patches `window.flip`
so an overlay is drawn on top of the env render whenever input mode is
active. The implementation is best-effort: if pyglet is missing or the
viewer's window cannot be reached, `attach()` returns False and the
visualization can fall back to terminal input.
"""

import os
import sys

try:
    import pyglet
    from pyglet import gl
    from pyglet.window import key as _key
    _PYGLET_OK = True
except Exception:
    pyglet = None
    gl = None
    _key = None
    _PYGLET_OK = False


class BPInstructionInput:
    def __init__(self, instruction_list=None):
        self.current_instruction = None
        self._buffer = ""
        self._active = False
        self._changed = False
        self._instruction_list = list(instruction_list or [])

        self._window = None
        self._title_label = None
        self._buffer_label = None
        self._hint_labels = []
        self._orig_flip = None

    # ------------------------------------------------------------------
    # Wiring
    # ------------------------------------------------------------------
    def attach(self, env):
        """Hook the BP env's pyglet window. Returns True on success.

        Idempotent: re-attaching the same window across episodes is a
        no-op (without that, `self._orig_flip = window.flip` would
        capture the already-patched flip from a previous attach and the
        next flip call would recurse infinitely).
        """
        if not _PYGLET_OK:
            return False
        viewer = getattr(getattr(env, "unwrapped", env), "viewer", None)
        window = getattr(viewer, "window", None) if viewer is not None else None
        if window is None:
            return False

        # Already attached to this exact window — handlers are stacked
        # via push_handlers and flip is already patched, so just bail.
        if self._window is window:
            return True

        self._window = window
        window.push_handlers(
            on_key_press=self._on_key_press,
            on_text=self._on_text,
        )

        # Build labels lazily on first overlay draw — the window's GL
        # context must be current for pyglet to upload the font texture.
        # Patch window.flip so any frame the env paints triggers an
        # overlay redraw on top of the buffered geoms BEFORE the flip
        # actually swaps. This keeps the overlay glued to the rendered
        # frame instead of flickering with the back buffer.
        self._orig_flip = window.flip

        def patched_flip():
            try:
                if self._active:
                    self._draw_overlay()
            finally:
                self._orig_flip()

        window.flip = patched_flip
        return True

    # ------------------------------------------------------------------
    # State accessors
    # ------------------------------------------------------------------
    def is_active(self):
        """True while the user is typing the instruction."""
        return self._active

    def consume_change(self):
        c = self._changed
        self._changed = False
        return c

    # ------------------------------------------------------------------
    # Pyglet event handlers
    # ------------------------------------------------------------------
    def _on_key_press(self, symbol, modifiers):
        # Toggle into input mode on 't' (only when not already typing).
        if symbol == _key.T and not self._active:
            self._active = True
            self._buffer = ""
            print("\n" + "=" * 60)
            print("INSTRUCTION INPUT MODE — game paused")
            print("Type the instruction. ENTER to confirm, ESC to cancel.")
            print("Empty + ENTER clears the active instruction.")
            if self._instruction_list:
                print("Available phrasings:")
                for s in self._instruction_list:
                    print(f"  - {s}")
            print("=" * 60)
            return True

        if not self._active:
            return False

        if symbol in (_key.RETURN, _key.ENTER):
            new_inst = self._buffer.strip() or None
            self.current_instruction = new_inst
            self._active = False
            self._buffer = ""
            self._changed = True
            if new_inst is None:
                print(">> instruction CLEARED (none active)\n")
            else:
                print(f">> instruction set to: '{new_inst}'\n")
            return True

        if symbol == _key.ESCAPE:
            self._active = False
            self._buffer = ""
            print(">> instruction input cancelled\n")
            return True

        if symbol == _key.BACKSPACE:
            self._buffer = self._buffer[:-1]
            return True

        return False

    def _on_text(self, text):
        if not self._active:
            return False
        # Drop newline characters; ENTER is handled in on_key_press above.
        for ch in text:
            if ch in ("\r", "\n"):
                continue
            self._buffer += ch
        return True

    # ------------------------------------------------------------------
    # In-window overlay (pyglet immediate-mode GL + Label)
    # ------------------------------------------------------------------
    def _ensure_labels(self):
        if self._title_label is not None:
            return
        font = "Arial"
        self._title_label = pyglet.text.Label(
            text="INSTRUCTION INPUT MODE",
            font_name=font, font_size=18, bold=True,
            color=(255, 230, 90, 255),
            x=20, y=10,  # placeholder; positioned in _draw_overlay
            anchor_x="left", anchor_y="bottom",
        )
        self._buffer_label = pyglet.text.Label(
            text="",
            font_name=font, font_size=20,
            color=(120, 255, 120, 255),
            x=20, y=10,
            anchor_x="left", anchor_y="bottom",
        )
        hint_lines = [
            "ENTER to confirm  —  empty + ENTER clears  —  ESC to cancel",
        ]
        if self._instruction_list:
            hint_lines.append("examples: " + ", ".join(self._instruction_list[:4]))
        for text in hint_lines:
            self._hint_labels.append(pyglet.text.Label(
                text=text,
                font_name=font, font_size=12,
                color=(220, 220, 220, 255),
                x=20, y=10,
                anchor_x="left", anchor_y="bottom",
            ))

    def _draw_overlay(self):
        if self._window is None:
            return
        self._ensure_labels()

        w = self._window.width
        h = self._window.height
        panel_h = 140
        panel_y = 0  # bottom strip

        # Translucent dark rectangle behind the text. Pyglet's classic-
        # control viewer leaves us with a fairly bare GL state; just push
        # what we need and blend manually.
        gl.glEnable(gl.GL_BLEND)
        gl.glBlendFunc(gl.GL_SRC_ALPHA, gl.GL_ONE_MINUS_SRC_ALPHA)
        gl.glColor4f(0.05, 0.05, 0.05, 0.85)
        gl.glBegin(gl.GL_QUADS)
        gl.glVertex2f(0, panel_y)
        gl.glVertex2f(w, panel_y)
        gl.glVertex2f(w, panel_y + panel_h)
        gl.glVertex2f(0, panel_y + panel_h)
        gl.glEnd()
        gl.glDisable(gl.GL_BLEND)

        # Lay the labels inside the panel.
        self._title_label.x = 20
        self._title_label.y = panel_y + panel_h - 32
        self._title_label.draw()

        self._buffer_label.text = (self._buffer + "|")
        self._buffer_label.x = 20
        self._buffer_label.y = panel_y + panel_h - 78
        self._buffer_label.draw()

        # Hint lines
        for i, lbl in enumerate(self._hint_labels):
            lbl.x = 20
            lbl.y = panel_y + 10 + i * 18
            lbl.draw()


def is_pyglet_available():
    return _PYGLET_OK
