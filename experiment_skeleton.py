# experiment_skeleton.py
"""
Skeleton script for custom stimulus-presentation and neurofeedback paradigms.

Uses Pygame for rendering and PyLSL to push two-channel string markers
(event name, value) and to read class predictions from a feedback stream.
Runs a baseline phase, a fixed number of trials and an end-of-run marker.
"""

import sys
import time
import math
import numpy as np
import pygame
from pylsl import StreamInfo, StreamOutlet, StreamInlet, resolve_byprop, local_clock

# --- CONFIGURATION ---
MARKER_STREAM_NAME = "OpenSesame_Markers"
FEEDBACK_STREAM_NAME = "Neurofeedback_Out"
SCREEN_WIDTH = 1024
SCREEN_HEIGHT = 768
FULLSCREEN = False

BASELINE_DURATION = 40.0  # Seconds
TOTAL_TRIALS = 5
TRIAL_DURATION = 6.0      # Duration of stimulus presentation
ITI_DURATION = 3.0        # Inter-Trial Interval (Rest)

def main():
    """Run the baseline phase and all trials, emitting LSL markers throughout.

    Each trial shows a fixation cross, a pulsing task stimulus, requests and
    waits up to 2.5 s for a feedback prediction, displays it, then rests.
    Escape or closing the window quits immediately.
    """
    # 1. Initialize Pygame display and sound
    pygame.init()
    pygame.mixer.init()
    
    flags = pygame.FULLSCREEN if FULLSCREEN else 0
    screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT), flags)
    pygame.display.set_caption("fNIRS Neurofeedback Paradigm - Custom Skeleton")
    font = pygame.font.SysFont("Arial", 36)
    clock = pygame.time.Clock()

    # Colors
    BG_COLOR = (15, 15, 15)
    WHITE = (240, 240, 240)
    GREEN = (0, 230, 120)
    RED = (230, 60, 60)
    BLUE = (0, 140, 255)

    # 2. Setup LSL Marker Outlet
    print("🎭 Initializing LSL Marker Stream...")
    marker_info = StreamInfo(MARKER_STREAM_NAME, 'Markers', 2, 0, 'string', 'custom_paradigm_01')
    marker_outlet = StreamOutlet(marker_info)

    # 3. Feedback inlet is resolved lazily (0.2 s timeout) at each feedback request until found
    print("🎧 Looking for Neurofeedback Outlet on network...")
    feedback_inlet = None

    def check_events():
        """Quit Pygame and exit the process on window close or Escape."""
        for event in pygame.event.get():
            if event.type == pygame.QUIT or (event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE):
                pygame.quit()
                sys.exit()

    def draw_text(text, color=WHITE, y_offset=0):
        """Clear the screen and draw centred text, offset vertically by y_offset."""
        screen.fill(BG_COLOR)
        txt_surface = font.render(text, True, color)
        rect = txt_surface.get_rect(center=(SCREEN_WIDTH // 2, SCREEN_HEIGHT // 2 + y_offset))
        screen.blit(txt_surface, rect)
        pygame.display.flip()

    def draw_fixation_cross(color=WHITE):
        """Clear the screen and draw a centred fixation cross."""
        screen.fill(BG_COLOR)
        cx, cy = SCREEN_WIDTH // 2, SCREEN_HEIGHT // 2
        size = 30
        pygame.draw.line(screen, color, (cx - size, cy), (cx + size, cy), 5)
        pygame.draw.line(screen, color, (cx, cy - size), (cx, cy + size), 5)
        pygame.display.flip()

    # --- PHASE 1: BASELINE RECORDING ---
    print(f"⏳ Starting {BASELINE_DURATION}s Baseline Recording...")
    marker_outlet.push_sample(["baseline_start", "0"])
    
    start_time = time.time()
    while time.time() - start_time < BASELINE_DURATION:
        check_events()
        remaining = int(BASELINE_DURATION - (time.time() - start_time))
        draw_text(f"Baseline Phase: Relax ({remaining}s)", BLUE)
        clock.tick(60)

    marker_outlet.push_sample(["baseline_end", "0"])
    marker_outlet.push_sample(["run_onset", "0"])
    print("🟢 Baseline Complete. Beginning Trial Loops.")

    # --- PHASE 2: EXPERIMENTAL TRIALS ---
    for trial in range(1, TOTAL_TRIALS + 1):
        print(f"\n🎬 Trial {trial}/{TOTAL_TRIALS}")
        
        marker_outlet.push_sample(["trial_index", str(trial)])
        time.sleep(0.05)

        # 1. Fixation Cue
        draw_fixation_cross(WHITE)
        time.sleep(1.5)

        # 2. Stimulus Presentation (e.g., Task Cue)
        marker_outlet.push_sample(["trial_onset", "0"])
        stim_start = time.time()
        
        while time.time() - stim_start < TRIAL_DURATION:
            check_events()
            screen.fill(BG_COLOR)
            cx, cy = SCREEN_WIDTH // 2, SCREEN_HEIGHT // 2
            # Pulsing circle as a placeholder active-task stimulus
            radius = int(80 + 20 * math.sin(time.time() * 5))
            pygame.draw.circle(screen, GREEN, (cx, cy), radius)
            txt = font.render(f"Trial {trial}: Perform Mental Motor Imagery", True, WHITE)
            screen.blit(txt, txt.get_rect(center=(cx, cy - 150)))
            pygame.display.flip()
            clock.tick(60)

        # 3. Request Feedback & Listen for Class Prediction
        print(f"  ↳ Requesting Neurofeedback Prediction...")
        marker_outlet.push_sample(["request_feedback", "0"])
        
        # Resolve the feedback inlet if not already connected
        if feedback_inlet is None:
            streams = resolve_byprop("name", FEEDBACK_STREAM_NAME, timeout=0.2)
            if streams:
                feedback_inlet = StreamInlet(streams[0])

        draw_text("Decoding Brain State...", WHITE)
        
        pred_class, confidence = None, 0.0
        fb_timeout = time.time() + 2.5 # Wait up to 2.5s for prediction
        while time.time() < fb_timeout:
            check_events()
            if feedback_inlet:
                sample, _ = feedback_inlet.pull_sample(timeout=0.0)
                if sample:
                    pred_class = sample[1]
                    print(f"  ↳ Received Neurofeedback Prediction: Class {pred_class}")
                    break
            time.sleep(0.02)

        # Show the feedback result (or a timeout message) for 2 s
        fb_display_start = time.time()
        while time.time() - fb_display_start < 2.0:
            check_events()
            if pred_class is not None:
                color = GREEN if pred_class == "1" else BLUE
                draw_text(f"Feedback Output: Class {pred_class}", color)
            else:
                draw_text("Feedback Signal Processing Timeout", RED)
            clock.tick(60)

        # 4. End Trial & ITI Rest
        marker_outlet.push_sample(["trial_offset", "0"])
        rest_start = time.time()
        while time.time() - rest_start < ITI_DURATION:
            check_events()
            draw_text("Rest (ITI)", WHITE)
            clock.tick(60)

    # --- PHASE 3: RUN COMPLETE ---
    marker_outlet.push_sample(["run_offset", "0"])
    draw_text("Experiment Complete!", GREEN)
    time.sleep(3.0)
    pygame.quit()
    print("🏁 Experiment script execution cleanly ended.")

if __name__ == "__main__":
    main()