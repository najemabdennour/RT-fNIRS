
import sys, time, json, os, pygame
from pylsl import StreamInfo, StreamOutlet, StreamInlet, resolve_byprop

COLOR_MAP = {
    "Green": (0, 230, 120),
    "Red": (230, 60, 60),
    "Blue": (70, 130, 230),
    "Yellow": (230, 210, 60),
    "White": (240, 240, 240),
    "Orange": (240, 150, 40),
    "Purple": (170, 90, 220),
}

def run():
    config_path = sys.argv[1]
    with open(config_path, 'r') as f:
        cfg = json.load(f)

    pygame.init()
    pygame.mixer.init()

    w, h = cfg.get("screen_width", 1024), cfg.get("screen_height", 768)
    flags = pygame.FULLSCREEN if cfg.get("fullscreen", False) else 0
    screen = pygame.display.set_mode((w, h), flags)
    pygame.display.set_caption("fNIRS Stimulus Renderer")
    font = pygame.font.SysFont("Arial", cfg.get("font_size", 36))
    clock = pygame.time.Clock()

    # Customizable system markers (session/trial lifecycle events). Per-step
    # markers - trial_onset, trial_offset, request_feedback, etc. - are
    # driven entirely by each step's own 'marker' field, set in the table.
    markers_cfg = cfg.get("markers", {})
    M_BASELINE_START = markers_cfg.get("baseline_start", "baseline_start")
    M_BASELINE_END = markers_cfg.get("baseline_end", "baseline_end")
    M_RUN_ONSET = markers_cfg.get("run_onset", "run_onset")
    M_RUN_OFFSET = markers_cfg.get("run_offset", "run_offset")
    M_TRIAL_INDEX = markers_cfg.get("trial_index", "trial_index")
    M_TARGET_CLASS = markers_cfg.get("target_class", "target_class")
    trial_target_sequence = cfg.get("trial_target_sequence", [])

    # Feedback Window display constants. Style and label visibility are set
    # per-step (see each step's "feedback_style"/"show_feedback_label"
    # fields, chosen in the step table's Content column) - only the chance
    # level and the two Disk-style reference-circle radii are fixed here.
    CHANCE_LEVEL = 0.5
    CHANCE_LEVEL_RADIUS = 50
    MAX_RADIUS = 100
    HBAR_W, HBAR_H = 400, 40   # Progress Bar (Horizontal) dimensions
    VBAR_W, VBAR_H = 60, 300   # Progress Bar (Vertical) dimensions

    def circle_radius(decoding_prob, chance_level, chance_level_radius):
        # Transform decoding probability to circle size - matches chance_level_radius
        # exactly when decoding_prob == chance_level, and grows/shrinks from there.
        if chance_level <= 0:
            return 0
        return (decoding_prob * chance_level_radius) / chance_level

    marker_outlet = StreamOutlet(StreamInfo('OpenSesame_Markers', 'Markers', 2, 0, 'string', 'py_runner_01'))
    feedback_inlet = None

    font_cache = {}
    def get_font(size):
        if size not in font_cache:
            font_cache[size] = pygame.font.SysFont("Arial", max(int(size), 6))
        return font_cache[size]

    def check_exit():
        for e in pygame.event.get():
            if e.type == pygame.QUIT or (e.type == pygame.KEYDOWN and e.key == pygame.K_ESCAPE):
                pygame.quit(); sys.exit()

    def draw_shape(shape, color, size):
        cx, cy = w // 2, h // 2
        if shape == "Circle":
            pygame.draw.circle(screen, color, (cx, cy), size)
        elif shape == "Square":
            pygame.draw.rect(screen, color, pygame.Rect(cx - size, cy - size, size * 2, size * 2))
        elif shape == "Rectangle":
            pygame.draw.rect(screen, color, pygame.Rect(cx - size, cy - size // 2, size * 2, size))
        elif shape == "Triangle":
            pygame.draw.polygon(screen, color, [(cx, cy - size), (cx - size, cy + size), (cx + size, cy + size)])
        elif shape == "Diamond":
            pygame.draw.polygon(screen, color, [(cx, cy - size), (cx + size, cy), (cx, cy + size), (cx - size, cy)])
        else:
            pygame.draw.circle(screen, color, (cx, cy), size)

    def render_screen(step, prob_value=None):
        stype = step.get("type", "")
        screen.fill((15, 15, 15))
        cx, cy = w // 2, h // 2
        color = COLOR_MAP.get(step.get("color", "White"), (240, 240, 240))

        if stype == "Fixation Cross":
            size = step.get("size", 30)
            pygame.draw.line(screen, color, (cx - size, cy), (cx + size, cy), 5)
            pygame.draw.line(screen, color, (cx, cy - size), (cx, cy + size), 5)
        elif stype == "Text Prompt":
            step_font = get_font(step.get("size", cfg.get("font_size", 36)))
            txt = step_font.render(step.get("content", ""), True, color)
            screen.blit(txt, txt.get_rect(center=(cx, cy)))
        elif stype == "Shape":
            shape_color = COLOR_MAP.get(step.get("color", "Green"), (0, 230, 120))
            draw_shape(step.get("shape", "Circle"), shape_color, step.get("size", 80))
        elif stype == "Image":
            img_path = step.get("content", "")
            if img_path and os.path.exists(img_path):
                img = pygame.image.load(img_path)
                img = pygame.transform.scale(img, (w // 2, h // 2))
                screen.blit(img, img.get_rect(center=(cx, cy)))
            else:
                txt = font.render("[Image not found]", True, (230, 60, 60))
                screen.blit(txt, txt.get_rect(center=(cx, cy)))
        elif stype == "Feedback Window":
            feedback_style = step.get("feedback_style", "Text")
            show_label = step.get("show_feedback_label", True)

            # Vertical half-extent of whatever this style actually draws, so
            # the "Feedback" label sits clear above it regardless of style -
            # previously fixed to MAX_RADIUS (the Disk style's own radius),
            # which the taller Vertical Progress Bar grew past and covered.
            if feedback_style == "Progress Bar (Horizontal)":
                content_half_height = HBAR_H // 2
            elif feedback_style == "Progress Bar (Vertical)":
                content_half_height = VBAR_H // 2
            elif feedback_style == "Disk":
                content_half_height = MAX_RADIUS
            else:  # "Text"
                content_half_height = 30

            if show_label:
                step_font = get_font(step.get("size", cfg.get("font_size", 36)))
                label = step_font.render("Feedback", True, color)
                screen.blit(label, label.get_rect(center=(cx, cy - content_half_height - 40)))

            if prob_value is None:
                status_font = get_font(28)
                status = status_font.render("Waiting for decoder...", True, (200, 200, 200))
                screen.blit(status, status.get_rect(center=(cx, cy)))
            elif feedback_style == "Progress Bar (Horizontal)":
                show_pct = step.get("show_percentage", True)
                bar_w, bar_h = HBAR_W, HBAR_H
                bar_x, bar_y = cx - bar_w // 2, cy - bar_h // 2
                pygame.draw.rect(screen, (60, 60, 60), pygame.Rect(bar_x, bar_y, bar_w, bar_h))
                fill_w = int(bar_w * max(0.0, min(1.0, prob_value)))
                pygame.draw.rect(screen, (0, 230, 120), pygame.Rect(bar_x, bar_y, fill_w, bar_h))
                pygame.draw.rect(screen, (240, 240, 240), pygame.Rect(bar_x, bar_y, bar_w, bar_h), 3)
                chance_x = bar_x + int(bar_w * CHANCE_LEVEL)
                pygame.draw.line(screen, (230, 210, 60), (chance_x, bar_y - 8), (chance_x, bar_y + bar_h + 8), 3)
                if show_pct:
                    pct_font = get_font(24)
                    pct_txt = pct_font.render(f"{prob_value * 100:.1f}%", True, (240, 240, 240))
                    screen.blit(pct_txt, pct_txt.get_rect(center=(cx, bar_y + bar_h + 30)))
            elif feedback_style == "Progress Bar (Vertical)":
                show_pct = step.get("show_percentage", True)
                bar_w, bar_h = VBAR_W, VBAR_H
                bar_x, bar_y = cx - bar_w // 2, cy - bar_h // 2
                pygame.draw.rect(screen, (60, 60, 60), pygame.Rect(bar_x, bar_y, bar_w, bar_h))
                fill_h = int(bar_h * max(0.0, min(1.0, prob_value)))
                fill_y = bar_y + bar_h - fill_h  # fill grows upward from the bottom
                pygame.draw.rect(screen, (0, 230, 120), pygame.Rect(bar_x, fill_y, bar_w, fill_h))
                pygame.draw.rect(screen, (240, 240, 240), pygame.Rect(bar_x, bar_y, bar_w, bar_h), 3)
                chance_y = bar_y + bar_h - int(bar_h * CHANCE_LEVEL)
                pygame.draw.line(screen, (230, 210, 60), (bar_x - 8, chance_y), (bar_x + bar_w + 8, chance_y), 3)
                if show_pct:
                    pct_font = get_font(24)
                    pct_txt = pct_font.render(f"{prob_value * 100:.1f}%", True, (240, 240, 240))
                    screen.blit(pct_txt, pct_txt.get_rect(center=(cx, bar_y + bar_h + 30)))
            elif feedback_style == "Disk":
                show_pct = step.get("show_percentage", True)
                # Clamped so a very high-confidence trial can't draw a circle far off-screen.
                perf_radius = max(0, min(circle_radius(prob_value, CHANCE_LEVEL, CHANCE_LEVEL_RADIUS), MAX_RADIUS * 2))
                pygame.draw.circle(screen, (230, 60, 60), (cx, cy), int(perf_radius))            # Participant's performance
                pygame.draw.circle(screen, (230, 210, 60), (cx, cy), int(CHANCE_LEVEL_RADIUS), 2)  # Chance-level reference
                pygame.draw.circle(screen, (240, 240, 240), (cx, cy), int(MAX_RADIUS), 4)          # Maximum performance reference
                if show_pct:
                    pct_font = get_font(24)
                    pct_txt = pct_font.render(f"{prob_value * 100:.1f}%", True, (240, 240, 240))
                    screen.blit(pct_txt, pct_txt.get_rect(center=(cx, cy + MAX_RADIUS + 30)))
            else:  # "Text" - always shows the percentage; that's its whole purpose
                pct_font = get_font(48)
                pct_txt = pct_font.render(f"{prob_value * 100:.1f}%", True, (255, 215, 0))
                screen.blit(pct_txt, pct_txt.get_rect(center=(cx, cy)))

        pygame.display.flip()

    # --- Pre-Baseline Grace Countdown ---
    # Gives downstream LSL consumers (like the Neurofeedback Engine) time to
    # finish resolving and connecting to these streams before 'baseline_start'
    # fires. Without this, the very first marker sample can be pushed and
    # gone before a late-connecting inlet is listening, which silently
    # produces a 0-frame baseline.
    pre_grace = cfg.get("pre_baseline_delay", 10)
    t_grace = time.time()
    while time.time() - t_grace < pre_grace:
        check_exit()
        rem = int(pre_grace - (time.time() - t_grace)) + 1
        render_screen({"type": "Text Prompt", "content": f"Get Ready... Starting in {rem}s"})
        clock.tick(60)

    # Baseline Phase
    b_dur = cfg.get("baseline_duration", 40)
    marker_outlet.push_sample([M_BASELINE_START, "0"])
    t0 = time.time()
    while time.time() - t0 < b_dur:
        check_exit()
        rem = int(b_dur - (time.time() - t0))
        render_screen({"type": "Text Prompt", "content": f"Baseline Phase ({rem}s)"})
        clock.tick(60)

    marker_outlet.push_sample([M_BASELINE_END, "0"])
    marker_outlet.push_sample([M_RUN_ONSET, "0"])

    # Trial Loops
    for trial in range(1, cfg.get("total_trials", 5) + 1):
        marker_outlet.push_sample([M_TRIAL_INDEX, str(trial)])

        # One target_class marker per trial, sent before any of its steps
        # run, so the Neurofeedback Engine's current_target_class is set for
        # the whole trial (e.g. alternating/counterbalanced classes across a
        # run) before that trial's Feedback Window requests a prediction.
        if trial_target_sequence:
            trial_target = trial_target_sequence[(trial - 1) % len(trial_target_sequence)]
            marker_outlet.push_sample([M_TARGET_CLASS, str(int(trial_target))])

        for step in cfg.get("steps", []):
            marker = step.get("marker", "")
            if marker:
                marker_outlet.push_sample([marker, "0"])

            if step.get("type") == "Feedback Window":
                if feedback_inlet is None:
                    streams = resolve_byprop("name", "Neurofeedback_Out", timeout=0.2)
                    if streams: feedback_inlet = StreamInlet(streams[0])

                prob_value = None
                t_end = time.time() + step.get("duration", 2.0)
                while time.time() < t_end:
                    check_exit()
                    if feedback_inlet:
                        s, _ = feedback_inlet.pull_sample(timeout=0.0)
                        if s and s[0] == "probability":
                            prob_value = float(s[1])
                    render_screen(step, prob_value=prob_value)
                    clock.tick(60)
            else:
                t_end = time.time() + step.get("duration", 2.0)
                while time.time() < t_end:
                    check_exit()
                    render_screen(step)
                    clock.tick(60)

    marker_outlet.push_sample([M_RUN_OFFSET, "0"])
    pygame.quit()

if __name__ == "__main__":
    run()
