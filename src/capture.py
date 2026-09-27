import cv2
import time


def open_camera(source=0):
    """Open a video source.

    source: int -> webcam index. Uses the DirectShow backend, which opens
                   almost instantly on Windows; the default MSMF backend
                   often stalls for several seconds.
            str -> path to a video file (no backend hint needed).
    """
    if isinstance(source, int):
        cap = cv2.VideoCapture(source, cv2.CAP_DSHOW)
    else:
        cap = cv2.VideoCapture(source)

    if not cap.isOpened():
        cap.release()
        raise RuntimeError("Could not open camera source " + str(source) + chr(10) + _diagnose())
    return cap


def _diagnose():
    """Scan nearby indices and name the usual suspects.

    The two common causes on Windows: another app holds the device (Windows
    Camera, Teams, Zoom, OBS), or the index shifted after the webcam was
    replugged. Both show up here.
    """
    lines = ["camera probe:"]
    for idx in range(4):
        for name, backend in (("DSHOW", cv2.CAP_DSHOW), ("MSMF", cv2.CAP_MSMF)):
            c = cv2.VideoCapture(idx, backend)
            if c.isOpened():
                ok, frame = c.read()
                if ok and frame is not None:
                    state = f"open, {frame.shape[1]}x{frame.shape[0]}"
                else:
                    state = "opens but no frames -> another app is holding it"
            else:
                state = "closed"
            c.release()
            lines.append(f"  index {idx} {name:<5} {state}")
    lines.append("If your usual index opens but gives no frames, close Windows Camera / "
                 "Teams / Zoom / OBS and retry.")
    lines.append("If a different index works, confirm it is the SAME physical webcam "
                 "before recording - pass --source N.")
    return chr(10).join(lines)


def main(source=0):
    cap = open_camera(source)

    last_report = time.perf_counter()
    frames = 0
    printed_shape = False

    try:
        while True:
            # frame is a NumPy array, (height, width, 3), uint8, channels in BGR order.
            ok, frame = cap.read()
            if not ok:
                print("Frame read failed - camera disconnected or in use by another app.")
                break

            if not printed_shape:
                print(f"frame.shape = {frame.shape}    frame.dtype = {frame.dtype}")
                # What the driver claims, which is often nominal or plain wrong -
                # compare it against the measured FPS below.
                print(f"driver-reported FPS = {cap.get(cv2.CAP_PROP_FPS)}")
                printed_shape = True

            frames += 1
            now = time.perf_counter()
            elapsed = now - last_report
            if elapsed >= 1.0:
                print(f"FPS: {frames / elapsed:.1f}")
                frames = 0
                last_report = now

            cv2.imshow("capture - press q to quit", frame)

            # waitKey is what pumps the OS window event queue; without it the
            # window never paints. The & 0xFF masks off high bits some platforms
            # set, so the comparison against ord('q') behaves consistently.
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
        # Runs even on an exception, so the camera never stays locked.
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main(0)
