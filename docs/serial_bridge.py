#!/usr/bin/env python3
"""Bridge stdin/stdout <-> serial port. stdlib only (termios), no pyserial.
Usage: serial_bridge.py /dev/cu.usbserial-X [baud]
Stdin '\n' is sent as '\r' (serial consoles want CR)."""
import os
import select
import sys
import termios

port = sys.argv[1]
baud = getattr(termios, 'B' + (sys.argv[2] if len(sys.argv) > 2 else '115200'))

fd = os.open(port, os.O_RDWR | os.O_NOCTTY | os.O_NONBLOCK)
a = termios.tcgetattr(fd)
a[0] = 0                       # iflag: raw
a[1] = 0                       # oflag: raw
a[2] = termios.CS8 | termios.CREAD | termios.CLOCAL
a[3] = 0                       # lflag: raw
a[4] = baud
a[5] = baud
a[6][termios.VMIN] = 0
a[6][termios.VTIME] = 0
termios.tcsetattr(fd, termios.TCSANOW, a)

out = sys.stdout.buffer
stdin_fd = sys.stdin.fileno()
print(f"[bridge] open {port} @ {sys.argv[2] if len(sys.argv) > 2 else '115200'}", flush=True)

while True:
    r, _, _ = select.select([fd, stdin_fd], [], [], 1)
    if fd in r:
        try:
            data = os.read(fd, 4096)
        except BlockingIOError:
            data = b''
        if data:
            out.write(data)
            out.flush()
    if stdin_fd in r:
        line = os.read(stdin_fd, 4096)
        if not line:
            break
        os.write(fd, line.replace(b'\n', b'\r'))
