"""
collect_data.py
----------------
JOB: Collect 3 readings that describe your network's health right now.

1. wifi_signal   -> how strong your wifi signal is (in %)
2. router_ping   -> how long it takes to reach your router (in ms)
3. external_ping -> how long it takes to reach the internet, e.g. Google DNS (in ms)

We use ping for router/external because it works the same way on any laptop.
Wifi signal reading is OS-specific (Windows vs Linux/Mac use different commands),
so we keep it simple with a helper function you can adjust for your OS.

v2 change:
 - If a Windows laptop is NOT connected to any Wi-Fi, the signal is now
   reported as 0 (the truth), instead of a random number.
"""

import subprocess
import platform
import re
import random


def ping(host, timeout_ms=1000):
    """
    Ping a host once and return the latency in milliseconds.
    Returns None if the host doesn't reply (i.e. it's down).
    """
    system = platform.system().lower()

    if "windows" in system:
        cmd = ["ping", "-n", "1", "-w", str(timeout_ms), host]
    else:  # Linux / Mac
        cmd = ["ping", "-c", "1", "-W", str(timeout_ms // 1000 or 1), host]

    try:
        output = subprocess.run(cmd, capture_output=True, text=True, timeout=3).stdout
    except Exception:
        return None

    # Look for a number like "time=23.4 ms" or "time=23ms"
    match = re.search(r"time[=<]([\d.]+)", output)
    if match:
        return float(match.group(1))
    return None  # ping failed / host unreachable


def get_wifi_signal_strength():
    """
    Returns wifi signal strength as a percentage (0-100).
    - Windows: reads it with netsh. If the laptop is disconnected from Wi-Fi, returns 0.
    - Linux: reads it with nmcli.
    - Anything else (e.g. a desktop with no Wi-Fi hardware): falls back to a
      random-but-realistic value so the project can still run for testing.
    """
    system = platform.system().lower()

    try:
        if "windows" in system:
            output = subprocess.run(
                ["netsh", "wlan", "show", "interfaces"],
                capture_output=True, text=True, timeout=3
            ).stdout
            match = re.search(r"Signal\s*:\s*(\d+)%", output)
            if match:
                return int(match.group(1))
            # Wi-Fi adapter exists but is not connected to any network -> signal is 0
            if re.search(r"State\s*:\s*disconnected", output, re.IGNORECASE):
                return 0

        elif "linux" in system:
            output = subprocess.run(
                ["nmcli", "-f", "SIGNAL", "dev", "wifi"],
                capture_output=True, text=True, timeout=3
            ).stdout
            lines = [l.strip() for l in output.splitlines() if l.strip().isdigit()]
            if lines:
                return int(lines[0])
    except Exception:
        pass

    # Fallback so the project still runs during dev/testing without real wifi hardware
    return random.randint(40, 100)


def get_default_gateway():
    """
    Auto-detects the current network's gateway (router) IP address,
    so this keeps working correctly even if you switch between wifi,
    a phone hotspot, or a different network on demo day.
    """
    system = platform.system().lower()
    try:
        if "windows" in system:
            output = subprocess.run(["ipconfig"], capture_output=True, text=True, timeout=3).stdout
            # "Default Gateway" often shows an IPv6 address first, with the real
            # IPv4 gateway on the line right after it - so scan a couple of lines.
            lines = output.splitlines()
            for i, line in enumerate(lines):
                if "Default Gateway" in line:
                    for nearby_line in lines[i:i + 3]:
                        match = re.search(r"\b(\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3})\b", nearby_line)
                        if match:
                            return match.group(1)
                    break
        else:  # Linux / Mac
            output = subprocess.run(["ip", "route"], capture_output=True, text=True, timeout=3).stdout
            match = re.search(r"default via ([\d.]+)", output)
            if match:
                return match.group(1)
    except Exception:
        pass
    return None  # couldn't detect - collect_reading() will handle this gracefully


def collect_reading():
    """
    Collects one full reading: wifi signal + router ping + external ping.
    Returns a dictionary - easy to log and easy to feed into the ML model.
    """
    gateway = get_default_gateway()
    router_latency = ping(gateway) if gateway else None

    return {
        "wifi_signal": get_wifi_signal_strength(),
        "router_ping": router_latency or 999,   # 999 = treat "no reply / no gateway found" as very bad
        "external_ping": ping("8.8.8.8") or 999,
    }


if __name__ == "__main__":
    # Quick manual test: run this file directly to see one reading printed
    reading = collect_reading()
    print("Current network reading:", reading)