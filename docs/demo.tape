Output assets/demo.gif

Set FontSize 14
Set Width 1200
Set Height 620
Set Padding 16
Set Theme Dracula

# Build a 60-file sample repo so the demo shows real numbers past ChatGPT's 40-file cap.
# Use echo (auto-newline) — no backslashes or inner double-quotes inside the Type strings.
Type "rm -rf /tmp/ctxfeed-demo && mkdir -p /tmp/ctxfeed-demo/src"
Enter
Sleep 1s
Type "for i in $(seq 1 60); do echo def fn_$i: > /tmp/ctxfeed-demo/src/mod_$i.py; echo '    return '$i >> /tmp/ctxfeed-demo/src/mod_$i.py; done"
Enter
Sleep 2s
Type "echo '# ctxfeed-demo' > /tmp/ctxfeed-demo/README.md"
Enter
Type "echo '[project]' > /tmp/ctxfeed-demo/pyproject.toml"
Enter
Type "cd /tmp/ctxfeed-demo"
Enter
Sleep 1s

# --- Star-able moment #1: files accepted (60+ vs ChatGPT's 40) ---
Type "ctxfeed init --repo ."
Enter
Sleep 5s

# --- Star-able moment #2: per-query token cost vs Opus ---
Type "ctxfeed cost --repo ."
Enter
Sleep 5s
