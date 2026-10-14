# Profile MAZ search or self-play (MuZero by default) and summarize the device trace.
prof-analyze game mode *args:
    uv run --frozen python scripts/profile_muzero.py {{game}} {{mode}} {{args}}

# Capture a trace without printing the text analysis.
prof game mode *args:
    uv run --frozen python scripts/profile_muzero.py {{game}} {{mode}} --no-analyze {{args}}

# Serve TensorBoard's Profile tab; viewer dependencies stay outside the project venv.
xp logdir="/tmp/maz-profiles" port="6007" *args:
    uv tool run --from tensorboard==2.20.0 --with xprof==2.23.2 --with setuptools==69.5.1 tensorboard --logdir {{quote(logdir)}} --port {{quote(port)}} --host localhost {{args}}
