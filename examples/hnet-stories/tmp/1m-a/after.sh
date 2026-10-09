#!/usr/bin/env bash
# usage: setsid nohup ./after.sh <pid-file> > /dev/null 2>&1 < /dev/null &
# Waits until the process of <pid-file> (a screen.pid, a queue.pid of another
# workspace) exits, then runs ./queue.sh here. So a queue starts right after
# a GPU job of another directory, and never beside it.
cd "$(dirname "$0")"
pid=$(cat "$1")
while kill -0 "$pid" 2>/dev/null; do sleep 20; done
exec bash ./queue.sh
