tmux set -g mouse on
tmux show -g | sed 's/^/set -g /' > ~/.tmux.conf
tmux source ~/.tmux.conf

pip install -e /u/home/code/dltools/
pre-commit install
