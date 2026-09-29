"""Create a read-only tmux monitor; the supervisor runs independently of tmux."""
import argparse
from pathlib import Path
import shlex
import subprocess
import sys

ROOT = Path(__file__).resolve().parent


def tmux(*arguments, check=True):
    return subprocess.run(['tmux', *map(str, arguments)], check=check, capture_output=True, text=True)


def command(*arguments):
    return shlex.join(map(str, arguments))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--session', default='ouro-workspace')
    args = parser.parse_args()
    session = args.session
    if tmux('has-session', '-t', f'={session}', check=False).returncode == 0:
        print(f'tmux attach -t {session}')
        return
    fit = ROOT / 'results/jlens-family-1000'
    evaluation = ROOT / 'results/workspace-full'
    window = tmux('new-session', '-d', '-P', '-F', '#{window_id}', '-s', session,
        '-n', 'overview', '-x', '132', '-y', '40', '-c', ROOT,
        command(sys.executable, '-u', ROOT/'ouro_dashboard.py')).stdout.strip()
    if tmux('display-message', '-p', '-t', window, '#{window_index}').stdout.strip() != '1':
        tmux('move-window', '-s', window, '-t', f'{session}:1')
    options = {
        'base-index':'1', 'renumber-windows':'on', 'mouse':'on', 'status':'on',
        'status-interval':'5', 'status-justify':'left',
        'status-style':'bg=#0b1220,fg=#94a3b8',
        'status-left':'#[bg=#22d3ee,fg=#0b1220,bold] OURO #[default] ',
        'status-left-length':'20', 'status-right-length':'75',
        'status-right':'#[fg=#22d3ee] GPUs 0–3 #[fg=#64748b]│#[fg=#cbd5e1] %H:%M %Z ',
        'window-status-format':' #I:#W ',
        'window-status-current-format':'#[bg=#1e293b,fg=#e2e8f0,bold] #I:#W #[default]',
        'message-style':'bg=#1e293b,fg=#e2e8f0',
    }
    for option, value in options.items():
        tmux('set-option', '-t', session, option, value)
    tmux('new-window', '-d', '-t', f'{session}:2', '-n', 'fit-logs', '-c', ROOT,
         command('tail', '-n', '8', '-F', fit/'supervisor.log'))
    tmux('split-window', '-d', '-v', '-l', '75%', '-t', f'{session}:2', '-c', ROOT,
         command('tail', '-n', '3', '-F', *[fit/'workers'/f'worker-{rank}'/'worker.log' for rank in range(8)]))
    tmux('new-window', '-d', '-t', f'{session}:3', '-n', 'eval-logs', '-c', ROOT,
         command('tail', '-n', '12', '-F', *[evaluation/f'worker-{rank}.log' for rank in range(4)]))
    for index in range(1, 4):
        tmux('set-option', '-w', '-t', f'{session}:{index}', 'pane-border-style', 'fg=#1e293b')
        tmux('set-option', '-w', '-t', f'{session}:{index}', 'pane-active-border-style', 'fg=#22d3ee')
        tmux('set-option', '-w', '-t', f'{session}:{index}', 'automatic-rename', 'off')
        tmux('set-option', '-w', '-t', f'{session}:{index}', 'remain-on-exit', 'on')
    tmux('select-window', '-t', f'{session}:1')
    print(f'tmux attach -t {session}')


if __name__ == '__main__':
    main()
