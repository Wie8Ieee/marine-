import argparse
from pathlib import Path
p=argparse.ArgumentParser(); p.add_argument('--model'); p.add_argument('--control',type=Path); p.add_argument('--config'); p.add_argument('--repo'); p.add_argument('--commit'); p.add_argument('--mode'); p.add_argument('--environment'); a=p.parse_args()
a.control.mkdir(parents=True,exist_ok=True)
with (a.control/'order.txt').open('a',encoding='utf-8') as stream: stream.write(a.model+'\n')
raise SystemExit(17 if (a.control/f'fail_{a.model}').exists() else 0)
