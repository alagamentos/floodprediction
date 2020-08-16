import os
from subprocess import PIPE, run

command = 'git diff --cached --name-only'

files = run(command.split(' '), stdout=PIPE) \
          .stdout.decode('unicode_escape') \
          .encode('latin1').decode('utf-8') \
          .replace('"', '').splitlines()

for file in files:
  # Check whether file was deleted or it's not a Jupyter Notebook
  if not os.path.exists(file) or '.ipynb' not in file:
    continue

  dirname = f'{os.path.dirname(file)}/py'

  if not os.path.exists(dirname):
    os.mkdir(dirname)

  strReturn = f'jupyter nbconvert "{file}" --to="python" --output-dir={dirname}'
  os.system(strReturn)
