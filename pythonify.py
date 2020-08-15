import os

files = os.popen('git diff --cached --name-only').read().splitlines()

for file in files:
  if os.path.exists(file) and '.ipynb' not in file:
    continue

  dirname = f'{os.path.dirname(file)}/py'

  if not os.path.exists(dirname):
    os.mkdir(dirname)

  strReturn = f'jupyter nbconvert "{file}" --to="python" --output-dir={dirname}'
  os.system(strReturn)
