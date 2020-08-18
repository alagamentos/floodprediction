import os
import re
from subprocess import PIPE, run

# Get paths from all staged files
command = 'git diff --cached --name-only'
files = run(command.split(' '), stdout=PIPE).stdout.decode('unicode_escape') \
          .encode('latin1').decode('utf-8').replace('"', '').splitlines()

for file in files:
  # Check whether file was deleted or it's not a Jupyter Notebook
  if not os.path.exists(file) or '.ipynb' not in file:
    continue

  py_dirname = f'{os.path.dirname(file)}/py'
  py_filename = os.path.basename(file).replace('.ipynb', '.py')

  if not os.path.exists(py_dirname):
    os.mkdir(py_dirname)

  # Convert Jupyter Notebooks to Python files
  command = f'jupyter nbconvert "{file}" --to="python" --output-dir={py_dirname}'
  os.system(command)

  # Remove line enumeration to avoid undesired file changes
  newFileData = ''

  with open(f'{py_dirname}/{py_filename}', 'r+') as filedata:
    for line in filedata:
      if re.match('# In\[(\d|\s)+\]:', line):
        line = '# In[ ]:\n'

      newFileData += line

    filedata.seek(0)
    filedata.truncate(0)
    filedata.write(newFileData)
