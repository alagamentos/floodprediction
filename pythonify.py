import os

root = os.getcwd() + '/develop/Notebooks'
directories = []

for r, d, f in os.walk(root):
  directories.extend(d)

  for file in f:
    if '.ipynb' in file:
      if not os.path.exists(r + '/py'):
        os.mkdir(r + '/py')

      string = f'jupyter nbconvert "{r}/{file}" --to="python" --output-dir={r}/py'
      os.system(string)
