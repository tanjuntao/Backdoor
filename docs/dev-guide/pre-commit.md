# Use pre-commit to automatically format and check code linting before git commit

`pre-commit` is a tool to do some extra procedures before git commit. We can use it to format our codes and check code linting.

1. Install `pre-commit` in the project environment

   `pip install pre-commit`

2. The `pre-commit` config has been written in `.pre-commit-config.yaml` and does three things:

   - Format imports with `isort`
   - Format modified codes with `black`
   - Check code linting with `flake8`
   
   Nothing needs to be done in this step. The config file is already written.

3. Install `pre-commit` in `.git\hooks\pre-commit`

   `pre-commit install`

After the steps above, whenever you submit a commit, the codes first get formatted by `isort` and `black` and then checked by `flake8`.



**Notice:**

- If your codes are formatted by `isort` or `black`, the pre-commit stage will fail because the codes are modified.
- If the commit still fails, perhaps `flake8` gives some warnings. Please check console outputs.
-  **You need to commit again if any of the situations above happen.**