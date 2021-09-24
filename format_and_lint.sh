ROOT_DIR="ai_economist/"

echo -e "\n\nRunning ISORT to sort imports ..."
isort --multi-line=3 --trailing-comma --force-grid-wrap=0 --use-parentheses --line-width=88 $ROOT_DIR

echo -e "\n\nRunning BLACK to format code ..."
black --line-length 88 $ROOT_DIR

echo -e "\n\nRunning FLAKE8 check to see if there are Python syntax errors ..."
flake8 --count --select=E9,F63,F7,F82 --show-source --statistics $ROOT_DIR

echo -e "\n\nRunning FLAKE8 formatting check ..."
flake8 --ignore=E203,C901,W503,F401 --count --max-complexity=15 --max-line-length=88 --statistics $ROOT_DIR

echo -e "\n\nRunning PYLINT check ..."
pylint --disable \
bad-continuation,\
duplicate-code,\
import-error,\
invalid-name,\
missing-module-docstring,\
missing-function-docstring,\
no-self-use,\
too-few-public-methods,\
too-many-arguments,\
too-many-branches,\
too-many-instance-attributes,\
too-many-lines,\
too-many-locals,\
too-many-nested-blocks,\
too-many-public-methods,\
too-many-statements, \
$ROOT_DIR

echo -e "\nPlease verify that FLAKE8 and PYLINT run successfully (above). If there are any errors, please fix them."

echo -e "\n\nRunning PYTEST ..."
pytest
