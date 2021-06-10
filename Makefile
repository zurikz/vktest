.PHONY: ipykernel
ipykernel:
	poetry run python -m ipykernel install --user --name vktest --display-name "vktest"