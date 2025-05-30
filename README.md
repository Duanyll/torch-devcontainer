# Torch-Devcontainer

A modern project template for PyTorch CUDA projects, designed to streamline the setup and management of development environments. Benefits include:

- **Consistent Development Environment**: Use `uv` and Docker to ensure reproducible environments across different platforms (Linux, Windows, WSL).
- **Easy Container Management**: Use `container.sh` to build and run Docker containers with a consistent setup, faster than standard Dev Containers.
- **Visual Studio Code Integration**: Seamlessly integrate with Visual Studio Code for environment creation, intellisense, and debugging.
- **User-Permission Mounting and Caching**: Match user permissions with the host system, and cache Python packages to save bandwidth and speed up builds.

## Getting Started

1. Download or clone the repository.
2. Open the repository in Visual Studio Code.
3. Rename the `example` folder, `project.name`, `tool.scikit-build.wheel.packages`, `tool.scikit-build.cmake.define.MODULE_DIR` in `pyproject.toml` to your actual project name.
4. Set up the development environment as described below.
5. Run the example script to verify everything is working:
   ```bash
   uv run -m example
   ```
5. Start developing your PyTorch CUDA project!

## Enter Development Container

This template is designed to maintain a consistent development environment with or without Docker, with or without Visual Studio Code.

### With Docker and Visual Studio Code

Make sure you have Docker and `nvidia-container-toolkit` installed on your system.

1. Install the [Dev Containers extension](https://marketplace.visualstudio.com/items?itemName=ms-vscode-remote.remote-containers) in Visual Studio Code.
2. Run `bash container.sh setup` in the terminal to build the Docker image and create actual configuration files for dev container.
3. Open the command palette (Ctrl+Shift+P) and select `Dev Containers: Reopen in Container`.
4. Wait for the container to build and start. You should now be inside the development container.

### With Docker in Terminal

Make sure you have Docker and `nvidia-container-toolkit` installed on your system.

1. Run `bash container.sh setup` in the terminal to build the Docker image.
2. You can now run the container with the command:
   ```bash
   bash container.sh <command>
   ```
   Replace `<command>` with the command you want to run inside the container, such as `bash` to start a shell session. Or to run Python module:
   ```bash
   bash container.sh uv run -m example
   ```
   The container is created with `--rm -it` options, so it will be removed after you exit the container.

The container created by `container.sh` tries to keep same environment as the one created by Visual Studio Code Dev Containers extension.

### Without Docker

If you prefer not to use Docker, you can still work with the project in your local environment. Just ensure you have CUDA Toolkit (Runtime and Compiler, if you need custom operators) installed. However, switching the version of CUDA Toolkit may not be as convenient as using Docker.

You also need to install `uv` into your system. It is strongly discouraged to install `uv` via `pip` or `pipx`. Instead, install uv using the following command:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

This template is also tested to work on Windows natively. To set up the development environment, first install [Visual Studio](https://visualstudio.microsoft.com/) with the C++ development workload, then install CUDA Toolkit. Try to install latest version of Visual Studio and CUDA Toolkit to ensure compatibility. `uv` can be installed via Scoop installer:

```bash
scoop install uv
```

## Python Package Management

This template uses [uv](https://astral.sh/uv/) for Python package management, which is a modern alternative to `pip` and `pipx`. It provides a more consistent and reliable way to manage Python packages across different environments. `pip` is not present in the container, so you should use `uv` instead.

To restore the Python environment, run the following command:

```bash
uv sync --dev --no-install-project
```

To add new packages, use:

```bash
uv add <package_name>
```

See the `pyproject.toml` file to understand how `uv` treat PyTorch installation with specific CUDA version. 

Try to run every python command with `uv run` instead of calling `python` directly, to let `uv` ensures a reliable environment for you. For example, to run a Python script, use:

```bash
uv run script.py
```

## C++ / CUDA Extensions Development

Development of C++/CUDA extensions is supported in this template. It works with the Visual Studio [CMake Tools extension](https://marketplace.visualstudio.com/items?itemName=ms-vscode.cmake-tools) out of the box. You can condfigure and build C++/CUDA extensions with all default options in its GUI dashboard, or use the command line:

```bash
cmake -S . -B build -G "Ninja"
cmake --build build
```

It copies the built shared libraries to the `example` folder, so you can import them in Python code directly.

The project also allows you to manage builds with `uv` and `scikit-build`. To build the project, run:

```bash
uv build
```

Which will produce a installable wheel package in the `dist` folder. You can also use

```bash
uv pip install -e .
```

to build the extension and install Python package in editable mode. Note that running consequent `uv build` commands is much slower than running bare `cmake` commands in terminal or in VS Code, since `uv` does not utilize the CMake cache. 

### Debug C++ Extensions

Make sure you built the C++ extension with `-DCMAKE_BUILD_TYPE=Debug` option, which can be set in the CMake Tools extension GUI.

As demonstrated in the `example/__main__.py` file, you can debug python code and C++ extensions in Visual Studio Code simultaneously. To do this, first launch the Python debugger and break after importing the C++ extension. Then, launch another debugging session with "(gdb) Attach" profile to attach to the running Python process. You can then set breakpoints in your C++ code and debug it as you would normally do in Visual Studio Code.

### Jupyter Notebooks

You can use Visual Studio Code to work with Jupyter Notebooks in this project. Make sure you have the [Jupyter extension](https://marketplace.visualstudio.com/items?itemName=ms-toolsai.jupyter) installed. Optionally, you can setup the `nbstripout` git hook to avoid committing potentially large notebook outputs. To do this, run:

```bash
nbstripout --install
```