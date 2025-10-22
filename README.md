# GRACE CLI

GRACE CLI is a tool for processing NIfTI (.nii or .nii.gz) files using [GRACE model](https://github.com/lab-smile/GRACE), batch processing is also supported.

## Prerequisites

Choose one of the following options:

### Option 1: Local Installation
- Python 3.1x
- Ability to create virtual environments (`python3-venv`)

### Option 2: Docker Installation
- Docker
- Docker Compose (optional, for easier usage)

## Installation

1. Clone this repository:
```bash
git clone [repository-url]
cd grace-cli
```

2. Make sure the run script is executable:
```bash
chmod +x run.sh
```

3. Download `GRACE.pth` file from the following build to `grace-cli` directory.
```bash
https://github.com/lab-smile/GRACE/releases/tag/v1.0.1
```

## Usage

### Using Local Installation

The tool can be run using the provided shell script:

```bash
./run.sh <input_nifti_file.nii.gz> or <folder_path_to_nifti_images>
```

For example:
```bash
./run.sh sample_image.nii.gz
./run.sh ./input_folder
```

### Using Docker

You can run the tool using Docker in two ways:

#### Using Docker directly:

1. Build the Docker image:
```bash
docker build -t grace-cli .
```

2. Run the container:
```bash
docker run -v $(pwd):/app grace-cli <input_nifti_file.nii.gz>
```

For example:
```bash
docker run -v $(pwd):/app grace-cli sample_image.nii.gz
```

#### Using Docker compose:
To run the repo with the following command, you need to change the command argument in the `docker-compose.yml` file. (For example: ['python', 'grace.py', 'input.nii'])
```bash
docker compose up --build
```


### What the script does:

1. Creates a Python virtual environment
2. Installs all required dependencies
3. Processes the input NIfTI file(s)
4. Outputs the results in the `outputs` folder in the current directory.

### Output

The processed files will be saved in the `outputs` directory with the following naming convention:
- `<input_filename>_pred_GRACE.nii(.gz)`: NIfTI format output

## Error Cases

The script will show an error message if:
- No input file is provided
- The input file doesn't exist
- The input file is not a .nii.gz or .nii file
- Input is neither a file nor a folder with NIfTI images

## Dependencies

All required Python packages are listed in `requirements.txt` and will be automatically installed in the virtual environment when running the script.

## Notes

- The script automatically handles the creation and cleanup of the Python virtual environment
- Each run creates a fresh virtual environment to ensure consistency
- GPU support is available if CUDA is properly configured on your system
- Change `python` command in `run.sh` if command installed on your machine is `python3.x`
- If you are running on hipergator make sure you have >=python3.10 loaded, you can load it using `module load python/3.10`
