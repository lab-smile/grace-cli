# GRACE CLI

GRACE CLI is a tool for processing NIfTI (.nii or .nii.gz) files using [GRACE model](https://github.com/lab-smile/GRACE), batch processing is also supported. You can check out the full details of how this tool works here: [GRACE-CLI Preview](https://youtu.be/0YU7Yd-mK2g). Check out the full playlist of these tools here: [GRACE Playlist](https://youtube.com/playlist?list=PLqPrlYT4iwKwsmmxTC7lvWbxwOp6Y5Gav&si=p2jHJoiH2_y8iyMU)

## Prerequisites

- Python 3.1x
- Ability to create virtual environments (`python3-venv`)
- Docker (Optional)

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
./run.sh <input_nifti_file.nii.gz> or <folder_path_to_nifti_images> [Other Options eg. --num_gpu 5 --spatial_size 256]
```

All available options are listed below:

| Argument                 | Type    | Default        | Description                                                                        |
| ------------------------ | ------- | -------------- | ---------------------------------------------------------------------------------- |
| `--input_path`           | *str*   | —              | Path to input NIfTI file or a folder (required as the first argument)              |
| `--output_dir`           | *str*   | `"outputs"`    | Directory to save outputs                                                          |
| `--model_path`           | *str*   | `"GRACE.pth"`  | Path to model weights file                                                         |
| `--spatial_size`         | *int*   | `64`           | One patch dimension                                                                |
| `--num_classes`          | *int*   | `12`           | Number of output classes                                                           |
| `--num_gpu`              | *int*   | `1`            | Number of GPUs to use                                                              |
| `--a_min_value`          | *float* | `0`            | Minimum intensity value for fixed normalization                                    |
| `--a_max_value`          | *float* | `255`          | Maximum intensity value for fixed normalization                                    |
| `--complexity_threshold` | *float* | `10000`        | Mean threshold to choose percentile normalization                                  |
| `--histogram_threshold`  | *float* | `400`          | Histogram spread threshold to choose percentile normalization                      |


For example:
```bash
./run.sh sample_image.nii.gz --complexity_threshold 12000
./run.sh ./input_folder --output_dir '/path/to/output' --num_gpu 4
```

### Using Docker

You can run the tool using Docker in three ways:

#### Using Docker directly:

1. Build the Docker image:
```bash
docker build -t grace-cli .
```

2. Run the container:
```bash
docker run -v $(pwd):/app grace-cli <input_nifti_file.nii.gz> [Additional Options]
```

For example:
```bash
docker run -v $(pwd):/app grace-cli sample_image.nii.gz [--num_gpu 2 --spatial_size 128 etc. (Optional)]
```

#### Using Docker compose:
To run the repo with the following command, you need to change the command argument in the `docker-compose.yml` file. (For example: ['python', 'grace.py', 'input.nii'])
```bash
docker compose up --build
```

#### Using our published docker hub image
You can use the published docker hub image `nikmk26/grace-cli:latest`

```bash
docker run -v $(pwd):/app nikmk26/grace-cli:latest <input_nifti_file.nii.gz>
```

### What the script does:

1. Creates a Python virtual environment
2. Installs all required dependencies
3. Processes the input NIfTI file(s)
4. Outputs the results in the `outputs` folder in the current directory.

### Output

The processed files will be saved in the `outputs` directory with the following naming convention:
- `<input_filename>_pred_GRACE.nii(.gz)`: NIfTI format output


## Dependencies

All required Python packages are listed in `requirements.txt` and will be automatically installed in the virtual environment when running the script.

## Notes

- The script automatically handles the creation of the Python virtual environment
- If you are running on HPCs like hipergator make sure you have >=python3.9 loaded, you can load it using `module load python/3.10` (example)
