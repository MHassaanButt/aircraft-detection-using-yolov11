import argparse
import os
from pathlib import Path
import torch
from ultralytics import YOLO
import pandas as pd
import matplotlib.pyplot as plt
import shutil
from datetime import datetime
import numpy as np
import PIL
from PIL import Image
import ast
from tqdm import tqdm
import yaml
import subprocess
import sys
import pkg_resources

def ensure_latest_ultralytics():
    """
    Ensures the latest version of ultralytics is installed.
    Returns the installed version.
    """
    try:
        # Attempt to upgrade ultralytics to the latest version
        subprocess.check_call([sys.executable, "-m", "pip", "install", "--upgrade", "ultralytics"])
        
        # Get the installed version
        installed_version = pkg_resources.get_distribution('ultralytics').version
        print(f"Using ultralytics version: {installed_version}")
        
        return installed_version
    except Exception as e:
        print(f"Error updating ultralytics: {e}")
        print("Attempting to proceed with currently installed version...")
        try:
            return pkg_resources.get_distribution('ultralytics').version
        except:
            print("Ultralytics not found. Installing...")
            subprocess.check_call([sys.executable, "-m", "pip", "install", "ultralytics"])
            return pkg_resources.get_distribution('ultralytics').version
def parse_args():
    parser = argparse.ArgumentParser(description='Preprocess data and train YOLOv11 on custom dataset')
    
    # Data preprocessing parameters
    parser.add_argument('--data-dir', type=str, required=True,
                        help='Path to the input data directory')
    parser.add_argument('--output-dir', type=str, default='processed_data',
                        help='Path to save processed data')
    parser.add_argument('--tile-size', type=int, default=512,
                        help='Size of image tiles')
    parser.add_argument('--tile-overlap', type=int, default=64,
                        help='Overlap between tiles')
    parser.add_argument('--truncated-percent', type=float, default=0.3,
                        help='Minimum visible part of object to be included')
    parser.add_argument('--val-fold', type=int, default=1,
                        help='Which fold to use for validation (1-5)')
    
    # Training parameters
     # Modified weights argument to be more flexible
    parser.add_argument('--weights', type=str, default='auto',
                        help='Path to weights file or "auto" for latest YOLO model')
    parser.add_argument('--epochs', type=int, default=50,
                        help='Number of training epochs')
    parser.add_argument('--batch-size', type=int, default=16,
                        help='Batch size')
    parser.add_argument('--device', type=str, default='',
                        help='Device to run on (e.g., cuda:0 or cpu)')
    parser.add_argument('--name', type=str, default='',
                        help='Custom name for the experiment')
    
    return parser.parse_args()

def setup_directories(base_dir):
    dirs = {
        'train_images': base_dir / 'train' / 'images',
        'train_labels': base_dir / 'train' / 'labels',
        'val_images': base_dir / 'val' / 'images',
        'val_labels': base_dir / 'val' / 'labels'
    }
    for dir_path in dirs.values():
        dir_path.mkdir(parents=True, exist_ok=True)
    return dirs

def process_annotations(data_dir):
    def parse_geometry(x):
        return ast.literal_eval(x.rstrip('\r\n'))
    
    df = pd.read_csv(data_dir / "annotations.csv", converters={'geometry': parse_geometry})
    
    def get_bounds(geometry):
        try:
            arr = np.array(geometry).T
            xmin, ymin = np.min(arr, axis=1)
            xmax, ymax = np.max(arr, axis=1)
            return (xmin, ymin, xmax, ymax)
        except:
            return np.nan
    
    df['bounds'] = df['geometry'].apply(get_bounds)
    df['width'] = df['bounds'].apply(lambda x: abs(x[2] - x[0]) if isinstance(x, tuple) else np.nan)
    df['height'] = df['bounds'].apply(lambda x: abs(x[3] - x[1]) if isinstance(x, tuple) else np.nan)
    
    return df

def tag_is_inside_tile(bounds, x_start, y_start, width, height, truncated_percent):
    x_min, y_min, x_max, y_max = bounds
    x_min, y_min, x_max, y_max = x_min - x_start, y_min - y_start, x_max - x_start, y_max - y_start
    
    if (x_min > width) or (x_max < 0.0) or (y_min > height) or (y_max < 0.0):
        return None
    
    x_max_trunc = min(x_max, width)
    x_min_trunc = max(x_min, 0)
    if (x_max_trunc - x_min_trunc) / (x_max - x_min) < truncated_percent:
        return None
    
    y_max_trunc = min(y_max, height)
    y_min_trunc = max(y_min, 0)
    if (y_max_trunc - y_min_trunc) / (y_max - y_min) < truncated_percent:
        return None
    
    x_center = (x_min_trunc + x_max_trunc) / 2.0 / width
    y_center = (y_min_trunc + y_max_trunc) / 2.0 / height
    x_extend = (x_max_trunc - x_min_trunc) / width
    y_extend = (y_max_trunc - y_min_trunc) / height
    
    return (0, x_center, y_center, x_extend, y_extend)

def process_data(args, dirs):
    data_dir = Path(args.data_dir)
    img_list = list(data_dir.glob('images/*.jpg'))
    print(f"Found {len(img_list)} images in {data_dir}")
    
    # Process annotations
    df = process_annotations(data_dir)
    
    # Determine validation set
    unique_images = df['image_id'].unique()
    num_fold = 5
    val_indexes = unique_images[len(unique_images) * args.val_fold // num_fold:
                               len(unique_images) * (args.val_fold + 1) // num_fold]
    
    # Process images
    for img_path in tqdm(img_list, desc="Processing images"):
        img = Image.open(img_path)
        img_width, img_height = img.size
        
        # Get annotations for this image
        img_labels = df[df["image_id"] == img_path.name]
        
        # Calculate tiles
        x_tiles = (img_width + args.tile_size - 1) // args.tile_size
        y_tiles = (img_height + args.tile_size - 1) // args.tile_size
        
        for x in range(x_tiles):
            for y in range(y_tiles):
                x_start = x * args.tile_size
                y_start = y * args.tile_size
                
                # Determine if this is a validation or training image
                is_val = img_path.name in val_indexes
                image_dir = dirs['val_images'] if is_val else dirs['train_images']
                label_dir = dirs['val_labels'] if is_val else dirs['train_labels']
                
                # Create tile filename
                tile_name = f"{img_path.stem}_{x_start}_{y_start}"
                
                # Extract and save tile
                tile = img.crop((x_start, y_start, 
                                min(x_start + args.tile_size, img_width),
                                min(y_start + args.tile_size, img_height)))
                
                if tile.size != (args.tile_size, args.tile_size):
                    new_tile = Image.new('RGB', (args.tile_size, args.tile_size))
                    new_tile.paste(tile, (0, 0))
                    tile = new_tile
                
                tile.save(image_dir / f"{tile_name}.jpg")
                
                # Process annotations for this tile
                found_tags = [
                    tag_is_inside_tile(bounds, x_start, y_start, args.tile_size, args.tile_size, args.truncated_percent)
                    for bounds in img_labels['bounds']
                ]
                found_tags = [tag for tag in found_tags if tag is not None]
                
                # Save labels
                with open(label_dir / f"{tile_name}.txt", 'w') as f:
                    for tag in found_tags:
                        f.write(' '.join(map(str, tag)) + '\n')

def create_yaml_config(output_dir):
    """
    Create YAML configuration file with absolute paths.
    """
    output_dir = Path(output_dir).resolve()  # Get absolute path
    
    config = {
        'train': str(output_dir / 'train'),  # Absolute path to train directory
        'val': str(output_dir / 'val'),      # Absolute path to val directory
        'nc': 1,
        'names': ['Aircraft']
    }
    
    # Verify directories exist
    for path in [config['train'], config['val']]:
        if not Path(path).exists():
            raise ValueError(f"Directory not found: {path}")
    
    yaml_path = output_dir / 'data.yaml'
    with open(yaml_path, 'w') as f:
        yaml.dump(config, f)
    
    print(f"Created YAML config at {yaml_path}")
    print("Config contents:")
    for key, value in config.items():
        print(f"{key}: {value}")
    
    return yaml_path

def generate_experiment_name(args, ultralytics_version):
    if args.weights.lower() == 'auto':
        model_name = get_latest_yolo_model().split('.')[0]  # Remove .pt extension
    else:
        model_name = Path(args.weights).stem
    
    custom_name = f"_{args.name}" if args.name else ""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    ultralytics_ver = ultralytics_version.replace('.', '-')  # Make version string path-friendly
    
    return f"{model_name}_ultr{ultralytics_ver}_e{args.epochs}_i{args.tile_size}{custom_name}_{timestamp}"


def get_latest_yolo_model():
    """
    Determines the latest available YOLO model from ultralytics.
    Returns the model name as a string.
    """
    from ultralytics import YOLO
    
    # You might want to adjust this based on your needs
    # This example uses YOLOv8n, but you could also check for v11 or other versions
    try:
        # Try to get YOLOv11 if available
        model = YOLO('yolo11n.pt')
        return 'yolo11n.pt'
    except:
        try:
            # Fallback to YOLOv8 if v11 is not available
            model = YOLO('yolov8n.pt')
            return 'yolov8n.pt'
        except Exception as e:
            print(f"Error loading YOLO model: {e}")
            return None

def train_model(args, yaml_path, exp_dir):
    # Ensure latest ultralytics version
    ultralytics_version = ensure_latest_ultralytics()
    
    # Determine weights to use
    if args.weights.lower() == 'auto':
        model_weights = get_latest_yolo_model()
        if model_weights is None:
            raise ValueError("Could not determine latest YOLO model")
        print(f"Using latest YOLO model: {model_weights}")
    else:
        model_weights = args.weights
    
    # Initialize model
    from ultralytics import YOLO
    model = YOLO(model_weights)
    
    # Ensure yaml_path is absolute
    yaml_path = Path(yaml_path).resolve()
    train_args = {
        'data': str(yaml_path),
        'epochs': args.epochs,
        'imgsz': args.tile_size,
        'batch': args.batch_size,
        'project': str(exp_dir.resolve()),  # Ensure absolute path
        'name': 'train',
        'exist_ok': True
    }
    
    if args.device:
        train_args['device'] = args.device
    
    # Print training configuration
    print("\nTraining configuration:")
    for key, value in train_args.items():
        print(f"{key}: {value}")
    
    try:
        results = model.train(**train_args)
        return model, results, ultralytics_version
    except Exception as e:
        print(f"Training error: {str(e)}")
        # Check if directories exist
        yaml_config = yaml.safe_load(open(yaml_path))
        for key in ['train', 'val']:
            path = Path(yaml_config[key])
            if not path.exists():
                print(f"Directory not found: {path}")
        raise

def plot_metrics(results_csv, exp_dir):
    df = pd.read_csv(results_csv)
    metrics = {
        'mAP50': 'metrics/mAP50(B)',
        'mAP50-95': 'metrics/mAP50-95(B)',
        'Precision': 'metrics/precision(B)',
        'Recall': 'metrics/recall(B)',
    }
    
    for name, column in metrics.items():
        plt.figure(figsize=(10, 6), dpi=500)
        plt.plot(df['epoch'], df[column])
        plt.title(f'{name} over Epochs')
        plt.xlabel('Epoch')
        plt.ylabel(name)
        plt.grid(True)
        plt.savefig(exp_dir / f'{name}_curve.png', dpi=500)
        plt.close()

def main(args):
    # Setup directories
    output_dir = Path(args.output_dir).resolve()  # Ensure absolute path
    print(f"Processing data to: {output_dir}")
    
    dirs = setup_directories(output_dir)
    
    # Process data
    process_data(args, dirs)
    
    # Verify directory contents
    for name, dir_path in dirs.items():
        files = list(dir_path.glob('*'))
        print(f"{name} contains {len(files)} files")
    
    # Create YAML config
    try:
        yaml_path = create_yaml_config(output_dir)
    except ValueError as e:
        print(f"Error creating YAML config: {e}")
        return
    
    # Setup experiment directory
    exp_dir = Path("experiments").resolve()
    exp_dir.mkdir(parents=True, exist_ok=True)
    
    # Train model
    try:
        model, results, ultralytics_version = train_model(args, yaml_path, exp_dir)
        
        # Setup experiment directory with version info
        exp_name = generate_experiment_name(args, ultralytics_version)
        exp_dir = Path("experiments") / exp_name
        exp_dir.mkdir(parents=True, exist_ok=True)
        
        # Move results from temp directory to final directory
        if (Path("temp_results") / "train").exists():
            shutil.move(str(Path("temp_results") / "train"), str(exp_dir))
        
        # Save configuration with version info
        with open(exp_dir / "config.txt", "w") as f:
            f.write(f"ultralytics_version: {ultralytics_version}\n")
            for arg in vars(args):
                f.write(f"{arg}: {getattr(args, arg)}\n")
        
        print(f"Training completed successfully. Results saved to {exp_dir}")
        
        # Plot and save metrics
        results_csv = exp_dir / 'train' / "results.csv"
        if results_csv.exists():
            plot_metrics(results_csv, exp_dir)
        
        # Validate the model and save metrics
        metrics = model.val()
        with open(exp_dir / "validation_metrics.txt", "w") as f:
            f.write(f"mAP50-95: {metrics.box.map}\n")
            f.write(f"mAP50: {metrics.box.map50}\n")
            f.write(f"mAP75: {metrics.box.map75}\n")
    
    except Exception as e:
        print(f"An error occurred during training: {e}")
    finally:
        # Cleanup temp directory
        if Path("temp_results").exists():
            shutil.rmtree("temp_results")

# Add a utility function to check dataset
def verify_dataset(yaml_path):
    """Verify that the dataset exists and is correctly structured."""
    with open(yaml_path) as f:
        data = yaml.safe_load(f)
    
    for split in ['train', 'val']:
        path = Path(data[split])
        images_path = path / 'images'
        labels_path = path / 'labels'
        
        if not images_path.exists():
            raise ValueError(f"{split} images path does not exist: {images_path}")
        if not labels_path.exists():
            raise ValueError(f"{split} labels path does not exist: {labels_path}")
        
        num_images = len(list(images_path.glob('*.jpg')))
        num_labels = len(list(labels_path.glob('*.txt')))
        
        print(f"{split} set: {num_images} images, {num_labels} labels")
        
if __name__ == "__main__":
    args = parse_args()
    main(args)