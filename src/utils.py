import os
import random
import numpy as np
import torch


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_device():
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def setup_paths(data_dir, model_dir, docs_dir, template_path, output_path):
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    
    data_dir = os.path.abspath(os.path.join(project_root, data_dir))
    model_dir = os.path.abspath(os.path.join(project_root, model_dir))
    docs_dir = os.path.abspath(os.path.join(project_root, docs_dir))
    
    if os.path.isabs(template_path):
        template_path_abs = template_path
    else:
        template_path_abs = os.path.abspath(os.path.join(project_root, template_path))
    
    if os.path.isabs(output_path):
        output_path_abs = output_path
    else:
        output_path_abs = os.path.abspath(os.path.join(project_root, output_path))
    
    output_dir = os.path.dirname(output_path_abs)
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(model_dir, exist_ok=True)
    os.makedirs(docs_dir, exist_ok=True)
    
    return {
        'project_root': project_root,
        'data_dir': data_dir,
        'model_dir': model_dir,
        'docs_dir': docs_dir,
        'template_path': template_path_abs,
        'output_path': output_path_abs,
        'output_dir': output_dir
    }


def count_parameters(model):
    return sum(p.numel() for p in model.parameters())


def print_paths(paths):
    print(f"\nProject root: {paths['project_root']}")
    print(f"Data directory: {paths['data_dir']}")
    print(f"Model directory: {paths['model_dir']}")
    print(f"Output directory: {paths['output_dir']}")
    print(f"Docs directory: {paths['docs_dir']}")
    print(f"Template path: {paths['template_path']}")
    print(f"Output path: {paths['output_path']}")
