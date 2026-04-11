import os
import pickle
import json
from pathlib import Path

def create_instructions_json_from_variation_descriptions(
    demo_root,
    output_dir="instructions/peract",
    task_list=None,
    splits=['train', 'val']
):
    """
    Extract task descriptions from variation_descriptions.pkl files 
    and create an instructions.json file.
    
    Args:
        demo_root: Path to RLBench demonstration root (e.g., data/train_data)
        output_dir: Output directory for instructions.json
        task_list: List of tasks to process. If None, auto-discover from folders
        splits: Which data splits to search through
    """
    
    # Auto-discover tasks if not provided
    if task_list is None:
        task_list = []
        for split in splits:
            split_path = Path(demo_root) / split
            if split_path.exists():
                task_list.extend([d.name for d in split_path.iterdir() if d.is_dir()])
        task_list = sorted(set(task_list))  # Remove duplicates
    
    print(f"Found tasks: {task_list}")
    
    # Extract descriptions
    all_instructions = {}
    
    for task in task_list:
        print(f"\nProcessing task: {task}")
        var2text = {}
        
        for split in splits:
            # Path to all_variations/episodes
            episodes_path = Path(demo_root) / f'{split}_data'/ task / "all_variations" / "episodes"
            
            if not episodes_path.exists():
                print(f"  Warning: {episodes_path} not found")
                continue
            
            # Find all episode folders
            episode_folders = sorted([
                d for d in episodes_path.iterdir() 
                if d.is_dir() and d.name.startswith('ep')
            ])
            
            print(f"  Found {len(episode_folders)} episodes in {split}")
            
            for ep_folder in episode_folders:
                # Read variation number
                var_number_path = ep_folder / "variation_number.pkl"
                if not var_number_path.exists():
                    continue
                
                with open(var_number_path, 'rb') as f:
                    var_num = pickle.load(f)
                
                # Skip if we already have this variation
                if var_num in var2text:
                    continue
                
                # Read variation descriptions
                descriptions_path = ep_folder / "variation_descriptions.pkl"
                if not descriptions_path.exists():
                    print(f"    Warning: No descriptions in {ep_folder}")
                    continue
                
                with open(descriptions_path, 'rb') as f:
                    descriptions = pickle.load(f)
                
                # Store with variation number as key
                var2text[var_num] = descriptions
                print(f"    Variation {var_num}: {len(descriptions)} descriptions")
        
        all_instructions[task] = var2text
    
    # Convert variation numbers to strings (JSON requirement)
    all_instructions_str_keys = {
        task: {str(var): descs for var, descs in var_dict.items()}
        for task, var_dict in all_instructions.items()
    }
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Save as JSON
    output_file = os.path.join(output_dir, "instructions.json")
    with open(output_file, 'w') as f:
        json.dump(all_instructions_str_keys, f, indent=2)
    
    print(f"\n✅ Successfully created {output_file}")
    print(f"Total tasks: {len(all_instructions)}")
    for task, var_dict in all_instructions.items():
        print(f"  {task}: {len(var_dict)} variations")
    
    return all_instructions_str_keys


if __name__ == "__main__":
    # Example usage for ManiGaussian data
    demo_root = "/home/olivier/Desktop/ManiGaussian/data/"  # Adjust to your path
    
    # Optional: specify exact tasks
    tasks = [
        'close_jar', 'open_drawer', 'sweep_to_dustpan_of_size', 
        'meat_off_grill', 'turn_tap', 'slide_block_to_color_target', 
        'put_item_in_drawer', 'reach_and_drag', 'push_buttons', 'stack_blocks'
    ]
    
    instructions = create_instructions_json_from_variation_descriptions(
        demo_root=demo_root,
        output_dir="instructions/peractnerf",
        task_list=tasks,
        splits=['train', 'val']
    )
    
    # Verify the structure
    print("\n--- Sample from instructions.json ---")
    for task in list(instructions.keys())[:2]:
        print(f"\n{task}:")
        for var_id in list(instructions[task].keys())[:2]:
            descs = instructions[task][var_id]
            print(f"  Variation {var_id}:")
            for desc in descs[:2]:  # Show first 2 descriptions
                print(f"    - {desc}")