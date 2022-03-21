import argparse
import gzip
from pathlib import Path

import habitat
from habitat.config.default import _C
from habitat.sims import make_sim

from habitat.datasets.pointnav.pointnav_generator import generate_pointnav_episode, generate_pointnav_episode_radius

parser = argparse.ArgumentParser(description='Create point goal navigation task dataset for The Beacon 3D scan data.')

# Scene dataset path.
parser.add_argument('--scene_dataset', help='Scene dataset path containing the *.glb files.')

# Task complexity.
parser.add_argument('--max-distance', default=30, type=int, help='Maximum shortest path distance in meters.')
parser.add_argument('--max-steps', default=500, type=int, help='Maximum number of episode steps.')

# Dataset split. Default values are based on the MP3D PointNav dataset in Habitat.
parser.add_argument('--train-episodes', default=1, type=int, help='Number of training episodes per scene.') # default = 5000
parser.add_argument('--valid-episodes', default=1, type=int, help='Number of validation episodes per scene.') # default = 50
parser.add_argument('--test-episodes', default=1, type=int, help='Number of testing episodes per scene.') # default = 150

# Output folder.
parser.add_argument('--output', default='./scenarios/datasets/hall2', help='Dataset root folder.')

# Parse arguments.
args = parser.parse_args()

#
radius = 0
start = [0, -12]
end = [0, 12]

# Setup output folders.
path = Path(args.output)
path.mkdir(parents=True, exist_ok=False)

# Create splits.
scenes_path = Path(args.scene_dataset)

splits = [('train', args.train_episodes), ('val', args.valid_episodes), ('test', args.test_episodes)]
scenes = ['room3x3']

max_steps = args.max_steps
max_distance = args.max_distance

for split, size in splits:
    print(f"Creating split: {split}")

    # Setup split folder.
    (path / split).mkdir(parents=True, exist_ok=False)

    # Create an empty split task data set.
    dataset = habitat.Dataset()
    dataset.episodes = []

    # with gzip.open(path / split / f'{split}.json.gz', 'wb') as f:
    #     f.write(dataset.to_json().encode())

    # Create a task dataset for each scene.
    for scene in scenes:
        # Setup simulator.
        config = _C.clone()
        config.SIMULATOR.SCENE = str(scenes_path / f"{scene}.glb")
        sim = make_sim(config.SIMULATOR.TYPE, config=config.SIMULATOR)

        # Setup episode generator.
        generator = generate_pointnav_episode_radius(
            sim,
            shortest_path_max_steps=max_steps,
            furthest_dist_limit=max_distance,
            num_episodes=size,
            start=start,
            end=end,
            point_radius=radius
        )

        # Create scene dataset.
        dataset = habitat.Dataset()
        dataset.episodes = [e for e in generator]

        # Store scene dataset.
        with gzip.open(path / split / f'{split}.json.gz', 'wb') as f:
            f.write(dataset.to_json().encode())

        sim.close()

