import os
import numpy as np
from PIL import Image, ImageDraw

IMAGE_SIZE = 128

TRAIN_SAMPLES = 2000
VAL_SAMPLES = 400
TEST_SAMPLES = 600

SEQUENCE_LENGTH = 5

OUTPUT_DIR = "dataset/data"

os.makedirs(OUTPUT_DIR, exist_ok=True)

def create_frame(x, y, angle, goal_x, goal_y, obstacles):
    """
    Create a top-down simulated robot camera frame.
    """

    img = Image.new(
        "RGB",
        (IMAGE_SIZE, IMAGE_SIZE),
        (240, 240, 240)
    )

    draw = ImageDraw.Draw(img)

    for i in range(0, IMAGE_SIZE, 16):

        draw.line(
            [(i, 0), (i, IMAGE_SIZE)],
            fill=(220, 220, 220)
        )

        draw.line(
            [(0, i), (IMAGE_SIZE, i)],
            fill=(220, 220, 220)
        )

    for ox, oy, size in obstacles:

        draw.rectangle(
            [
                (ox - size, oy - size),
                (ox + size, oy + size)
            ],
            fill=(80, 80, 80)
        )

    # --------------------------------------------------------
    # Goal
    # --------------------------------------------------------

    goal_size = 5

    draw.ellipse(
        [
            goal_x - goal_size,
            goal_y - goal_size,
            goal_x + goal_size,
            goal_y + goal_size
        ],
        fill=(0, 180, 0)
    )

    robot_size = 7

    draw.ellipse(
        [
            x - robot_size,
            y - robot_size,
            x + robot_size,
            y + robot_size
        ],
        fill=(30, 100, 220)
    )

    direction_length = 12

    dx = np.cos(angle) * direction_length
    dy = np.sin(angle) * direction_length

    draw.line(
        [
            (x, y),
            (x + dx, y + dy)
        ],
        fill=(255, 0, 0),
        width=2
    )

    return np.array(
        img,
        dtype=np.uint8
    )

def generate_sequence():


    x = np.random.uniform(
        20,
        108
    )

    y = np.random.uniform(
        20,
        108
    )

    goal_x = np.random.uniform(
        20,
        108
    )

    goal_y = np.random.uniform(
        20,
        108
    )

    angle = np.random.uniform(
        -np.pi,
        np.pi
    )

    speed = np.random.uniform(
        2.0,
        5.0
    )

    obstacles = []

    for _ in range(3):

        ox = np.random.uniform(
            20,
            108
        )

        oy = np.random.uniform(
            20,
            108
        )

        size = np.random.uniform(
            5,
            10
        )

        obstacles.append(
            (ox, oy, size)
        )

    frames = []

    positions = []

    # Need one extra position
    # for target movement

    for t in range(
        SEQUENCE_LENGTH + 1
    ):

        positions.append(
            (x, y)
        )

        frame = create_frame(
            x,
            y,
            angle,
            goal_x,
            goal_y,
            obstacles
        )

        frames.append(frame)

        angle += np.random.normal(
            0,
            0.10
        )

        x += np.cos(angle) * speed
        y += np.sin(angle) * speed

        if x < 10 or x > 118:

            angle = np.pi - angle

        if y < 10 or y > 118:

            angle = -angle

        x = np.clip(
            x,
            10,
            118
        )

        y = np.clip(
            y,
            10,
            118
        )

    x1, y1 = positions[-2]

    x2, y2 = positions[-1]

    dx = (
        x2 - x1
    ) / IMAGE_SIZE

    dy = (
        y2 - y1
    ) / IMAGE_SIZE

    target = np.array(
        [dx, dy],
        dtype=np.float32
    )

    frames = np.array(
        frames[:SEQUENCE_LENGTH]
    )

    return frames, target

def generate_dataset(
    number_of_samples,
    filename
):

    print(
        f"\nGenerating {number_of_samples} samples..."
    )

    X = []

    y = []

    for i in range(
        number_of_samples
    ):

        frames, target = generate_sequence()

        X.append(frames)

        y.append(target)

        if (
            (i + 1) % 500 == 0
        ):

            print(
                f"Generated "
                f"{i + 1}/{number_of_samples}"
            )

    X = np.array(
        X,
        dtype=np.uint8
    )

    y = np.array(
        y,
        dtype=np.float32
    )

    output_path = os.path.join(
        OUTPUT_DIR,
        filename
    )

    np.savez_compressed(
        output_path,
        X=X,
        y=y
    )

    print(
        f"Saved: {output_path}"
    )

    print(
        f"X shape: {X.shape}"
    )

    print(
        f"y shape: {y.shape}"
    )
    
if __name__ == "__main__":

    np.random.seed(42)

    generate_dataset(
        TRAIN_SAMPLES,
        "train.npz"
    )

    generate_dataset(
        VAL_SAMPLES,
        "val.npz"
    )

    generate_dataset(
        TEST_SAMPLES,
        "test.npz"
    )

    print(
        "\nDataset generation completed successfully!"
    )