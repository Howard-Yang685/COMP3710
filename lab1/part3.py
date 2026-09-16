import torch
import matplotlib
matplotlib.use("QtAgg")
import matplotlib.pyplot as plt

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- Sierpinski Triangle ---

num_points = 200000

# Triangle vertices
vertices = torch.tensor([
    [0.0, 0.0],
    [1.0, 0.0],
    [0.5, 0.866]
], device=device)

# Start many points at random positions
points = torch.rand(num_points, 2, device=device)

# Apply the chaos-game rule
for i in range(20):
    choices = torch.randint(0, 3, (num_points,), device=device)
    selected_vertices = vertices[choices]

    points = (points + selected_vertices) / 2

# Move to CPU for plotting
points = points.cpu().numpy()

plt.scatter(points[:, 0], points[:, 1], s=0.1)
plt.title("Sierpinski Triangle")
plt.axis("equal")
plt.axis("off")
plt.tight_layout()
plt.show()