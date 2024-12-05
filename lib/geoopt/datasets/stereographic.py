import itertools
import torch.utils.data


class StereographicTreeDataset(torch.utils.data.Dataset):
    """
    Implementation of a synthetic dataset by hierarchical diffusion.

    Adopted from https://github.com/emilemathieu/pvae/blob/ca5c4997a90839fc8960ec812df4cbf83da55781/pvae/datasets/datasets.py

    Parameters
    ----------
    dim: int
        dimension of the input sample
    depth: int:
        depth of the tree; the root corresponds to the depth 0
    numberOfChildren: int
        Number of children of each node in the tree
    numberOfsiblings: int
        Number of noisy observations obtained from the nodes of the tree
    sigma_children: float
        noise
    """

    def __init__(
        self,
        ball,
        dim,
        depth,
        numberOfChildren=2,
        dist_children=1,
        sigma_sibling=2,
        numberOfsiblings=1,
    ):
        self.dim = int(dim)
        self.ball = ball
        self.root = ball.origin(self.dim)
        self.sigma_sibling = sigma_sibling
        self.depth = int(depth)
        self.dist_children = dist_children
        self.numberOfChildren = int(numberOfChildren)
        self.numberOfsiblings = int(numberOfsiblings)
        self.__class_counter = itertools.count()
        self.origin_data, self.origin_labels, self.data, self.labels = map(
            torch.detach, self.bst()
        )
        self.num_classes = self.origin_labels.max().item() + 1

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        data, labels = self.data[idx], self.labels[idx]
        return data, labels, labels.max(-1).values

    def get_directions(self, parent_value, numberOfChildren):
        if self.dim == 2 and numberOfChildren == 2:
            direction = torch.randn(self.dim)
            parent_value_n = parent_value / parent_value.norm().clamp_min(1e-15)
            direction -= parent_value_n @ direction * parent_value_n
            return torch.stack([direction, -direction])
        else:
            directions = torch.randn(numberOfChildren, self.dim)
            parent_value_n = parent_value / parent_value.norm().clamp_min(1e-15)
            directions -= (directions @ parent_value_n)[:, None] * parent_value_n
            return directions

    def get_children(self, parent_value, parent_label, current_depth, offspring=True):
        if offspring:
            numberOfChildren = self.numberOfChildren
            sigma = self.dist_children
        else:
            numberOfChildren = self.numberOfsiblings
            sigma = self.sigma_sibling
        if offspring:
            directions = self.get_directions(parent_value, numberOfChildren)
            child_values = self.ball.geodesic_unit(
                torch.tensor(sigma), parent_value, directions
            )
            children = []
            for child in child_values:
                child_label = parent_label.clone()
                child_label[current_depth] = next(self.__class_counter)
                children.append((child, child_label))
        else:
            children = []
            for _ in range(numberOfChildren):
                child_value = self.ball.random(
                    self.dim, mean=parent_value, std=sigma**0.5
                )
                child_label = parent_label.clone()
                children.append((child_value, child_label))
        return children

    def bst(self):
        label = -torch.ones(self.depth + 1, dtype=torch.long)
        label[0] = next(self.__class_counter)
        queue = [(self.root, label, 0)]
        visited = []
        labels_visited = []
        values_clones = []
        labels_clones = []
        while len(queue) > 0:
            current_node, current_label, current_depth = queue.pop(0)
            visited.append(current_node)
            labels_visited.append(current_label)
            if current_depth < self.depth:
                children = self.get_children(current_node, current_label, current_depth)
                for child in children:
                    queue.append((child[0], child[1], current_depth + 1))
            if current_depth <= self.depth:
                clones = self.get_children(
                    current_node, current_label, current_depth, False
                )
                for clone in clones:
                    values_clones.append(clone[0])
                    labels_clones.append(clone[1])
        length = int(
            ((self.numberOfChildren) ** (self.depth + 1) - 1)
            / (self.numberOfChildren - 1)
        )
        images = torch.cat([i for i in visited]).reshape(length, self.dim)
        labels_visited = torch.cat([i for i in labels_visited]).reshape(
            length, self.depth + 1
        )[:, : self.depth]
        values_clones = torch.cat([i for i in values_clones]).reshape(
            self.numberOfsiblings * length, self.dim
        )
        labels_clones = torch.cat([i for i in labels_clones]).reshape(
            self.numberOfsiblings * length, self.depth + 1
        )
        return images, labels_visited, values_clones, labels_clones
