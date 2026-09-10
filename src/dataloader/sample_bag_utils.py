from pathlib import Path
from typing import Tuple, Union, Sequence, Dict, List, Optional

import h5py
import torch
from .dataclasses import Scan, Slide


def zero_pad_tensor(
    tensor: torch.Tensor | List, bag_size: int, device: torch.device
) -> torch.Tensor | List:
    sizes = []
    if isinstance(tensor, List):
        return tensor + [""] * (bag_size - len(tensor))
    else:
        sizes.append(bag_size - tensor.shape[0])
        if len(tensor.shape) > 1:
            sizes.append(tensor.shape[1])
        return torch.cat(
            (
                tensor,
                torch.zeros(sizes, device=device),
            )
        )


# TODO: Merge sample_bag and sample_bag_streamed into one function with a flag for streamed or not
def sample_bag_streamed(
    tensors: Dict[str, Union[torch.Tensor, List]],
    bag_size: int,
    scan: Scan,
    permutation: torch.Tensor,
) -> Dict:
    device = torch.device("cpu")

    bag_idxs = scan.possible_indicies[permutation[:bag_size]]

    output = {}
    bag_samples = read_h5(scan.path, field_name="features", indicies=bag_idxs)
    # If the full bag has less samples than the desired bag size, we pad with zeros
    output["features"] = zero_pad_tensor(bag_samples, bag_size, device)
    output["size"] = torch.tensor(min(bag_size, len(bag_idxs)), device=device)

    for name, tensor in tensors.items():
        # Get the samples from the bag
        if isinstance(tensor, List):
            # If it's a list, we assume it's a list of strings (tile names)
            bag_samples = [tensor[i] for i in bag_idxs]
        else:
            bag_samples = tensor[bag_idxs]
        # If the full bag has less samples than the desired bag size, we pad with zeros
        zero_padded_bag = zero_pad_tensor(bag_samples, bag_size, device)
        output[name] = zero_padded_bag

    return output


def sample_bag(
    tensors: Dict[str, torch.Tensor],
    bag_size: int,
    permutation: torch.Tensor,
    possible_indicies: torch.Tensor,
) -> Dict:
    """Since PyTorch doesn't yet (?) support tensors of varying sizes, and some of the bags might have fewer tiles than the bag size we set,
    we need to pad these bags with zeros. These are later ignored by the model, so it doesn't impact performance.
    """

    device = tensors["features"][0].device
    bag_idxs = possible_indicies[permutation[:bag_size]]

    output = {}

    for name, tensor in tensors.items():
        # Get the samples from the bag
        if isinstance(tensor, List):
            # If it's a list, we assume it's a list of strings (tile names)
            bag_samples = [tensor[i] for i in bag_idxs]
        else:
            bag_samples = tensor[bag_idxs]
        # If the full bag has less samples than the desired bag size, we pad with zeros
        zero_padded_bag = zero_pad_tensor(bag_samples, bag_size, device)
        output[name] = zero_padded_bag

    output["size"] = torch.tensor(min(bag_size, len(bag_idxs)), device=device)
    # We also return the size of the bag, so that it can be ignored by the model
    return output


def read_h5(
    path: Union[str, Path],
    field_name: str = "features",
    indicies: Optional[torch.Tensor] = None,
) -> torch.Tensor | List[str]:
    """Reads the features from an h5 file, ignoring all other information stored in the file."""
    with h5py.File(path, "r") as f:
        ds = f[field_name]

        if indicies is None:
            field_value_np = ds[:]
        else:
            idx = indicies.to("cpu").long()
            sort_perm = torch.argsort(idx)
            idx_sorted = idx[sort_perm].numpy()
            vals_sorted = ds[idx_sorted]

            inv_perm = torch.empty_like(sort_perm)

            inv_perm[sort_perm] = torch.arange(len(sort_perm))
            field_value_np = vals_sorted[inv_perm.numpy()]

        # Check if field value is str
        if field_name == "tile_names":
            field_value = [str(s.decode("utf-8")) for s in field_value_np]
            raise ValueError("Just making sure this isn't running...")
        else:
            field_value = torch.from_numpy(field_value_np)

        return field_value


# def read_h5(
#     path: Union[str, Path],
#     field_name: str = "features",
#     indicies: Optional[torch.Tensor] = None,
# ) -> torch.Tensor | List[str]:
#     """
#     Reads the features from an h5 file.
#     OPTIMIZATION: Reads the entire dataset into RAM first (fast sequential read),
#     then slices in memory. This avoids slow HDF5 random seek operations.
#     """
#     with h5py.File(path, "r") as f:
#         ds = f[field_name]

#         # 1. READ EVERYTHING (Sequential Read = Fast)
#         full_data = ds[:]

#         if indicies is None:
#             field_value_np = full_data
#         else:
#             # 2. SLICE IN RAM (Instant)
#             # We don't need to sort indices here; RAM supports random access.
#             idx_cpu = indicies.to("cpu").numpy()
#             field_value_np = full_data[idx_cpu]

#         # Check if field value is str (for tile_names)
#         if field_name == "tile_names":
#             field_value = [str(s.decode("utf-8")) for s in field_value_np]
#             raise ValueError("Just making sure this isn't running...")
#         else:
#             field_value = torch.from_numpy(field_value_np)

#         return field_value
