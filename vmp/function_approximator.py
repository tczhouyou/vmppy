from copy import deepcopy
from pydantic import BaseModel, Field, model_validator, ValidationInfo

from typing import Any, Dict, List, Optional, Tuple, Union, Literal
import numpy as np
from functools import lru_cache
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from rff import rff
from omegaconf import OmegaConf


from .type_utils import check_type

from vmp.data_type import MVN
import logging


logger = logging.getLogger()
logger.setLevel(logging.INFO)



class BaseConfig(BaseModel):
    dim: int
    in_dim: int = 1

class KRBFConfig(BaseConfig):
    num_kernels: int = 10
    kernel_std: float = 0.1
    center_offsets: float = 0.2
    return_type: Literal["mvn", "point"] = "point"

class FixedBoundaryFCNNConfig(BaseConfig):
    hidden_sizes: List[int] = Field(default_factory=lambda: [64, 64])
    activations: Union[List[str], str] = "relu"
    optimizer: str = "adam"
    learning_rate: float = 1e-3
    train_epoch: int = 100
    stop_threshold: float = 1e-3
    batch_size: int = 32
    boundary_start: Optional[List[float]] = None
    boundary_end: Optional[List[float]] = None
    device: str = "cuda" if torch.cuda.is_available() else "cpu"

class FixedBoundaryGRUConfig(BaseConfig):
    hidden_size: int = 128
    num_layers: int = 2
    optimizer: str = "adam"
    learning_rate: float = 1e-3
    train_epoch: int = 100
    stop_threshold: float = 1e-3
    batch_size: int = 32
    subseq_len: int = 32
    boundary_start: Optional[List[float]] = None
    boundary_end: Optional[List[float]] = None
    device: str = "cuda" if torch.cuda.is_available() else "cpu"

class HashMappingConfig(BaseConfig):
    table_size: int = 1000



class FunctionApproximator(nn.Module):
    def __init__(self, in_dim: int, dim: int) -> None:
        super().__init__()
        self.in_dim = in_dim
        self.dim = dim

    def __call__(self, phase: float) -> np.ndarray:
        pass

    def learn(self, x: np.ndarray, y: np.ndarray) -> None:
        pass

    def save_model(self, path):
        torch.save(self.state_dict(), path)

    def load_model(self, path):
        self.load_state_dict(torch.load(path))


class HashMapping(FunctionApproximator):
    def __init__(self, dim: int) -> None:
        """
        Args:
            in_dim (int): Input dimension (will be always 1 in this case)
            dim (int): Output dimension
        """
        super().__init__(1, dim)
        self.hash_table = {}  # To store key-value pairs
        self.x_values = []  # To store the x values for interpolation

    def _hash(self, x: float) -> int:
        return float(x) 

    def __call__(self, phase: Union[float, np.ndarray]) -> np.ndarray:
        """Retrieve or interpolate value based on the input.
        
        Args:
            phase (Union[float, np.ndarray]): Input value
        
        Returns:
            np.ndarray: The interpolated or exact value
        """
        if isinstance(phase, np.ndarray):
            return np.array([self._get_value(p) for p in phase])
        return self._get_value(phase)

    def _get_value(self, x: float) -> np.ndarray:
        """Retrieve the value for a given input x, with interpolation if necessary.
        
        Args:
            x (float): Input value
            
        Returns:
            np.ndarray: Interpolated or exact value from the hash table
        """
        if x in self.x_values:
            hash_key = self._hash(x)
            return self.hash_table[hash_key]
        
        # Interpolate if x is not in the hash table
        return self._interpolate(x)

    def _interpolate(self, x: float) -> np.ndarray:
        """Interpolate between two nearest x-values to get the value for x.
        
        Args:
            x (float): Input value
        
        Returns:
            np.ndarray: Interpolated output
        """
        sorted_x = sorted(self.x_values)
        idx = np.searchsorted(sorted_x, x)

        if idx == 0:
            return self.hash_table[self._hash(sorted_x[0])]
        elif idx == len(sorted_x):
            return self.hash_table[self._hash(sorted_x[-1])]

        x_left, x_right = sorted_x[idx - 1], sorted_x[idx]
        y_left = self.hash_table[self._hash(x_left)]
        y_right = self.hash_table[self._hash(x_right)]

        # Linear interpolation
        alpha = (x - x_left) / (x_right - x_left)
        return (1 - alpha) * y_left + alpha * y_right

    def learn(self, x: np.ndarray, y: np.ndarray) -> None:
        """Populate the hash table with key-value pairs from the training data.
        
        Args:
            x (np.ndarray): Input data, shape (num_demos, num_samples, 1)
            y (np.ndarray): Output data, shape (num_demos, num_samples, dim)
        """
        assert x.shape[0] == y.shape[0] == 1, "Hash mapping only supports one demo"
        assert x.shape[1] == y.shape[1], "Number of inputs must match number of outputs"
        x = x.squeeze(0)
        y = y.squeeze(0)

        for i in range(x.shape[0]):
            hash_key = self._hash(x[i])
            self.hash_table[hash_key] = y[i]
            self.x_values.append(float(x[i]))

    def save_hash_table(self, path: str) -> None:
        """Save the hash table to a file."""
        np.savez(path, hash_table=self.hash_table, x_values=self.x_values)

    def load_hash_table(self, path: str) -> None:
        """Load the hash table from a file."""
        data = np.load(path, allow_pickle=True)
        self.hash_table = data['hash_table'].item()
        self.x_values = data['x_values'].tolist()


class KRBF(FunctionApproximator):
    def __init__(
        self, 
        dim: int,
        in_dim: int = 1,
        num_kernels: int = 10,
        kernel_std: float = 0.1,
        center_offsets: float = 0.2,
        return_type: str = 'point'
    ) -> None:
        """Kernelized Radial Basis Function (KRBF) for function approximation.
        
        Args:
            dim (int): Dimension of the function
            num_kernels (int, optional): Number of kernels. Defaults to 10.
            kernel_std (float, optional): Standard deviation of the kernel. Defaults to 0.1.
            center_offsets (float, optional): Offset of the centers. Defaults to 0.2.
            return_type (str, optional): Return type. Defaults to 'mvn'. It can be {'mvn', 'point'}
        """
        super().__init__(in_dim, dim)
        self.num_kernels = num_kernels
        self.inv_var = -0.5 / (kernel_std ** 2)
        self.centers = np.linspace(
            1 + center_offsets,
            0 - center_offsets,
            num_kernels
        )
        self.dim = self.dim
        self.learned_weights = MVN(center=np.zeros(self.dim * num_kernels), cov=np.eye(self.dim * num_kernels))
        self.weights = deepcopy(self.learned_weights)
        self.return_type = return_type
        assert self.return_type in ['mvn', 'point'], "Invalid return type"

    @lru_cache(maxsize=128)
    def __psi_cached__(self, can_values: Tuple[float]) -> np.ndarray:
        can_values = np.array(can_values)  # Convert back to NumPy array
        return np.exp(np.square(can_values - self.centers) * self.inv_var)

    def __psi__(self, can_value: Union[float, np.ndarray]) -> np.ndarray:
        """Get kernel evaluation.
        
        Args:
            can_value (Union[float, np.ndarray]): Canonical system value
                it can be (num_samples,) or (num_samples, 1)
        Returns:
            np.ndarray: Kernel evaluation (num_samples, num_kernels)
        """
        if np.ndim(can_value) == 0:
            can_value = np.array([[can_value]])
        elif np.ndim(can_value) == 1:
            can_value = can_value[..., np.newaxis]

        can_values = tuple(map(tuple, can_value))
        return self.__psi_cached__(can_values)

    def __dpsi__(self, can_value: Union[float, np.ndarray]) -> np.ndarray:
        fx = self.__psi__(can_value)
        dft = - np.multiply(fx, 2 * (can_value - self.centers) * self.inv_var)
        return dft

    def __Psi__(self, can_value: Union[float, np.ndarray]) -> np.ndarray:
        """Get kernel evaluation.
        
        Args:
            can_value (Union[float, np.ndarray]): Canonical system value
                it can be (num_samples,) or (num_samples, 1)
        Returns:
            np.ndarray: Kernel evaluation (num_samples * dim, num_kernels * dim)
        """
        psi = self.__psi__(can_value)
        n_samples, n_kernel = psi.shape
        Psi = np.zeros((n_samples*self.dim, n_kernel*self.dim))
        for i in range(self.dim):
            Psi[i*n_samples:(i+1)*n_samples, i*n_kernel:(i+1)*n_kernel] = psi

        return Psi

    def __call__(self, phase: Union[float, np.ndarray]) -> np.ndarray:
        Psi = self.__Psi__(phase)
        traj_mean = Psi @ self.weights.center
        
        cov = self.weights.cov
        if cov.ndim == 0:
            traj_cov = np.diag(np.ones_like(traj_mean) * cov)
        else:
            traj_cov = Psi @ cov @ Psi.T

        mvn = MVN(center=traj_mean, cov=traj_cov)
        if self.return_type == 'point':
            center = mvn.center.reshape(-1, self.dim, order='F')
            return center
        elif self.return_type == 'mvn':
            return mvn
        else:
            raise ValueError("Invalid return type")

    def learn(self, x: np.ndarray, y: np.ndarray) -> None:
        """Learn the weights.

        Args:
            x (np.ndarray): x is a (num_demos, num_samples, ) array
            y (np.ndarray): y is a (num_demos, num_samples, dim) array
        """

        assert x.ndim == 3, "x should be a 3D array, (num_demos, num_samples, 1)"
        assert y.ndim == 3, "y should be a 3D array, (num_demos, num_samples, dim)"
        assert x.shape[0] == y.shape[0], "Number of demos should be the same"
        assert x.shape[1] == y.shape[1], "Number of samples for each demo should be the same"
        
        num_demos = x.shape[0]
        weights = []
        for i in range(num_demos):
            x_i = x[i, :, :]
            y_i = y[i, :, :]
            weight = np.linalg.lstsq(self.__Psi__(x_i), y_i.flatten(order='F'), rcond=None)[0]
            weights.append(weight)

        weights = np.array(weights)  # (num_demos, num_kernel * dim)
        avg_weight = np.mean(weights, axis=0) # (num_kernel * dim,)
        cov_weight = np.cov(weights, rowvar=False) # (num_kernel * dim, num_kernel * dim)
        self.learned_weights = MVN(center=avg_weight, cov=cov_weight)
        self.weights = deepcopy(self.learned_weights)

    def save_learned_weights_to_csv(self, path: str) -> None:
        np.savetxt(path, self.learned_weights.center.reshape(1,-1), delimiter=",")
    

def get_activation(activation: str):
    if activation == 'relu':
        return torch.nn.ReLU()
    elif activation == 'tanh':
        return torch.nn.Tanh()
    elif activation == 'sigmoid':
        return torch.nn.Sigmoid()
    else:
        raise ValueError("Invalid activation function")
    
def get_optimizer(parameters: torch.Tensor, optimizer: str, learning_rate: float):
    if optimizer == 'adam':
        return torch.optim.Adam(parameters, lr=learning_rate, weight_decay=1e-5)
    elif optimizer == 'sgd':
        return torch.optim.SGD(parameters, lr=learning_rate, weight_decay=1e-5)
    else:
        raise ValueError("Invalid optimizer")


class FixedBoundaryFCNN(FunctionApproximator):
    def __init__(
        self, 
        dim: int, 
        in_dim: int = 1, 
        hidden_sizes: List[float] = [64, 64], 
        activations: Union[List[str], str] = 'relu',
        optimizer: str = "adam",
        learning_rate: float = 1e-3,
        train_epoch: int = 100,
        stop_threshold: float = 1e-3,
        batch_size: int = 32,
        boundary_start: Optional[np.ndarray] = None,
        boundary_end: Optional[np.ndarray] = None,
        device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
    ) -> None:
        """Fully connected neural network for function approximation.

        Args:
            in_dim (int): Input dimension
            dim (int): Output dimension
            hidden_sizes (List[float]): List of hidden layer sizes
            activations (Union[List[str], str], optional): Activation function. Defaults to 'relu'.
            optimizer (str, optional): Optimizer. Defaults to "adam".
            learning_rate (float, optional): Learning rate. Defaults to 1e-3.
            train_epoch (int, optional): Number of training epochs. Defaults to 100.
            stop_threshold (float, optional): Stop threshold for training. Defaults to 1e-3.
            batch_size (int, optional): Batch size. Defaults to 32.
            device (str, optional): Device. Defaults to 'cuda' if torch.cuda.is_available() else 'cpu'.
        """
        super().__init__(in_dim, dim)

        # Define the model
        if isinstance(activations, str):
            activations = [activations] * len(hidden_sizes)

        layers = []
        for i, hidden_size in enumerate(hidden_sizes):
            layers.append(torch.nn.Linear(in_dim, hidden_size))
            if i == 1:
                layers.append(rff.layers.GaussianEncoding(1.0, hidden_size, hidden_size // 2))
            layers.append(
                get_activation(activations.pop(0))
            )
            in_dim = hidden_size

        self.model = torch.nn.Sequential(*layers, torch.nn.Linear(in_dim, dim))
        self.optimizer = get_optimizer(self.model.parameters(), optimizer, learning_rate)
        self.loss_fn = torch.nn.MSELoss()
        self.train_epoch = train_epoch
        self.stop_threshold = stop_threshold
        self.batch_size = batch_size
        self.device = torch.device(device)
        self.model.to(self.device)

        self.boundary_start = boundary_start
        self.boundary_end = boundary_end

        if self.boundary_start is None:
            self.boundary_start = np.zeros(self.dim)
        if self.boundary_end is None:
            self.boundary_end = np.zeros(self.dim)

        self._initialize_weights()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        boundary_start = torch.tensor(self.boundary_start, dtype=torch.float32).to(self.device).unsqueeze(0)
        boundary_end = torch.tensor(self.boundary_end, dtype=torch.float32).to(self.device).unsqueeze(0)
        y = self.model(x)

        # phase variable goes from 1 to 0
        y = x * boundary_start + (1 - x) * boundary_end + y * x * (1-x)
        return y
    
    def _initialize_weights(self):
        """Initialize weights of the model using Xavier (Glorot) uniform initialization."""
        for layer in self.model:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)
                if layer.bias is not None:
                    nn.init.zeros_(layer.bias)

    def learn(self, x: np.ndarray, y: np.ndarray) -> None:
        """Train the FCNN model.

        Args:
            x (np.ndarray): Input data, shape (num_demos, num_samples, in_dim)
            y (np.ndarray): Target data, shape (num_demos, num_samples, dim)
        """
        x_tensor = torch.tensor(x.reshape(-1, self.in_dim), dtype=torch.float32).to(self.device)
        y_tensor = torch.tensor(y.reshape(-1, self.dim), dtype=torch.float32).to(self.device)
        self.batch_size = min(self.batch_size, x_tensor.shape[0])

        dataset = TensorDataset(x_tensor, y_tensor)
        dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

        for epoch in range(self.train_epoch):
            total_loss = 0
            for batch_x, batch_y in dataloader:
                self.optimizer.zero_grad()
                predictions = self.forward(batch_x)
                loss = self.loss_fn(predictions, batch_y)
                loss.backward()
                self.optimizer.step()
                total_loss += loss.item()
            
            avg_loss = total_loss / len(dataloader)
            if avg_loss < self.stop_threshold:
                print(f"Training stopped at epoch {epoch} with average loss {avg_loss}")
                break
            if epoch % 10 == 0:
                print(f"Epoch {epoch}, Loss: {avg_loss}")

    def __call__(self, phase: Union[float, np.ndarray]) -> np.ndarray:
        if np.ndim(phase) == 0:
            phase = np.array([[phase]]) # (1, 1)
        elif np.ndim(phase) == 1:
            phase = phase[..., np.newaxis] # (n, 1)
        
        phase_tensor = torch.tensor(phase, dtype=torch.float32).to(self.device)
        with torch.no_grad():
            output = self.model(phase_tensor).cpu().numpy()

        boundary_start = np.expand_dims(self.boundary_start, axis=0)
        boundary_end = np.expand_dims(self.boundary_end, axis=0)
        output = phase * boundary_start + (1 - phase) * boundary_end + output * phase * (1-phase)
        return output # (n|1, dim)


class FixedBoundaryGRU(FunctionApproximator):
    def __init__(
        self, 
        dim: int, 
        in_dim: int = 1,
        hidden_size: int=128, 
        num_layers: int=2, 
        optimizer: str = "adam",
        learning_rate: float = 1e-3,
        train_epoch: int = 100,
        stop_threshold: float = 1e-3,
        batch_size: int = 32,
        subseq_len: int = 32,
        boundary_start: Optional[np.ndarray] = None,
        boundary_end: Optional[np.ndarray] = None,
        device='cuda' if torch.cuda.is_available() else 'cpu'
    ):
        super().__init__(in_dim, dim)
        
        # Store boundary values
        self.boundary_start = boundary_start
        self.boundary_end = boundary_end
        if self.boundary_start is None:
            self.boundary_start = np.zeros(self.dim)
        if self.boundary_end is None:
            self.boundary_end = np.zeros(self.dim)

        # Define GRU model
        self.init_layer = nn.Linear(in_dim, hidden_size)
        self.rff = rff.layers.GaussianEncoding(1.0, hidden_size, hidden_size // 2)
        self.gru = nn.GRU(input_size=hidden_size, hidden_size=hidden_size, num_layers=num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, dim)

        # Define optimizer
        self.optimizer = get_optimizer(self.parameters(), optimizer, learning_rate)
        self.loss_fn = nn.MSELoss()
        self.batch_size = batch_size
        self.subseq_len = subseq_len
        self.train_epoch = train_epoch
        self.stop_threshold = stop_threshold

        # Initialize device
        self.device = torch.device(device)
        self.to(self.device)

        self.hidden_state = None

    def create_subsequences(self, x: np.ndarray, y: np.ndarray, subseq_len: int, step=1):
        """Split a long sequence into subsequences with a fixed length.
        
        Args:
            x (np.ndarray): Original sequence of shape (num_demos, num_samples, in_dim).
            y (np.ndarray): : Original sequence of shape (num_demos, num_samples, in_dim).
            subseq_len (int): Length of each subsequence (window size).
            step (int, optional): Step size for the sliding window. Defaults to 1.
        
        Returns:
            Tuple[np.ndarray, np.ndarray]: Tuple containing subsequences and corresponding targets.
        """
        subsequences = []
        targets = []
        
        for i in range(0, x.shape[1] - subseq_len + 1, step):
            subseq = x[:, i:i + subseq_len]
            subsequences.append(subseq)
            target = y[:, i:i + subseq_len]
            targets.append(target)
        
        subsequences = np.stack(subsequences, axis=1) # (num_demos, num_subseq, subseq_len, in_dim)
        targets = np.stack(targets, axis=1) # (num_demos, num_subseq, subseq_len, in_dim)
        return subsequences, targets

    def forward(self, x, hidden_state = None):
        # Pass input through GRU layer
        embed = self.init_layer(x)
        gru_out, hidden_state = self.gru(embed, hidden_state)  # Shape: (batch_size, seq_len, hidden_size)
        output = self.fc(gru_out)  # Shape: (batch_size, seq_len, dim)
        
        boundary_start = torch.tensor(self.boundary_start, dtype=torch.float32).to(self.device).unsqueeze(0).unsqueeze(0)
        boundary_end = torch.tensor(self.boundary_end, dtype=torch.float32).to(self.device).unsqueeze(0).unsqueeze(0)
        
        # Apply fixed boundary conditions
        output = x * boundary_start + (1 - x) * boundary_end + output * x * (1-x)
        return output, hidden_state

    def learn(self, x: np.ndarray, y: np.ndarray):
        """Train the GRU model.

        Args:
            x (np.ndarray or torch.Tensor): Input sequence data (num_demos, num_samples, 1).
            y (np.ndarray or torch.Tensor): Target sequence data (num_demos, num_samples, dim).
            num_epochs (int, optional): Number of training epochs. Defaults to 100.
            learning_rate (float, optional): Learning rate for the optimizer. Defaults to 1e-3.
        """
        # Convert to tensor if numpy array is provided
        num_samples = x.shape[1]
        if num_samples / self.subseq_len < 1: 
            raise ValueError(
                f"The number of trajectory samples {num_samples} should be greater than subsequence length {self.subseq_len}"
            )
        x, y = self.create_subsequences(x, y, self.subseq_len, step=1)
        x = x.reshape(-1, x.shape[-2], x.shape[-1])
        y = y.reshape(-1, y.shape[-2], y.shape[-1])
        self.batch_size = min(self.batch_size, x.shape[0])

        x_tensor = torch.tensor(x, dtype=torch.float32).to(self.device)
        y_tensor = torch.tensor(y, dtype=torch.float32).to(self.device)
        
        self.train()
        dataset = TensorDataset(x_tensor, y_tensor)
        dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

        # Training loop
        loss = 0
        for epoch in range(self.train_epoch):
            total_loss = 0
            for batch_x, batch_y in dataloader:
                self.optimizer.zero_grad()
                predictions, _ = self.forward(batch_x)
                loss = self.loss_fn(predictions, batch_y)
                loss.backward()
                self.optimizer.step()
                total_loss += loss.item()
      
            avg_loss = total_loss / len(dataloader)
            if avg_loss < self.stop_threshold:
                print(f"Training stopped at epoch {epoch} with average loss {avg_loss}")
                break
            if epoch % 10 == 0:
                print(f"Epoch {epoch}, Loss: {avg_loss}")

    def __call__(self, phase: Union[float, np.ndarray]) -> np.ndarray:
        """Inference method to pass data through the model."""
        self.eval()  # Set model to evaluation mode
        if isinstance(phase, float):
            phase = np.array([[phase]]) # (1, 1)
        else:
            phase = phase[..., np.newaxis] # (n, 1)
        
        # Get batch size
        phase = torch.tensor(phase, dtype=torch.float32).to(self.device).unsqueeze(0) # (1, 1|n, 1)
        with torch.no_grad():
            output, self.hidden_state = self.forward(phase, self.hidden_state)

        output = output.squeeze(0).cpu().numpy() # (1|n, dim)
        return output


class FunctionApproximatorConfig(BaseModel):
    type: Literal["krbf", "fbfcnn", "fbgru", "hash"] = "krbf"
    config: Union[KRBFConfig, FixedBoundaryFCNNConfig, FixedBoundaryGRUConfig, HashMappingConfig] = KRBFConfig(dim=1)

    @model_validator(mode="before")
    def set_shape_modulation_config(cls, data: Dict[str, Any]) -> Dict[str, Any]:
        if not isinstance(data, dict):
            data = OmegaConf.to_container(data)
    
        values = data if isinstance(data, dict) else {}
        config_type = values.get("type", "krbf")

        if "config" not in values:
            values["config"] = {"dim": 1}

        if check_type(values["config"], Union[KRBFConfig, FixedBoundaryFCNNConfig, FixedBoundaryGRUConfig, HashMappingConfig]):
            return values

        if config_type == "krbf":
            values["config"] = KRBFConfig(**values["config"])
        elif config_type == "fbfcnn":
            values["config"] = FixedBoundaryFCNNConfig(**values["config"])
        elif config_type == "fbgru":
            values["config"] = FixedBoundaryGRUConfig(**values["config"])
        elif config_type == "hash":
            values["config"] = HashMappingConfig(**values["config"])
        else:
            raise ValueError(f"Unsupported type {config_type}")
        return values

        
class FunctionApproximatorFactory():
    @staticmethod
    def get_function_approximator(config: FunctionApproximatorConfig) -> FunctionApproximator:
        if config.type == "krbf":
            return KRBF(**config.config.model_dump())
        elif config.type == "fbfcnn":
            return FixedBoundaryFCNN(**config.config.model_dump())
        elif config.type == "fbgru":
            return FixedBoundaryGRU(**config.config.model_dump())
        elif config.type == "hash":
            return HashMapping(**config.config.model_dump())
        else:
            raise ValueError(f"Unsupported type {config.type}")