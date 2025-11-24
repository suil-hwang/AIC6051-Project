"""Neural A* search
Author: Ryo Yonetani
Affiliation: OSX
"""
from __future__ import annotations

from functools import partial

import torch
import torch.nn as nn

from . import encoder
from .differentiable_astar import AstarOutput, DifferentiableAstar
from .pq_astar import pq_astar


class VanillaAstar(nn.Module):
    def __init__(
        self,
        g_ratio: float = 0.5,
        use_differentiable_astar: bool = True,
    ):
        """
        Vanilla A* search

        Args:
            g_ratio (float, optional): ratio between g(v) + h(v). Set 0 to perform as best-first search. Defaults to 0.5.
            use_differentiable_astar (bool, optional): if the differentiable A* is used instead of standard A*. Defaults to True.

        Examples:
            >>> planner = VanillaAstar()
            >>> outputs = planner(map_designs, start_maps, goal_maps)
            >>> histories = outputs.histories
            >>> paths = outputs.paths

        Note:
            For perform inference on a large map, set use_differentiable_astar = False to peform a faster A* with priority queue
        """

        super().__init__()
        self.astar = DifferentiableAstar(
            g_ratio=g_ratio,
            Tmax=1.0,
        )
        self.g_ratio = g_ratio
        self.use_differentiable_astar = use_differentiable_astar

    def perform_astar(
        self,
        map_designs: torch.tensor,
        start_maps: torch.tensor,
        goal_maps: torch.tensor,
        obstacles_maps: torch.tensor,
        store_intermediate_results: bool = False,
    ) -> AstarOutput:

        astar = (
            self.astar
            if self.use_differentiable_astar
            else partial(pq_astar, g_ratio=self.g_ratio)
        )

        astar_outputs = astar(
            map_designs,
            start_maps,
            goal_maps,
            obstacles_maps,
            store_intermediate_results,
        )

        return astar_outputs

    def forward(
        self,
        map_designs: torch.tensor,
        start_maps: torch.tensor,
        goal_maps: torch.tensor,
        store_intermediate_results: bool = False,
    ) -> AstarOutput:
        """
        Perform A* search

        Args:
            map_designs (torch.tensor): map designs (obstacle maps or raw image)
            start_maps (torch.tensor): start maps indicating the start location with one-hot binary map
            goal_maps (torch.tensor): goal maps indicating the goal location with one-hot binary map
            store_intermediate_results (bool, optional): If the intermediate search results are stored in Astar output. Defaults to False.

        Returns:
            AstarOutput: search histories and solution paths, and optionally intermediate search results.
        """

        cost_maps = map_designs
        obstacles_maps = map_designs

        return self.perform_astar(
            cost_maps,
            start_maps,
            goal_maps,
            obstacles_maps,
            store_intermediate_results,
        )


class NeuralAstar(VanillaAstar):
    def __init__(
        self,
        g_ratio: float = 0.5,
        Tmax: float = 1.0,
        encoder_input: str = "m+",
        encoder_arch: str = "CNN",
        encoder_depth: int = 4,
        learn_obstacles: bool = False,
        const: float = None,
        use_differentiable_astar: bool = True,
    ):
        """
        Neural A* search

        Args:
            g_ratio (float, optional): ratio between g(v) + h(v). Set 0 to perform as best-first search. Defaults to 0.5.
            Tmax (float, optional): how much of the map the model explores during training. Set a small value (0.25) when training the model. Defaults to 1.0.
            encoder_input (str, optional): input format. Set "m+" to use the concatenation of map_design and (start_map + goal_map). Set "m" to use map_design only. Defaults to "m+".
            encoder_arch (str, optional): encoder architecture. Defaults to "CNN".
            encoder_depth (int, optional): depth of the encoder. Defaults to 4.
            learn_obstacles (bool, optional): if the obstacle is invisible to the model. Defaults to False.
            const (float, optional): learnable weight to be multiplied for h(v). Defaults to None.
            use_differentiable_astar (bool, optional): if the differentiable A* is used instead of standard A*. Defaults to True.

        Examples:
            >>> planner = NeuralAstar()
            >>> outputs = planner(map_designs, start_maps, goal_maps)
            >>> histories = outputs.histories
            >>> paths = outputs.paths

        Note:
            For perform inference on a large map, set use_differentiable_astar = False to peform a faster A* with priority queue
        """

        super().__init__()
        self.astar = DifferentiableAstar(
            g_ratio=g_ratio,
            Tmax=Tmax,
        )
        self.encoder_input = encoder_input
        encoder_arch = getattr(encoder, encoder_arch)
        self.encoder = encoder_arch(len(self.encoder_input), encoder_depth, const)
        self.learn_obstacles = learn_obstacles
        if self.learn_obstacles:
            print("WARNING: learn_obstacles has been set to True")
        self.g_ratio = g_ratio
        self.use_differentiable_astar = use_differentiable_astar

    def encode(
        self,
        map_designs: torch.tensor,
        start_maps: torch.tensor,
        goal_maps: torch.tensor,
    ) -> torch.tensor:
        """
        Encode the input problem

        Args:
            map_designs (torch.tensor): map designs (obstacle maps or raw image)
            start_maps (torch.tensor): start maps indicating the start location with one-hot binary map
            goal_maps (torch.tensor): goal maps indicating the goal location with one-hot binary map

        Returns:
            torch.tensor: predicted cost maps
        """
        inputs = map_designs
        if "+" in self.encoder_input:
            if map_designs.shape[-1] == start_maps.shape[-1]:
                inputs = torch.cat((inputs, start_maps + goal_maps), dim=1)
            else:
                upsampler = nn.UpsamplingNearest2d(map_designs.shape[-2:])
                inputs = torch.cat((inputs, upsampler(start_maps + goal_maps)), dim=1)
        cost_maps = self.encoder(inputs)

        return cost_maps

    def forward(
        self,
        map_designs: torch.tensor,
        start_maps: torch.tensor,
        goal_maps: torch.tensor,
        store_intermediate_results: bool = False,
    ) -> AstarOutput:
        """
        Perform neural A* search

        Args:
            map_designs (torch.tensor): map designs (obstacle maps or raw image)
            start_maps (torch.tensor): start maps indicating the start location with one-hot binary map
            goal_maps (torch.tensor): goal maps indicating the goal location with one-hot binary map
            store_intermediate_results (bool, optional): If the intermediate search results are stored in Astar output. Defaults to False.

        Returns:
            AstarOutput: search histories and solution paths, and optionally intermediate search results.
        """

        cost_maps = self.encode(map_designs, start_maps, goal_maps)
        obstacles_maps = (
            map_designs if not self.learn_obstacles else torch.ones_like(start_maps)
        )

        return self.perform_astar(
            cost_maps,
            start_maps,
            goal_maps,
            obstacles_maps,
            store_intermediate_results,
        )

class NeuralAstarField(VanillaAstar):
    def __init__(
        self,
        g_ratio: float = 0.5,
        Tmax: float = 1.0,
        encoder_input: str = "m+",
        encoder_arch: str = "CNN",
        encoder_depth: int = 4,
        learn_obstacles: bool = False,
        const: float = None,
        use_differentiable_astar: bool = True,
        # Distance field settings
        use_geodesic_loss: bool = True,
        use_obstacle_loss: bool = True,
        geodesic_loss_weight: float = 0.5,
        obstacle_loss_weight: float = 0.3,
        smooth_l1_beta: float = 1.0,
        use_log_space: bool = True,
    ):
        """
        Neural A* with Distance Field supervision (Improved)
        
        Args:
            smooth_l1_beta: Beta parameter for Smooth L1 loss
            use_log_space: Use log(1+x) transform for distance fields
        """
        super().__init__()
        self.astar = DifferentiableAstar(g_ratio=g_ratio, Tmax=Tmax)
        self.encoder_input = encoder_input
        
        # Shared encoder
        # NeuralAstarField always uses 3 input channels: [map_designs, start_maps, goal_maps]
        encoder_class = getattr(encoder, encoder_arch)
        input_channels = 3  # Always M + S + G for field-based model
        self.shared_encoder = encoder_class(
            input_channels, 
            encoder_depth, 
            const=None  # No const for shared features
        )
        
        # Infer feature dimension from the actual encoder output to avoid channel mismatches
        feature_dim = self._infer_feature_dim(self.shared_encoder.model)
        
        # Multi-head: 1x1 convolutions
        self.cost_head = nn.Sequential(
            nn.Conv2d(feature_dim, 1, 1),
            nn.Sigmoid(),
        )
        
        if use_geodesic_loss:
            self.geodesic_head = nn.Sequential(
                nn.Conv2d(feature_dim, 1, 1),
                nn.ReLU(),  # Distance >= 0
            )
        else:
            self.geodesic_head = None
        
        if use_obstacle_loss:
            self.obstacle_head = nn.Sequential(
                nn.Conv2d(feature_dim, 1, 1),
                nn.ReLU(),  # Distance >= 0
            )
        else:
            self.obstacle_head = None
        
        # Apply const if provided
        if const is not None:
            self.const = nn.Parameter(torch.ones(1) * const)
        else:
            self.const = 1.0
        
        self.learn_obstacles = learn_obstacles
        self.g_ratio = g_ratio
        self.use_differentiable_astar = use_differentiable_astar
        
        self.use_geodesic_loss = use_geodesic_loss
        self.use_obstacle_loss = use_obstacle_loss
        self.geodesic_loss_weight = geodesic_loss_weight
        self.obstacle_loss_weight = obstacle_loss_weight
        self.smooth_l1_beta = smooth_l1_beta
        self.use_log_space = use_log_space

    def encode(
        self,
        map_designs: torch.tensor,
        start_maps: torch.tensor,
        goal_maps: torch.tensor,
    ) -> tuple:
        """
        Encode input and predict cost map + distance fields
        
        Returns:
            tuple: (cost_maps, geodesic_pred, obstacle_pred)
        """
        # Prepare input: [M, S, G] (3 channels)
        if map_designs.shape[-1] == start_maps.shape[-1]:
            # Same resolution
            inputs = torch.cat((map_designs, start_maps, goal_maps), dim=1)
        else:
            # Need upsampling
            upsampler = nn.UpsamplingNearest2d(map_designs.shape[-2:])
            inputs = torch.cat(
                (map_designs, upsampler(start_maps), upsampler(goal_maps)), 
                dim=1
            )
        
        # Shared feature extraction
        features = self.shared_encoder.model(inputs)  # [B, C, H, W]
        
        # Multi-head prediction
        cost_maps = self.cost_head(features) * self.const
        
        geodesic_pred = None
        if self.geodesic_head is not None:
            geodesic_pred = self.geodesic_head(features)
        
        obstacle_pred = None
        if self.obstacle_head is not None:
            obstacle_pred = self.obstacle_head(features)
        
        return cost_maps, geodesic_pred, obstacle_pred

    def forward(
        self,
        map_designs: torch.tensor,
        start_maps: torch.tensor,
        goal_maps: torch.tensor,
        store_intermediate_results: bool = False,
    ) -> tuple:
        """
        Returns:
            tuple: (AstarOutput, geodesic_pred, obstacle_pred)
        """
        cost_maps, geodesic_pred, obstacle_pred = self.encode(
            map_designs, start_maps, goal_maps
        )
        
        obstacles_maps = (
            map_designs if not self.learn_obstacles else torch.ones_like(start_maps)
        )
        
        astar_output = self.perform_astar(
            cost_maps,
            start_maps,
            goal_maps,
            obstacles_maps,
            store_intermediate_results,
        )
        
        return astar_output, geodesic_pred, obstacle_pred

    @staticmethod
    def _infer_feature_dim(encoder_model: nn.Module) -> int:
        """
        Determine the number of channels produced by the encoder model.

        Looks for the last BatchNorm2d or Conv2d layer, which carry the output
        channel count for Sequential-based encoders.
        """
        for module in reversed(list(encoder_model.modules())):
            if isinstance(module, nn.BatchNorm2d):
                return module.num_features
            if isinstance(module, nn.Conv2d):
                return module.out_channels
        raise ValueError("Unable to determine encoder output channels")
