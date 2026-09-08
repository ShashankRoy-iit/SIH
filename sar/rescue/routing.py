"""Safe Ground & Amphibious Access Route Planning for SAR Operations.

Generates safe, obstacle-avoidant transit corridors from incident base / staging
areas to identified survivor locations, accounting for flood depth, terrain
slope, fire plumes, powerlines, and structural collapse zones.
"""

from __future__ import annotations

import heapq
import math
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

from sar.core.geo import GeoPoint, local_to_wgs84


class RescueTeamType(str, Enum):
    """Operational modality of the ground / water rescue unit."""

    FOOT_RESCUE_TEAM = "foot_team"            # Search & Rescue squad on foot
    AMPHIBIOUS_RESCUE_BOAT = "rescue_boat"     # Motorized zodiac / flood rescue boat
    OFFROAD_4X4_VEHICLE = "offroad_4x4"       # High-clearance rescue vehicle
    ALL_TERRAIN_UGV = "all_terrain_ugv"       # Autonomous tracked ground robot


@dataclass
class TeamCapability:
    """Operational constraints for a specific rescue asset."""

    team_type: RescueTeamType
    nominal_speed_ms: float
    max_wading_depth_m: float
    min_water_depth_m: float       # For boats
    max_slope_deg: float
    rubble_passable: bool
    label: str


TEAM_CAPABILITIES: Dict[RescueTeamType, TeamCapability] = {
    RescueTeamType.FOOT_RESCUE_TEAM: TeamCapability(
        team_type=RescueTeamType.FOOT_RESCUE_TEAM,
        nominal_speed_ms=1.2,
        max_wading_depth_m=0.5,
        min_water_depth_m=0.0,
        max_slope_deg=28.0,
        rubble_passable=True,
        label="SAR Foot Response Team",
    ),
    RescueTeamType.AMPHIBIOUS_RESCUE_BOAT: TeamCapability(
        team_type=RescueTeamType.AMPHIBIOUS_RESCUE_BOAT,
        nominal_speed_ms=4.5,
        max_wading_depth_m=10.0,
        min_water_depth_m=0.25,
        max_slope_deg=4.0,
        rubble_passable=False,
        label="Amphibious Rescue Zodiac",
    ),
    RescueTeamType.OFFROAD_4X4_VEHICLE: TeamCapability(
        team_type=RescueTeamType.OFFROAD_4X4_VEHICLE,
        nominal_speed_ms=7.0,
        max_wading_depth_m=0.4,
        min_water_depth_m=0.0,
        max_slope_deg=18.0,
        rubble_passable=False,
        label="4x4 Emergency Response Vehicle",
    ),
    RescueTeamType.ALL_TERRAIN_UGV: TeamCapability(
        team_type=RescueTeamType.ALL_TERRAIN_UGV,
        nominal_speed_ms=2.2,
        max_wading_depth_m=0.6,
        min_water_depth_m=0.0,
        max_slope_deg=35.0,
        rubble_passable=True,
        label="Autonomous Tracked UGV",
    ),
}


@dataclass
class RouteWaypoint:
    """One waypoint along a computed rescue route."""

    north_m: float
    east_m: float
    terrain_elevation_m: float = 0.0
    water_depth_m: float = 0.0
    cumulative_distance_m: float = 0.0
    estimated_arrival_s: float = 0.0
    hazard_warning: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "north_m": round(self.north_m, 1),
            "east_m": round(self.east_m, 1),
            "elevation_m": round(self.terrain_elevation_m, 1),
            "water_depth_m": round(self.water_depth_m, 2),
            "cumulative_distance_m": round(self.cumulative_distance_m, 1),
            "eta_s": round(self.estimated_arrival_s, 1),
            "hazard_warning": self.hazard_warning,
        }


@dataclass
class RescueRoute:
    """Complete turn-by-turn safe navigation corridor for rescue response."""

    route_id: str
    target_id: int
    team_type: RescueTeamType
    start_point_ned: Tuple[float, float, float]
    target_point_ned: Tuple[float, float, float]
    waypoints: List[RouteWaypoint] = field(default_factory=list)
    total_distance_m: float = 0.0
    estimated_transit_time_s: float = 0.0
    safety_score: float = 1.0           # 0..1 (1 = fully clear of hazards)
    is_feasible: bool = True
    warnings: List[str] = field(default_factory=list)

    @property
    def estimated_eta_s(self) -> float:
        return self.estimated_transit_time_s

    def to_dict(self) -> Dict[str, Any]:
        return {
            "route_id": self.route_id,
            "target_id": self.target_id,
            "team_type": self.team_type.value,
            "start_ned": [round(x, 1) for x in self.start_point_ned],
            "target_ned": [round(x, 1) for x in self.target_point_ned],
            "total_distance_m": round(self.total_distance_m, 1),
            "estimated_transit_time_s": round(self.estimated_transit_time_s, 1),
            "estimated_transit_time_min": round(self.estimated_transit_time_s / 60.0, 1),
            "safety_score": round(self.safety_score, 2),
            "is_feasible": self.is_feasible,
            "waypoints_count": len(self.waypoints),
            "waypoints": [wp.to_dict() for wp in self.waypoints],
            "warnings": list(self.warnings),
        }


class GroundRescueRouter:
    """A* Graph search routing engine for ground and water rescue deployments."""

    def __init__(self, cell_size_m: float = 5.0) -> None:
        self.cell_size_m = float(cell_size_m)

    def plan_route(
        self,
        start_ned: Tuple[float, float, float],
        target_ned: Tuple[float, float, float],
        team_type: RescueTeamType,
        world: Optional[Any] = None,
        hazard_map: Optional[Any] = None,
        target_id: int = 0,
    ) -> RescueRoute:
        """Find the optimal safe route from start to target."""
        cap = TEAM_CAPABILITIES.get(team_type, TEAM_CAPABILITIES[RescueTeamType.FOOT_RESCUE_TEAM])
        sn, se = float(start_ned[0]), float(start_ned[1])
        tn, te = float(target_ned[0]), float(target_ned[1])

        # Grid bounds
        margin = 60.0
        min_n = min(sn, tn) - margin
        max_n = max(sn, tn) + margin
        min_e = min(se, te) - margin
        max_e = max(se, te) + margin

        res = self.cell_size_m
        rows = int(math.ceil((max_n - min_n) / res)) + 1
        cols = int(math.ceil((max_e - min_e) / res)) + 1

        def coord_to_cell(n: float, e: float) -> Tuple[int, int]:
            r = int(np.clip(round((n - min_n) / res), 0, rows - 1))
            c = int(np.clip(round((e - min_e) / res), 0, cols - 1))
            return r, c

        def cell_to_coord(r: int, c: int) -> Tuple[float, float]:
            return min_n + r * res, min_e + c * res

        start_cell = coord_to_cell(sn, se)
        target_cell = coord_to_cell(tn, te)

        # Pre-sample fields for the search box
        rr, cc = np.meshgrid(np.arange(rows), np.arange(cols), indexing="ij")
        grid_n = min_n + rr * res
        grid_e = min_e + cc * res

        # Terrain & Water
        if world is not None:
            elev = np.nan_to_num(world.terrain_z(grid_n, grid_e))
            water = np.nan_to_num(world.water_depth(grid_n, grid_e))
            struct = np.nan_to_num(world.structure_height(grid_n, grid_e))
            debris = np.nan_to_num(world.debris_density(grid_n, grid_e))
        else:
            elev = np.zeros((rows, cols), dtype=np.float32)
            water = np.zeros((rows, cols), dtype=np.float32)
            struct = np.zeros((rows, cols), dtype=np.float32)
            debris = np.zeros((rows, cols), dtype=np.float32)

        # Hazards from HazardMap
        if hazard_map is not None:
            # query hazard severity
            haz_cost = np.zeros((rows, cols), dtype=np.float32)
            for r in range(rows):
                for c in range(cols):
                    q = hazard_map.query(grid_n[r, c], grid_e[r, c])
                    haz_cost[r, c] = float(q.get("severity", 0.0))
        else:
            haz_cost = np.zeros((rows, cols), dtype=np.float32)

        # A* Search Priority Queue
        # (f_cost, g_cost, (r, c))
        open_set = []
        heapq.heappush(open_set, (0.0, 0.0, start_cell))
        came_from: Dict[Tuple[int, int], Tuple[int, int]] = {}
        g_scores = {start_cell: 0.0}

        # 8-connected neighbors
        neighbors = [
            (-1, 0, 1.0), (1, 0, 1.0), (0, -1, 1.0), (0, 1, 1.0),
            (-1, -1, 1.414), (-1, 1, 1.414), (1, -1, 1.414), (1, 1, 1.414),
        ]

        found = False
        warnings = []

        while open_set:
            f, g, current = heapq.heappop(open_set)
            if current == target_cell:
                found = True
                break

            if g > g_scores.get(current, float("inf")):
                continue

            cr, cc_idx = current
            curr_n, curr_e = cell_to_coord(cr, cc_idx)

            for dr, dc, dist_mult in neighbors:
                nr, nc = cr + dr, cc_idx + dc
                if not (0 <= nr < rows and 0 <= nc < cols):
                    continue

                w_depth = float(water[nr, nc])
                s_height = float(struct[nr, nc])
                d_level = float(debris[nr, nc])
                h_sev = float(haz_cost[nr, nc])

                # Capability gating
                if team_type == RescueTeamType.AMPHIBIOUS_RESCUE_BOAT:
                    if w_depth < cap.min_water_depth_m and (nr, nc) != target_cell:
                        continue  # Boat cannot travel on dry land
                else:
                    if w_depth > cap.max_wading_depth_m and (nr, nc) != target_cell:
                        continue  # Too deep for ground unit

                if s_height > 1.5 and not cap.rubble_passable:
                    continue  # Impassable structure

                # Movement cost calculation
                base_step = res * dist_mult
                speed_factor = 1.0

                if team_type != RescueTeamType.AMPHIBIOUS_RESCUE_BOAT:
                    if w_depth > 0.1:
                        speed_factor *= max(0.2, 1.0 - w_depth / max(cap.max_wading_depth_m, 0.1))
                    if d_level > 0.3:
                        speed_factor *= 0.6
                else:
                    if w_depth >= 1.0:
                        speed_factor = 1.2  # Faster in deep water

                cost_mult = 1.0 / max(speed_factor, 0.1)
                cost_mult += h_sev * 4.0  # Heavy penalty for hazardous zones

                tentative_g = g + base_step * cost_mult

                next_node = (nr, nc)
                if tentative_g < g_scores.get(next_node, float("inf")):
                    g_scores[next_node] = tentative_g
                    came_from[next_node] = current
                    h_dist = math.hypot(nr - target_cell[0], nc - target_cell[1]) * res
                    heapq.heappush(open_set, (tentative_g + h_dist, tentative_g, next_node))

        # Reconstruct path
        path_cells = []
        if found:
            curr = target_cell
            while curr in came_from:
                path_cells.append(curr)
                curr = came_from[curr]
            path_cells.append(start_cell)
            path_cells.reverse()
        else:
            # Fallback direct straight line path if blocked
            warnings.append(f"Direct corridor planned: obstacles or deep water detected for {cap.label}")
            path_cells = [start_cell, target_cell]

        # Build Waypoints
        waypoints: List[RouteWaypoint] = []
        cum_dist = 0.0
        cum_time = 0.0
        prev_pt = None

        # Downsample waypoints to reduce packet size
        step = max(1, len(path_cells) // 16)
        sampled_cells = path_cells[::step]
        if path_cells and path_cells[-1] not in sampled_cells:
            sampled_cells.append(path_cells[-1])

        total_haz = 0.0
        for r, c in sampled_cells:
            wn, we = cell_to_coord(r, c)
            w_elev = float(elev[r, c])
            w_water = float(water[r, c])
            w_haz = float(haz_cost[r, c])
            total_haz += w_haz

            if prev_pt is not None:
                d = math.hypot(wn - prev_pt[0], we - prev_pt[1])
                cum_dist += d
                speed = cap.nominal_speed_ms
                if w_water > 0.1 and team_type != RescueTeamType.AMPHIBIOUS_RESCUE_BOAT:
                    speed *= 0.5
                cum_time += d / max(speed, 0.2)

            prev_pt = (wn, we)
            warn = None
            if w_haz >= 3.0:
                warn = "Passing near active critical hazard zone"
            elif w_water > 0.4 and team_type != RescueTeamType.AMPHIBIOUS_RESCUE_BOAT:
                warn = f"Wading water depth {w_water:.1f} m"

            waypoints.append(RouteWaypoint(
                north_m=wn,
                east_m=we,
                terrain_elevation_m=w_elev,
                water_depth_m=w_water,
                cumulative_distance_m=cum_dist,
                estimated_arrival_s=cum_time,
                hazard_warning=warn,
            ))

        safety_score = float(np.clip(1.0 - total_haz / max(len(waypoints) * 3.0, 1.0), 0.1, 1.0))

        return RescueRoute(
            route_id=f"route_{target_id}_{team_type.value}",
            target_id=target_id,
            team_type=team_type,
            start_point_ned=(sn, se, float(start_ned[2])),
            target_point_ned=(tn, te, float(target_ned[2])),
            waypoints=waypoints,
            total_distance_m=cum_dist,
            estimated_transit_time_s=cum_time,
            safety_score=safety_score,
            is_feasible=found,
            warnings=warnings,
        )
