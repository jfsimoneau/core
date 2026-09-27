"""Data update coordinator for the Proximity integration."""

from collections import defaultdict, deque
from dataclasses import dataclass, field
from datetime import datetime
import logging
import math
from typing import cast, override

from homeassistant.components.device_tracker import (
    DOMAIN as DEVICE_TRACKER_DOMAIN,
    DeviceTrackerEntityStateAttribute,
)
from homeassistant.components.person import (
    DOMAIN as PERSON_DOMAIN,
    PersonEntityStateAttribute,
)
from homeassistant.components.zone import (
    DOMAIN as ZONE_DOMAIN,
    ENTITY_ID_HOME,
    ZoneEntityStateAttribute,
)
from homeassistant.config_entries import ConfigEntry
from homeassistant.const import (
    ATTR_NAME,
    CONF_UNIT_OF_MEASUREMENT,
    CONF_ZONE,
    STATE_HOME,
    EntityStateAttribute,
)
from homeassistant.core import (
    CALLBACK_TYPE,
    Event,
    EventStateChangedData,
    HomeAssistant,
    State,
    callback,
)
from homeassistant.helpers import entity_registry as er
from homeassistant.helpers.event import async_call_later
from homeassistant.helpers.issue_registry import IssueSeverity, async_create_issue
from homeassistant.helpers.update_coordinator import DataUpdateCoordinator
from homeassistant.util import dt as dt_util
from homeassistant.util.location import distance as haversine_distance

from .const import (
    ATTR_DIR_OF_TRAVEL,
    ATTR_DIST_TO,
    ATTR_IN_IGNORED_ZONE,
    ATTR_NEAREST,
    ATTR_SPEED,
    CONF_IGNORED_ZONES,
    CONF_TOLERANCE,
    CONF_TRACKED_ENTITIES,
    DEFAULT_DIR_OF_TRAVEL,
    DEFAULT_DIST_TO_ZONE,
    DEFAULT_NEAREST,
    DEFAULT_SPEED,
    DEFAULT_TOLERANCE,
    DOMAIN,
)

_LOGGER = logging.getLogger(__name__)

type ProximityConfigEntry = ConfigEntry[ProximityDataUpdateCoordinator]

# Movement window tuning

POSITION_WINDOW_SIZE: int = 8
"""Number of position samples retained per entity (sliding window)."""

SPEED_THRESHOLD_MS: float = 0.5
"""Default stationary threshold used when no config entry value is present.
The active value is read from CONF_TOLERANCE (m/s) and stored per-coordinator
in self.speed_threshold."""

DOT_THRESHOLD_COS: float = 0.5
"""cos(60 deg) - minimum absolute cosine between v_move and v_zone for the
direction to be resolved as 'towards' or 'away_from'.  Below this threshold
(roughly perpendicular movement, e.g. orbiting) the last valid direction is
preserved instead."""

STALE_THRESHOLD_S: float = 60.0
"""Seconds of GPS silence before a synthetic stationary sample is injected.
With POSITION_WINDOW_SIZE = 8 the speed decays to zero after at most 7
injections (~7 minutes of silence)."""

_EARTH_RADIUS_M: float = 6_371_000.0


# Data structures


@dataclass
class PositionSample:
    """A single GPS position sample with its wall-clock timestamp."""

    timestamp: datetime
    latitude: float
    longitude: float


@dataclass
class EntityMovementState:
    """Per-entity movement state persisted across coordinator refreshes.

    Not exposed directly to sensors; sensors read from ProximityData.
    """

    name: str
    samples: deque[PositionSample] = field(
        default_factory=lambda: deque(maxlen=POSITION_WINDOW_SIZE)
    )
    distance_to_zone: int | None = None
    speed: float | None = None
    direction: str | None = None
    # Preserved when movement is perpendicular to the zone vector.
    last_valid_direction: str | None = None
    in_ignored_zone: bool = False


@dataclass
class ProximityData:
    """ProximityCoordinatorData class."""

    proximity: dict[str, str | int | float | None]
    entities: dict[str, dict[str, str | int | float | None]]


DEFAULT_PROXIMITY_DATA: dict[str, str | int | float | None] = {
    ATTR_DIST_TO: DEFAULT_DIST_TO_ZONE,
    ATTR_DIR_OF_TRAVEL: DEFAULT_DIR_OF_TRAVEL,
    ATTR_NEAREST: DEFAULT_NEAREST,
    ATTR_SPEED: DEFAULT_SPEED,
}


# Zone membership helpers (preserved from original)


def _tracked_in_zones(state: State) -> list[str] | None:
    """Return the zone membership of a tracked entity state.

    Only person and device_tracker entities report zone membership; each
    exposes it under its own platform enum.  Any other domain returns None.
    """
    if state.domain == PERSON_DOMAIN:
        return state.attributes.get(PersonEntityStateAttribute.IN_ZONES)
    if state.domain == DEVICE_TRACKER_DOMAIN:
        return state.attributes.get(DeviceTrackerEntityStateAttribute.IN_ZONES)
    return None


# Coordinator


class ProximityDataUpdateCoordinator(DataUpdateCoordinator[ProximityData]):
    """Proximity data update coordinator.

    Design

    The coordinator is purely event-driven (update_interval=None).  Distance
    is recomputed on every tracked-entity state-change event.  Speed and
    direction are derived from a sliding window of the last POSITION_WINDOW_SIZE
    real GPS samples.

    Speed
        Weighted average of haversine segment speeds across the window.
        Determines stationarity: speed < SPEED_THRESHOLD_MS -> "stationary".

    Direction
        Dot product of v_move (first->last sample, local equirectangular) and
        v_zone (last sample -> zone centre).  Perpendicular movement (|cos a| <=
        DOT_THRESHOLD_COS) preserves the last valid direction, handling the
        "orbiting" case without introducing a new state.

    Decay
        When a real GPS update arrives and speed >= threshold, a one-shot timer
        (async_call_later) is scheduled for STALE_THRESHOLD_S seconds.  On
        fire it injects a synthetic stationary sample and refreshes; if speed
        is still >= threshold it reschedules itself.  The timer is cancelled
        immediately when the next real GPS fix arrives.  This means timers only
        run for moving entities, and stop automatically once the entity is
        stationary.
    """

    config_entry: ProximityConfigEntry

    def __init__(self, hass: HomeAssistant, config_entry: ProximityConfigEntry) -> None:
        """Initialize the Proximity coordinator."""
        self.ignored_zone_ids: list[str] = config_entry.data[CONF_IGNORED_ZONES]
        self.tracked_entities: list[str] = config_entry.data[CONF_TRACKED_ENTITIES]
        self.proximity_zone_id: str = config_entry.data[CONF_ZONE]
        self.unit_of_measurement: str = config_entry.data.get(
            CONF_UNIT_OF_MEASUREMENT, hass.config.units.length_unit
        )
        self.speed_threshold: float = float(
            config_entry.data.get(CONF_TOLERANCE, DEFAULT_TOLERANCE)
        )
        self.entity_mapping: dict[str, list[str]] = defaultdict(list)

        super().__init__(
            hass,
            _LOGGER,
            config_entry=config_entry,
            name=config_entry.title,
            # Event-driven only; decay is handled by async_call_later timers.
            update_interval=None,
        )

        self.data = ProximityData(dict(DEFAULT_PROXIMITY_DATA), {})

        # Per-entity movement history.
        self._movement: dict[str, EntityMovementState] = {}
        # One pending decay timer per entity; value is the cancel callback.
        self._decay_timers: dict[str, CALLBACK_TYPE] = {}

    # Public helpers

    @callback
    def async_add_entity_mapping(self, tracked_entity_id: str, entity_id: str) -> None:
        """Register a tracked-entity -> proximity-sensor mapping."""
        self.entity_mapping[tracked_entity_id].append(entity_id)

    # Event handlers

    async def async_check_proximity_state_change(
        self,
        event: Event[EventStateChangedData],
    ) -> None:
        """Handle a state-changed event for a tracked entity.

        Sequence

        1. Cancel any pending decay timer - a real GPS fix just arrived.
        2. If the new state carries coordinates, record a position sample.
        3. Refresh all sensors.
        4. If the entity is still moving, schedule the next decay timer.
        """
        data = event.data
        entity_id: str = data["entity_id"]
        new_state: State | None = data["new_state"]

        # A real update arrived - cancel the decay timer immediately so we
        # don't inject a stale sample on top of a fresh one.
        self._cancel_decay(entity_id)

        if new_state is not None:
            lat = new_state.attributes.get(EntityStateAttribute.LATITUDE)
            lon = new_state.attributes.get(EntityStateAttribute.LONGITUDE)
            if lat is not None and lon is not None:
                self._add_position_sample(entity_id, float(lat), float(lon))

        await self.async_refresh()

        # Schedule decay only when the entity is actually moving, so timers
        # never pile up for stationary or unavailable entities.
        if entity_id in self._movement:
            speed = self._movement[entity_id].speed or 0.0
            if speed >= self.speed_threshold:
                self._schedule_decay(entity_id)

    async def async_check_tracked_entity_change(
        self, event: Event[er.EventEntityRegistryUpdatedData]
    ) -> None:
        """Handle entity-registry updates for tracked entities."""
        data = event.data
        if data["action"] == "remove":
            self._create_removed_tracked_entity_issue(data["entity_id"])

        if data["action"] == "update" and "entity_id" in data["changes"]:
            old_id: str = data["old_entity_id"]
            new_id: str = data["entity_id"]

            # Migrate movement history and decay timer to the renamed entity.
            if old_id in self._movement:
                self._movement[new_id] = self._movement.pop(old_id)
            if old_id in self._decay_timers:
                self._decay_timers[new_id] = self._decay_timers.pop(old_id)

            self.hass.config_entries.async_update_entry(
                self.config_entry,
                data={
                    **self.config_entry.data,
                    CONF_TRACKED_ENTITIES: [
                        e for e in (*self.tracked_entities, new_id) if e != old_id
                    ],
                },
            )

    # Decay timer management

    def _schedule_decay(self, entity_id: str) -> None:
        """Schedule a decay tick for *entity_id* after STALE_THRESHOLD_S seconds.

        Any previously scheduled timer for this entity is cancelled first so
        only one timer per entity is ever live.
        """
        self._cancel_decay(entity_id)

        @callback
        def _on_decay(_now: datetime) -> None:
            # Remove from dict before creating the task so _cancel_decay called
            # from inside _async_decay_entity (via async_check_proximity_*) is
            # a no-op and doesn't try to cancel an already-fired timer.
            self._decay_timers.pop(entity_id, None)
            self.hass.async_create_task(
                self._async_decay_entity(entity_id),
                name=f"proximity_decay_{entity_id}",
            )

        self._decay_timers[entity_id] = async_call_later(
            self.hass, STALE_THRESHOLD_S, _on_decay
        )
        _LOGGER.debug(
            "%s: decay timer scheduled for %s (%.0f s)",
            self.name,
            entity_id,
            STALE_THRESHOLD_S,
        )

    def _cancel_decay(self, entity_id: str) -> None:
        """Cancel any pending decay timer for *entity_id*."""
        if cancel := self._decay_timers.pop(entity_id, None):
            cancel()

    async def _async_decay_entity(self, entity_id: str) -> None:
        """Inject a synthetic stationary sample and refresh.

        Called by the decay timer.  After refreshing, reschedules itself if
        the weighted-average speed is still above the stationary threshold,
        allowing a gradual decay over multiple STALE_THRESHOLD_S ticks.
        """
        if entity_id not in self._movement:
            return
        mov = self._movement[entity_id]
        if not mov.samples:
            return

        last = mov.samples[-1]
        age = (dt_util.utcnow() - last.timestamp).total_seconds()
        mov.samples.append(
            PositionSample(
                timestamp=dt_util.utcnow(),
                latitude=last.latitude,
                longitude=last.longitude,
            )
        )
        _LOGGER.debug(
            "%s: decay sample injected for %s (%.0f s since last GPS fix)",
            self.name,
            entity_id,
            age,
        )

        await self.async_refresh()

        # Keep decaying until speed drops below the threshold.
        if entity_id in self._movement:
            speed = self._movement[entity_id].speed or 0.0
            if speed >= self.speed_threshold:
                self._schedule_decay(entity_id)

    # Zone membership (preserved from original)

    def _tracked_entity_in_zone(self, zone: State, tracked_entity_state: State) -> bool:
        """Return whether the tracked entity is currently in the proximity zone.

        Modern entity-based trackers and person entities always report zone
        membership authoritatively in the ``in_zones`` attribute (a list of
        zone entity IDs), so a present, empty list genuinely means "in no zone".

        The state-based fallback below is a temporary shim for two deprecated
        producers whose ``in_zones`` cannot be trusted as authoritative:

        - Legacy (non-entity) device trackers omit ``in_zones`` entirely
          (deprecated, removed in HA Core 2027.5).
        - Trackers using the deprecated ``location_name`` report an empty
          ``in_zones`` while their state still names their location
          (deprecated, removed in HA Core 2027.7).

        For both, an empty or absent ``in_zones`` does not imply "in no zone",
        so we fall back to matching the tracked entity state against the zone's
        friendly name, plus an explicit home-zone check.  Once both deprecations
        are gone, ``in_zones`` is authoritative and this method should reduce to
        the membership check alone.
        """
        if in_zones := _tracked_in_zones(tracked_entity_state):
            return zone.entity_id in in_zones

        # Remove once legacy device trackers (2027.5) and location_name (2027.7)
        # are gone; see detailed comment above.
        zone_friendly_name = zone.attributes.get(EntityStateAttribute.FRIENDLY_NAME)
        return (
            zone_friendly_name is not None
            and tracked_entity_state.state.lower() == zone_friendly_name.lower()
        ) or (
            tracked_entity_state.state == STATE_HOME
            and zone.entity_id == ENTITY_ID_HOME
        )

    # Position sample management

    def _add_position_sample(
        self, entity_id: str, latitude: float, longitude: float
    ) -> None:
        """Append a real GPS sample for *entity_id*, lazily creating its state."""
        if entity_id not in self._movement:
            self._movement[entity_id] = EntityMovementState(name="")
        self._movement[entity_id].samples.append(
            PositionSample(
                timestamp=dt_util.utcnow(),
                latitude=latitude,
                longitude=longitude,
            )
        )

    # Distance

    def _calc_distance_to_zone(
        self,
        zone: State,
        tracked_entity_state: State,
        latitude: float | None,
        longitude: float | None,
    ) -> int | None:
        """Return distance from the entity to the zone edge in metres.

        Returns 0 when the entity is inside the zone (using authoritative zone
        membership where available, with a legacy fallback - see
        _tracked_entity_in_zone).  Returns None when no coordinates are
        available and the entity is not inside the zone.
        """
        if self._tracked_entity_in_zone(zone, tracked_entity_state):
            _LOGGER.debug(
                "%s: %s in zone -> distance=0",
                self.name,
                tracked_entity_state.entity_id,
            )
            return 0

        if latitude is None or longitude is None:
            _LOGGER.debug(
                "%s: %s has no coordinates -> distance=None",
                self.name,
                tracked_entity_state.entity_id,
            )
            return None

        distance_to_centre = haversine_distance(
            zone.attributes[EntityStateAttribute.LATITUDE],
            zone.attributes[EntityStateAttribute.LONGITUDE],
            latitude,
            longitude,
        )
        # Zones always have lat/lon, so distance_to_centre is never None here.
        assert distance_to_centre is not None

        zone_radius: float = zone.attributes[ZoneEntityStateAttribute.RADIUS]
        if zone_radius >= distance_to_centre:
            return 0
        return round(distance_to_centre - zone_radius)

    # Speed

    @staticmethod
    def _calc_speed(samples: deque[PositionSample]) -> float | None:
        """Return the weighted average speed (m/s) over the position window.

        Each segment's speed is weighted by the recency of its later endpoint:
            w[i] = 1 / (1 + age_seconds(sample[i]))

        Using actual haversine distance between consecutive positions means
        that slow lateral drifts and GPS jitter both contribute correctly.
        """
        sample_list = list(samples)
        if len(sample_list) < 2:
            return None

        now = sample_list[-1].timestamp
        total_weight = 0.0
        weighted_sum = 0.0

        for i in range(1, len(sample_list)):
            dt = (
                sample_list[i].timestamp - sample_list[i - 1].timestamp
            ).total_seconds()
            if dt <= 0:
                continue

            seg_dist = haversine_distance(
                sample_list[i - 1].latitude,
                sample_list[i - 1].longitude,
                sample_list[i].latitude,
                sample_list[i].longitude,
            )
            if seg_dist is None:
                continue

            age = (now - sample_list[i].timestamp).total_seconds()
            w = 1.0 / (1.0 + age)
            weighted_sum += w * (seg_dist / dt)
            total_weight += w

        return weighted_sum / total_weight if total_weight > 0 else None

    # Direction

    @staticmethod
    def _to_cartesian(
        ref_lat: float, ref_lon: float, lat: float, lon: float
    ) -> tuple[float, float]:
        """Equirectangular projection centred on (ref_lat, ref_lon) -> metres.

        x grows eastward, y grows northward.  Accurate enough for the
        distances (< ~100 km) typical of proximity zones.
        """
        cos_ref = math.cos(math.radians(ref_lat))
        x = math.radians(lon - ref_lon) * cos_ref * _EARTH_RADIUS_M
        y = math.radians(lat - ref_lat) * _EARTH_RADIUS_M
        return x, y

    def _calc_direction(
        self,
        samples: deque[PositionSample],
        zone_lat: float,
        zone_lon: float,
        last_valid: str | None,
    ) -> str | None:
        """Resolve direction using the dot product of v_move and v_zone.

        Both vectors are expressed in a local equirectangular frame centred on
        the newest sample so that longitude distortion is minimised.

        v_move = pos[-1] - pos[0]
            Cumulative displacement over the window.  Equivalent to the vector
            sum of all consecutive segment vectors (telescoping sum), so
            oscillating back-and-forth motion partially cancels out.

        v_zone = zone_centre - pos[-1]
            Points from the current position toward the zone centre.

        Decision

        cos a = (v_move . v_zone) / (|v_move| * |v_zone|)

          cos a > +DOT_THRESHOLD_COS  ->  "towards"
          cos a < -DOT_THRESHOLD_COS  ->  "away_from"
          |cos a| <=  DOT_THRESHOLD_COS  ->  perpendicular (e.g. orbiting);
                                            return *last_valid* unchanged.

        The perpendicular case returning last_valid means an entity that has
        been heading towards the zone and starts orbiting it will keep showing
        "towards" until it actually moves away or arrives.
        """
        sample_list = list(samples)
        if len(sample_list) < 2:
            return last_valid

        last = sample_list[-1]
        first = sample_list[0]
        ref_lat, ref_lon = last.latitude, last.longitude

        # In the local frame last maps to (0, 0), so v_move = (0,0) - first_local.
        x_first, y_first = ProximityDataUpdateCoordinator._to_cartesian(
            ref_lat, ref_lon, first.latitude, first.longitude
        )
        vx_move = -x_first
        vy_move = -y_first

        # v_zone = zone_centre - last (last is the origin).
        vx_zone, vy_zone = ProximityDataUpdateCoordinator._to_cartesian(
            ref_lat, ref_lon, zone_lat, zone_lon
        )

        mag_move = math.hypot(vx_move, vy_move)
        mag_zone = math.hypot(vx_zone, vy_zone)

        if mag_move < 1.0:
            # Net displacement under 1 m - no directional signal.
            return last_valid
        if mag_zone < 1.0:
            # Entity is essentially at the zone centre.
            return last_valid

        cos_theta = (vx_move * vx_zone + vy_move * vy_zone) / (mag_move * mag_zone)

        if abs(cos_theta) <= DOT_THRESHOLD_COS:
            _LOGGER.debug(
                "%s: movement perpendicular to zone (cos a=%.2f) -> keeping '%s'",
                ref_lat,
                cos_theta,
                last_valid,
            )
            return last_valid

        return "towards" if cos_theta > 0 else "away_from"

    # Core update

    @override
    async def _async_update_data(self) -> ProximityData:
        """Recalculate proximity data for every tracked entity.

        Called on every state-change event and on every decay-timer tick.

        Correctness properties

        * entities_data is built from scratch on every run - no partial
          mutation of self.data, so self.data stays fully consistent after
          each successful update even if an exception is raised midway.
        * Distance uses _tracked_entity_in_zone (authoritative in_zones
          attribute with legacy fallback) then raw GPS coordinates.
        * Direction uses the window-based dot-product algorithm, not a
          snapshot of two consecutive states, so it is stable against
          single-sample noise and perpendicular motion.
        """
        if (zone_state := self.hass.states.get(self.proximity_zone_id)) is None:
            _LOGGER.debug(
                "%s: zone %s does not exist -> reset",
                self.name,
                self.proximity_zone_id,
            )
            return ProximityData(dict(DEFAULT_PROXIMITY_DATA), {})

        zone_lat: float = zone_state.attributes[EntityStateAttribute.LATITUDE]
        zone_lon: float = zone_state.attributes[EntityStateAttribute.LONGITUDE]

        # Build fresh - never mutate self.data in place.
        entities_data: dict[str, dict[str, str | int | float | None]] = {}

        for entity_id in self.tracked_entities:
            tracked_state = self.hass.states.get(entity_id)
            if tracked_state is None:
                self._movement.pop(entity_id, None)
                self._cancel_decay(entity_id)
                _LOGGER.debug("%s: %s does not exist -> skipped", self.name, entity_id)
                continue

            # Lazily initialise movement state and seed with current position.
            if entity_id not in self._movement:
                _LOGGER.debug("%s: %s is new -> add", self.name, entity_id)
                self._movement[entity_id] = EntityMovementState(name=tracked_state.name)
                lat = tracked_state.attributes.get(EntityStateAttribute.LATITUDE)
                lon = tracked_state.attributes.get(EntityStateAttribute.LONGITUDE)
                if lat is not None and lon is not None:
                    self._movement[entity_id].samples.append(
                        PositionSample(
                            timestamp=dt_util.utcnow(),
                            latitude=float(lat),
                            longitude=float(lon),
                        )
                    )

            mov = self._movement[entity_id]
            mov.name = tracked_state.name

            # Distance
            lat = tracked_state.attributes.get(EntityStateAttribute.LATITUDE)
            lon = tracked_state.attributes.get(EntityStateAttribute.LONGITUDE)
            dist = self._calc_distance_to_zone(
                zone_state,
                tracked_state,
                float(lat) if lat is not None else None,
                float(lon) if lon is not None else None,
            )
            mov.distance_to_zone = dist

            # Speed
            mov.speed = self._calc_speed(mov.samples)

            # Direction
            if dist == 0:
                direction: str | None = "arrived"
                mov.last_valid_direction = direction
            elif dist is None or mov.speed is None:
                direction = None
            elif mov.speed < self.speed_threshold:
                direction = "stationary"
            else:
                direction = self._calc_direction(
                    mov.samples,
                    zone_lat,
                    zone_lon,
                    mov.last_valid_direction,
                )
                if direction not in (None, "stationary"):
                    mov.last_valid_direction = direction

            mov.direction = direction

            # Ignored-zone flag
            mov.in_ignored_zone = (
                f"{ZONE_DOMAIN}.{tracked_state.state.lower()}" in self.ignored_zone_ids
            )

            entities_data[entity_id] = {
                ATTR_NAME: mov.name,
                ATTR_DIST_TO: mov.distance_to_zone,
                ATTR_DIR_OF_TRAVEL: mov.direction,
                ATTR_SPEED: round(mov.speed, 2) if mov.speed is not None else None,
                ATTR_IN_IGNORED_ZONE: mov.in_ignored_zone,
            }

            _LOGGER.debug(
                "%s: %-40s  dist=%-8s  speed=%5.2f m/s  dir=%s",
                self.name,
                entity_id,
                mov.distance_to_zone,
                mov.speed,
                mov.direction,
            )

        # Legacy proximity sensor (nearest non-ignored entity)
        proximity_data: dict[str, str | int | float | None] = dict(
            DEFAULT_PROXIMITY_DATA
        )

        for entity_data in entities_data.values():
            if entity_data[ATTR_IN_IGNORED_ZONE] or entity_data[ATTR_DIST_TO] is None:
                continue

            current_dist = cast(int, entity_data[ATTR_DIST_TO])

            if isinstance(proximity_data[ATTR_DIST_TO], str):
                # First eligible entity (sentinel is a str "not set").
                _LOGGER.debug("set first entity_data: %s", entity_data)
                proximity_data = {
                    ATTR_DIST_TO: current_dist,
                    ATTR_DIR_OF_TRAVEL: entity_data[ATTR_DIR_OF_TRAVEL],
                    ATTR_NEAREST: str(entity_data[ATTR_NAME]),
                    ATTR_SPEED: entity_data[ATTR_SPEED],
                }
                continue

            nearest_dist = cast(int, proximity_data[ATTR_DIST_TO])

            if nearest_dist > current_dist:
                _LOGGER.debug("set closer entity_data: %s", entity_data)
                proximity_data = {
                    ATTR_DIST_TO: current_dist,
                    ATTR_DIR_OF_TRAVEL: entity_data[ATTR_DIR_OF_TRAVEL],
                    ATTR_NEAREST: str(entity_data[ATTR_NAME]),
                    ATTR_SPEED: entity_data[ATTR_SPEED],
                }
            elif nearest_dist == current_dist:
                _LOGGER.debug("set equally close entity_data: %s", entity_data)
                proximity_data[ATTR_NEAREST] = (
                    f"{proximity_data[ATTR_NEAREST]}, {entity_data[ATTR_NAME]!s}"
                )

        return ProximityData(proximity_data, entities_data)

    # Lifecycle

    @override
    async def async_shutdown(self) -> None:
        """Cancel all pending decay timers before shutting down."""
        for entity_id in list(self._decay_timers):
            self._cancel_decay(entity_id)
        await super().async_shutdown()

    # Issue reporting

    def _create_removed_tracked_entity_issue(self, entity_id: str) -> None:
        """Create a repair issue for a removed tracked entity."""
        async_create_issue(
            self.hass,
            DOMAIN,
            f"tracked_entity_removed_{entity_id}",
            is_fixable=True,
            is_persistent=True,
            severity=IssueSeverity.WARNING,
            translation_key="tracked_entity_removed",
            translation_placeholders={"entity_id": entity_id, "name": self.name},
        )
