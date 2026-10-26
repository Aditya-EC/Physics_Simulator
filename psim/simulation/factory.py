"""Building a ready-to-run engine from a scene (shared by the GUI session controller and batch runs)."""
from __future__ import annotations

from ..processes.process_scheduler import Diagnostic, ExecutionPlan, build_plan
from ..recording.recording import Recording
from .conditions import attach_conditions
from .settings import SimulationSettings
from .simulation_engine import SimulationEngine
from .state import capture_bodies
from .world import EngineWorld, capture_frames, capture_static, global_values

NOTHING_TO_SIMULATE = ("Nothing to simulate: assign at least one enabled process to an object "
                       "(Processes ▸ Assign Process).")


def plan_run(scene, settings: SimulationSettings) -> ExecutionPlan:
    """The execution plan of the scene's process assignments plus the settings' stop conditions."""
    plan = build_plan(scene, scene.assignments)
    attach_conditions(plan, scene, settings)
    return plan


def build_engine(scene, settings: SimulationSettings, record: bool | None = None,
                 plan: ExecutionPlan | None = None) -> tuple[ExecutionPlan, SimulationEngine | None, Recording | None]:
    """Plan, capture the scene and create the engine (with a recording when `record`, default: the settings).

    Returns (plan, None, None) when the plan has errors or nothing is simulated (an error diagnostic is
    added for the latter). Raises SimulationError for invalid settings or numerical set-up problems.
    """
    settings.check()
    plan = plan if plan is not None else plan_run(scene, settings)
    if not plan.ok:
        return plan, None, None
    if not plan.body_ids:
        plan.diagnostics.append(Diagnostic("error", NOTHING_TO_SIMULATE))
        return plan, None, None
    bodies, state = capture_bodies(scene, plan.body_ids)
    statics = capture_static(scene)
    world = EngineWorld(statics, plan.body_ids, global_values(scene), settings.delta_time,
                        capture_frames(scene, statics))
    engine = SimulationEngine(SimulationSettings.from_dict(settings.to_dict()), plan, bodies, state, world)
    rec = None
    if settings.record_enabled if record is None else record:
        rec = Recording(int(settings.record_max_mb * 2 ** 20), settings.record_overflow,
                        settings.record_every_n_steps)
        rec.meta["settings"] = settings.to_dict()
        rec.meta["names"] = {o.id: o.name for o in scene.objects.values()}
        rec.meta["types"] = {o.id: o.type_id for o in scene.objects.values()}
        engine.attach_recording(rec, settings.record_process_outputs)
    return plan, engine, rec
