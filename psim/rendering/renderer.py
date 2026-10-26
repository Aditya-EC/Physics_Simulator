"""Frame renderer: draws the scene with OpenGL 3.3 core.

Pass order: shadow map -> background -> opaque surfaces -> ground shadow ->
grids -> transparent surfaces -> edges / axes / overlays -> placement ghost
-> gizmo (always on top). Text overlays are painted afterwards by the
viewport using QPainter.
"""
from __future__ import annotations

import math
import time
from dataclasses import dataclass, field

import numpy as np
from OpenGL import GL

from .. import numerics
from ..scene import geometry as g
from ..scene.scene_manager import Scene, SceneError
from . import coordinate_axes as axes_geo
from . import physics_overlay
from . import grid as grid_geo
from . import shaders
from .camera import OrbitCamera
from .gizmo import HOVER_COLOR, Handle
from .gl_resources import GpuCache, LineBuffer, MeshBuffer, ShaderProgram, ShadowMap
from .vector_arrows import ARROW_COLORS, arrow_lines, arrow_vectors
from .view_settings import ViewSettings


@dataclass
class GizmoDraw:
    handles: list[Handle]
    model: np.ndarray
    hover: str | None = None
    active: str | None = None


@dataclass
class GhostDraw:
    mesh: g.Mesh
    model: np.ndarray
    color: tuple[float, float, float] = (0.45, 0.70, 1.0)


@dataclass
class FrameInput:
    scene: Scene
    view: ViewSettings
    camera: OrbitCamera
    width_px: int
    height_px: int
    default_fbo: int
    hover_id: str | None = None
    gizmo: GizmoDraw | None = None
    trajectories: list = field(default_factory=list)   # TrajectoryPath, built from the recording
    contacts: list = field(default_factory=list)       # (point, normal, force) of the running simulation
    ghost: GhostDraw | None = None
    dpr: float = 1.0


@dataclass
class FrameStats:
    objects_drawn: int = 0
    triangles: int = 0
    draw_calls: int = 0
    cpu_ms: float = 0.0
    shadow: bool = False


@dataclass
class Renderer:
    gpu: GpuCache = field(default_factory=GpuCache)
    ready: bool = False
    error: str = ""
    gl_info: dict = field(default_factory=dict)

    def initialize(self) -> None:
        try:
            self.mesh_prog = ShaderProgram(shaders.MESH_VS, shaders.MESH_FS)
            self.depth_prog = ShaderProgram(shaders.DEPTH_VS, shaders.DEPTH_FS)
            self.line_prog = ShaderProgram(shaders.LINE_VS, shaders.LINE_FS, shaders.LINE_GS)
            self.bg_prog = ShaderProgram(shaders.BACKGROUND_VS, shaders.BACKGROUND_FS)
            self.catcher_prog = ShaderProgram(shaders.MESH_VS, shaders.SHADOW_CATCHER_FS)
        except Exception as exc:  # pragma: no cover - driver dependent
            self.error = str(exc)
            return
        self.empty_vao = GL.glGenVertexArrays(1)
        self.lines = LineBuffer()
        self.shadow = ShadowMap(2048)
        self.gl_info = {
            "vendor": _gl_str(GL.GL_VENDOR), "renderer": _gl_str(GL.GL_RENDERER),
            "version": _gl_str(GL.GL_VERSION), "glsl": _gl_str(GL.GL_SHADING_LANGUAGE_VERSION),
        }
        self._grid_cache: dict = {}
        self._ground_mesh = None
        self._origin_marker = g.sphere(1.0, 24, 12)
        self.ready = True

    def shutdown(self) -> None:
        if not self.ready:
            return
        self.gpu.release_all()
        self.lines.delete()
        self.shadow.delete()
        for p in (self.mesh_prog, self.depth_prog, self.line_prog, self.bg_prog, self.catcher_prog):
            p.delete()
        self.ready = False

    # ------------------------------------------------------------------------------------------
    def render(self, f: FrameInput) -> FrameStats:
        stats = FrameStats()
        if not self.ready:
            return stats
        t0 = time.perf_counter()
        self._dpr = max(f.dpr, 1.0)
        self.gpu.collect()
        scene, view, cam, env = f.scene, f.view, f.camera, f.view.environment
        w, h = max(f.width_px, 1), max(f.height_px, 1)
        aspect = w / h

        visible = [o for o in scene.objects.values() if scene.is_effectively_visible(o.id)]
        models = {o.id: scene.world_matrix(o.id) for o in visible}
        bounds = scene.world_bounds([o.id for o in visible])
        if bounds is not None:
            lo, hi = bounds
            center = (lo + hi) / 2
            radius = max(float(np.linalg.norm(hi - lo)) / 2, 1.0)
        else:
            center, radius = np.zeros(3), 1.0
        cam.scene_radius = max(radius + float(np.linalg.norm(center)), view.grids["XY"].extent)
        viewproj = cam.view_projection(aspect)
        eye = cam.eye()

        # ---- light ---------------------------------------------------------------------------
        az, el = math.radians(env.light_azimuth_deg), math.radians(env.light_elevation_deg)
        light_dir = np.array([math.cos(el) * math.cos(az), math.cos(el) * math.sin(az), math.sin(el)])
        shadow_on = env.shadows and self.shadow.complete and bool(visible)
        light_vp = np.eye(4)
        if shadow_on:
            r = radius * 1.8 + 1.0
            lc = center.copy()
            lc[2] = max(lc[2] * 0.5, 0.0) if env.ground_shadow else lc[2]
            lview = np.asarray(numerics.look_at(lc + light_dir * r * 2, lc, [0, 0, 1] if abs(light_dir[2]) < 0.99
                                                else [0, 1, 0]))
            lproj = np.asarray(numerics.orthographic(-r, r, -r, r, 0.01, r * 4))
            light_vp = lproj @ lview
            GL.glBindFramebuffer(GL.GL_FRAMEBUFFER, self.shadow.fbo)
            GL.glViewport(0, 0, self.shadow.size, self.shadow.size)
            GL.glClear(GL.GL_DEPTH_BUFFER_BIT)
            GL.glEnable(GL.GL_DEPTH_TEST)
            GL.glDisable(GL.GL_CULL_FACE)
            GL.glEnable(GL.GL_POLYGON_OFFSET_FILL)
            GL.glPolygonOffset(2.0, 4.0)
            self.depth_prog.use()
            for o in visible:
                if not o.show_surface or o.opacity < 0.35 or o.wireframe:
                    continue
                self.depth_prog.mat4("uLightMVP", light_vp @ models[o.id])
                self.gpu.mesh(o.mesh()).draw()
                stats.draw_calls += 1
            GL.glDisable(GL.GL_POLYGON_OFFSET_FILL)
            stats.shadow = True
        GL.glBindFramebuffer(GL.GL_FRAMEBUFFER, f.default_fbo)
        GL.glViewport(0, 0, w, h)

        # ---- background -------------------------------------------------------------------------
        GL.glDisable(GL.GL_CULL_FACE)
        GL.glClearColor(*env.background_bottom, 1.0)
        GL.glClear(GL.GL_COLOR_BUFFER_BIT | GL.GL_DEPTH_BUFFER_BIT)
        GL.glDisable(GL.GL_DEPTH_TEST)
        self.bg_prog.use()
        self.bg_prog.vec3("uTop", env.background_top)
        self.bg_prog.vec3("uBottom", env.background_bottom)
        GL.glBindVertexArray(self.empty_vao)
        GL.glDrawArrays(GL.GL_TRIANGLES, 0, 3)
        GL.glEnable(GL.GL_DEPTH_TEST)
        GL.glDepthFunc(GL.GL_LEQUAL)

        # ---- mesh program common uniforms ----------------------------------------------------------
        mp = self.mesh_prog
        mp.use()
        mp.mat4("uViewProj", viewproj)
        mp.mat4("uLightVP", light_vp)
        mp.vec3("uCamPos", eye)
        mp.vec3("uLightDir", light_dir)
        mp.f("uLightIntensity", env.light_intensity)
        mp.f("uAmbient", env.ambient)
        mp.vec3("uSkyColor", (0.80, 0.84, 0.92))
        mp.vec3("uGroundColor", (0.36, 0.34, 0.33))
        mp.f("uEnvStrength", 0.85)
        mp.i("uShadows", int(shadow_on))
        mp.f("uShadowTexel", 1.0 / self.shadow.size)
        GL.glActiveTexture(GL.GL_TEXTURE0)
        GL.glBindTexture(GL.GL_TEXTURE_2D, self.shadow.tex)
        mp.i("uShadowMap", 0)
        mp.vec3("uHighlight", env.highlight_color)
        mp.i("uUnlit", 0)

        selected = set(scene.selection)
        opaque = [o for o in visible if o.show_surface and o.opacity >= 0.999]
        transparent = [o for o in visible if o.show_surface and o.opacity < 0.999]

        GL.glDisable(GL.GL_BLEND)
        GL.glDepthMask(GL.GL_TRUE)
        for o in opaque:
            self._draw_object(o, models[o.id], selected, f.hover_id, stats)

        # ---- ground shadow catcher --------------------------------------------------------------
        GL.glEnable(GL.GL_BLEND)
        GL.glBlendFunc(GL.GL_SRC_ALPHA, GL.GL_ONE_MINUS_SRC_ALPHA)
        if shadow_on and env.ground_shadow:
            ext = max(view.grids["XY"].extent, radius * 3)
            if self._ground_mesh is None or self._ground_mesh[0] != ext:
                v, idx = grid_geo.ground_quad(ext)
                self._ground_mesh = (ext, g.Mesh(v, np.tile([0, 0, 1], (4, 1)), idx, solid=False))
            cp = self.catcher_prog
            cp.use()
            cp.mat4("uModel", np.eye(4))
            cp.mat4("uViewProj", viewproj)
            cp.mat3("uNormalMat", np.eye(3))
            cp.mat4("uLightVP", light_vp)
            cp.i("uShadowMap", 0)
            cp.f("uShadowTexel", 1.0 / self.shadow.size)
            cp.f("uStrength", 0.42)
            cp.vec3("uFadeCenter", [center[0], center[1], 0.0])
            cp.f("uFadeRadius", radius * 2.5 + 2.0)
            GL.glDepthMask(GL.GL_FALSE)
            GL.glEnable(GL.GL_POLYGON_OFFSET_FILL)
            GL.glPolygonOffset(-1.0, -1.0)
            self.gpu.mesh(self._ground_mesh[1]).draw()
            GL.glDisable(GL.GL_POLYGON_OFFSET_FILL)
            stats.draw_calls += 1

        # ---- grids ------------------------------------------------------------------------------------
        GL.glDepthMask(GL.GL_FALSE)
        for key, gs in view.grids.items():
            if not gs.visible:
                continue
            gl = self._grid_lines(key, gs)
            fade_center = cam.target.copy()
            fade = gs.extent
            for kind, width in (("minor", max(gs.line_width * 0.6, 0.6)), ("major", gs.line_width),
                                ("origin", gs.line_width * 1.6)):
                if kind in gl and len(gl[kind][0]):
                    self._lines(gl[kind][0], gl[kind][1], viewproj, w, h, width, fade_center=fade_center,
                                fade_radius=fade * 1.15)
                    stats.draw_calls += 1

        # ---- trajectories (depth-tested, drawn before transparent surfaces) ---------------------------------
        for path in f.trajectories:
            a, c = path.segments()
            if len(a):
                self._lines(a, c, viewproj, w, h, path.width * f.dpr)
                stats.draw_calls += 1

        # ---- transparent surfaces (back to front) --------------------------------------------------------
        if transparent:
            mp.use()
            def depth(o):
                return float(np.dot(models[o.id][:3, 3] - eye, cam.forward()))
            GL.glDepthMask(GL.GL_FALSE)
            GL.glEnable(GL.GL_CULL_FACE)
            for o in sorted(transparent, key=depth, reverse=True):
                for face in (GL.GL_FRONT, GL.GL_BACK):
                    GL.glCullFace(face)
                    self._draw_object(o, models[o.id], selected, f.hover_id, stats)
            GL.glDisable(GL.GL_CULL_FACE)
        GL.glDepthMask(GL.GL_TRUE)

        # ---- edges, selection outlines, local frames, boxes ---------------------------------------------
        for o in visible:
            edge_sel = o.id in selected
            if o.show_edges or edge_sel:
                col = env.highlight_color if edge_sel else [0.06, 0.07, 0.08]
                buf = self.gpu.edges(o.mesh())
                self._line_buffer(buf, viewproj @ models[o.id], models[o.id], w, h, 2.0 if edge_sel else 1.2,
                                  color=[*col, 0.95 if edge_sel else 0.85])
                stats.draw_calls += 1
        pts, cols = [], []
        for o in visible:
            ov = o.overlays
            show_axes = (ov.get("local_axes") or env.show_local_axes) and "local_axes" in o.typedef.overlays
            p, q = scene.world_pose(o.id)
            if show_axes:
                a, c = axes_geo.frame_lines(p, q, o.axes_length())
                pts.append(a)
                cols.append(c)
            if ov.get("bbox"):
                lo_, hi_ = o.mesh().bounds()
                a, c = axes_geo.box_lines(models[o.id], lo_, hi_, (0.95, 0.80, 0.30))
                pts.append(a)
                cols.append(c)
            for _q, start, vec in arrow_vectors(scene, o):
                a, c = arrow_lines(start, vec, cam.forward(), ARROW_COLORS[_q])
                pts.append(a)
                cols.append(c)
            if ov.get("com") and o.typedef.physical:
                com = models[o.id] @ np.append(o.mass_properties()["com"], 1.0)
                size = cam.world_per_pixel(com[:3], h) * 9
                a, c = axes_geo.cross_lines(com[:3], size, (0.98, 0.98, 0.98))
                pts.append(a)
                cols.append(c)
        for fr in scene.frames:  # named reference frames
            if not fr.visible:
                continue
            try:
                p, q = scene.frame_world_pose(fr.id)
            except SceneError:
                continue  # detached (parent deleted); reported in Errors & Warnings
            a, c = axes_geo.frame_lines(p, q, fr.axis_length)
            pts.append(a)
            cols.append(c)
            a, c = axes_geo.cross_lines(p, cam.world_per_pixel(p, h) * 6, fr.color)
            pts.append(a)
            cols.append(c)
        marker = cam.world_per_pixel(cam.target, h) * 6
        a, c = physics_overlay.collider_lines(scene, env.show_colliders)
        pts += a
        cols += c
        if env.show_joints:
            a, c = physics_overlay.joint_lines(scene, marker)
            pts += a
            cols += c
        a, c = physics_overlay.measurement_lines(scene)
        pts += a
        cols += c
        if env.show_contacts:
            a, c = physics_overlay.contact_lines(f.contacts, marker)
            pts += a
            cols += c
        if pts:
            GL.glDisable(GL.GL_DEPTH_TEST)
            self._lines(np.vstack(pts), np.vstack(cols), viewproj, w, h, 2.0)
            GL.glEnable(GL.GL_DEPTH_TEST)
            stats.draw_calls += 1

        # ---- global axes ---------------------------------------------------------------------------------
        ax = view.axes
        if ax.visible:
            a, c = axes_geo.global_axes(ax)
            self._lines(a, c, viewproj, w, h, ax.line_width)
            mp.use()
            mp.mat4("uViewProj", viewproj)
            mp.i("uUnlit", 0)
            for cone, model, color in axes_geo.axis_heads(ax):
                self._draw_mesh(cone, model, color, 1.0, 0.0, 0.4, False, 0.0, stats)
            if ax.show_origin_marker:
                s = ax.length * 0.035
                marker = np.diag([s, s, s, 1.0])
                marker[:3, 3] = ax.origin
                self._draw_mesh(self._origin_marker, marker, (0.92, 0.92, 0.92), 1.0, 0.0, 0.3, False, 0.0, stats)

        # ---- placement ghost ---------------------------------------------------------------------------------
        if f.ghost is not None:
            mp.use()
            GL.glDepthMask(GL.GL_FALSE)
            self._draw_mesh(f.ghost.mesh, f.ghost.model, f.ghost.color, 0.45, 0.0, 0.5, False, 0.6, stats)
            GL.glDepthMask(GL.GL_TRUE)
            buf = self.gpu.edges(f.ghost.mesh)
            self._line_buffer(buf, viewproj @ f.ghost.model, f.ghost.model, w, h, 1.5, color=[0.55, 0.78, 1.0, 0.9])

        # ---- gizmo ---------------------------------------------------------------------------------------------
        if f.gizmo is not None:
            GL.glClear(GL.GL_DEPTH_BUFFER_BIT)
            mp.use()
            mp.mat4("uViewProj", viewproj)
            mp.i("uUnlit", 1)
            for hd in f.gizmo.handles:
                active = f.gizmo.active
                if active and hd.name != active:
                    alpha = 0.25
                elif hd.plane is not None:
                    alpha = 0.55
                else:
                    alpha = 1.0
                color = HOVER_COLOR if hd.name in (f.gizmo.hover, active) else hd.color
                GL.glDepthMask(GL.GL_TRUE if alpha >= 0.9 else GL.GL_FALSE)
                self._draw_mesh(hd.mesh, f.gizmo.model, color, alpha, 0.0, 0.5, False, 0.0, stats)
            GL.glDepthMask(GL.GL_TRUE)
            mp.i("uUnlit", 0)

        GL.glBindVertexArray(0)
        GL.glUseProgram(0)
        GL.glDisable(GL.GL_DEPTH_TEST)
        GL.glDisable(GL.GL_BLEND)
        GL.glPolygonMode(GL.GL_FRONT_AND_BACK, GL.GL_FILL)
        stats.cpu_ms = (time.perf_counter() - t0) * 1000.0
        return stats

    # ---- helpers ----------------------------------------------------------------------------------------
    def _draw_object(self, o, model, selected, hover_id, stats: FrameStats) -> None:
        hl = 1.0 if o.id in selected else (0.45 if o.id == hover_id else 0.0)
        if o.wireframe:
            GL.glPolygonMode(GL.GL_FRONT_AND_BACK, GL.GL_LINE)
        self._draw_mesh(o.mesh(), model, o.color, o.opacity, o.metallic, o.roughness, o.typedef.uses_vertex_colors,
                        hl, stats)
        if o.wireframe:
            GL.glPolygonMode(GL.GL_FRONT_AND_BACK, GL.GL_FILL)
        stats.objects_drawn += 1

    def _draw_mesh(self, mesh: g.Mesh, model, color, opacity, metallic, roughness, vertex_colors, highlight,
                   stats: FrameStats) -> None:
        mp = self.mesh_prog
        mp.mat4("uModel", model)
        m3 = np.asarray(model, float)[:3, :3]
        try:
            nm = np.linalg.inv(m3).T
        except np.linalg.LinAlgError:
            nm = np.eye(3)
        mp.mat3("uNormalMat", nm)
        mp.vec3("uBaseColor", color)
        mp.f("uOpacity", opacity)
        mp.f("uMetallic", metallic)
        mp.f("uRoughness", roughness)
        mp.i("uUseVertexColor", int(vertex_colors))
        mp.f("uHighlightAmount", highlight)
        if opacity < 0.999:
            GL.glEnable(GL.GL_BLEND)
            GL.glBlendFunc(GL.GL_SRC_ALPHA, GL.GL_ONE_MINUS_SRC_ALPHA)
        buf: MeshBuffer = self.gpu.mesh(mesh)
        buf.draw()
        stats.draw_calls += 1
        stats.triangles += mesh.triangle_count

    def _grid_lines(self, key, gs):
        sig = (gs.plane, gs.major_spacing, gs.subdivisions, gs.extent, tuple(gs.color), gs.opacity, gs.emphasize_origin)
        cached = self._grid_cache.get(key)
        if cached is None or cached[0] != sig:
            cached = (sig, grid_geo.grid_lines(gs))
            self._grid_cache[key] = cached
        return cached[1]

    def _lines(self, pts, cols, viewproj, w, h, width, model=None, fade_center=None, fade_radius=0.0) -> None:
        self.lines.upload(pts, cols)
        model = np.eye(4) if model is None else model
        self._line_buffer(self.lines, viewproj @ model, model, w, h, width, fade_center=fade_center,
                          fade_radius=fade_radius)

    def _line_buffer(self, buf: LineBuffer, mvp, model, w, h, width, color=None, fade_center=None,
                     fade_radius=0.0) -> None:
        lp = self.line_prog
        lp.use()
        lp.mat4("uMVP", mvp)
        lp.mat4("uModel", model)
        lp.vec2("uViewport", (w, h))
        lp.f("uWidth", width * getattr(self, "_dpr", 1.0))
        lp.vec3("uFadeCenter", fade_center if fade_center is not None else (0, 0, 0))
        lp.f("uFadeRadius", fade_radius)
        GL.glEnable(GL.GL_BLEND)
        GL.glBlendFunc(GL.GL_SRC_ALPHA, GL.GL_ONE_MINUS_SRC_ALPHA)
        if color is not None:
            GL.glBindVertexArray(buf.vao)
            GL.glDisableVertexAttribArray(1)
            GL.glVertexAttrib4f(1, *[float(c) for c in color])
            buf.draw()
            GL.glBindVertexArray(buf.vao)
            GL.glEnableVertexAttribArray(1)
        else:
            buf.draw()
        self.mesh_prog.use()


def _gl_str(name) -> str:
    try:
        v = GL.glGetString(name)
        return v.decode() if isinstance(v, bytes) else str(v)
    except Exception:  # pragma: no cover
        return "unknown"
