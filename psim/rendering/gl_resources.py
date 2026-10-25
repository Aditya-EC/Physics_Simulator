"""Thin wrappers around OpenGL objects (programs, vertex buffers, shadow map)."""
from __future__ import annotations

import ctypes
import weakref

import numpy as np
from OpenGL import GL

from ..scene.geometry import Mesh


class ShaderError(RuntimeError):
    pass


class ShaderProgram:
    def __init__(self, vs: str, fs: str, gs: str | None = None) -> None:
        shaders = [self._compile(vs, GL.GL_VERTEX_SHADER), self._compile(fs, GL.GL_FRAGMENT_SHADER)]
        if gs:
            shaders.append(self._compile(gs, GL.GL_GEOMETRY_SHADER))
        self.id = GL.glCreateProgram()
        for s in shaders:
            GL.glAttachShader(self.id, s)
        GL.glLinkProgram(self.id)
        ok = GL.glGetProgramiv(self.id, GL.GL_LINK_STATUS)
        for s in shaders:
            GL.glDetachShader(self.id, s)
            GL.glDeleteShader(s)
        if not ok:
            log = GL.glGetProgramInfoLog(self.id)
            raise ShaderError(f"Shader link failed: {log.decode() if isinstance(log, bytes) else log}")
        self._loc: dict[str, int] = {}

    @staticmethod
    def _compile(src: str, kind) -> int:
        s = GL.glCreateShader(kind)
        GL.glShaderSource(s, src)
        GL.glCompileShader(s)
        if not GL.glGetShaderiv(s, GL.GL_COMPILE_STATUS):
            log = GL.glGetShaderInfoLog(s)
            raise ShaderError(f"Shader compile failed: {log.decode() if isinstance(log, bytes) else log}")
        return s

    def use(self) -> None:
        GL.glUseProgram(self.id)

    def loc(self, name: str) -> int:
        if name not in self._loc:
            self._loc[name] = GL.glGetUniformLocation(self.id, name)
        return self._loc[name]

    def mat4(self, name: str, m: np.ndarray) -> None:
        GL.glUniformMatrix4fv(self.loc(name), 1, GL.GL_TRUE, np.ascontiguousarray(m, np.float32))

    def mat3(self, name: str, m: np.ndarray) -> None:
        GL.glUniformMatrix3fv(self.loc(name), 1, GL.GL_TRUE, np.ascontiguousarray(m, np.float32))

    def vec3(self, name: str, v) -> None:
        GL.glUniform3f(self.loc(name), *[float(c) for c in v[:3]])

    def vec2(self, name: str, v) -> None:
        GL.glUniform2f(self.loc(name), float(v[0]), float(v[1]))

    def f(self, name: str, v: float) -> None:
        GL.glUniform1f(self.loc(name), float(v))

    def i(self, name: str, v: int) -> None:
        GL.glUniform1i(self.loc(name), int(v))

    def delete(self) -> None:
        GL.glDeleteProgram(self.id)


class MeshBuffer:
    """VAO with interleaved position / normal / colour and an index buffer."""

    STRIDE = 9 * 4

    def __init__(self, mesh: Mesh) -> None:
        colors = mesh.colors if mesh.colors is not None else np.ones((len(mesh.vertices), 3), np.float32)
        data = np.ascontiguousarray(np.hstack([mesh.vertices, mesh.normals, colors]), np.float32)
        idx = np.ascontiguousarray(mesh.indices.reshape(-1), np.uint32)
        self.count = int(idx.size)
        self.vao = GL.glGenVertexArrays(1)
        self.vbo, self.ebo = GL.glGenBuffers(2)
        GL.glBindVertexArray(self.vao)
        GL.glBindBuffer(GL.GL_ARRAY_BUFFER, self.vbo)
        GL.glBufferData(GL.GL_ARRAY_BUFFER, data.nbytes, data, GL.GL_STATIC_DRAW)
        GL.glBindBuffer(GL.GL_ELEMENT_ARRAY_BUFFER, self.ebo)
        GL.glBufferData(GL.GL_ELEMENT_ARRAY_BUFFER, idx.nbytes, idx, GL.GL_STATIC_DRAW)
        for loc, off in ((0, 0), (1, 12), (2, 24)):
            GL.glEnableVertexAttribArray(loc)
            GL.glVertexAttribPointer(loc, 3, GL.GL_FLOAT, GL.GL_FALSE, self.STRIDE, ctypes.c_void_p(off))
        GL.glBindVertexArray(0)

    def draw(self) -> None:
        GL.glBindVertexArray(self.vao)
        GL.glDrawElements(GL.GL_TRIANGLES, self.count, GL.GL_UNSIGNED_INT, None)

    def delete(self) -> None:
        GL.glDeleteVertexArrays(1, [self.vao])
        GL.glDeleteBuffers(2, [self.vbo, self.ebo])


class LineBuffer:
    """Dynamic GL_LINES buffer with per-vertex RGBA colour."""

    def __init__(self) -> None:
        self.vao = GL.glGenVertexArrays(1)
        self.vbo = GL.glGenBuffers(1)
        self.count = 0
        self._capacity = 0
        GL.glBindVertexArray(self.vao)
        GL.glBindBuffer(GL.GL_ARRAY_BUFFER, self.vbo)
        GL.glEnableVertexAttribArray(0)
        GL.glVertexAttribPointer(0, 3, GL.GL_FLOAT, GL.GL_FALSE, 28, ctypes.c_void_p(0))
        GL.glEnableVertexAttribArray(1)
        GL.glVertexAttribPointer(1, 4, GL.GL_FLOAT, GL.GL_FALSE, 28, ctypes.c_void_p(12))
        GL.glBindVertexArray(0)

    def upload(self, points: np.ndarray, colors: np.ndarray) -> None:
        points = np.asarray(points, np.float32).reshape(-1, 3)
        colors = np.asarray(colors, np.float32).reshape(-1, 4)
        if len(points) == 0:
            self.count = 0
            return
        if len(colors) == 1 and len(points) > 1:
            colors = np.repeat(colors, len(points), axis=0)
        data = np.ascontiguousarray(np.hstack([points, colors]), np.float32)
        self.count = len(points)
        GL.glBindBuffer(GL.GL_ARRAY_BUFFER, self.vbo)
        if data.nbytes > self._capacity:
            self._capacity = max(data.nbytes, 2 * self._capacity, 1024)
            GL.glBufferData(GL.GL_ARRAY_BUFFER, self._capacity, None, GL.GL_DYNAMIC_DRAW)
        if data.nbytes:
            GL.glBufferSubData(GL.GL_ARRAY_BUFFER, 0, data.nbytes, data)

    def draw(self) -> None:
        if self.count:
            GL.glBindVertexArray(self.vao)
            GL.glDrawArrays(GL.GL_LINES, 0, self.count)

    def delete(self) -> None:
        GL.glDeleteVertexArrays(1, [self.vao])
        GL.glDeleteBuffers(1, [self.vbo])


class GpuCache:
    """GPU buffers keyed by Mesh object identity; released when the mesh is garbage collected."""

    def __init__(self) -> None:
        self._meshes: dict[int, tuple[weakref.ref, MeshBuffer]] = {}
        self._edges: dict[int, tuple[weakref.ref, LineBuffer]] = {}

    def mesh(self, mesh: Mesh) -> MeshBuffer:
        entry = self._meshes.get(id(mesh))
        if entry is None or entry[0]() is not mesh:
            if entry is not None:
                entry[1].delete()
            entry = (weakref.ref(mesh), MeshBuffer(mesh))
            self._meshes[id(mesh)] = entry
        return entry[1]

    def edges(self, mesh: Mesh) -> LineBuffer:
        entry = self._edges.get(id(mesh))
        if entry is None or entry[0]() is not mesh:
            if entry is not None:
                entry[1].delete()
            buf = LineBuffer()
            buf.upload(mesh.feature_edges(), np.array([[1, 1, 1, 1]], np.float32))
            entry = (weakref.ref(mesh), buf)
            self._edges[id(mesh)] = entry
        return entry[1]

    def collect(self) -> None:
        """Delete buffers whose meshes no longer exist (call with the context current)."""
        for store in (self._meshes, self._edges):
            for k in [k for k, (ref, _) in store.items() if ref() is None]:
                store.pop(k)[1].delete()

    def release_all(self) -> None:
        for store in (self._meshes, self._edges):
            for _, buf in store.values():
                buf.delete()
            store.clear()

    @property
    def mesh_count(self) -> int:
        return len(self._meshes)


class ShadowMap:
    def __init__(self, size: int = 2048) -> None:
        self.size = size
        self.tex = GL.glGenTextures(1)
        GL.glBindTexture(GL.GL_TEXTURE_2D, self.tex)
        GL.glTexImage2D(GL.GL_TEXTURE_2D, 0, GL.GL_DEPTH_COMPONENT24, size, size, 0, GL.GL_DEPTH_COMPONENT,
                        GL.GL_FLOAT, None)
        GL.glTexParameteri(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_MIN_FILTER, GL.GL_NEAREST)
        GL.glTexParameteri(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_MAG_FILTER, GL.GL_NEAREST)
        GL.glTexParameteri(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_WRAP_S, GL.GL_CLAMP_TO_BORDER)
        GL.glTexParameteri(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_WRAP_T, GL.GL_CLAMP_TO_BORDER)
        GL.glTexParameterfv(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_BORDER_COLOR, np.array([1, 1, 1, 1], np.float32))
        self.fbo = GL.glGenFramebuffers(1)
        GL.glBindFramebuffer(GL.GL_FRAMEBUFFER, self.fbo)
        GL.glFramebufferTexture2D(GL.GL_FRAMEBUFFER, GL.GL_DEPTH_ATTACHMENT, GL.GL_TEXTURE_2D, self.tex, 0)
        GL.glDrawBuffer(GL.GL_NONE)
        GL.glReadBuffer(GL.GL_NONE)
        self.complete = GL.glCheckFramebufferStatus(GL.GL_FRAMEBUFFER) == GL.GL_FRAMEBUFFER_COMPLETE

    def delete(self) -> None:
        GL.glDeleteFramebuffers(1, [self.fbo])
        GL.glDeleteTextures(1, [self.tex])
