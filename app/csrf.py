"""CSRF 防护：写操作必须带自定义请求头 X-Robot-Client。

背景（产品规划 v4 P1-6）：服务绑定 127.0.0.1 挡住了局域网，但挡不住家人浏览器里
恶意网页发起的"简单请求"（表单 POST / 无自定义头的 fetch）。要求自定义头后，
跨站请求会先触发 CORS 预检，而本服务不返回任何 CORS 允许头，预检即失败。
同源页面（web.html）通过全局 fetch 包装自动带头，对用户无感。
"""
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import JSONResponse

HEADER = "X-Robot-Client"
WRITE_METHODS = {"POST", "PUT", "PATCH", "DELETE"}


class CSRFMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        if request.method in WRITE_METHODS and not request.headers.get(HEADER):
            return JSONResponse({"detail": "缺少客户端标识头，请从机器人页面操作。"}, status_code=403)
        return await call_next(request)
