import logging

import pytest
from starlette.requests import Request
from starlette.responses import JSONResponse

from src.core.error_handling import request_id_var, set_request_id
from src.core.logging import DecoratedTextFormatter, JSONFormatter
from src.core.logging_utils import build_log_extra
from src.core.middleware import RequestIDMiddleware


def test_build_log_extra_omits_none_fields():
    extra = build_log_extra(workflow="ocr", ocr_mode="tables", ignored=None)
    assert extra == {"extra_fields": {"workflow": "ocr", "ocr_mode": "tables"}}


def test_decorated_text_formatter_appends_extra_fields():
    formatter = DecoratedTextFormatter("%(levelname)s %(message)s")
    record = logging.LogRecord(
        name="test",
        level=logging.INFO,
        pathname=__file__,
        lineno=10,
        msg="workflow_execution_completed",
        args=(),
        exc_info=None,
    )
    record.extra_fields = {"workflow": "ocr", "sections": 3}

    rendered = formatter.format(record)

    assert "workflow_execution_completed" in rendered
    assert "workflow=ocr" in rendered
    assert "sections=3" in rendered


def test_json_formatter_includes_request_id_and_extra_fields():
    formatter = JSONFormatter()
    record = logging.LogRecord(
        name="test",
        level=logging.INFO,
        pathname=__file__,
        lineno=25,
        msg="workflow_execution_completed",
        args=(),
        exc_info=None,
    )
    record.request_id = "req-123"
    record.extra_fields = {"workflow": "ocr", "sections": 2}

    rendered = formatter.format(record)

    assert '"request_id": "req-123"' in rendered
    assert '"workflow": "ocr"' in rendered
    assert '"sections": 2' in rendered


@pytest.mark.asyncio
async def test_request_id_middleware_resets_contextvar():
    middleware = RequestIDMiddleware(app=lambda scope, receive, send: None)
    request = Request(
        {
            "type": "http",
            "method": "GET",
            "path": "/ping",
            "headers": [],
            "query_string": b"",
            "scheme": "http",
            "server": ("testserver", 80),
            "client": ("testclient", 123),
            "root_path": "",
            "http_version": "1.1",
        }
    )

    seen_request_id = {}

    async def call_next(_: Request):
        seen_request_id["value"] = request_id_var.get()
        return JSONResponse({"ok": True})

    response = await middleware.dispatch(request, call_next)

    assert response.status_code == 200
    assert response.headers["X-Request-ID"]
    assert seen_request_id["value"] == response.headers["X-Request-ID"]
    assert request_id_var.get() == ""


@pytest.mark.asyncio
async def test_request_id_middleware_uses_route_override():
    middleware = RequestIDMiddleware(app=lambda scope, receive, send: None)
    request = Request(
        {
            "type": "http",
            "method": "POST",
            "path": "/extract-json",
            "headers": [(b"x-request-id", b"header-id")],
            "query_string": b"",
            "scheme": "http",
            "server": ("testserver", 80),
            "client": ("testclient", 123),
            "root_path": "",
            "http_version": "1.1",
        }
    )

    async def call_next(req: Request):
        req.state.request_id = "body-id"
        set_request_id("body-id")
        return JSONResponse({"ok": True})

    response = await middleware.dispatch(request, call_next)

    assert response.headers["X-Request-ID"] == "body-id"
