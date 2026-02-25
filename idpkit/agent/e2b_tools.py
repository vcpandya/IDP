"""E2B code execution and browser tools for IDA agent.

Provides sandbox-based Python code execution and browser automation via E2B.
"""

from __future__ import annotations

import base64
import logging
import os
from typing import Any

logger = logging.getLogger(__name__)

E2B_TIMEOUT = 60
SANDBOX_TIMEOUT = 300


def _get_api_key() -> str | None:
    return os.getenv("E2B_API_KEY")


async def execute_python(code: str, timeout: int = E2B_TIMEOUT) -> dict[str, Any]:
    from e2b_code_interpreter import AsyncSandbox

    code = (code or "").strip()
    if not code:
        return {"error": "No code provided."}

    api_key = _get_api_key()
    if not api_key:
        return {"error": "E2B code execution is not configured. E2B_API_KEY is not set."}

    sandbox = None
    try:
        sandbox = await AsyncSandbox.create(api_key=api_key, timeout=SANDBOX_TIMEOUT)
        execution = await sandbox.run_code(code, timeout=timeout)

        stdout = "".join(line.line for line in (execution.logs.stdout or []))
        stderr = "".join(line.line for line in (execution.logs.stderr or []))

        charts = []
        if execution.results:
            for result in execution.results:
                if hasattr(result, "png") and result.png:
                    charts.append({
                        "type": "image/png",
                        "data": result.png,
                    })
                elif hasattr(result, "svg") and result.svg:
                    charts.append({
                        "type": "image/svg+xml",
                        "data": result.svg,
                    })

        text_results = []
        if execution.results:
            for result in execution.results:
                if hasattr(result, "text") and result.text:
                    text_results.append(result.text)

        error_info = None
        if execution.error:
            error_info = {
                "name": getattr(execution.error, "name", "Error"),
                "value": getattr(execution.error, "value", str(execution.error)),
                "traceback": getattr(execution.error, "traceback", ""),
            }

        return {
            "stdout": stdout[:10000] if stdout else "",
            "stderr": stderr[:5000] if stderr else "",
            "results": text_results[:10],
            "charts": charts[:5],
            "error": error_info,
            "success": error_info is None,
        }

    except Exception as exc:
        logger.error("E2B execute_python failed: %s", exc)
        return {"error": f"Code execution failed: {exc}", "success": False}
    finally:
        if sandbox:
            try:
                await sandbox.kill()
            except Exception:
                pass


async def install_package(package_name: str) -> dict[str, Any]:
    from e2b_code_interpreter import AsyncSandbox

    package_name = (package_name or "").strip()
    if not package_name:
        return {"error": "No package name provided."}

    api_key = _get_api_key()
    if not api_key:
        return {"error": "E2B code execution is not configured. E2B_API_KEY is not set."}

    sandbox = None
    try:
        sandbox = await AsyncSandbox.create(api_key=api_key, timeout=SANDBOX_TIMEOUT)
        install_code = f"import subprocess; result = subprocess.run(['pip', 'install', '{package_name}'], capture_output=True, text=True); print(result.stdout[-2000:] if len(result.stdout) > 2000 else result.stdout); print(result.stderr[-1000:] if len(result.stderr) > 1000 else result.stderr) if result.returncode != 0 else None"
        execution = await sandbox.run_code(install_code, timeout=120)

        stdout = "".join(line.line for line in (execution.logs.stdout or []))
        stderr = "".join(line.line for line in (execution.logs.stderr or []))

        success = execution.error is None and ("Successfully installed" in stdout or "already satisfied" in stdout.lower())

        return {
            "package": package_name,
            "success": success,
            "stdout": stdout[:3000],
            "stderr": stderr[:1000] if stderr else "",
        }

    except Exception as exc:
        logger.error("E2B install_package failed: %s", exc)
        return {"error": f"Package installation failed: {exc}", "success": False}
    finally:
        if sandbox:
            try:
                await sandbox.kill()
            except Exception:
                pass


async def browse_web(url: str, task: str = "", timeout: int = 120) -> dict[str, Any]:
    from e2b_code_interpreter import AsyncSandbox

    url = (url or "").strip()
    if not url:
        return {"error": "No URL provided."}

    api_key = _get_api_key()
    if not api_key:
        return {"error": "E2B browser use is not configured. E2B_API_KEY is not set."}

    task_instruction = task.strip() if task else "Extract the main content of the page."

    browser_code = f'''
import subprocess
subprocess.run(["pip", "install", "playwright"], capture_output=True, text=True)
subprocess.run(["playwright", "install", "chromium"], capture_output=True, text=True)

import asyncio
from playwright.async_api import async_playwright

async def browse():
    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True, args=["--no-sandbox", "--disable-setuid-sandbox"])
        context = await browser.new_context(
            viewport={{"width": 1280, "height": 720}},
            user_agent="Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
        )
        page = await context.new_page()

        try:
            await page.goto("{url}", wait_until="domcontentloaded", timeout=30000)
            await page.wait_for_timeout(2000)

            title = await page.title()

            content = await page.evaluate("""() => {{
                const selectors = ['article', 'main', '[role="main"]', '.content', '#content', '.post', '.article'];
                for (const sel of selectors) {{
                    const el = document.querySelector(sel);
                    if (el && el.innerText.trim().length > 100) {{
                        return el.innerText.trim();
                    }}
                }}
                return document.body.innerText.trim();
            }}""")

            links = await page.evaluate("""() => {{
                const anchors = Array.from(document.querySelectorAll('a[href]'));
                return anchors.slice(0, 20).map(a => ({{
                    text: a.innerText.trim().substring(0, 100),
                    href: a.href
                }})).filter(l => l.text && l.href.startsWith('http'));
            }}""")

            meta = await page.evaluate("""() => {{
                const desc = document.querySelector('meta[name="description"]');
                const og = document.querySelector('meta[property="og:description"]');
                return {{
                    description: desc ? desc.content : (og ? og.content : ''),
                    url: window.location.href
                }};
            }}""")

            screenshot_bytes = await page.screenshot(type="png", full_page=False)
            import base64
            screenshot_b64 = base64.b64encode(screenshot_bytes).decode("utf-8")

            print("__RESULT_JSON__")
            import json
            print(json.dumps({{
                "title": title,
                "content": content[:15000] if content else "",
                "links": links,
                "meta": meta,
                "screenshot": screenshot_b64,
                "url": "{url}",
                "success": True
            }}))

        except Exception as e:
            print("__RESULT_JSON__")
            import json
            print(json.dumps({{"error": str(e), "url": "{url}", "success": False}}))
        finally:
            await browser.close()

asyncio.run(browse())
'''

    sandbox = None
    try:
        sandbox = await AsyncSandbox.create(api_key=api_key, timeout=SANDBOX_TIMEOUT)
        execution = await sandbox.run_code(browser_code, timeout=timeout)

        stdout = "".join(line.line for line in (execution.logs.stdout or []))
        stderr = "".join(line.line for line in (execution.logs.stderr or []))

        if execution.error:
            return {
                "error": f"Browser execution failed: {getattr(execution.error, 'value', str(execution.error))}",
                "stderr": stderr[:3000],
                "success": False,
                "url": url,
            }

        import json as _json
        if "__RESULT_JSON__" in stdout:
            json_part = stdout.split("__RESULT_JSON__", 1)[1].strip()
            try:
                result = _json.loads(json_part)
                if result.get("content"):
                    result["content"] = result["content"][:10000]
                return result
            except _json.JSONDecodeError:
                pass

        return {
            "url": url,
            "content": stdout[:10000] if stdout else "No content extracted.",
            "stderr": stderr[:2000] if stderr else "",
            "success": True,
        }

    except Exception as exc:
        logger.error("E2B browse_web failed: %s", exc)
        return {"error": f"Browser browsing failed: {exc}", "success": False}
    finally:
        if sandbox:
            try:
                await sandbox.kill()
            except Exception:
                pass
