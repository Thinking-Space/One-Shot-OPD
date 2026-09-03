"""HumanEval+ subprocess 沙箱执行器 (v2, 修复版)

相比 v1 的改进：
  1. timeout 由外层 cap 到 60s，不再随 test case 数线性增长
  2. preexec_fn 使用 resource.setrlimit 设置内存限制 (4GB)，能穿透 exec
  3. 使用 start_new_session=True 创建新进程组，timeout 时 kill 整个进程组
  4. 增加 RLIMIT_CPU 限制单进程 CPU 时间
"""
import os
import platform
import resource
import signal
import subprocess
from tempfile import TemporaryDirectory

from ._base_imports import BASE_IMPORTS

_ERROR_MSG_PREFIX = "Failed to execute program: "
_MAX_TIMEOUT = 60  # 硬上限 60 秒


def _preexec_fn():
    """在 fork 后 exec 前执行，设置 resource limits（穿透 exec）."""
    # 4GB 内存限制
    mem_limit = 4 * 1024 * 1024 * 1024
    try:
        resource.setrlimit(resource.RLIMIT_AS, (mem_limit, mem_limit))
        if platform.uname().system != "Darwin":
            resource.setrlimit(resource.RLIMIT_DATA, (mem_limit, mem_limit))
    except (ValueError, resource.error):
        pass
    # CPU 时间限制 60s（防止 signal 失效时的兜底）
    try:
        resource.setrlimit(resource.RLIMIT_CPU, (60, 65))
    except (ValueError, resource.error):
        pass


def run_test(code: str, test: str = None, timeout: int = _MAX_TIMEOUT) -> tuple[bool, str]:
    """在 subprocess 中执行 solution + test code.

    Args:
        code: 模型生成的代码.
        test: evalplus test 代码（含 inputs/results + check 函数 + check 调用）.
        timeout: 超时秒数，硬上限 60s.

    Returns:
        (passed: bool, output_or_error: str)
    """
    timeout = min(timeout, _MAX_TIMEOUT)

    if not test:
        raise ValueError("No test provided.")

    env = os.environ.copy()
    env["OMP_NUM_THREADS"] = "1"

    code_to_run = f"{BASE_IMPORTS}\n\n{code}\n\n{test}\n"

    with TemporaryDirectory() as tmpdir:
        solution_path = os.path.join(tmpdir, "solution.py")
        with open(solution_path, "w") as f:
            f.write(code_to_run)

        command = ["python3", solution_path]
        try:
            result = subprocess.run(
                command,
                cwd=tmpdir,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                env=env,
                check=False,
                timeout=timeout,
                start_new_session=True,
                preexec_fn=_preexec_fn,
            )
            stderr = result.stderr.decode().strip()
            stdout = result.stdout.decode()
            if result.returncode == 0:
                return True, stdout
            return False, _ERROR_MSG_PREFIX + f"STDOUT:\n{stdout}\n\nSTDERR:\n{stderr}"

        except subprocess.TimeoutExpired as e:
            # kill 整个进程组
            try:
                os.killpg(os.getpgid(e.process.pid), signal.SIGKILL)
            except (ProcessLookupError, OSError, AttributeError):
                pass
            return False, _ERROR_MSG_PREFIX + f"Execution timed out after {timeout} seconds."
        except Exception as e:
            return False, _ERROR_MSG_PREFIX + f"Exception: {str(e)}"
