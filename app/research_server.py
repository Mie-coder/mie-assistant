"""小咩拥有的本地研究服务；自动选端口，退出时只清理自己启动的进程。"""

import json
import os
from pathlib import Path
import signal
import socket
import subprocess
import sys
import tempfile
import time

import httpx


# 当前文件在 mie_assistant/app/ 中；向上两层得到小咩项目根目录。
# 后面用绝对路径找研究员代码和 .env，不依赖用户在哪个目录启动小咩。
ROOT = Path(__file__).resolve().parents[1]


class LocalResearchServer:
    # main.py 中的使用方式：
    # with LocalResearchServer() as server:
    #     agent = build_agent(research_url=server.url)
    #     MieApp(...).run()
    #
    # LocalResearchServer() -> __init__：先记住配置，还没有启动服务。
    # 进入 with            -> __enter__：启动服务，准备好后交出 server。
    # 离开 with            -> __exit__：关闭自己启动的服务，清理临时文件。
    def __init__(self, *, graph_path="app/researcher.py:graph", env_file=ROOT / ".env"):
        # 冒号前是 Python 文件，冒号后是这个文件里的变量名。
        # 即：让服务加载 app/researcher.py 中已经创建好的 graph 对象。
        self.graph_path = graph_path
        # 服务进程读取这份模型配置；传入 None 时不加载文件，供离线测试使用。
        self.env_file = env_file
        # 启动后会保存子进程对象，退出时通过它找到并关闭服务。
        self.process = None

    def __enter__(self):
        # 第 1 步：准备本次运行专用的临时目录，放配置、日志和服务运行数据。
        # 它不是研究员源码目录，退出时删除它不会删除项目代码或用户资料。
        self.directory = tempfile.TemporaryDirectory(prefix="mie-research-")
        directory = Path(self.directory.name)
        self.log_path = directory / "server.log"
        self.log = self.log_path.open("w+")
        # 第 2 步：请操作系统选一个当时空闲的本机端口。
        # 端口写 0 表示自动分配；127.0.0.1 表示只允许本机访问。
        # 这个临时 socket 随 with 关闭，后面由真正的研究服务使用选出的端口。
        with socket.socket() as sock:
            sock.bind(("127.0.0.1", 0))
            port = sock.getsockname()[1]
        # main.py 会把这个地址交给主 Agent，用来发送后台任务请求。
        self.url = f"http://127.0.0.1:{port}"
        # 第 3 步：生成课件中的 langgraph.json。
        # dependencies：让服务能加载小咩项目代码。
        # graphs：把 researcher.py:graph 注册为 researcher，供 graph_id 定位。
        config = {"dependencies": [str(ROOT)], "graphs": {"researcher": str(ROOT / self.graph_path)}}
        if self.env_file is not None:
            config["env"] = str(self.env_file)
        config_path = directory / "langgraph.json"
        config_path.write_text(json.dumps(config))
        # 复制当前进程环境给服务使用，并关闭 CLI 使用统计；不修改父进程环境。
        env = {**os.environ, "LANGGRAPH_CLI_NO_ANALYTICS": "1"}
        try:
            # 第 4 步：相当于替用户在另一个终端运行 langgraph dev。
            # sys.executable 使用小咩当前的 Python，保证两边使用同一个虚拟环境。
            # Popen 启动独立进程后就返回，研究服务会在后台继续运行。
            # 4 是并发执行槽位数；不自动开浏览器，也不监听源码变化自动重启。
            self.process = subprocess.Popen(
                [sys.executable, "-m", "langgraph_cli", "dev", "--config", str(config_path),
                 "--host", "127.0.0.1", "--port", str(port), "--n-jobs-per-worker", "4",
                 "--no-browser", "--no-reload"],
                # 日志写入文件，避免服务日志冲进小咩聊天界面。
                cwd=directory, env=env, stdout=self.log, stderr=subprocess.STDOUT,
                # 单独创建进程组，退出时能清理它及其子进程，不影响其他服务。
                start_new_session=True,
            )
            # 第 5 步：启动命令返回，不等于服务已经准备好；需要实际请求验证。
            # 单次请求最多等 1 秒，不使用系统代理访问本机；总共最多等 40 秒。
            with httpx.Client(base_url=self.url, timeout=1, trust_env=False) as client:
                deadline = time.monotonic() + 40
                while time.monotonic() < deadline:
                    # poll() 返回 None 表示进程还活着；其他值表示它已退出。
                    if self.process.poll() is not None:
                        break
                    try:
                        # 这里的 search 是查询已注册的 Agent，不是上网搜索资料。
                        # 找到 researcher 才算准备好；此时还没有派发研究任务。
                        response = client.post("/assistants/search", json={"graph_id": "researcher"})
                        if response.is_success and response.json():
                            # 这个 self 就是 with ... as server 中的 server。
                            # 返回后，main.py 才继续创建主 Agent、打开聊天界面。
                            return self
                    except httpx.HTTPError:
                        # 服务刚启动时可能暂时连不上，稍等后再试。
                        pass
                    time.sleep(.15)
            raise RuntimeError("本地研究服务未启动，请确认已执行 uv sync --locked。")
        except BaseException:
            # 保留仅本次启动的诊断日志，避免在聊天或错误正文中回显配置。
            with tempfile.NamedTemporaryFile(prefix="mie-research-error-", suffix=".log", delete=False) as saved:
                saved.write(self.log_path.read_bytes())
                error_log = saved.name
            # __enter__ 没有成功返回时，Python 不会自动调用 __exit__。
            # 因此启动失败需要在这里手动清理已创建的进程和临时目录。
            self.__exit__(None, None, None)
            raise RuntimeError(f"本地研究服务启动失败，诊断日志：{error_log}") from None

    def __exit__(self, exc_type, exc, tb):
        # 正常退出聊天界面，或 with 内抛出异常，都会来到这里。
        # exc_type 为 None 表示正常离开；非 None 表示携带异常退出。
        try:
            if self.process is not None and self.process.poll() is None:
                # 先请求服务正常停止，最多给它 8 秒收尾。
                os.killpg(self.process.pid, signal.SIGTERM)
                try:
                    self.process.wait(timeout=8)
                except subprocess.TimeoutExpired:
                    # 超时仍不退出才强制结束；wait() 用来等待并回收子进程。
                    os.killpg(self.process.pid, signal.SIGKILL)
                    self.process.wait(timeout=3)
        finally:
            # 无论停止过程是否出错，都执行文件清理。
            self.log.close()
            if exc_type is not None:
                # 异常退出时先把日志复制到另一个临时文件，保留诊断依据。
                with tempfile.NamedTemporaryFile(prefix="mie-research-error-", suffix=".log", delete=False) as saved:
                    saved.write(self.log_path.read_bytes())
                    print(f"研究服务诊断日志：{saved.name}", file=sys.stderr)
            # 删除本次服务的临时目录；上面另存的异常日志不在这个目录里。
            self.directory.cleanup()
