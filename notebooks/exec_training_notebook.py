"""
Execute notebooks/03_Model_Training.ipynb with the 'roadrisk' (venv) kernel and
embed the outputs in place. Uses jupyter_client (no nbconvert required).

Usage:  python exec_training_notebook.py [notebook_path]
"""
import base64
import json
import os
import sys
from queue import Empty
import time

import nbformat as nbf
from jupyter_client import KernelManager

NotebookNode = nbf.NotebookNode

HERE = os.path.dirname(os.path.abspath(__file__))
PROJECT = os.path.dirname(HERE)

# Make the venv kernelspec discoverable
venv_share = os.path.join(PROJECT, ".venv", "share", "jupyter")
os.environ["JUPYTER_PATH"] = venv_share + os.pathsep + os.environ.get("JUPYTER_PATH", "")

KERNEL = "roadrisk"
TIMEOUT = 900  # generous per-cell timeout (seconds)


def _sanitize_data(data):
    """Convert raw message data into a JSON-serialisable mime bundle."""
    out = {}
    for k, v in data.items():
        if isinstance(v, bytes):
            out[k] = base64.b64encode(v).decode("ascii")
        elif isinstance(v, (dict, list)):
            out[k] = json.loads(json.dumps(v, default=str))
        else:
            out[k] = v
    return out


def _collect(kc, msg_id, timeout):
    """Collect iopub outputs for a cell until idle status; returns (outputs, errored)."""
    outputs = []
    done = False
    errored = False
    while not done:
        try:
            msg = kc.get_iopub_msg(timeout=timeout)
        except Empty:
            raise TimeoutError(f"Timed out waiting for cell {msg_id}")
        if msg.get("parent_header", {}).get("msg_id") != msg_id:
            continue
        mtype = msg["msg_type"]
        content = msg["content"]
        if mtype == "status":
            done = content["execution_state"] == "idle"
        elif mtype == "stream":
            outputs.append(
                {"output_type": "stream", "name": content["name"], "text": content["text"]}
            )
        elif mtype == "execute_result":
            outputs.append(
                {
                    "output_type": "execute_result",
                    "execution_count": content.get("execution_count"),
                    "data": _sanitize_data(content["data"]),
                    "metadata": {},
                }
            )
        elif mtype == "display_data":
            outputs.append(
                {
                    "output_type": "display_data",
                    "data": _sanitize_data(content["data"]),
                    "metadata": content.get("metadata", {}),
                }
            )
        elif mtype == "error":
            errored = True
            outputs.append(
                {
                    "output_type": "error",
                    "ename": content["ename"],
                    "evalue": content["evalue"],
                    "traceback": content["traceback"],
                }
            )
    # Confirm via the shell reply whether the cell errored
    try:
        reply = kc.get_shell_msg(timeout=timeout)
        if reply.get("msg_type") == "execute_reply":
            status = reply["content"].get("status")
            if status == "error":
                errored = True
    except Empty:
        pass
    return outputs, errored


def execute_notebook(nb_path, kernel_name=KERNEL, timeout=TIMEOUT):
    nb = nbf.read(nb_path, as_version=4)

    km = KernelManager(kernel_name=kernel_name)
    km.start_kernel()
    kc = km.client()
    kc.start_channels()
    try:
        kc.wait_for_ready(timeout=120)
    except Exception as exc:  # noqa: BLE001
        km.shutdown_kernel(now=True)
        raise RuntimeError(f"Kernel failed to start: {exc}") from exc

    print(f"Kernel '{kernel_name}' ready.")
    start = time.time()
    failures = 0
    run_count = 0
    try:
        for cell in nb.cells:
            if cell["cell_type"] != "code" or not cell.get("source", "").strip():
                continue
            run_count += 1
            source = cell["source"]
            if isinstance(source, list):
                source = "".join(source)
            msg_id = kc.execute(source)
            outputs, errored = _collect(kc, msg_id, timeout)
            cell["outputs"] = [NotebookNode(o) for o in outputs]
            cell["execution_count"] = run_count
            if errored:
                failures += 1
                ename = next((o["ename"] for o in outputs if o["output_type"] == "error"), "?")
                print(f"  [ERROR] cell {run_count}: {ename}")
            else:
                print(f"  [ok] cell {run_count} (in {time.time()-start:.1f}s cumulative)")
    finally:
        kc.stop_channels()
        km.shutdown_kernel(now=True)

    nbf.write(nb, nb_path)
    print(f"Executed {run_count} code cell(s), {failures} error(s) in {time.time()-start:.1f}s.")
    print(f"Saved executed notebook -> {nb_path}")
    return failures


if __name__ == "__main__":
    path = sys.argv[1] if len(sys.argv) > 1 else os.path.join(HERE, "03_Model_Training.ipynb")
    sys.exit(1 if execute_notebook(path) else 0)
