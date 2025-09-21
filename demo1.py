"""
Demo 1: Streamlit AI Weather Lab Environment Checker

Run: streamlit run demo1.py
"""

import streamlit as st
import platform
import sys
import subprocess
import shutil
import importlib
import json
import re
from pathlib import Path
from datetime import datetime, timezone
import psutil

st.set_page_config(page_title="AI Weather Lab — Environment Checker", layout="centered")

# --- Helper functions ---

def run_cmd(cmd, timeout=8):
    """Run a shell command and return (ok, stdout, stderr, returncode)"""
    try:
        completed = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=timeout, text=True)
        out = completed.stdout.strip()
        err = completed.stderr.strip()
        ok = completed.returncode == 0
        return ok, out, err, completed.returncode
    except FileNotFoundError:
        return False, "", f"{cmd[0]}: not found", 127
    except subprocess.TimeoutExpired:
        return False, "", "Timed out", -1
    except Exception as e:
        return False, "", str(e), -2


def get_system_info():
    info = {}
    info['platform'] = platform.system()
    info['platform_release'] = platform.release()
    info['architecture'] = platform.machine()
    info['python_version'] = platform.python_version()
    info['cpu_count'] = shutil.os.cpu_count()
    if psutil:
        info['total_ram_bytes'] = psutil.virtual_memory().total
    else:
        info['total_ram_bytes'] = None
    return info


def check_nvidia_smi():
    return run_cmd(["nvidia-smi"], timeout=6)


def detect_wsl():
    try:
        with open('/proc/version', 'r') as f:
            v = f.read()
        if 'Microsoft' in v or 'microsoft' in v:
            return True, v
        else:
            return False, v
    except Exception:
        return False, None





def check_docker():
    if not shutil.which('docker'):
        return False, '', 'docker not found in PATH', 127
    ok_ver, out_ver, err_ver, rc_ver = run_cmd(['docker', '--version'], timeout=6)
    ok_run, out_run, err_run, rc_run = run_cmd(['docker', 'run', '--rm', 'hello-world'], timeout=30)
    ok = ok_ver and ok_run
    out = out_ver + '\n' + out_run
    err = err_ver + '\n' + err_run
    return ok, out, err, (rc_ver or rc_run)





def human_bytes(n):
    if n is None:
        return 'Unknown (psutil not installed)'
    for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
        if n < 1024.0:
            return f"{n:3.1f} {unit}"
        n /= 1024.0
    return f"{n:.1f} PB"


# Keys to store current session check results
keys = [
    'sysinfo',
    'sys_checked',
    'gpu_result',  # tuple (ok,out,err,rc) from automatic nvidia-smi
    'gpu_paste_text',
    'gpu_paste_detected',
    'conda_result',
    'docker_result',
    'libs_result'
]
for k in keys:
    if k not in st.session_state:
        st.session_state[k] = None



st.title("AI Weather Lab — Environment Checker")
st.caption("Guided checklist to verify WSL, GPU drivers, Conda, Docker and core Python packages")

st.sidebar.header("Steps")
steps = [
    "System Health Check",
    "WSL & GPU Driver Verification",
    "Docker Functionality Test",
    "Documentation "
]
for s in steps:
    st.sidebar.write(f"• {s}")

# 1) System Health Check
### 
### TODO: ADD MINIMUM REQUIREMENTS CHECK ###
### 

st.header("1 — System Health Check")
st.markdown("*Check OS, CPU, RAM, and GPU to ensure the machine meets minimum requirements for training AI models.*")
col1, col2 = st.columns([3,1])
with col1:
    if st.button('Run system checks'):
        info = get_system_info()
        st.session_state['sysinfo'] = info
        st.session_state['sys_checked'] = True
        st.success('System check completed and cached for summary')

    if st.session_state['sysinfo']:
        info = st.session_state['sysinfo']
        st.markdown(f"**Platform:** {info['platform']} {info['platform_release']} ({info['architecture']})")
        st.markdown(f"**Python:** {info['python_version']}")
        st.markdown(f"**CPUs:** {info['cpu_count']}")
        st.markdown(f"**RAM:** {human_bytes(info['total_ram_bytes'])}")
    else:
        st.info('Click "Run system checks" to detect OS, CPU, RAM and Python version — results are cached for the summary')
with col2:
    if st.session_state['sys_checked']:
        st.metric(label='System checked', value='Yes')
    else:
        st.metric(label='System checked', value='No')

st.markdown('---')

# 2. WSL & GPU Driver Verification
st.header("2 — WSL & GPU Driver Verification")
st.markdown("*We confirm your Linux environment inside Windows (WSL) can see the GPU. This ensures deep learning workloads can run on the GPU from Linux, not just from Windows.*")

with st.expander("What is WSL?"):
    st.write("WSL (Windows Subsystem for Linux) lets you run a Linux environment on Windows without a virtual machine. We use it to check GPU availability for Linux-based workflows.")


st.subheader('Try automatic nvidia-smi')
if st.button('Run nvidia-smi (automated)'):
    ok, out, err, rc = check_nvidia_smi()
    st.session_state['gpu_result'] = (ok, out, err, rc)
    if ok:
        st.success('nvidia-smi ran successfully — GPU drivers look present on this host (cached)')
        st.code(out)
    else:
        st.error('Failed to run nvidia-smi automatically on this host — cached result saved')
        if err:
            st.code(err)
        st.info('If you are using WSL, open a WSL terminal and run `nvidia-smi` there; then paste the output on the right')


st.subheader('WSL manual paste')
with st.expander("Quick manual WSL GPU check — step-by-step", expanded=False):
    st.markdown("""
**1)** Open Windows PowerShell (or Command Prompt).  
   - **PowerShell:** Press `Windows key`, type `PowerShell`, then click **Windows PowerShell**.  
   - **Command Prompt:** Press `Windows key`, type `cmd`, then click **Command Prompt**.

**2)** List installed WSL distros and check WSL version:
```
wsl -l -v
```

**3)** Start a shell in your distro (replace `<distro>` with the name shown):
```
wsl -d <distro>
```

**4)** Inside the WSL shell, run:
```
nvidia-smi
```
You should see a header beginning with `NVIDIA-SMI` and a table showing GPU name, memory usage, and driver version.

**5)** Copy the entire output and paste it into the box below.

If `nvidia-smi` fails, ensure you have WSL2 enabled and have installed the NVIDIA drivers for WSL: https://developer.nvidia.com/cuda/wsl
""")
    pasted = st.text_area('Paste output of `nvidia-smi` from your WSL terminal here (or leave empty)', height=160, key='gpu_paste_text')
    if pasted:
        st.session_state['gpu_paste_text'] = pasted
        found = bool(re.search(r'NVIDIA-SMI', pasted, re.IGNORECASE))
        st.session_state['gpu_paste_detected'] = found
        if found:
            st.success('Detected NVIDIA-SMI header in pasted output — GPU drivers in WSL appear configured (cached)')
            st.code('\n'.join(pasted.splitlines()[:20]))
        else:
            st.warning('Could not find typical nvidia-smi header in the pasted output. Please ensure you pasted the whole output.')

st.markdown('---')


# 3) Docker Functionality Test

st.header('3 — Docker Functionality Test')
st.markdown("*Docker allows running software in isolated containers. This ensures reproducible environments and simplifies running pre-packaged AI tools.*")
with st.expander("Quick manual Docker check — step-by-step", expanded=False):
    st.markdown("""
**1)** Open a terminal.

**2)** Verify Docker installation:
```
docker --version
```
**3)** Run a test container:
```
docker run hello-world
```
You should see a message confirming Docker can run containers.

**Docker documentation:** https://docs.docker.com/get-started/
""")

st.write('This will attempt to run `docker run --rm hello-world` which pulls a small image and runs it.')
if st.button('Run Docker hello-world'):
    st.session_state['docker_result'] = check_docker()
    ok = st.session_state['docker_result'][0]
    if ok:
        st.success('Docker ran hello-world successfully — cached for summary')
        st.code(st.session_state['docker_result'][1])
    else:
        st.error('Docker test failed or docker not found — cached result saved')
        if st.session_state['docker_result'][1]:
            st.code(st.session_state['docker_result'][1])
        if st.session_state['docker_result'][2]:
            st.code(st.session_state['docker_result'][2])
    st.info('If you receive permission errors, ensure your user can access the Docker daemon (e.g., add to docker group on Linux)')
else:
    if st.session_state['docker_result']:
        ok = st.session_state['docker_result'][0]
        if ok:
            st.success('Cached: Docker OK')
        else:
            st.error('Cached: Docker check failed')
        st.code(st.session_state['docker_result'][1])
    else:
        st.write('Press the button to attempt Docker checks — results will be cached')

st.markdown('---')




st.markdown("""### 4 — Useful documentation links

- [WSL (Windows Subsystem for Linux) docs](https://learn.microsoft.com/en-us/windows/wsl/)
- [NVIDIA CUDA on WSL guide](https://developer.nvidia.com/cuda/wsl)
- [Xarray documentation](https://docs.xarray.dev/en/stable/api.html)
- [Pytorch documentation](https://docs.pytorch.org/docs/stable/index.html)
- [Anemoi documentation](https://anemoi.readthedocs.io/projects/inference/en/latest/)
- [WeatherbenchX documentation](https://weatherbench-x.readthedocs.io/en/latest/)
- [Docker docs (installation and troubleshooting)](https://docs.docker.com/)
""")

st.markdown('---')
