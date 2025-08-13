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


def check_conda():
    if not shutil.which('conda'):
        return False, '', 'conda not found in PATH', 127
    ok_ver, out_ver, err_ver, rc_ver = run_cmd(['conda', '--version'], timeout=6)
    ok_env, out_env, err_env, rc_env = run_cmd(['conda', 'env', 'list'], timeout=8)
    return (ok_ver and ok_env), (out_ver + '\n' + out_env), (err_ver + '\n' + err_env), (rc_ver or rc_env)


def check_docker():
    if not shutil.which('docker'):
        return False, '', 'docker not found in PATH', 127
    ok_ver, out_ver, err_ver, rc_ver = run_cmd(['docker', '--version'], timeout=6)
    ok_run, out_run, err_run, rc_run = run_cmd(['docker', 'run', '--rm', 'hello-world'], timeout=30)
    ok = ok_ver and ok_run
    out = out_ver + '\n' + out_run
    err = err_ver + '\n' + err_run
    return ok, out, err, (rc_ver or rc_run)


def check_python_libs(lib_list):
    results = {}
    for lib in lib_list:
        try:
            m = importlib.import_module(lib)
            ver = getattr(m, '__version__', None)
            extra = ''
            if lib == 'torch':
                try:
                    cuda_available = m.cuda.is_available()
                    extra = f"CUDA available: {cuda_available}"
                except Exception:
                    extra = "CUDA check failed"
            results[lib] = {'installed': True, 'version': str(ver), 'extra': extra}
        except Exception as e:
            results[lib] = {'installed': False, 'version': None, 'extra': str(e)}
    return results


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
    "Conda Environment Check",
    "Docker Functionality Test",
    "Core Python Library Check",
    "Summary & Docs"
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

col1, col2 = st.columns(2)

with col1:
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

with col2:
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
# 3) Conda Environment Check
###
### TODO: ADD conda env name ###
###

st.header('3 — Conda Environment Check')
st.markdown("*Conda is a package and environment manager that lets you isolate dependencies for different projects. Using the correct Conda environment ensures the AI models run with the exact library versions and settings they need for reliable results.*")
with st.expander("Quick manual Conda environment check — step-by-step", expanded=False):
    st.markdown("""
**1)** Open a terminal (PowerShell, Command Prompt, or WSL shell).

**2)** Activate the course environment (replace `<env_name>` with the given name):
```
conda activate <env_name>
```
If you get an error, you may need to first run:
```
conda init
```
then restart your shell.

**3)** Verify installed packages:
```
conda list
```
Scroll through and confirm key packages like `xarray`, `torch`, and `weatherbench2` are present.

**4)** (Optional) Check Python version:
```
python --version
```
It should match the course requirements.

**Official Conda documentation:** https://docs.conda.io/projects/conda/en/latest/user-guide/index.html
""")
if st.button('Run Conda checks'):
    st.session_state['conda_result'] = check_conda()
    ok = st.session_state['conda_result'][0]
    if ok:
        st.success('Conda appears available — result cached for summary')
        st.code(st.session_state['conda_result'][1])
    else:
        st.error('Conda checks failed or conda is not found — cached result saved')
        if st.session_state['conda_result'][1]:
            st.code(st.session_state['conda_result'][1])
        if st.session_state['conda_result'][2]:
            st.code(st.session_state['conda_result'][2])
    st.info('If conda is not found, ensure conda is installed and on PATH, or activate the environment inside the terminal you use for development')
else:
    if st.session_state['conda_result']:
        ok = st.session_state['conda_result'][0]
        if ok:
            st.success('Cached: Conda available')
        else:
            st.error('Cached: Conda missing or check failed')
        st.code(st.session_state['conda_result'][1])
    else:
        st.write('Click "Run Conda checks" to try `conda --version` and `conda env list` from this Streamlit process — results will be cached')

st.markdown('---')

# 4) Docker Functionality Test

st.header('4 — Docker Functionality Test')
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

# 5) Core Python Library Check

st.header('5 — Core Python Library Check')
st.markdown("*Ensure core Python libraries like xarray, torch, and weatherbench2 are installed and accessible, as they are essential for running the model and the notebooks.*")
st.write('This attempts to import the main libraries used in the course: xarray, torch, weatherbench2')
libs_to_check = ['xarray', 'torch', 'weatherbench2']
if st.button('Run Python imports'):
    st.session_state['libs_result'] = check_python_libs(libs_to_check)
    st.success('Import checks ran and results cached for summary')

if st.session_state['libs_result']:
    for lib, r in st.session_state['libs_result'].items():
        if r['installed']:
            st.success(f"{lib} installed — version: {r['version']} {r['extra']}")
        else:
            st.error(f"{lib} NOT installed — error: {r['extra']}")
else:
    st.write('Click to run import checks in the current Streamlit/Python process — results will be cached')

st.markdown('---')

# 6) Summary and Resources

st.header('6 — Summary & Resources (reads cached results)')
st.markdown("*Collect all check results to quickly identify if the system is ready or requires further setup.*")

if st.button('Generate summary report'):
    # Build the summary from cached session state only
    summary = {
        'timestamp': datetime.now(timezone.utc).isoformat(),
        'sysinfo': st.session_state.get('sysinfo'),
        'gpu_auto_check': None,
        'gpu_wsl_paste': None,
        'conda': None,
        'docker': None,
        'libs': st.session_state.get('libs_result')
    }

    if st.session_state.get('gpu_result'):
        ok, out, err, rc = st.session_state['gpu_result']
        summary['gpu_auto_check'] = {'ok': ok, 'out': out[:1000], 'err': err[:1000], 'rc': rc}
    if st.session_state.get('gpu_paste_text'):
        summary['gpu_wsl_paste'] = {
            'detected_header': bool(st.session_state.get('gpu_paste_detected')),
            'sample': st.session_state.get('gpu_paste_text')[:1000]
        }
    if st.session_state.get('conda_result'):
        ok, out, err, rc = st.session_state['conda_result']
        summary['conda'] = {'ok': ok, 'out': out[:1000], 'err': err[:1000], 'rc': rc}
    if st.session_state.get('docker_result'):
        ok, out, err, rc = st.session_state['docker_result']
        summary['docker'] = {'ok': ok, 'out': out[:1000], 'err': err[:1000], 'rc': rc}

    # Derive high-level status
    checks_run = {
        'system': bool(st.session_state.get('sysinfo')),
        'gpu_auto': bool(st.session_state.get('gpu_result')),
        'gpu_wsl_paste': bool(st.session_state.get('gpu_paste_text')),
        'conda': bool(st.session_state.get('conda_result')),
        'docker': bool(st.session_state.get('docker_result')),
        'libs': bool(st.session_state.get('libs_result'))
    }

    def ok_bool(obj):
        if not obj:
            return False
        if isinstance(obj, tuple) or isinstance(obj, list):
            return bool(obj[0])
        return bool(obj)

    derived_ok = {
        'system': checks_run['system'],
        'gpu_auto': ok_bool(st.session_state.get('gpu_result')),
        'gpu_wsl_paste': bool(st.session_state.get('gpu_paste_detected')),
        'conda': ok_bool(st.session_state.get('conda_result')),
        'docker': ok_bool(st.session_state.get('docker_result')),
        'libs': all([v.get('installed', False) for v in (st.session_state.get('libs_result') or {}).values()]) if st.session_state.get('libs_result') else False
    }

    overall_ok = all(derived_ok.values())

    # UI summary
    if overall_ok:
        st.success('All cached checks look good 👍')
    else:
        st.warning('Some checks are missing or reported issues — review sections below')

    st.subheader('Quick status')
    c1, c2, c3 = st.columns(3)
    c1.metric('System', 'OK' if derived_ok['system'] else 'Missing')
    c2.metric('GPU', 'OK' if (derived_ok['gpu_auto'] or derived_ok['gpu_wsl_paste']) else 'Missing')
    c3.metric('Conda', 'OK' if derived_ok['conda'] else 'Missing')
    c1.metric('Docker', 'OK' if derived_ok['docker'] else 'Missing')
    c2.metric('Python libs', 'OK' if derived_ok['libs'] else 'Missing')

    st.markdown('---')

    # Expanders for each cached result
    with st.expander('System info (cached)'):
        if summary['sysinfo']:
            st.json(summary['sysinfo'])
        else:
            st.info('System check was not run — go to step 1 and run system checks')

    with st.expander('GPU automatic check (cached)'):
        if summary['gpu_auto_check']:
            st.write('OK' if summary['gpu_auto_check']['ok'] else 'Error')
            st.code(summary['gpu_auto_check']['out'] or summary['gpu_auto_check']['err'])
        else:
            st.info('Automatic nvidia-smi was not run from this Streamlit process')

    with st.expander('GPU WSL paste (cached)'):
        if summary['gpu_wsl_paste']:
            st.write('NVIDIA-SMI header detected' if summary['gpu_wsl_paste']['detected_header'] else 'Header not detected')
            st.code(summary['gpu_wsl_paste']['sample'])
        else:
            st.info('No pasted WSL output found in this session')

    with st.expander('Conda check (cached)'):
        if summary['conda']:
            st.write('OK' if summary['conda']['ok'] else 'Error')
            st.code(summary['conda']['out'] or summary['conda']['err'])
        else:
            st.info('Conda check not run in this session')

    with st.expander('Docker check (cached)'):
        if summary['docker']:
            st.write('OK' if summary['docker']['ok'] else 'Error')
            st.code(summary['docker']['out'] or summary['docker']['err'])
        else:
            st.info('Docker check not run in this session')

    with st.expander('Python libraries (cached)'):
        if summary['libs']:
            st.json(summary['libs'])
        else:
            st.info('Import checks not run in this session')

    # Final JSON + download
    summary_md = json.dumps({'derived_ok': derived_ok, 'summary': summary}, indent=2)
    st.download_button('Download JSON summary (cached)', data=summary_md, file_name='env_summary_cached.json', mime='application/json')

else:
    st.info('Click "Generate summary report" to assemble a report from cached results')

st.markdown("""### Useful documentation links

- [WSL (Windows Subsystem for Linux) docs](https://learn.microsoft.com/en-us/windows/wsl/)
- [NVIDIA CUDA on WSL guide](https://developer.nvidia.com/cuda/wsl)
- [Conda documentation](https://docs.conda.io/)
- [Docker docs (installation and troubleshooting)](https://docs.docker.com/)
""")

st.markdown('---')
