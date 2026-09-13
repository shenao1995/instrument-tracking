"""Prepare project-local official CUDA/MSVC/Windows SDK files; build nvdiffrast.

No administrator installer, driver change, registry edit, or global PATH change.
Downloads come from NVIDIA, Microsoft's VS distribution/NuGet, and NVlabs GitHub.
Source and toolchain provenance are saved in .tools/provenance.json.
"""
from __future__ import annotations
import argparse
import hashlib
import io
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tarfile
import zipfile
from urllib.request import Request, urlopen

ROOT = Path(__file__).resolve().parent
TOOLS = ROOT/".tools"
CACHE = TOOLS/"cache"
NVDIFFRAST_COMMIT = "253ac4fcea7de5f396371124af597e6cc957bfae"


def get_json(url):
    return json.loads(urlopen(Request(url,headers={"User-Agent":"instrument-pose-renderer-setup"}),timeout=60).read())


def download(url, filename, sha256=None):
    path = CACHE/filename
    CACHE.mkdir(parents=True,exist_ok=True)
    if not path.exists():
        print(f"Downloading {filename}",flush=True)
        temporary = path.with_suffix(path.suffix+".part")
        with urlopen(Request(url,headers={"User-Agent":"instrument-pose-renderer-setup"}),timeout=60) as response, temporary.open("wb") as dst:
            shutil.copyfileobj(response,dst,1024*1024)
        temporary.replace(path)
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    if sha256 and digest.lower() != sha256.lower():
        raise ValueError(f"SHA256 mismatch for {path}")
    return path, {"url":url,"sha256":digest,"file":str(path.relative_to(ROOT))}


def extract(path, destination, strip_root=False, prefix=None):
    destination = destination.resolve()
    with zipfile.ZipFile(path) as archive:
        for entry in archive.infolist():
            if entry.is_dir():
                continue
            name = entry.filename.replace("\\","/")
            if prefix:
                if not name.startswith(prefix):
                    continue
                name = name[len(prefix):]
            elif strip_root:
                if "/" not in name:
                    continue
                name = name.split("/",1)[1]
            target = (destination/name).resolve()
            if destination not in target.parents:
                raise ValueError("Archive path escapes target")
            if target.exists() and target.stat().st_size == entry.file_size:
                continue
            target.parent.mkdir(parents=True,exist_ok=True)
            with archive.open(entry) as src,target.open("wb") as dst:
                shutil.copyfileobj(src,dst)


def prepare():
    provenance_path = TOOLS/"provenance.json"
    provenance = json.loads(provenance_path.read_text()) if provenance_path.exists() else []
    cuda = get_json("https://developer.download.nvidia.com/compute/cuda/redist/redistrib_12.6.3.json")
    for package in ("cuda_nvcc","cuda_cudart","cuda_cccl"):
        item = cuda[package]["windows-x86_64"]
        path,record = download("https://developer.download.nvidia.com/compute/cuda/redist/"+item["relative_path"],Path(item["relative_path"]).name,item["sha256"])
        provenance.append(record)
        extract(path,TOOLS/"cuda",strip_root=True)
    # PyTorch's CUDA headers transitively include BLAS/sparse/solver headers.
    # NVIDIA's dev-only Conda packages avoid downloading duplicate runtime DLLs.
    import zstandard
    for package,version in (("libcusparse-dev","12.5.4.2"),("libcublas-dev","12.6.4.1"),("libcusolver-dev","11.7.1.2")):
        release = get_json(f"https://api.anaconda.org/release/nvidia/{package}/{version}")
        item = next(d for d in release["distributions"] if d["basename"].startswith("win-64/"))
        path,record = download("https:"+item["download_url"],Path(item["basename"]).name,item["sha256"])
        provenance.append(record)
        with zipfile.ZipFile(path) as archive:
            payload = next(n for n in archive.namelist() if n.startswith("pkg-") and n.endswith(".tar.zst"))
            with zstandard.ZstdDecompressor().stream_reader(io.BytesIO(archive.read(payload))) as stream:
                with tarfile.open(fileobj=stream,mode="r|") as contents:
                    for entry in contents:
                        if not entry.isfile() or not entry.name.startswith("Library/include/"):
                            continue
                        destination = (TOOLS/"cuda/include").resolve()
                        target = (destination/entry.name[len("Library/include/"):]).resolve()
                        if destination not in target.parents:
                            raise ValueError("Header path escapes CUDA include directory")
                        target.parent.mkdir(parents=True,exist_ok=True)
                        with contents.extractfile(entry) as src,target.open("wb") as dst:
                            shutil.copyfileobj(src,dst)
    manifest_path = CACHE/"vs_manifest.json"
    if not manifest_path.exists():
        channel = get_json("https://aka.ms/vs/17/release/channel")
        info = next(item for item in channel["channelItems"] if item["id"] == "Microsoft.VisualStudio.Manifests.VisualStudio")["payloads"][0]
        _,record = download(info["url"],manifest_path.name,info["sha256"])
        provenance.append(record)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8-sig"))
    wanted = {
        "Microsoft.VC.14.38.17.8.Tools.HostX64.TargetX64.base":None,
        "Microsoft.VC.14.38.17.8.Tools.HostX64.TargetX64.Res.base":"en-US",
        "Microsoft.VC.14.38.17.8.CRT.Headers.base":None,
        "Microsoft.VC.14.38.17.8.CRT.x64.Desktop.base":None,
        "Microsoft.VC.14.38.17.8.CRT.x64.Store.base":None,
    }
    for ident,language in wanted.items():
        package = next(p for p in manifest["packages"] if p["id"] == ident and p.get("language") == language)
        for item in package["payloads"]:
            path,record = download(item["url"],Path(item["fileName"]).name,item["sha256"])
            provenance.append(record)
            extract(path,TOOLS/"msvc",prefix="Contents/")
    sdk_version = "10.0.22621.3233"
    for package in ("microsoft.windows.sdk.cpp","microsoft.windows.sdk.cpp.x64"):
        filename = f"{package}.{sdk_version}.nupkg"
        path,record = download(f"https://api.nuget.org/v3-flatcontainer/{package}/{sdk_version}/{filename}",filename)
        provenance.append(record)
        extract(path,TOOLS/"winsdk"/package)
    source_path = ROOT/".reference/nvdiffrast"
    if not (source_path/"setup.py").exists():
        commit = NVDIFFRAST_COMMIT
        path,record = download(f"https://codeload.github.com/NVlabs/nvdiffrast/zip/{commit}",f"nvdiffrast-{commit}.zip")
        record["commit"] = commit
        provenance.append(record)
        extract(path,source_path,strip_root=True)
    # NVCC checks this layout marker even with --use-local-env. All actual
    # INCLUDE/LIB/PATH setup is provided by build_environment, not by this file.
    marker = TOOLS/"msvc/VC/Auxiliary/Build/vcvarsall.bat"
    marker.parent.mkdir(parents=True,exist_ok=True)
    marker.write_text("@rem Project-local toolchain: environment supplied by setup_nvdiffrast_windows.py\n",encoding="ascii")
    unique = {item["file"]:item for item in provenance}
    provenance_path.write_text(json.dumps(list(unique.values()),indent=2),encoding="utf-8")
    print("Project-local dependencies extracted.",flush=True)


def build_environment():
    import torch
    if torch.version.cuda != "12.6":
        raise RuntimeError("This local toolchain targets torch CUDA 12.6; use a matching toolkit for your torch build")
    env = os.environ.copy()
    toolset = sorted((TOOLS/"msvc/VC/Tools/MSVC").iterdir())[-1]
    sdk_root = TOOLS/"winsdk/microsoft.windows.sdk.cpp/c"
    sdk_x64 = TOOLS/"winsdk/microsoft.windows.sdk.cpp.x64/c"
    sdk_version = sorted((sdk_root/"Include").iterdir())[-1].name
    include = [toolset/"include"]+[sdk_root/"Include"/sdk_version/name for name in ("ucrt","shared","um","winrt")]
    lib = [toolset/"lib/x64",sdk_x64/"ucrt/x64",sdk_x64/"um/x64"]
    bins = [toolset/"bin/Hostx64/x64",TOOLS/"cuda/bin",ROOT/".venv/Scripts",sdk_root/"bin"/sdk_version/"x64"]
    env.update({"CUDA_HOME":str(TOOLS/"cuda"),"CUDA_PATH":str(TOOLS/"cuda"),
                "INCLUDE":";".join(map(str,include)),"LIB":";".join(map(str,lib)),
                "PATH":";".join(map(str,bins))+";"+env["PATH"],
                "DISTUTILS_USE_SDK":"1","MSSdk":"1","VSCMD_ARG_TGT_ARCH":"x64",
                "TORCH_CUDA_ARCH_LIST":env.get("TORCH_CUDA_ARCH_LIST","8.9"),"MAX_JOBS":env.get("MAX_JOBS","4")})
    for path in include+lib:
        if not path.exists():
            raise FileNotFoundError(path)
    return env


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepare-only",action="store_true")
    parser.add_argument("--build-only",action="store_true")
    args = parser.parse_args()
    if os.name != "nt":
        raise RuntimeError("This helper targets Windows x64")
    if not args.build_only:
        prepare()
    if not args.prepare_only:
        env = build_environment()
        subprocess.run([str(TOOLS/"cuda/bin/nvcc.exe"),"--version"],env=env,check=True)
        subprocess.run([sys.executable,"-m","pip","install","--no-build-isolation","--no-deps",str(ROOT/".reference/nvdiffrast")],env=env,check=True)
