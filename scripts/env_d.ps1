# Redirect every temp/cache/config location to the D: drive for this PowerShell session.
# Usage (from repo root):  . .\scripts\env_d.ps1
$SYS = "D:\MV Project\_sys"

$vars = [ordered]@{
    TEMP                  = "$SYS\tmp"
    TMP                   = "$SYS\tmp"
    TMPDIR                = "$SYS\tmp"
    PIP_CACHE_DIR         = "$SYS\pip_cache"
    MPLCONFIGDIR          = "$SYS\mpl"
    TORCH_HOME            = "$SYS\torch"
    JOBLIB_TEMP_FOLDER    = "$SYS\joblib"
    XDG_CACHE_HOME        = "$SYS\cache"
    HF_HOME               = "$SYS\cache"
    PYTHONPYCACHEPREFIX   = "$SYS\pycache"
    PYTHONUSERBASE        = "$SYS\userbase"
    JUPYTER_CONFIG_DIR    = "$SYS\jupyter\config"
    JUPYTER_DATA_DIR      = "$SYS\jupyter\data"
    JUPYTER_RUNTIME_DIR   = "$SYS\jupyter\runtime"
    IPYTHONDIR            = "$SYS\ipython"
    OMP_NUM_THREADS       = "1"
    MKL_NUM_THREADS       = "1"
    OPENBLAS_NUM_THREADS  = "1"
}

foreach ($name in $vars.Keys) {
    $value = $vars[$name]
    if ($value -like "D:\*") { New-Item -ItemType Directory -Force -Path $value | Out-Null }
    Set-Item -Path "Env:$name" -Value $value
    Write-Host ("{0,-22} = {1}" -f $name, $value)
}
