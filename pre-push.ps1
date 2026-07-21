<#
    Windows equivalent of ./pre-push (the bash script) for boost-multi.

    Builds and tests the project with whatever compiler toolchains are actually
    installed on this machine, mirroring the spirit of the Linux/macOS script's
    multi-compiler matrix (several build dirs, warnings-as-errors, a sanitizer
    build, a static-analysis build, a formatting check) using the MSVC/LLVM
    tools that ship with Visual Studio.

    Detected on this machine: cl.exe (MSVC) + bundled Ninja + bundled
    clang-format/clang-tidy, via "Visual Studio Build Tools 2022".
    clang-cl.exe and ccache were NOT found; the script skips those steps
    with a warning instead of failing. To add them later:

      winget install Microsoft.VisualStudio.2022.BuildTools --override `
        "--add Microsoft.VisualStudio.Workload.VCTools --add Microsoft.VisualStudio.Component.VC.Llvm.Clang"
      winget install ccache   # or: scoop install ccache / choco install ccache

    Usage:
      pwsh -File .\pre-push.ps1                    # build + test everything
      pwsh -File .\pre-push.ps1 multi_array_ref     # limit to one target/ctest filter
#>

[CmdletBinding()]
param(
    [Parameter(Position = 0)]
    [string]$Target
)

$ErrorActionPreference = 'Stop'

$repoRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $repoRoot

function Write-Section([string]$Message) {
    Write-Host "`n=== $Message ===" -ForegroundColor Cyan
}

function Find-VsInstallation {
    $vswhere = Join-Path ${env:ProgramFiles(x86)} 'Microsoft Visual Studio\Installer\vswhere.exe'
    if (-not (Test-Path $vswhere)) { return $null }
    $installPath = & $vswhere -latest -products * `
        -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 `
        -property installationPath
    if (-not $installPath) { return $null }
    return $installPath.Trim()
}

function Import-VsDevEnvironment([string]$VsPath) {
    $vsDevCmd = Join-Path $VsPath 'Common7\Tools\VsDevCmd.bat'
    if (-not (Test-Path $vsDevCmd)) { throw "VsDevCmd.bat not found at $vsDevCmd" }
    # Use %ComSpec% rather than a bare 'cmd.exe': the latter depends on the
    # invoking shell's PATH containing System32, which isn't always true
    # (trimmed/restricted PATH, unusual launch context, etc.), whereas
    # ComSpec is set by Windows itself to cmd.exe's full path unconditionally.
    $comSpec = if ($env:ComSpec) { $env:ComSpec } else { Join-Path $env:SystemRoot 'System32\cmd.exe' }
    $envDump = & $comSpec /c "`"$vsDevCmd`" -arch=x64 -host_arch=x64 -no_logo && set"
    foreach ($line in $envDump) {
        if ($line -match '^([A-Za-z_][A-Za-z0-9_]*)=(.*)$') {
            [System.Environment]::SetEnvironmentVariable($Matches[1], $Matches[2], 'Process')
        }
    }
}

# ---- locate toolchain ----------------------------------------------------
$vsPath = Find-VsInstallation
if (-not $vsPath) {
    Write-Error "No Visual Studio C++ toolset found via vswhere. Install the 'Desktop development with C++' workload."
    exit 1
}
Write-Section "Using Visual Studio at $vsPath"
Import-VsDevEnvironment $vsPath

$ninja       = Join-Path $vsPath 'Common7\IDE\CommonExtensions\Microsoft\CMake\Ninja\ninja.exe'
$clangFormat = Join-Path $vsPath 'VC\Tools\Llvm\x64\bin\clang-format.exe'
$clangCl     = Join-Path $vsPath 'VC\Tools\Llvm\x64\bin\clang-cl.exe'

# Full path to the cl.exe that Import-VsDevEnvironment just put on PATH (from
# *this* $vsPath), so it can be pinned explicitly below instead of leaving
# CMake to cache whatever cl.exe/clang-cl.exe it first resolved. Otherwise, if
# a build dir was configured before a newer/older VS install appeared (e.g.
# after installing a preview alongside 2022), CMakeCache.txt keeps pointing at
# the old compiler binary while the environment's INCLUDE/LIB (re-imported
# fresh every run, above) point at the *new* install's MSVC STL headers --
# silent version-mismatch UB at best, a hard STL1000 static_assert at worst.
$clExe = (Get-Command cl.exe -ErrorAction SilentlyContinue).Source

if (Test-Path $ninja) {
    $env:PATH = "$(Split-Path $ninja);$env:PATH"
    $env:CMAKE_GENERATOR = 'Ninja'
    $env:NINJA_STATUS    = '[%f/%t||%r] '  # matches .gitlab-ci-correaa.yml's NINJA_STATUS
    Write-Host "Generator: Ninja ($ninja)"
} else {
    Write-Warning 'Bundled Ninja not found; falling back to the default Visual Studio generator (multi-config).'
}

$env:CMAKE_COLOR_DIAGNOSTICS = 'ON'

# ---- vcpkg (for find_package(Boost) to succeed) -----------------------------
$vcpkgToolchainArgs = @()
$vcpkgRoot = 'C:\vcpkg'
$vcpkgToolchain = Join-Path $vcpkgRoot 'scripts\buildsystems\vcpkg.cmake'
if (Test-Path $vcpkgToolchain) {
    $vcpkgToolchainArgs = @("-DCMAKE_TOOLCHAIN_FILE=$vcpkgToolchain")
    Write-Host "vcpkg detected at $vcpkgRoot; passing its toolchain file so find_package(Boost) can succeed."
} else {
    Write-Warning "vcpkg not found at $vcpkgRoot; Boost will likely not be found, so tests will not be built (see CMakeLists.txt's find_package(Boost) warning)."
}

$ccache = Get-Command ccache.exe -ErrorAction SilentlyContinue
if ($ccache) {
    $env:CMAKE_CXX_COMPILER_LAUNCHER = 'ccache'
    Write-Host 'ccache detected, using as compiler launcher.'
} else {
    Write-Warning 'ccache not found; builds will not be cached. (winget install ccache)'
}

# ---- parallelism -----------------------------------------------------------
$physCores = (Get-CimInstance Win32_Processor | Measure-Object -Property NumberOfCores -Sum).Sum
if (-not $physCores) { $physCores = [Environment]::ProcessorCount }
$env:CMAKE_BUILD_PARALLEL_LEVEL = "$physCores"
$env:CTEST_PARALLEL_LEVEL       = "$physCores"
$env:CTEST_OUTPUT_ON_FAILURE    = '1'

# ---- target / filter arg ----------------------------------------------------
$buildTargetArgs = @()
$ctestFilterArgs = @()
if ($Target) {
    Write-Host "Target/filter: $Target"
    $buildTargetArgs = @('--target', $Target)
    $ctestFilterArgs = @('-R', $Target)
} else {
    Write-Host 'No target arg; building/testing everything.'
}

$script:failed = @()
$script:noTests = @()

function Invoke-Variant {
    param(
        [string]$Name,
        [string]$BuildDir,
        [string[]]$ConfigureArgs,
        [string]$Config = 'Debug'
    )
    Write-Section $Name
    try {
        # Always reconfigure (cheap no-op when nothing changed) instead of only
        # configuring on first creation of $BuildDir: skipping it left stale
        # CMakeCache.txt entries (e.g. a compiler path from a VS install that's
        # since been superseded) silently un-synced with the freshly re-imported
        # environment above, see $clExe comment.
        # --no-warn-unused-cli: on a reconfigure of an *existing* cache,
        # CMAKE_TOOLCHAIN_FILE is legitimately re-derived from the cache rather
        # than re-consumed (it only matters before the first project()), so
        # CMake would otherwise flag our always-passed -DCMAKE_TOOLCHAIN_FILE
        # as "unused" every run.
        & cmake --no-warn-unused-cli -S . -B $BuildDir @ConfigureArgs @vcpkgToolchainArgs
        if ($LASTEXITCODE -ne 0) { throw 'configure failed' }

        & cmake --build $BuildDir --config $Config @buildTargetArgs
        if ($LASTEXITCODE -ne 0) { throw 'build failed' }

        $ctestOutput = & ctest --test-dir $BuildDir -C $Config @ctestFilterArgs 2>&1
        $ctestOutput | ForEach-Object { Write-Host $_ }
        if ($LASTEXITCODE -ne 0) {
            & ctest --test-dir $BuildDir -C $Config --rerun-failed @ctestFilterArgs
            if ($LASTEXITCODE -ne 0) { throw 'tests failed' }
        }
        if ($ctestOutput -match 'No tests were found') {
            Write-Warning "$Name`: no tests were actually compiled/run (likely CMakeLists.txt's find_package(Boost) failed to locate Boost -- see the 'Cannot find Boost' warning above). This variant only verified the headers compile, nothing was tested."
            $script:noTests += $Name
        }
    } catch {
        Write-Warning "$Name FAILED: $_"
        $script:failed += $Name
    }
}

# Pin cl.exe by full path (recomputed above from the current $vsPath) on every
# configure, the same way the clang-cl variant below already pins clang-cl --
# otherwise CMake just keeps whatever compiler path got cached the first time
# this build dir was configured, even after a newer VS install changes which
# cl.exe/headers Import-VsDevEnvironment puts on PATH.
$msvcCompilerArgs = @()
if ($clExe) {
    $msvcCompilerArgs = @("-DCMAKE_C_COMPILER=$clExe", "-DCMAKE_CXX_COMPILER=$clExe")
} else {
    Write-Warning 'cl.exe not found on PATH after importing the VS dev environment; letting CMake auto-detect the compiler (may pick up a stale cached one on reconfigure).'
}

# ---- variant 1: MSVC debug, warnings as errors ------------------------------
Invoke-Variant -Name 'MSVC (cl.exe) Debug -WX' -BuildDir '.build.msvc' -Config 'Debug' -ConfigureArgs (@(
    '-DCMAKE_BUILD_TYPE=Debug',
    '-DCMAKE_COMPILE_WARNING_AS_ERROR=ON'
) + $msvcCompilerArgs)

# ---- variant 2: MSVC + AddressSanitizer -------------------------------------
# RelWithDebInfo (not Debug) because CMake's default Debug flags add /RTC1,
# which MSVC refuses to combine with /fsanitize=address.
# /Zi + linker /DEBUG: without debug info, cl.exe emits warning C5072 ("ASAN
# enabled without debug information emission") and ASAN error reports fall
# back to raw addresses instead of symbolized file/line stack traces.
Invoke-Variant -Name 'MSVC (cl.exe) AddressSanitizer' -BuildDir '.build.msvc.asan' -Config 'RelWithDebInfo' -ConfigureArgs (@(
    '-DCMAKE_BUILD_TYPE=RelWithDebInfo',
    '-DCMAKE_CXX_FLAGS=/fsanitize=address /Zi',
    '-DCMAKE_EXE_LINKER_FLAGS=/DEBUG',
    '-DCMAKE_SHARED_LINKER_FLAGS=/DEBUG'
) + $msvcCompilerArgs)

# ---- variant 3: MSVC release, C++23 ------------------------------------------
# NOTE: clang-tidy static analysis was tried here but disabled again: the
# bundled clang-tidy (19.1.5, same version-drift problem as clang-format
# above) misdiagnoses "cannot use throw/try with exceptions disabled" even
# though /EHsc is passed -- a known clang-tidy/cl.exe driver-mode rough edge
# -- on top of extra misc-include-cleaner complaints not seen on the repo's
# pinned Linux toolchain. Re-enable via CMAKE_CXX_CLANG_TIDY once a matching
# clang-tidy version is confirmed available.
Invoke-Variant -Name 'MSVC (cl.exe) Release C++23' -BuildDir '.build.msvc.c++23' -Config 'Release' -ConfigureArgs (@(
    '-DCMAKE_BUILD_TYPE=Release',
    '-DCMAKE_CXX_STANDARD=23'
) + $msvcCompilerArgs)

# ---- optional variant: clang-cl, only if the LLVM component is installed ---
# NOTE: "-T ClangCL" only works with the Visual Studio generator; with the
# Ninja generator (forced above via CMAKE_GENERATOR when bundled Ninja is
# found), clang-cl must be selected directly via CMAKE_<LANG>_COMPILER.
if (Test-Path $clangCl) {
    Invoke-Variant -Name 'clang-cl Debug -WX' -BuildDir '.build.clangcl' -Config 'Debug' -ConfigureArgs @(
        '-DCMAKE_BUILD_TYPE=Debug',
        '-DCMAKE_COMPILE_WARNING_AS_ERROR=ON',
        "-DCMAKE_C_COMPILER=$clangCl",
        "-DCMAKE_CXX_COMPILER=$clangCl"
    )
} else {
    Write-Warning 'clang-cl.exe not found; skipping clang-cl variant. (Install the "C++ Clang Compiler for Windows" component to enable it.)'
}

# ---- clang-format check ------------------------------------------------------
# Files are checked out with CRLF (core.autocrlf=true), but clang-format's
# style rules are LF-based, so a naive comparison flags every line ending in
# every file as a "diff". Normalize both sides to LF before comparing.
function Test-ClangFormatFile {
    param(
        [string]$Path,
        [string]$ClangFormatExe,
        [string]$StyleFile
    )
    $originalLF = (Get-Content -Raw -LiteralPath $Path) -replace "`r`n", "`n"

    $psi = [System.Diagnostics.ProcessStartInfo]::new()
    $psi.FileName = $ClangFormatExe
    $psi.ArgumentList.Add("--style=file:$StyleFile")
    $psi.ArgumentList.Add("--assume-filename=$Path")
    $psi.RedirectStandardInput = $true
    $psi.RedirectStandardOutput = $true
    $psi.StandardInputEncoding = [System.Text.Encoding]::UTF8
    $psi.StandardOutputEncoding = [System.Text.Encoding]::UTF8
    $psi.UseShellExecute = $false

    $proc = [System.Diagnostics.Process]::Start($psi)
    $proc.StandardInput.Write($originalLF)
    $proc.StandardInput.Close()
    $formattedLF = $proc.StandardOutput.ReadToEnd() -replace "`r`n", "`n"
    $proc.WaitForExit()

    return $formattedLF -eq $originalLF
}

Write-Section 'clang-format check'
# The repo's committed style baseline is produced by clang-format-21 (see the
# Linux pre-push script, which specifically requires clang-format-21 and
# otherwise skips the check). Different clang-format versions disagree on
# edge cases (e.g. "double (&)[2]" vs "double(&)[2]"), so running any other
# version here would flag near every file as misformatted even though it
# isn't. Mirror the Linux script: only enforce the check on a matching
# major version; otherwise skip with a warning instead of failing.
$requiredClangFormatMajor = 21
$clangFormatVersionOk = $false
if (Test-Path $clangFormat) {
    $versionOutput = & $clangFormat --version
    if ($versionOutput -match 'version (\d+)') {
        $foundMajor = [int]$Matches[1]
        $clangFormatVersionOk = ($foundMajor -eq $requiredClangFormatMajor)
        if (-not $clangFormatVersionOk) {
            Write-Warning "clang-format-$requiredClangFormatMajor not installed (found version $foundMajor at $clangFormat); skipping formatting check to avoid false positives from version drift."
        }
    }
}
if ($clangFormatVersionOk) {
    $styleFile = Join-Path $repoRoot '.clang-format'
    $files = Get-ChildItem -Path 'include', 'test' -Recurse -Include '*.hpp', '*.cpp' -ErrorAction SilentlyContinue
    $bad = @()
    foreach ($f in $files) {
        if (-not (Test-ClangFormatFile -Path $f.FullName -ClangFormatExe $clangFormat -StyleFile $styleFile)) {
            $bad += $f.FullName
        }
    }
    if ($bad.Count -gt 0) {
        Write-Warning "clang-format found formatting issues in:`n$($bad -join "`n")"
        $script:failed += 'clang-format'
    } else {
        Write-Host 'clang-format: OK'
    }
} elseif (-not (Test-Path $clangFormat)) {
    Write-Warning "clang-format not found at $clangFormat; skipping."
}

# ---- summary ------------------------------------------------------------------
if ($script:noTests.Count -gt 0) {
    Write-Host "`nNOTE: these variants compiled but ran no tests (Boost not found by CMake): $($script:noTests -join ', ')" -ForegroundColor Yellow
    Write-Host "      Install Boost so CMake can find it, e.g.: vcpkg install boost-serialization && cmake ... -DCMAKE_TOOLCHAIN_FILE=<vcpkg>/scripts/buildsystems/vcpkg.cmake" -ForegroundColor Yellow
}
if ($script:failed.Count -gt 0) {
    Write-Host "`nFAILED: $($script:failed -join ', ')" -ForegroundColor Red
    exit 666
} else {
    Write-Host "`nAll variants passed." -ForegroundColor Green
    exit 0
}
