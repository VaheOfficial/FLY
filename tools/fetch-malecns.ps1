<#
.SYNOPSIS
    Download the MaleCNS v1.0 flat-connectome tables that connectome-convert reads.

.DESCRIPTION
    Fetches the neuron annotations, neurotransmitter predictions, and connection
    weights tables from Janelia's public Google Cloud bucket into data/malecns-v1.0
    and verifies each file's MD5 against the bucket's published checksum.
    Files that already exist and verify are skipped, so re-running is cheap.

    Total download is about 1.1 GB. The data is CC-BY 4.0; cite the MaleCNS
    release (https://male-cns.janelia.org) if you redistribute anything derived
    from it.

.PARAMETER Destination
    Directory to download into. Defaults to data/malecns-v1.0 under the repo root.

.EXAMPLE
    .\tools\fetch-malecns.ps1
#>
[CmdletBinding()]
param(
    [string]$Destination = (Join-Path $PSScriptRoot '..\data\malecns-v1.0')
)

$ErrorActionPreference = 'Stop'

$Base = 'https://storage.googleapis.com/flyem-male-cns/v1.0/connectome-data/flat-connectome'

# Name, size in bytes, and base64 MD5 as reported by the bucket's object metadata.
$Files = @(
    @{ Name = 'body-annotations-male-cns-v1.0-minconf-0.5.feather';  Size = 14483314;   Md5 = 'UKdxh3DFciDxYLpPQxq4ng==' }
    @{ Name = 'body-neurotransmitters-male-cns-v1.0.feather';        Size = 43282834;   Md5 = 'PYQrEv5cSe763lKNfdJKHw==' }
    @{ Name = 'connectome-weights-male-cns-v1.0-minconf-0.5.feather'; Size = 1051241946; Md5 = '8w6dzKJc/QIb8eez2XVZng==' }
)

function Get-Md5Base64([string]$Path) {
    $md5 = [System.Security.Cryptography.MD5]::Create()
    $stream = [System.IO.File]::OpenRead($Path)
    try {
        [Convert]::ToBase64String($md5.ComputeHash($stream))
    } finally {
        $stream.Dispose()
        $md5.Dispose()
    }
}

$Destination = [System.IO.Path]::GetFullPath($Destination)
New-Item -ItemType Directory -Force -Path $Destination | Out-Null
Write-Host "Destination: $Destination"

$failed = 0
foreach ($f in $Files) {
    $target = Join-Path $Destination $f.Name
    $sizeMb = [math]::Round($f.Size / 1MB, 1)

    if (Test-Path $target) {
        if ((Get-Item $target).Length -eq $f.Size -and (Get-Md5Base64 $target) -eq $f.Md5) {
            Write-Host "ok       $($f.Name) ($sizeMb MB, already present)"
            continue
        }
        Write-Host "stale    $($f.Name) (checksum mismatch, re-downloading)"
        Remove-Item $target
    }

    Write-Host "fetching $($f.Name) ($sizeMb MB)"
    $partial = "$target.part"
    try {
        # curl is present on every supported Windows 10+ install and streams
        # large files without buffering them in memory.
        & curl.exe --fail --silent --show-error --location --output $partial "$Base/$($f.Name)"
        if ($LASTEXITCODE -ne 0) { throw "curl exited with $LASTEXITCODE" }

        $got = Get-Md5Base64 $partial
        if ($got -ne $f.Md5) {
            throw "checksum mismatch: expected $($f.Md5), got $got"
        }
        Move-Item -Force $partial $target
        Write-Host "verified $($f.Name)"
    } catch {
        Write-Host "FAILED   $($f.Name): $_" -ForegroundColor Red
        if (Test-Path $partial) { Remove-Item $partial }
        $failed++
    }
}

if ($failed -gt 0) {
    Write-Host "$failed file(s) failed." -ForegroundColor Red
    exit 1
}
Write-Host 'All MaleCNS tables present and verified.'
