Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

$sourceRoot = [System.IO.Path]::GetFullPath(
    "D:\Johsnon Lab\GCaMP6s_EX37x_Days_Repeating"
)
$destinationRoot = [System.IO.Path]::GetFullPath(
    "C:\Users\mzinn1\Desktop\DailyRecordings_LateTimepoints"
)

function Assert-PathUnderRoot {
    param(
        [Parameter(Mandatory = $true)][string]$Path,
        [Parameter(Mandatory = $true)][string]$Root
    )

    $fullPath = [System.IO.Path]::GetFullPath($Path)
    $fullRoot = [System.IO.Path]::GetFullPath($Root).TrimEnd("\") + "\"
    if (-not $fullPath.StartsWith(
        $fullRoot,
        [System.StringComparison]::OrdinalIgnoreCase
    )) {
        throw "Path is outside the intended root: $fullPath"
    }
    return $fullPath
}

if (-not (Test-Path -LiteralPath $sourceRoot -PathType Container)) {
    throw "Source root does not exist: $sourceRoot"
}
if (-not (Test-Path -LiteralPath $destinationRoot -PathType Container)) {
    throw "Destination root does not exist: $destinationRoot"
}

# Normalize the existing late IOBP folders so the notebook parses recording ID
# and day in the same way as the BP and historical folders.
$renamedFolders = 0
$alreadyNormalizedFolders = 0
foreach ($day in 13, 21, 28) {
    foreach ($replicate in 1..5) {
        $oldName = "${day}-${replicate}_IOBP"
        $newName = "1-${replicate}_Day${day}"
        $oldPath = Assert-PathUnderRoot `
            -Path (Join-Path $destinationRoot "IOBP\$oldName") `
            -Root $destinationRoot
        $newPath = Assert-PathUnderRoot `
            -Path (Join-Path $destinationRoot "IOBP\$newName") `
            -Root $destinationRoot

        $oldExists = Test-Path -LiteralPath $oldPath -PathType Container
        $newExists = Test-Path -LiteralPath $newPath -PathType Container
        if ($oldExists -and $newExists) {
            throw "Both old and normalized IOBP folders exist: $oldPath and $newPath"
        }
        if ($oldExists) {
            Move-Item -LiteralPath $oldPath -Destination $newPath
            $renamedFolders += 1
        } elseif ($newExists) {
            $alreadyNormalizedFolders += 1
        } else {
            throw "Expected late IOBP folder was not found: $oldPath"
        }
    }
}

$metricsCopied = 0
$iscellCopied = 0
$totalBytesCopied = [int64]0
$recordingsCopied = 0

foreach ($treatment in "BP", "IOBP") {
    $sourceTreatment = Join-Path $sourceRoot $treatment
    $destinationTreatment = Assert-PathUnderRoot `
        -Path (Join-Path $destinationRoot $treatment) `
        -Root $destinationRoot
    New-Item -ItemType Directory -Path $destinationTreatment -Force | Out-Null

    $recordingFolders = Get-ChildItem -LiteralPath $sourceTreatment -Directory |
        Where-Object {
            $_.Name -notmatch "(?i)^(new[ _-]*data|metrics)$"
        } |
        Sort-Object Name

    foreach ($recordingFolder in $recordingFolders) {
        $sourceMetricsDirectory = Join-Path $recordingFolder.FullName "metrics"
        $metricsFiles = @(
            Get-ChildItem `
                -LiteralPath $sourceMetricsDirectory `
                -Filter "*_metrics.xlsx" `
                -File `
                -ErrorAction SilentlyContinue
        )

        if ($metricsFiles.Count -eq 0) {
            continue
        }
        if ($metricsFiles.Count -ne 1) {
            throw "Expected exactly one metrics workbook in $sourceMetricsDirectory"
        }

        $destinationRecording = Assert-PathUnderRoot `
            -Path (Join-Path $destinationTreatment $recordingFolder.Name) `
            -Root $destinationRoot
        $destinationMetrics = Assert-PathUnderRoot `
            -Path (Join-Path $destinationRecording "metrics") `
            -Root $destinationRoot
        New-Item -ItemType Directory -Path $destinationMetrics -Force | Out-Null

        $metricsDestinationPath = Assert-PathUnderRoot `
            -Path (Join-Path $destinationMetrics $metricsFiles[0].Name) `
            -Root $destinationRoot
        Copy-Item `
            -LiteralPath $metricsFiles[0].FullName `
            -Destination $metricsDestinationPath `
            -Force
        $metricsCopied += 1
        $recordingsCopied += 1
        $totalBytesCopied += $metricsFiles[0].Length

        $sourceIscell = Join-Path `
            $recordingFolder.FullName `
            "suite2p\plane0\iscell.npy"
        if (Test-Path -LiteralPath $sourceIscell -PathType Leaf) {
            $destinationPlane = Assert-PathUnderRoot `
                -Path (Join-Path $destinationRecording "suite2p\plane0") `
                -Root $destinationRoot
            New-Item -ItemType Directory -Path $destinationPlane -Force | Out-Null
            $destinationIscell = Assert-PathUnderRoot `
                -Path (Join-Path $destinationPlane "iscell.npy") `
                -Root $destinationRoot
            Copy-Item `
                -LiteralPath $sourceIscell `
                -Destination $destinationIscell `
                -Force
            $iscellCopied += 1
            $totalBytesCopied += (Get-Item -LiteralPath $sourceIscell).Length
        }
    }
}

if ($recordingsCopied -ne 79) {
    throw "Copied $recordingsCopied recording workbooks; expected 79."
}
if ($metricsCopied -ne 79) {
    throw "Copied $metricsCopied metrics workbooks; expected 79."
}
if ($iscellCopied -ne 78) {
    throw "Copied $iscellCopied iscell.npy files; expected 78."
}

$finalMetrics = @(
    Get-ChildItem `
        -LiteralPath $destinationRoot `
        -Recurse `
        -Filter "*_metrics.xlsx" `
        -File
)
$finalIscell = @(
    Get-ChildItem `
        -LiteralPath $destinationRoot `
        -Recurse `
        -Filter "iscell.npy" `
        -File
)

[pscustomobject]@{
    SourceRoot = $sourceRoot
    DestinationRoot = $destinationRoot
    RenamedLateIOBPFolders = $renamedFolders
    AlreadyNormalizedLateIOBPFolders = $alreadyNormalizedFolders
    HistoricalRecordingsCopied = $recordingsCopied
    MetricsWorkbooksCopied = $metricsCopied
    IscellFilesCopied = $iscellCopied
    BytesCopied = $totalBytesCopied
    FinalMetricsWorkbooks = $finalMetrics.Count
    FinalIscellFiles = $finalIscell.Count
} | Format-List
