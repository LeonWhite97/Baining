[CmdletBinding()]
param(
    [string]$OutputDirectory
)

$ErrorActionPreference = "Stop"
$ProgressPreference = "SilentlyContinue"

if ([string]::IsNullOrWhiteSpace($OutputDirectory)) {
    $OutputDirectory = Join-Path $PSScriptRoot "..\..\data\external\pcb_stability_samples"
}

Add-Type -AssemblyName System.Drawing
Add-Type -AssemblyName System.Web

$samples = @(
    @{ Id = "pcb_01_canonscan"; Title = "File:CanoScan 5600F main PCB top view.jpg"; Use = "large_top_view" },
    @{ Id = "pcb_02_samsung_ssd"; Title = "File:Samsung SSD 860 Pro internal top.jpg"; Use = "dense_top_view" },
    @{ Id = "pcb_03_acer_netbook"; Title = "File:Acer Aspire One ZG5 motherboard DA0ZG5MB8F0 - top view.jpg"; Use = "large_irregular_board" },
    @{ Id = "pcb_04_nikon_d80"; Title = "File:Nikon D80 mainboard.jpg"; Use = "dense_irregular_board" },
    @{ Id = "pcb_05_lg_front"; Title = "File:LG VN251S Motherboard Front.jpg"; Use = "small_board_front" },
    @{ Id = "pcb_06_lg_back"; Title = "File:LG VN251S Motherboard Back.jpg"; Use = "small_board_back" },
    @{ Id = "pcb_07_cat_front"; Title = "File:CAT-B25-Motherboard-Front-FL.jpg"; Use = "phone_board_front" },
    @{ Id = "pcb_08_cat_back"; Title = "File:CAT-B25-Motherboard-Back-FL.jpg"; Use = "phone_board_back" },
    @{ Id = "pcb_09_atmega"; Title = "File:Simple ATmega325 Development Board (8414307586).jpg"; Use = "sparse_top_view" },
    @{ Id = "pcb_10_wd_controller"; Title = "File:WD Blue WD40NMZW-11GX6S1 - controller - Marvell 88I1047-NDB2-9724.jpg"; Use = "square_top_view" }
)

function ConvertFrom-HtmlText {
    param([AllowNull()][string]$Value)

    if ([string]::IsNullOrWhiteSpace($Value)) {
        return ""
    }

    $withoutTags = [regex]::Replace($Value, "<[^>]+>", " ")
    return [System.Web.HttpUtility]::HtmlDecode($withoutTags).Trim() -replace "\s+", " "
}

function Get-MetadataValue {
    param($Metadata, [string]$Name)

    $property = $Metadata.PSObject.Properties[$Name]
    if ($null -eq $property -or $null -eq $property.Value) {
        return ""
    }
    return [string]$property.Value.value
}

function Export-NormalizedJpeg {
    param(
        [Parameter(Mandatory)][string]$SourcePath,
        [Parameter(Mandatory)][string]$DestinationPath
    )

    $source = [System.Drawing.Image]::FromFile($SourcePath)
    try {
        $canvas = New-Object System.Drawing.Bitmap 1920, 1080
        try {
            $canvas.SetResolution(96, 96)
            $graphics = [System.Drawing.Graphics]::FromImage($canvas)
            try {
                $graphics.Clear([System.Drawing.Color]::Black)
                $graphics.CompositingQuality = [System.Drawing.Drawing2D.CompositingQuality]::HighQuality
                $graphics.InterpolationMode = [System.Drawing.Drawing2D.InterpolationMode]::HighQualityBicubic
                $graphics.SmoothingMode = [System.Drawing.Drawing2D.SmoothingMode]::HighQuality
                $graphics.PixelOffsetMode = [System.Drawing.Drawing2D.PixelOffsetMode]::HighQuality

                $scale = [Math]::Min(1920.0 / $source.Width, 1080.0 / $source.Height)
                $width = [Math]::Max(1, [int][Math]::Round($source.Width * $scale))
                $height = [Math]::Max(1, [int][Math]::Round($source.Height * $scale))
                $x = [int][Math]::Floor((1920 - $width) / 2)
                $y = [int][Math]::Floor((1080 - $height) / 2)
                $graphics.DrawImage($source, $x, $y, $width, $height)
            }
            finally {
                $graphics.Dispose()
            }

            $jpegCodec = [System.Drawing.Imaging.ImageCodecInfo]::GetImageEncoders() |
                Where-Object MimeType -eq "image/jpeg"
            $encoderParameters = New-Object System.Drawing.Imaging.EncoderParameters 1
            $encoderParameters.Param[0] = New-Object System.Drawing.Imaging.EncoderParameter(
                [System.Drawing.Imaging.Encoder]::Quality,
                [long]92
            )
            try {
                $canvas.Save($DestinationPath, $jpegCodec, $encoderParameters)
            }
            finally {
                $encoderParameters.Dispose()
            }
        }
        finally {
            $canvas.Dispose()
        }
    }
    finally {
        $source.Dispose()
    }
}

$sourceDirectory = Join-Path $OutputDirectory "source"
$normalizedDirectory = Join-Path $OutputDirectory "normalized_1920x1080"
New-Item -ItemType Directory -Force -Path $sourceDirectory, $normalizedDirectory | Out-Null

$manifest = foreach ($sample in $samples) {
    $apiParameters = @{
        action = "query"
        format = "json"
        prop = "imageinfo"
        iiprop = "url|size|mime|extmetadata"
        iiurlwidth = "1920"
        titles = $sample.Title
    }
    $apiUrl = "https://commons.wikimedia.org/w/api.php?" + (($apiParameters.GetEnumerator() | ForEach-Object {
        "{0}={1}" -f [uri]::EscapeDataString($_.Key), [uri]::EscapeDataString($_.Value)
    }) -join "&")

    $response = Invoke-RestMethod -Uri $apiUrl -Method Get
    $page = $response.query.pages.PSObject.Properties.Value | Select-Object -First 1
    if ($null -eq $page.imageinfo) {
        throw "Wikimedia Commons metadata not found: $($sample.Title)"
    }

    $image = $page.imageinfo[0]
    $extension = [System.IO.Path]::GetExtension(([uri]$image.thumburl).AbsolutePath)
    if ([string]::IsNullOrWhiteSpace($extension)) {
        $extension = ".jpg"
    }
    $sourcePath = Join-Path $sourceDirectory ($sample.Id + $extension.ToLowerInvariant())
    $normalizedPath = Join-Path $normalizedDirectory ($sample.Id + ".jpg")

    if (-not (Test-Path $sourcePath)) {
        try {
            Invoke-WebRequest -Uri $image.thumburl -OutFile $sourcePath
        }
        catch {
            throw "Failed to download $($sample.Title). Wikimedia may be rate-limiting media downloads. Retry later or place the licensed image at '$sourcePath'. $($_.Exception.Message)"
        }
    }
    Export-NormalizedJpeg -SourcePath $sourcePath -DestinationPath $normalizedPath

    $downloadedImage = [System.Drawing.Image]::FromFile($sourcePath)
    try {
        $sourceWidth = $downloadedImage.Width
        $sourceHeight = $downloadedImage.Height
    }
    finally {
        $downloadedImage.Dispose()
    }

    $metadata = $image.extmetadata
    [PSCustomObject]@{
        id = $sample.Id
        intended_use = $sample.Use
        title = $page.title -replace "^File:", ""
        source_file = (Resolve-Path -Relative $sourcePath) -replace "\\", "/"
        normalized_file = (Resolve-Path -Relative $normalizedPath) -replace "\\", "/"
        source_width = $sourceWidth
        source_height = $sourceHeight
        quality_tier = if ([Math]::Max($sourceWidth, $sourceHeight) -ge 900) { "visual_baseline" } else { "low_resolution_pipeline_stress" }
        normalized_width = 1920
        normalized_height = 1080
        mime_type = $image.mime
        source_page = $image.descriptionurl
        download_url = $image.thumburl
        author = ConvertFrom-HtmlText (Get-MetadataValue $metadata "Artist")
        attribution = ConvertFrom-HtmlText (Get-MetadataValue $metadata "Attribution")
        license = Get-MetadataValue $metadata "LicenseShortName"
        license_url = Get-MetadataValue $metadata "LicenseUrl"
        source_sha256 = (Get-FileHash -Algorithm SHA256 -Path $sourcePath).Hash.ToLowerInvariant()
        normalized_sha256 = (Get-FileHash -Algorithm SHA256 -Path $normalizedPath).Hash.ToLowerInvariant()
    }
}

$manifest | ConvertTo-Json -Depth 5 | Set-Content -Encoding UTF8 (Join-Path $OutputDirectory "manifest.json")
$manifest | Export-Csv -NoTypeInformation -Encoding UTF8 (Join-Path $OutputDirectory "manifest.csv")

Write-Host "Prepared $($manifest.Count) PCB photos in $OutputDirectory"
Write-Host "Normalized copies: $normalizedDirectory"
